//! CUDA decode engine for the dense architectures (`llama`: MiniCPM5, `lfm3`). The whole model lives in VRAM,
//! one token at a time through the existing quantised matvec kernels plus the `dn_*` kernels in cuda/dense.cu.
use super::*;
use crate::dense::{Dense, Kind, LayerKind};

struct DL {
    attn_norm: Dp, ffn_norm: Dp, post_attn: Dp, post_ffw: Dp, swa: bool, conv: bool,
    wq: W, wk: W, wv: W, wo: W, wg: W, qn: Dp, kn: Dp, in_proj: W, out_proj: W, conv_w: Dp,
    gate: W, up: W, down: W,
}

pub struct DenseGpu {
    m: Dense, _ctx: Arc<CudaContext>, st: Arc<CudaStream>, f: HashMap<&'static str, CudaFunction>,
    lay: Vec<DL>, out_norm: Dp, out_w: W, ff: Dp,
    x: Dp, h: Dp, q: Dp, k: Dp, v: Dp, g: Dp, o: Dp, t1: Dp, t2: Dp, u: Dp, a: Dp, logits: Dp,
    xqb: Dp, xdb: Dp, xsb: Dp,
    kc: Vec<Dp>, vc: Vec<Dp>, conv: Vec<Dp>, pos: usize, ctx_len: usize,
}

impl DenseGpu {
    pub fn new(m: Dense, ctx_len: usize) -> DenseGpu {
        let ctx = CudaContext::new(0).expect("cuda");
        let st = ctx.default_stream();
        let inc = ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include/cuda"].iter().find(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists()).map(|s| s.to_string());
        let opts = CompileOptions { arch: Some("sm_86"), include_paths: inc.into_iter().collect(), ..Default::default() };
        let ptx = compile_ptx_with_opts(kernel_source(), opts).unwrap_or_else(|e| panic!("nvrtc: {e:?}"));
        let module = ctx.load_module(ptx).expect("module");
        let mut f = HashMap::new();
        for n in ["quant_x", "matvec_8", "matvec_12", "matvec_14", "matvec_11", "matvec_13", "matvec_2", "rmsnorm", "add_inplace", "swiglu", "dn_prep", "dn_cvt_half", "dn_attn", "dn_shortconv"] {
            f.insert(n, module.load_function(n).unwrap_or_else(|e| panic!("{n}: {e:?}")));
        }
        let mat = |name: &str| -> W { let (t, b) = m.bytes(name); let d = dup(b); W { cell: dup(&[d]), ty: t.ty, rows: t.dims[1], cols: t.dims[0], raw: d } };
        let none = W { cell: 0, ty: 0, rows: 0, cols: 0, raw: 0 };
        let vecd = |name: &str| -> Dp { dup(&m.vec_f32(name)) };
        let has = |n: &str| m.t.contains_key(n);
        let lfm = m.kind == Kind::Lfm3;
        let (nkv, hd) = (m.n_kv, m.head_dim);
        let mut lay = Vec::new(); let (mut kc, mut vc, mut conv) = (vec![], vec![], vec![]);
        for l in 0..m.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let is_conv = m.layers[l] == LayerKind::Conv;
            let swa = matches!(m.layers[l], LayerKind::Attn { swa: true });
            let mut d = DL { attn_norm: vecd(&p("attn_norm.weight")), ffn_norm: vecd(&p("ffn_norm.weight")), post_attn: 0, post_ffw: 0, swa, conv: is_conv,
                wq: none, wk: none, wv: none, wo: none, wg: none, qn: 0, kn: 0, in_proj: none, out_proj: none, conv_w: 0,
                gate: mat(&p("ffn_gate.weight")), up: mat(&p("ffn_up.weight")), down: mat(&p("ffn_down.weight")) };
            if lfm { d.post_attn = vecd(&p("post_attention_norm.weight")); d.post_ffw = vecd(&p("post_ffw_norm.weight")); }
            if is_conv {
                d.in_proj = mat(&p("shortconv.in_proj.weight")); d.out_proj = mat(&p("shortconv.out_proj.weight")); d.conv_w = vecd(&p("shortconv.conv.weight"));
                let n = m.n_embd * (m.l_cache - 1) * 4; let c = dalloc(n); dzero(c, n); conv.push(c); kc.push(0); vc.push(0);
            } else {
                d.wq = mat(&p("attn_q.weight")); d.wk = mat(&p("attn_k.weight")); d.wv = mat(&p("attn_v.weight")); d.wo = mat(&p("attn_output.weight"));
                if lfm { d.wg = mat(&p("attn_gate.weight")); d.qn = vecd(&p("attn_q_norm.weight")); d.kn = vecd(&p("attn_k_norm.weight")); }
                conv.push(0); kc.push(0); vc.push(0);
            }
            lay.push(d);
        }
        let out_w = mat(if has("output.weight") { "output.weight" } else { "token_embd.weight" });
        // KV caches (f16) sized to whatever VRAM is left, keeping 128 MiB spare.
        let n_attn = lay.iter().filter(|l| !l.conv).count();
        let (free, _) = result::mem_get_info().expect("meminfo");
        let per_tok = n_attn * nkv * hd * 2 * 2;
        let fit = if per_tok == 0 { ctx_len } else { free.saturating_sub(160 << 20) / per_tok };
        let ctx_len = ctx_len.min(fit).max(256);
        eprintln!("dense gpu: {} layers, kv ctx {} ({} MiB free after weights)", m.n_layer, ctx_len, free >> 20);
        for l in 0..m.n_layer { if !lay[l].conv { let n = ctx_len * nkv * hd * 2; kc[l] = dalloc(n); vc[l] = dalloc(n); } }
        let ff = if has("rope_freqs.weight") { vecd("rope_freqs.weight") } else { 0 };
        let wide = m.n_embd.max(m.n_head * hd).max(8192 * 2);
        let ffn = lay[0].gate.rows;
        let big = wide.max(ffn * 2).max(m.n_embd * 3);
        let fb = |n: usize| dalloc(n * 4);
        DenseGpu {
            _ctx: ctx, st, f, out_norm: vecd("output_norm.weight"), out_w, ff,
            x: fb(m.n_embd), h: fb(m.n_embd), q: fb(m.n_head * hd), k: fb(nkv * hd), v: fb(nkv * hd), g: fb(big), o: fb(m.n_head * hd), t1: fb(big), t2: fb(m.n_embd), u: fb(ffn), a: fb(ffn), logits: fb(m.vocab),
            xqb: dalloc(big.max(32768)), xdb: dalloc(big / 8 + 64), xsb: dalloc(big / 8 + 64),
            kc, vc, conv, pos: 0, ctx_len, lay, m,
        }
    }
    fn launch(&self, name: &str, grid: u32, block: u32, smem: u32, args: &[A]) {
        let mut b = self.st.launch_builder(&self.f[name]);
        for a in args { match a { P(p) => { b.arg(p); } I(i) => { b.arg(i); } F(x) => { b.arg(x); } } }
        unsafe { b.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem }).expect("launch"); }
    }
    fn qx(&self, x: Dp, n: usize) { let u = n / 32; self.launch("quant_x", u.div_ceil(8) as u32, 256, 0, &[P(x), P(self.xqb), P(self.xdb), P(self.xsb), I(u as i32)]); }
    /// out = W · x, with x quantised first.
    fn mv(&self, w: W, x: Dp, out: Dp) {
        self.qx(x, w.cols); self.mvq(w, out);
    }
    /// out = W · (already quantised x)
    fn mvq(&self, w: W, out: Dp) {
        let name = match w.ty { 2 => "matvec_2", 8 => "matvec_8", 11 => "matvec_11", 12 => "matvec_12", 13 => "matvec_13", 14 => "matvec_14", t => panic!("dense gpu: no matvec kernel for type {t}") };
        let units = w.cols / 32; let lpr = units.min(32); let rpb = 8 * (32 / lpr);
        self.launch(name, w.rows.div_ceil(rpb) as u32, 256, (units * 36 + units * 8) as u32,
            &[P(w.cell), P(self.xqb), P(self.xdb), P(self.xsb), P(out), I(w.rows as i32), I(w.cols as i32), I(0)]);
    }
    fn rms(&self, x: Dp, w: Dp, o: Dp, n: usize) { self.launch("rmsnorm", 1, 256, 0, &[P(x), P(w), P(o), I(n as i32), F(self.m.eps)]); }
    fn add(&self, x: Dp, y: Dp, n: usize) { self.launch("add_inplace", n.div_ceil(256) as u32, 256, 0, &[P(x), P(y), I(n as i32)]); }

    fn bind(&self) { self._ctx.bind_to_thread().expect("bind"); }
    pub fn pos(&self) -> usize { self.pos }
    pub fn reset(&mut self) { self.bind();
        self.pos = 0;
        for l in 0..self.lay.len() { if self.lay[l].conv { dzero(self.conv[l], self.m.n_embd * (self.m.l_cache - 1) * 4); } }
    }
    fn step(&mut self, token: u32, want: bool) -> Vec<f32> {
        assert!(self.pos < self.ctx_len, "dense gpu: context full");
        let m = &self.m; let (n, hd, nh, nkv) = (m.n_embd, m.head_dim, m.n_head, m.n_kv);
        let pos = self.pos as i32;
        let mut e = vec![0f32; n];
        { let (t, b) = m.bytes("token_embd.weight"); let rb = crate::quant::row_bytes(t.ty, n); crate::quant::dequant_row(t.ty, &b[token as usize * rb..(token as usize + 1) * rb], &mut e); }
        unsafe { result::memcpy_htod_sync(self.x, &e).unwrap() };
        let lfm = m.kind == Kind::Lfm3;
        for l in 0..m.n_layer {
            let d = &self.lay[l];
            self.rms(self.x, d.attn_norm, self.h, n);
            if d.conv {
                self.mv(d.in_proj, self.h, self.g);
                self.launch("dn_shortconv", n.div_ceil(128) as u32, 128, 0, &[P(self.g), P(d.conv_w), P(self.conv[l]), P(self.t1), I(n as i32), I(m.l_cache as i32)]);
                self.mv(d.out_proj, self.t1, self.t2);
            } else {
                self.qx(self.h, n);
                self.mvq(d.wq, self.q); self.mvq(d.wk, self.k); self.mvq(d.wv, self.v);
                if lfm { self.mvq(d.wg, self.g); }
                let base = if d.swa { m.rope_base_swa } else { m.rope_base };
                let ff = if d.swa { 0 } else { self.ff };
                let neox = m.neox as i32;
                let kslot = self.kc[l] + (self.pos * nkv * hd * 2) as u64;
                self.launch("dn_prep", nh as u32, 128, 0, &[P(self.q), P(self.q), I(0), I(hd as i32), I(m.rope_dim as i32), I(pos), F(base), P(ff), I(neox), P(d.qn), F(m.eps)]);
                self.launch("dn_prep", nkv as u32, 128, 0, &[P(self.k), P(kslot), I(1), I(hd as i32), I(m.rope_dim as i32), I(pos), F(base), P(ff), I(neox), P(d.kn), F(m.eps)]);
                self.launch("dn_cvt_half", (nkv * hd).div_ceil(128) as u32, 128, 0, &[P(self.v), P(self.vc[l] + (self.pos * nkv * hd * 2) as u64), I((nkv * hd) as i32)]);
                let nk = self.pos + 1;
                let t0 = if d.swa && m.window > 0 && nk > m.window { nk - m.window } else { 0 };
                self.launch("dn_attn", nh as u32, 256, 0, &[P(self.q), P(self.kc[l]), P(self.vc[l]), P(if lfm { self.g } else { 0 }), P(self.o), I(t0 as i32), I(nk as i32), I(nh as i32), I(nkv as i32), I(hd as i32)]);
                self.mv(d.wo, self.o, self.t2);
            }
            if lfm { self.rms(self.t2, d.post_attn, self.t2, n); }
            self.add(self.x, self.t2, n);
            self.rms(self.x, d.ffn_norm, self.h, n);
            self.qx(self.h, n);
            self.mvq(d.gate, self.a); self.mvq(d.up, self.u);
            let ffn = d.gate.rows;
            self.launch("swiglu", ffn.div_ceil(256) as u32, 256, 0, &[P(self.a), P(self.u), P(self.a), I(ffn as i32)]);
            self.mv(d.down, self.a, self.t2);
            if lfm { self.rms(self.t2, d.post_ffw, self.t2, n); }
            self.add(self.x, self.t2, n);
        }
        self.pos += 1;
        if !want { return Vec::new(); }
        self.rms(self.x, self.out_norm, self.h, n);
        self.mv(self.out_w, self.h, self.logits);
        let mut out = vec![0f32; m.vocab];
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut out, self.logits).unwrap() };
        out
    }
    pub fn forward(&mut self, t: u32) -> Vec<f32> { self.bind(); self.step(t, true) }
    pub fn prefill(&mut self, ids: &[u32]) -> Vec<f32> { self.bind();
        let mut l = vec![];
        for (i, &t) in ids.iter().enumerate() { l = self.step(t, i + 1 == ids.len()); }
        l
    }
    pub fn snap_recurrent(&mut self) -> Vec<u8> { self.bind();
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap() };
        let n = self.m.n_embd * (self.m.l_cache.max(1) - 1) * 4; let mut out = Vec::new();
        for l in 0..self.lay.len() { if self.lay[l].conv { let o = out.len(); out.resize(o + n, 0); unsafe { result::memcpy_dtoh_sync(&mut out[o..], self.conv[l]).unwrap() } } }
        out
    }
    pub fn restore_recurrent(&mut self, b: &[u8], pos: usize) { self.bind();
        let n = self.m.n_embd * (self.m.l_cache.max(1) - 1) * 4; let mut o = 0;
        for l in 0..self.lay.len() { if self.lay[l].conv && o + n <= b.len() { unsafe { result::memcpy_htod_sync(self.conv[l], &b[o..o + n]).unwrap() }; o += n; } }
        self.pos = pos;
    }
}
