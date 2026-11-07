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
    xqb: Dp, xdb: Dp, xsb: Dp, part: Dp,
    kc: Vec<Dp>, vc: Vec<Dp>, conv: Vec<Dp>, pos: usize, ctx_len: usize,
    bx: Dp, bh: Dp, bq: Dp, bk: Dp, bv: Dp, bg: Dp, bo: Dp, bt1: Dp, bt2: Dp, bu: Dp, ba: Dp, bxq: Dp, bxd: Dp, bxs: Dp, boff: Dp,
}
const CH: usize = 512;

impl DenseGpu {
    pub fn new(m: Dense, ctx_len: usize) -> DenseGpu {
        let ctx = CudaContext::new(0).unwrap_or_else(|e| { eprintln!("lastengine: no usable CUDA device: {e:?}"); std::process::exit(1) });
        let arch = cuda_arch(&ctx).unwrap_or_else(|e| { eprintln!("lastengine: {e}"); std::process::exit(1) });
        let st = ctx.default_stream();
        let inc = ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include/cuda"].iter().find(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists()).map(|s| s.to_string());
        let opts = CompileOptions { arch: Some(arch), include_paths: inc.into_iter().collect(), ..Default::default() };
        let ptx = compile_ptx_with_opts(kernel_source(), opts).unwrap_or_else(|e| panic!("nvrtc: {e:?}"));
        let module = ctx.load_module(ptx).expect("module");
        let mut f = HashMap::new();
        for n in ["quant_x", "matvec_8", "matvec_12", "matvec_14", "matvec_11", "matvec_13", "matvec_2", "rmsnorm", "add_inplace", "swiglu", "dn_prep", "dn_cvt_half", "dn_attn", "dn_flash_128", "dn_attn_split", "dn_attn_comb", "dn_shortconv", "dn_shortconv_b", "rmsnorm_b", "mmq_2", "mmq_8", "mmq_11", "mmq_12", "mmq_13", "mmq_14"] {
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
        let wide_cols = m.n_embd.max(m.n_head * hd).max(ffn);
        let fb = |n: usize| dalloc(n * 4);
        DenseGpu {
            _ctx: ctx, st, f, out_norm: vecd("output_norm.weight"), out_w, ff,
            x: fb(m.n_embd), h: fb(m.n_embd), q: fb(m.n_head * hd), k: fb(nkv * hd), v: fb(nkv * hd), g: fb(big), o: fb(m.n_head * hd), t1: fb(big), t2: fb(m.n_embd), u: fb(ffn), a: fb(ffn), logits: fb(m.vocab),
            part: dalloc(m.n_head * 128 * (hd + 2) * 4), xqb: dalloc(big.max(32768)), xdb: dalloc(big / 8 + 64), xsb: dalloc(big / 8 + 64),
            kc, vc, conv, pos: 0, ctx_len, lay,
            bx: dalloc(CH * m.n_embd * 4), bh: dalloc(CH * m.n_embd * 4), bq: dalloc(CH * m.n_head * hd * 4), bk: dalloc(CH * nkv * hd * 4), bv: dalloc(CH * nkv * hd * 4),
            bg: dalloc(CH * (3 * m.n_embd).max(m.n_head) * 4), bo: dalloc(CH * m.n_head * hd * 4), bt1: dalloc(CH * m.n_embd * 4), bt2: dalloc(CH * m.n_embd * 4), bu: dalloc(CH * ffn * 4), ba: dalloc(CH * ffn * 4),
            bxq: dalloc(CH * wide_cols), bxd: dalloc(CH * wide_cols / 8), bxs: dalloc(CH * wide_cols / 8), boff: dalloc(8), m,
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


    fn launch3(&self, name: &str, grid: (u32, u32, u32), block: u32, args: &[A]) {
        let mut b = self.st.launch_builder(&self.f[name]);
        for a in args { match a { P(p) => { b.arg(p); } I(i) => { b.arg(i); } F(x) => { b.arg(x); } } }
        unsafe { b.launch(LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 }).expect("launch"); }
    }
    fn launch3s(&self, name: &str, grid: (u32, u32, u32), block: u32, smem: u32, args: &[A]) {
        let mut b = self.st.launch_builder(&self.f[name]);
        for a in args { match a { P(p) => { b.arg(p); } I(i) => { b.arg(i); } F(x) => { b.arg(x); } } }
        unsafe { b.launch(LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: smem }).expect("launch"); }
    }
    /// Quantise `t` rows of `cols` floats from `x` for the batched GEMMs.
    fn bqx(&self, x: Dp, t: usize, cols: usize) { let u = t * cols / 32; self.launch("quant_x", u.div_ceil(8) as u32, 256, 0, &[P(x), P(self.bxq), P(self.bxd), P(self.bxs), I(u as i32)]); }
    /// out[t][rows] = W · quantised x[t][cols] (int8 tensor-core path).
    fn bmm(&self, w: W, out: Dp, t: usize) {
        let name = match w.ty { 2 => "mmq_2", 8 => "mmq_8", 11 => "mmq_11", 12 => "mmq_12", 13 => "mmq_13", 14 => "mmq_14", t => panic!("dense gpu: no mmq kernel for type {t}") };
        self.launch3(name, (w.rows.div_ceil(64) as u32, t.div_ceil(64) as u32, 1), 256, &[P(w.cell), P(self.boff), P(0), P(0), P(self.bxq), P(self.bxd), P(self.bxs), P(out), I(w.rows as i32), I(w.cols as i32)]);
    }
    fn brms(&self, x: Dp, w: Dp, o: Dp, t: usize) { self.launch("rmsnorm_b", t as u32, 256, 0, &[P(x), P(w), P(o), I(self.m.n_embd as i32), F(self.m.eps)]); }
    /// Process up to CH tokens as one batch; returns the logits of the last one.
    fn chunk(&mut self, ids: &[u32], want: bool) -> Vec<f32> {
        let t = ids.len(); assert!(t > 0 && t <= CH && self.pos + t <= self.ctx_len, "dense gpu: context full");
        let m = &self.m; let (n, hd, nh, nkv) = (m.n_embd, m.head_dim, m.n_head, m.n_kv);
        let mut e = vec![0f32; t * n];
        { let (ty, b) = m.bytes("token_embd.weight"); let rb = crate::quant::row_bytes(ty.ty, n);
          for (i, &id) in ids.iter().enumerate() { crate::quant::dequant_row(ty.ty, &b[id as usize * rb..(id as usize + 1) * rb], &mut e[i * n..(i + 1) * n]); } }
        unsafe { result::memcpy_htod_sync(self.bx, &e).unwrap(); result::memcpy_htod_sync(self.boff, &[0i32, t as i32]).unwrap(); }
        let lfm = m.kind == Kind::Lfm3; let pos0 = self.pos;
        for l in 0..m.n_layer {
            let d = &self.lay[l];
            self.brms(self.bx, d.attn_norm, self.bh, t);
            self.bqx(self.bh, t, n);
            if d.conv {
                self.bmm(d.in_proj, self.bg, t);
                self.launch("dn_shortconv_b", n.div_ceil(128) as u32, 128, 0, &[P(self.bg), P(d.conv_w), P(self.conv[l]), P(self.bt1), I(n as i32), I(m.l_cache as i32), I(t as i32)]);
                self.bqx(self.bt1, t, n);
                self.bmm(d.out_proj, self.bt2, t);
            } else {
                self.bmm(d.wq, self.bq, t); self.bmm(d.wk, self.bk, t); self.bmm(d.wv, self.bv, t);
                if lfm { self.bmm(d.wg, self.bg, t); }
                let base = if d.swa { m.rope_base_swa } else { m.rope_base };
                let ff = if d.swa { 0 } else { self.ff };
                let neox = m.neox as i32;
                let kslot = self.kc[l] + (pos0 * nkv * hd * 2) as u64;
                self.launch3("dn_prep", (nh as u32, t as u32, 1), 128, &[P(self.bq), P(self.bq), I(0), I(hd as i32), I(m.rope_dim as i32), I(pos0 as i32), F(base), P(ff), I(neox), P(d.qn), F(m.eps)]);
                self.launch3("dn_prep", (nkv as u32, t as u32, 1), 128, &[P(self.bk), P(kslot), I(1), I(hd as i32), I(m.rope_dim as i32), I(pos0 as i32), F(base), P(ff), I(neox), P(d.kn), F(m.eps)]);
                self.launch("dn_cvt_half", (t * nkv * hd).div_ceil(128) as u32, 128, 0, &[P(self.bv), P(self.vc[l] + (pos0 * nkv * hd * 2) as u64), I((t * nkv * hd) as i32)]);
                let win = if d.swa { m.window } else { 0 };
                if hd == 128 && std::env::var("LE_DN_FLASH").map_or(true, |v| v != "0") {
                    let smem = ((64 + 32) * (hd + 8) + hd * 40) * 2;
                    self.launch3s("dn_flash_128", (t.div_ceil(64) as u32, nh as u32, 1), 128, smem as u32, &[P(self.bq), P(self.kc[l]), P(self.vc[l]), P(if lfm { self.bg } else { 0 }), P(self.bo), I(pos0 as i32), I(t as i32), I(nh as i32), I(nkv as i32), I(win as i32)]);
                } else {
                self.launch3("dn_attn", (nh as u32, t as u32, 1), 256, &[P(self.bq), P(self.kc[l]), P(self.vc[l]), P(if lfm { self.bg } else { 0 }), P(self.bo), I(win as i32), I((pos0 + 1) as i32), I(nh as i32), I(nkv as i32), I(hd as i32)]);
                }
                self.bqx(self.bo, t, nh * hd);
                self.bmm(d.wo, self.bt2, t);
            }
            if lfm { self.brms(self.bt2, d.post_attn, self.bt2, t); }
            self.add(self.bx, self.bt2, t * n);
            self.brms(self.bx, d.ffn_norm, self.bh, t);
            self.bqx(self.bh, t, n);
            self.bmm(d.gate, self.ba, t); self.bmm(d.up, self.bu, t);
            let ffn = d.gate.rows;
            self.launch("swiglu", (t * ffn).div_ceil(256) as u32, 256, 0, &[P(self.ba), P(self.bu), P(self.ba), I((t * ffn) as i32)]);
            self.bqx(self.ba, t, ffn);
            self.bmm(d.down, self.bt2, t);
            if lfm { self.brms(self.bt2, d.post_ffw, self.bt2, t); }
            self.add(self.bx, self.bt2, t * n);
        }
        self.pos += t;
        if !want { unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap() }; return Vec::new(); }
        unsafe { result::memcpy_dtod_async(self.x, self.bx + ((t - 1) * n * 4) as u64, n * 4, self.st.cu_stream()).unwrap() };
        self.rms(self.x, self.out_norm, self.h, n);
        self.mv(self.out_w, self.h, self.logits);
        let mut out = vec![0f32; m.vocab];
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut out, self.logits).unwrap() };
        out
    }
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
                let win = if d.swa { m.window } else { 0 };
                let span = if win > 0 { nk.min(win) } else { nk };
                if hd == 128 && span >= std::env::var("LE_DN_SPLIT_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(512usize) && std::env::var("LE_DN_SPLIT").map_or(true, |v| v != "0") {
                    let sp = std::env::var("LE_DN_SP").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
                    self.launch3("dn_attn_split", (nh as u32, sp as u32, 1), 256, &[P(self.q), P(self.kc[l]), P(self.vc[l]), P(self.part), I(win as i32), I(nk as i32), I(nh as i32), I(nkv as i32), I(hd as i32), I(sp)]);
                    self.launch("dn_attn_comb", nh as u32, 128, 0, &[P(self.part), P(if lfm { self.g } else { 0 }), P(self.o), I(hd as i32), I(sp)]);
                } else {
                self.launch("dn_attn", nh as u32, 256, 0, &[P(self.q), P(self.kc[l]), P(self.vc[l]), P(if lfm { self.g } else { 0 }), P(self.o), I(win as i32), I(nk as i32), I(nh as i32), I(nkv as i32), I(hd as i32)]);
                }
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
        if std::env::var("LE_DENSE_SERIAL").is_ok() { for (i, &t) in ids.iter().enumerate() { l = self.step(t, i + 1 == ids.len()); } return l; }
        let mut i = 0;
        while i < ids.len() { let e = (i + CH).min(ids.len()); l = self.chunk(&ids[i..e], e == ids.len()); i = e; }
        l
    }
    pub fn export_state(&mut self) -> Vec<u8> { self.bind();
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap() };
        let (nkv, hd) = (self.m.n_kv, self.m.head_dim);
        let kvb = self.pos * nkv * hd * 2; let cb = self.m.n_embd * (self.m.l_cache.max(1) - 1) * 4;
        let mut out = (self.pos as u64).to_le_bytes().to_vec();
        let rd = |p: Dp, n: usize, out: &mut Vec<u8>| { let o = out.len(); out.resize(o + n, 0); if n > 0 { unsafe { result::memcpy_dtoh_sync(&mut out[o..], p).unwrap() } } };
        for l in 0..self.lay.len() { if self.lay[l].conv { rd(self.conv[l], cb, &mut out); } else { rd(self.kc[l], kvb, &mut out); rd(self.vc[l], kvb, &mut out); } }
        out
    }
    pub fn import_state(&mut self, b: &[u8]) -> Result<(), String> { self.bind();
        if b.len() < 8 { return Err("slot file too short".into()); }
        let pos = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        if pos > self.ctx_len { return Err(format!("slot holds {pos} tokens but the context is {}", self.ctx_len)); }
        let (nkv, hd) = (self.m.n_kv, self.m.head_dim);
        let kvb = pos * nkv * hd * 2; let cb = self.m.n_embd * (self.m.l_cache.max(1) - 1) * 4;
        let need: usize = 8 + self.lay.iter().map(|l| if l.conv { cb } else { 2 * kvb }).sum::<usize>();
        if b.len() != need { return Err("slot file does not match this model".into()); }
        let mut o = 8;
        let mut wr = |p: Dp, n: usize| { if n > 0 { unsafe { result::memcpy_htod_sync(p, &b[o..o + n]).unwrap() } } o += n; };
        for l in 0..self.lay.len() { if self.lay[l].conv { wr(self.conv[l], cb); } else { wr(self.kc[l], kvb); wr(self.vc[l], kvb); } }
        self.pos = pos; Ok(())
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
