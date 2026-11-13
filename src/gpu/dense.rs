//! CUDA decode engine for the dense architectures (`llama`: MiniCPM5, `lfm3`). The whole model lives in VRAM,
//! one token at a time through the existing quantised matvec kernels plus the `dn_*` kernels in cuda/dense.cu.
use super::*;
use crate::dense::{Dense, Kind, LayerKind};
use cudarc::driver::CudaGraph;
/// A captured graph is only ever launched by the thread that owns this DenseGpu (behind the engine's mutex).
pub struct GraphBox(CudaGraph);
unsafe impl Send for GraphBox {}

struct DL {
    attn_norm: Dp, ffn_norm: Dp, post_attn: Dp, post_ffw: Dp, swa: bool, conv: bool,
    wq: W, wk: W, wv: W, wo: W, wg: W, qn: Dp, kn: Dp, in_proj: W, out_proj: W, conv_w: Dp,
    gu: W, down: W, // gu = gate and up rows concatenated (rows = 2*ffn), one GEMM/matvec for both
}

pub struct DenseGpu {
    m: Dense, _ctx: Arc<CudaContext>, st: Arc<CudaStream>, f: HashMap<&'static str, CudaFunction>,
    lay: Vec<DL>, out_norm: Dp, out_w: W, ff: Dp,
    x: Dp, h: Dp, q: Dp, k: Dp, v: Dp, g: Dp, o: Dp, t1: Dp, t2: Dp, u: Dp, a: Dp, logits: Dp,
    xqb: Dp, xdb: Dp, xsb: Dp, part: Dp,
    /// Device-resident [pos, pos+1] for the decode step: kernels read the position and KV length here, so launch args stay constant per token (graph-capturable).
    dpos: Dp, fused_qkv: bool, rmsq: bool,
    /// CUDA graphs of the decode layer loop, keyed by the per-layer split-K choice.
    graphs: HashMap<Vec<bool>, GraphBox>,
    kc: Vec<Dp>, vc: Vec<Dp>, conv: Vec<Dp>, pos: usize, ctx_len: usize,
    bx: Dp, bh: Dp, bq: Dp, bk: Dp, bv: Dp, bg: Dp, bo: Dp, bt1: Dp, bt2: Dp, bgu: Dp, ba: Dp, bxq: Dp, bxd: Dp, bxs: Dp, boff: Dp,
}
const CH: usize = 512;

/// Split-K knobs, read from the environment once per process (the decode loop used to re-read them every layer).
fn dn_split_env() -> (usize, bool, usize) {
    static V: std::sync::OnceLock<(usize, bool, usize)> = std::sync::OnceLock::new();
    *V.get_or_init(|| (
        std::env::var("LE_DN_SPLIT_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(512usize),
        std::env::var("LE_DN_SPLIT").map_or(true, |v| v != "0"),
        std::env::var("LE_DN_SP").ok().and_then(|v| v.parse().ok()).unwrap_or(32usize),
    ))
}

/// LE_PROF=1: per-kernel-name wall time with a sync after each launch (profiling only; distorts overlap).
fn prof_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| std::env::var("LE_PROF").is_ok()) }
static PROF: std::sync::Mutex<Vec<(String, f64, usize)>> = std::sync::Mutex::new(Vec::new());
fn prof_add(name: &str, secs: f64) { let mut v = PROF.lock().unwrap(); match v.iter_mut().find(|e| e.0 == name) { Some(e) => { e.1 += secs; e.2 += 1; } None => v.push((name.to_string(), secs, 1)) } }
fn prof_dump() { let mut v = PROF.lock().unwrap(); v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); let tot: f64 = v.iter().map(|e| e.1).sum(); for e in v.iter().take(25) { eprintln!("prof {:>22} {:>9.2} ms {:>6} calls {:>5.1}%", e.0, e.1 * 1e3, e.2, 100.0 * e.1 / tot); } eprintln!("prof total {:.2} ms", tot * 1e3); v.clear(); }


/// LE_EVT=1: per-launch CUDA events on the stream (no host sync); summed at end of prefill.
fn evt_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| std::env::var("LE_EVT").is_ok()) }
thread_local! { static NO_EVT: std::cell::Cell<bool> = std::cell::Cell::new(false); }
thread_local! { static DEC: std::cell::RefCell<std::collections::BTreeMap<String,(f64,usize)>> = std::cell::RefCell::new(Default::default()); static DEC_WALL: std::cell::RefCell<(f64,f64,f64,usize)> = std::cell::RefCell::new((0.0,0.0,0.0,0)); }
thread_local! { static EVT: std::cell::RefCell<Vec<(String, std::sync::Arc<cudarc::driver::CudaEvent>, std::sync::Arc<cudarc::driver::CudaEvent>)>> = std::cell::RefCell::new(Vec::new()); }

impl DenseGpu {
    pub fn new(m: Dense, ctx_len: usize) -> DenseGpu {
        let ctx = CudaContext::new(0).unwrap_or_else(|e| { eprintln!("lastengine: no usable CUDA device: {e:?}"); std::process::exit(1) });
        let arch = cuda_arch(&ctx).unwrap_or_else(|e| { eprintln!("lastengine: {e}"); std::process::exit(1) });
        let st = ctx.new_stream().expect("stream"); // non-blocking: the legacy default stream cannot be graph-captured
        let inc = ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include/cuda"].iter().find(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists()).map(|s| s.to_string());
        let opts = CompileOptions { arch: Some(arch), include_paths: inc.into_iter().collect(), ..Default::default() };
        let ptx = compile_ptx_with_opts(kernel_source(), opts).unwrap_or_else(|e| panic!("nvrtc: {e:?}"));
        let module = ctx.load_module(ptx).expect("module");
        let mut f = HashMap::new();
        for n in ["quant_x", "matvec_8", "matvec_12", "matvec_14", "matvec_11", "matvec_13", "matvec_2", "rmsnorm", "add_inplace", "swiglu", "swiglu_gu", "dn_prep", "dn_cvt_half", "dn_cvt_half_d", "dn_attn", "dn_flash_128", "dn_flash_128b", "dn_qkv_dec", "rms_quant", "zero_f", "mmq128r2_2", "mmq128r2_8", "mmq128r2_11", "mmq128r2_12", "mmq128r2_13", "mmq128r2_14", "dn_attn_split", "dn_attn_gqa", "dn_attn_comb", "dn_shortconv", "dn_shortconv_b", "rmsnorm_b", "mmq_2", "mmq_8", "mmq_11", "mmq_12", "mmq_13", "mmq_14", "mmq128_2", "mmq128_8", "mmq128_11", "mmq128_12", "mmq128_13", "mmq128_14", "mmq128r_2", "mmq128r_8", "mmq128r_11", "mmq128r_12", "mmq128r_13", "mmq128r_14"] {
            f.insert(n, module.load_function(n).unwrap_or_else(|e| panic!("{n}: {e:?}")));
        }
        let mat = |name: &str| -> W { let (t, b) = m.bytes(name); let d = dup(b); W { cell: dup(&[d]), ty: t.ty, rows: t.dims[1], cols: t.dims[0], raw: d } };
        // Gate and up share cols and quant type, so their row-major bytes concatenate into one matrix with 2*ffn rows.
        let gate_up = |g: &str, u: &str| -> W {
            let (tg, bg) = m.bytes(g); let (tu, bu) = m.bytes(u);
            assert!(tg.ty == tu.ty && tg.dims == tu.dims, "dense gpu: gate/up type or shape differ, cannot fuse ({g}, {u})");
            let mut cat = Vec::with_capacity(bg.len() + bu.len()); cat.extend_from_slice(bg); cat.extend_from_slice(bu);
            let d = dup(&cat);
            W { cell: dup(&[d]), ty: tg.ty, rows: 2 * tg.dims[1], cols: tg.dims[0], raw: d }
        };
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
                gu: gate_up(&p("ffn_gate.weight"), &p("ffn_up.weight")), down: mat(&p("ffn_down.weight")) };
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
        let ffn = lay[0].gu.rows / 2;
        let big = wide.max(ffn * 2).max(m.n_embd * 3);
        let wide_cols = m.n_embd.max(m.n_head * hd).max(ffn);
        let fb = |n: usize| dalloc(n * 4);
        DenseGpu {
            _ctx: ctx, st, f, out_norm: vecd("output_norm.weight"), out_w, ff,
            x: fb(m.n_embd), h: fb(m.n_embd), q: fb(m.n_head * hd), k: fb(nkv * hd), v: fb(nkv * hd), g: fb(big), o: fb(m.n_head * hd), t1: fb(big), t2: fb(m.n_embd), u: fb(ffn), a: fb(2 * ffn), logits: fb(m.vocab),
            part: dalloc(m.n_head * 128 * (hd + 2) * 4), dpos: dalloc(8), fused_qkv: std::env::var("LE_DN_QKV").map_or(true, |v| v != "0"), rmsq: std::env::var("LE_RMSQ").map_or(true, |v| v != "0"), graphs: HashMap::new(), xqb: dalloc(big.max(32768)), xdb: dalloc(big / 8 + 64), xsb: dalloc(big / 8 + 64),
            kc, vc, conv, pos: 0, ctx_len, lay,
            bx: dalloc(CH * m.n_embd * 4), bh: dalloc(CH * m.n_embd * 4), bq: dalloc(CH * m.n_head * hd * 4), bk: dalloc(CH * nkv * hd * 4), bv: dalloc(CH * nkv * hd * 4),
            bg: dalloc(CH * (3 * m.n_embd).max(m.n_head) * 4), bo: dalloc(CH * m.n_head * hd * 4), bt1: dalloc(CH * m.n_embd * 4), bt2: dalloc(CH * m.n_embd * 4), bgu: dalloc(CH * 2 * ffn * 4), ba: dalloc(CH * ffn * 4),
            bxq: dalloc(CH * wide_cols), bxd: dalloc(CH * wide_cols / 8), bxs: dalloc(CH * wide_cols / 8), boff: dalloc(8), m,
        }
    }
    fn ev_begin(&self) -> Option<std::sync::Arc<cudarc::driver::CudaEvent>> {
        if !evt_on() || NO_EVT.with(|c| c.get()) { return None; }
        let e = self._ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("event");
        e.record(&self.st).expect("record"); Some(std::sync::Arc::new(e))
    }
    fn ev_end(&self, name: &str, start: std::sync::Arc<cudarc::driver::CudaEvent>) {
        let end = self._ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("event");
        end.record(&self.st).expect("record");
        EVT.with(|v| v.borrow_mut().push((name.to_string(), start, std::sync::Arc::new(end))));
    }
    fn launch(&self, name: &str, grid: u32, block: u32, smem: u32, args: &[A]) {
        let mut b = self.st.launch_builder(&self.f[name]);
        for a in args { match a { P(p) => { b.arg(p); } I(i) => { b.arg(i); } F(x) => { b.arg(x); } } }
        let ev = self.ev_begin();
        let t0 = std::time::Instant::now();
        unsafe { b.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem }).expect("launch"); }
        if let Some(e) = ev { self.ev_end(name, e); }
        if prof_on() { self.st.synchronize().unwrap(); prof_add(name, t0.elapsed().as_secs_f64()); }
    }
    fn rms_q(&self, x: Dp, w: Dp, o: Dp, n: usize) { self.launch("rms_quant", (n / 32).div_ceil(8).min(64) as u32, 256, 0, &[P(x), P(w), P(o), P(self.xqb), P(self.xdb), P(self.xsb), I(n as i32), F(self.m.eps)]); }
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
        let ev = self.ev_begin();
        let t0 = std::time::Instant::now();
        unsafe { b.launch(LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 }).expect("launch"); }
        if let Some(e) = ev { self.ev_end(name, e); }
        if prof_on() { self.st.synchronize().unwrap(); prof_add(name, t0.elapsed().as_secs_f64()); }
    }
    fn launch3s(&self, name: &str, grid: (u32, u32, u32), block: u32, smem: u32, args: &[A]) {
        let mut b = self.st.launch_builder(&self.f[name]);
        for a in args { match a { P(p) => { b.arg(p); } I(i) => { b.arg(i); } F(x) => { b.arg(x); } } }
        let ev = self.ev_begin();
        let t0 = std::time::Instant::now();
        unsafe { b.launch(LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: smem }).expect("launch"); }
        if let Some(e) = ev { self.ev_end(name, e); }
        if prof_on() { self.st.synchronize().unwrap(); prof_add(name, t0.elapsed().as_secs_f64()); }
    }
    /// Quantise `t` rows of `cols` floats from `x` for the batched GEMMs.
    fn bqx(&self, x: Dp, t: usize, cols: usize) { let u = t * cols / 32; self.launch("quant_x", u.div_ceil(8) as u32, 256, 0, &[P(x), P(self.bxq), P(self.bxd), P(self.bxs), I(u as i32)]); }
    /// out[t][rows] = W · quantised x[t][cols] (int8 tensor-core path).
    fn bmm(&self, w: W, out: Dp, t: usize) {
        let wide = std::env::var("LE_MMQ128").map_or(true, |v| v != "0");
        let name = match (w.ty, wide) { (2, false) => "mmq_2", (8, false) => "mmq_8", (11, false) => "mmq_11", (12, false) => "mmq_12", (13, false) => "mmq_13", (14, false) => "mmq_14",
            (2, true) => "mmq128_2", (8, true) => "mmq128_8", (11, true) => "mmq128_11", (12, true) => "mmq128_12", (13, true) => "mmq128_13", (14, true) => "mmq128_14",
            (t, _) => panic!("dense gpu: no mmq kernel for type {t}") };
        let tile = if wide { 128 } else { 64 };
        let rtile = wide && std::env::var("LE_MMQR").map_or(true, |v| v != "0");
        let r2 = std::env::var("LE_MMQR2").map_or(false, |v| v == "1");
        let name_r = format!("mmq128r{}_{}", if r2 { "2" } else { "" }, &name["mmq128_".len()..]);
        let (name, rrows) = if rtile { (name_r.as_str(), 128) } else { (name, 64) };
        // LE_MMQSK=S (opt-in, default 1): split-K over the 128-row kernel. Needs units % 4 == 0; output is zeroed then atomically accumulated.
        let sk = if rtile && w.cols % 128 == 0 { std::env::var("LE_MMQSK").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1).max(1) } else { 1 };
        if sk > 1 {
            self.launch("zero_f", (t * w.rows).div_ceil(256) as u32, 256, 0, &[P(out), I((t * w.rows) as i32)]);
            self.launch3(name, ((w.rows.div_ceil(rrows) * sk as usize) as u32, t.div_ceil(tile) as u32, 1), 256, &[P(w.cell), P(self.boff), P(0), P(0), P(self.bxq), P(self.bxd), P(self.bxs), P(out), I(w.rows as i32), I(w.cols as i32)]);
            return;
        }
        self.launch3(name, (w.rows.div_ceil(rrows) as u32, t.div_ceil(tile) as u32, 1), 256, &[P(w.cell), P(self.boff), P(0), P(0), P(self.bxq), P(self.bxd), P(self.bxs), P(out), I(w.rows as i32), I(w.cols as i32)]);
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
                self.launch3("dn_prep", (nh as u32, t as u32, 1), 128, &[P(self.bq), P(self.bq), I(0), I(hd as i32), I(m.rope_dim as i32), I(pos0 as i32), F(base), P(ff), I(neox), P(d.qn), F(m.eps), P(0), I(0)]);
                self.launch3("dn_prep", (nkv as u32, t as u32, 1), 128, &[P(self.bk), P(kslot), I(1), I(hd as i32), I(m.rope_dim as i32), I(pos0 as i32), F(base), P(ff), I(neox), P(d.kn), F(m.eps), P(0), I(0)]);
                self.launch("dn_cvt_half", (t * nkv * hd).div_ceil(128) as u32, 128, 0, &[P(self.bv), P(self.vc[l] + (pos0 * nkv * hd * 2) as u64), I((t * nkv * hd) as i32)]);
                let win = if d.swa { m.window } else { 0 };
                if hd == 128 && std::env::var("LE_DN_FLASH").map_or(true, |v| v != "0") {
                    let v2 = std::env::var("LE_DN_FLASH2").map_or(true, |v| v != "0");
                    let smem = if v2 { 2 * 64 * (hd + 8) * 2 } else { ((64 + 32) * (hd + 8) + hd * 40) * 2 };
                    let kname = if v2 { "dn_flash_128b" } else { "dn_flash_128" };
                    self.launch3s(kname, (t.div_ceil(64) as u32, nh as u32, 1), 128, smem as u32, &[P(self.bq), P(self.kc[l]), P(self.vc[l]), P(if lfm { self.bg } else { 0 }), P(self.bo), I(pos0 as i32), I(t as i32), I(nh as i32), I(nkv as i32), I(win as i32)]);
                } else {
                self.launch3("dn_attn", (nh as u32, t as u32, 1), 256, &[P(self.bq), P(self.kc[l]), P(self.vc[l]), P(if lfm { self.bg } else { 0 }), P(self.bo), I(win as i32), I((pos0 + 1) as i32), I(nh as i32), I(nkv as i32), I(hd as i32), P(0)]);
                }
                self.bqx(self.bo, t, nh * hd);
                self.bmm(d.wo, self.bt2, t);
            }
            if lfm { self.brms(self.bt2, d.post_attn, self.bt2, t); }
            self.add(self.bx, self.bt2, t * n);
            self.brms(self.bx, d.ffn_norm, self.bh, t);
            self.bqx(self.bh, t, n);
            self.bmm(d.gu, self.bgu, t);
            let ffn = d.gu.rows / 2;
            self.launch("swiglu_gu", (t * ffn).div_ceil(256) as u32, 256, 0, &[P(self.bgu), P(self.ba), I(ffn as i32), I((t * ffn) as i32)]);
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
    /// Runs all layers for one decode token. Only kernel launches here, so the sequence can be captured in a CUDA graph.
    /// `split[l]` picks split-K attention for layer l (decided on the host from the KV length).
    fn layers(&self, split: &[bool]) {
        let m = &self.m; let (n, hd, nh, nkv) = (m.n_embd, m.head_dim, m.n_head, m.n_kv);
        let lfm = m.kind == Kind::Lfm3;
        let gqa = nh / nkv == 8 && hd == 128 && std::env::var("LE_DN_GQA").map_or(true, |v| v != "0");
        for l in 0..m.n_layer {
            let d = &self.lay[l];
            let fused_attn = self.rmsq && !d.conv;
            if fused_attn { self.rms_q(self.x, d.attn_norm, self.h, n); } else { self.rms(self.x, d.attn_norm, self.h, n); }
            if d.conv {
                self.mv(d.in_proj, self.h, self.g);
                self.launch("dn_shortconv", n.div_ceil(128) as u32, 128, 0, &[P(self.g), P(d.conv_w), P(self.conv[l]), P(self.t1), I(n as i32), I(m.l_cache as i32)]);
                self.mv(d.out_proj, self.t1, self.t2);
            } else {
                if !fused_attn { self.qx(self.h, n); }
                self.mvq(d.wq, self.q); self.mvq(d.wk, self.k); self.mvq(d.wv, self.v);
                if lfm { self.mvq(d.wg, self.g); }
                let base = if d.swa { m.rope_base_swa } else { m.rope_base };
                let ff = if d.swa { 0 } else { self.ff };
                let neox = m.neox as i32;
                if self.fused_qkv { self.launch("dn_qkv_dec", (nh + nkv) as u32, 128, 0, &[P(self.q), P(self.k), P(self.v), P(self.kc[l]), P(self.vc[l]), P(self.dpos), I(self.pos as i32), I(nh as i32), I(nkv as i32), I(hd as i32), I(m.rope_dim as i32), F(base), P(ff), I(neox), P(d.qn), P(d.kn), F(m.eps)]); } else {
                self.launch("dn_prep", nh as u32, 128, 0, &[P(self.q), P(self.q), I(0), I(hd as i32), I(m.rope_dim as i32), I(self.pos as i32), F(base), P(ff), I(neox), P(d.qn), F(m.eps), P(self.dpos), I(0)]);
                self.launch("dn_prep", nkv as u32, 128, 0, &[P(self.k), P(self.kc[l]), I(1), I(hd as i32), I(m.rope_dim as i32), I(self.pos as i32), F(base), P(ff), I(neox), P(d.kn), F(m.eps), P(self.dpos), I((nkv * hd) as i32)]);
                self.launch("dn_cvt_half_d", (nkv * hd).div_ceil(128) as u32, 128, 0, &[P(self.v), P(self.vc[l]), P(self.dpos), I((nkv * hd) as i32)]);
                }
                let nk = self.pos + 1;
                let win = if d.swa { m.window } else { 0 };
                let (_, _, sp) = dn_split_env();
                if split[l] {
                    if gqa { self.launch3("dn_attn_gqa", (nkv as u32, sp as u32, 1), 256, &[P(self.q), P(self.kc[l]), P(self.vc[l]), P(self.part), I(win as i32), I(nk as i32), I(nh as i32), I(nkv as i32), I(hd as i32), I(sp as i32), P(self.dpos + 4)]); } else {
                        self.launch3("dn_attn_split", (nh as u32, sp as u32, 1), 256, &[P(self.q), P(self.kc[l]), P(self.vc[l]), P(self.part), I(win as i32), I(nk as i32), I(nh as i32), I(nkv as i32), I(hd as i32), I(sp as i32), P(self.dpos + 4)]); }
                    self.launch("dn_attn_comb", nh as u32, 128, 0, &[P(self.part), P(if lfm { self.g } else { 0 }), P(self.o), I(hd as i32), I(sp as i32)]);
                } else {
                self.launch("dn_attn", nh as u32, 256, 0, &[P(self.q), P(self.kc[l]), P(self.vc[l]), P(if lfm { self.g } else { 0 }), P(self.o), I(win as i32), I(nk as i32), I(nh as i32), I(nkv as i32), I(hd as i32), P(self.dpos + 4)]);
                }
                self.mv(d.wo, self.o, self.t2);
            }
            if lfm { self.rms(self.t2, d.post_attn, self.t2, n); }
            self.add(self.x, self.t2, n);
            if self.rmsq { self.rms_q(self.x, d.ffn_norm, self.h, n); } else { self.rms(self.x, d.ffn_norm, self.h, n); self.qx(self.h, n); }
            self.mvq(d.gu, self.a);
            let ffn = d.gu.rows / 2;
            self.launch("swiglu_gu", ffn.div_ceil(256) as u32, 256, 0, &[P(self.a), P(self.u), I(ffn as i32), I(ffn as i32)]);
            self.mv(d.down, self.u, self.t2);
            if lfm { self.rms(self.t2, d.post_ffw, self.t2, n); }
            self.add(self.x, self.t2, n);
        }
    }

    fn step(&mut self, token: u32, want: bool) -> Vec<f32> {
        assert!(self.pos < self.ctx_len, "dense gpu: context full");
        let m = &self.m; let (n, hd, nh, nkv) = (m.n_embd, m.head_dim, m.n_head, m.n_kv);
        let pos = self.pos as i32;
        let mut e = vec![0f32; n];
        { let (t, b) = m.bytes("token_embd.weight"); let rb = crate::quant::row_bytes(t.ty, n); crate::quant::dequant_row(t.ty, &b[token as usize * rb..(token as usize + 1) * rb], &mut e); }
        unsafe { result::memcpy_htod_sync(self.x, &e).unwrap(); result::memcpy_htod_sync(self.dpos, &[self.pos as i32, (self.pos + 1) as i32]).unwrap() };
        let lfm = m.kind == Kind::Lfm3;
        let (split_min, split_on, _) = dn_split_env();
        let nk = self.pos + 1;
        let split: Vec<bool> = (0..m.n_layer).map(|l| { if self.lay[l].conv { return false; } let win = if self.lay[l].swa { m.window } else { 0 }; let span = if win > 0 { nk.min(win) } else { nk }; m.head_dim == 128 && span >= split_min && split_on }).collect();
        if std::env::var("LE_DN_GRAPH").map_or(true, |v| v != "0") {
            if !self.graphs.contains_key(&split) {
                self.st.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).expect("graph begin");
                self.layers(&split);
                let g = self.st.end_capture(sys::CUgraphInstantiate_flags(0)).expect("graph end").expect("empty graph");
                self.graphs.insert(split.clone(), GraphBox(g));
            }
            self.graphs[&split].0.launch().expect("graph launch");
        } else { self.layers(&split); }
        self.pos += 1;
        if !want { return Vec::new(); }
        self.rms(self.x, self.out_norm, self.h, n);
        self.mv(self.out_w, self.h, self.logits);
        let mut out = vec![0f32; m.vocab];
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut out, self.logits).unwrap() };
        out
    }
    /// LE_EVT=1 decode: per-kernel event sums and host wall per token, printed every 64 tokens.
    /// Per-kernel times need LE_DN_GRAPH=0 (graph capture records events as graph nodes, not timings).
    pub fn forward(&mut self, t: u32) -> Vec<f32> { self.bind();
        if !evt_on() { return self.step(t, true); }
        // Graph mode: events inside graph capture are nodes, not timings, so use host wall only.
        let graph = std::env::var("LE_DN_GRAPH").map_or(true, |v| v != "0");
        NO_EVT.with(|c| c.set(graph));
        EVT.with(|v| v.borrow_mut().clear());
        let w0 = std::time::Instant::now();
        let w0e = if graph { None } else { self.ev_begin() };
        let out = self.step(t, true);
        NO_EVT.with(|c| c.set(false));
        let host_ms = w0.elapsed().as_secs_f64() * 1e3;
        let g_ms = match w0e { Some(a) => { let b = self._ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("event"); b.record(&self.st).expect("record"); b.synchronize().expect("sync"); a.elapsed_ms(&b).expect("elapsed") as f64 } None => host_ms };
        let mut ksum = 0f64;
        EVT.with(|v| for (n, a, b) in v.borrow().iter() { let ms = a.elapsed_ms(b).expect("elapsed") as f64; ksum += ms; DEC.with(|d| { let mut d = d.borrow_mut(); let e = d.entry(n.clone()).or_insert((0.0, 0)); e.0 += ms; e.1 += 1; }); });
        DEC_WALL.with(|d| { let mut d = d.borrow_mut(); d.0 += g_ms; d.1 += host_ms; d.2 += ksum; d.3 += 1; });
        let (n_done, wall_sum) = DEC_WALL.with(|d| { let d = d.borrow(); (d.3, d.0) });
        if n_done % 64 == 0 {
            DEC_WALL.with(|d| { let mut d = d.borrow_mut(); let (gw, hw, ks, n) = (d.0, d.1, d.2, d.3 as f64); eprintln!("evtdec {} tok: gpu wall {:.3} ms/tok, host {:.3} ms/tok, kernel sum {:.3} ms/tok, idle {:.3} ms/tok, {:.1} tok/s", n, gw / n, hw / n, ks / n, (gw - ks) / n, 1e3 * n / gw); d.0 = 0.0; d.1 = 0.0; d.2 = 0.0; d.3 = 0; });
            DEC.with(|d| { let mut rows: Vec<_> = d.borrow().iter().map(|(k, v)| (k.clone(), *v)).collect(); rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap()); let tot: f64 = rows.iter().map(|r| r.1 .0).sum(); for (n, (ms, c)) in rows.iter().take(12) { eprintln!("evtdec {:>20} {:>8.3} ms/tok {:>6.2} launches/tok {:>5.1}%", n, ms / 64.0, *c as f64 / 64.0, 100.0 * ms / tot); } d.borrow_mut().clear(); });
        }
        let _ = wall_sum;
        out
    }
    pub fn prefill(&mut self, ids: &[u32]) -> Vec<f32> { self.bind();
        let mut l = vec![];
        if std::env::var("LE_DENSE_SERIAL").is_ok() { for (i, &t) in ids.iter().enumerate() { l = self.step(t, i + 1 == ids.len()); } return l; }
        let wall0 = if evt_on() { EVT.with(|v| v.borrow_mut().clear()); self.ev_begin() } else { None };
        let mut i = 0;
        while i < ids.len() { let e = (i + CH).min(ids.len()); l = self.chunk(&ids[i..e], e == ids.len()); i = e; }
        if let Some(w0) = wall0 {
            let w1 = self._ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("event");
            w1.record(&self.st).expect("record"); w1.synchronize().expect("sync");
            let wall = w0.elapsed_ms(&w1).expect("elapsed") as f64;
            let mut agg: std::collections::BTreeMap<String, (f64, usize)> = Default::default(); let mut ksum = 0f64;
            EVT.with(|v| for (n, a, b) in v.borrow().iter() { let ms = a.elapsed_ms(b).expect("elapsed") as f64; ksum += ms; let e = agg.entry(n.clone()).or_insert((0.0, 0)); e.0 += ms; e.1 += 1; });
            let mut rows: Vec<_> = agg.into_iter().collect(); rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            eprintln!("evt wall {:.2} ms, kernel sum {:.2} ms, idle {:.2} ms ({:.1}%)", wall, ksum, wall - ksum, 100.0 * (wall - ksum) / wall);
            for (n, (ms, c)) in rows.iter().take(25) { eprintln!("evt {:>22} {:>9.2} ms {:>6} calls {:>5.1}%", n, ms, c, 100.0 * ms / ksum); }
            EVT.with(|v| v.borrow_mut().clear());
        }
        if prof_on() { prof_dump(); }
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
