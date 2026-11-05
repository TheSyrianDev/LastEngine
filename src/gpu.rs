//! CUDA backend for qwen35moe: non-expert weights resident in VRAM, routed experts through a VRAM slot cache fed from host memory.
use crate::cpu_ref::Model;
use cudarc::driver::{result, sys, sys::CUdeviceptr, CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::{collections::HashMap, sync::Arc};

type Dp = CUdeviceptr;
mod prefill;
fn mk_events() -> [sys::CUevent; 4] { let mut e = [std::ptr::null_mut(); 4]; for x in e.iter_mut() { unsafe { sys::cuEventCreate(x, 2).result().unwrap() }; } e }
fn dalloc(bytes: usize) -> Dp { unsafe { result::malloc_sync(bytes.max(16)).expect("cuda alloc") } }
fn dup<T>(data: &[T]) -> Dp { let p = dalloc(std::mem::size_of_val(data)); unsafe { result::memcpy_htod_sync(p, data).unwrap() }; p }
fn dzero(p: Dp, bytes: usize) { unsafe { result::memset_d8_sync(p, 0, bytes).unwrap() } }

#[derive(Clone, Copy)]
pub struct W { cell: Dp, ty: u32, rows: usize, cols: usize, raw: Dp }
struct Lay {
    attn_norm: Dp, post_norm: Dp, router: W, shg_in: W,
    sh_gate: W, sh_up: W, sh_down: W,
    full: bool, q: Option<W>, k: Option<W>, v: Option<W>, o: Option<W>, qn: Dp, kn: Dp,
    qkv: Option<W>, gate: Option<W>, beta: Option<W>, alpha: Option<W>, ssm_out: Option<W>, conv_w: Dp, a_log: Dp, dt_b: Dp, ssm_norm: Dp,
    ety: [u32; 3], ebytes: [usize; 3],
}
enum A { P(Dp), I(i32), F(f32) }
use A::*;

pub struct Gpu {
    pub m: Arc<Model>, _ctx: Arc<CudaContext>, st: Arc<CudaStream>, f: HashMap<&'static str, CudaFunction>,
    lay: Vec<Lay>, out_norm: Dp, out_w: W,
    x: Dp, h: Dp, t0: Dp, t1: Dp, t2: Dp, t3: Dp, y: Dp, logits: Dp, scores: Dp,
    idx: Dp, rw: Dp, shg: Dp, eg: Dp, eu: Dp, ea: Dp, ed: Dp, shg_t: Dp, shu_t: Dp, sha_t: Dp, shd_t: Dp,
    ptr_g: Dp, ptr_u: Dp, ptr_d: Dp, xqb: [Dp; 2], xdb: [Dp; 2], xsb: [Dp; 2],
    conv: Vec<Dp>, sdelta: Vec<Dp>, kc: Vec<Dp>, vc: Vec<Dp>,
    arena: Dp, slot_bytes: usize, n_slots: usize, slot_of: HashMap<(usize, usize), usize>, key_of: Vec<Option<(usize, usize)>>, last_use: Vec<u64>, tick: u64,
    cs: sys::CUstream, ev: sys::CUevent, stg: *mut u8, pf_pending: bool, store: *mut u8, store_off: Vec<usize>, ph: Dp, plog: Dp, pidx: Dp, prw: Dp, pf_extra: usize,
    pf: Option<Box<prefill::Pf>>, dbg: bool, dbg_pos: usize, mmq: bool, wave: u64, comp_ev: [sys::CUevent; 4], cp_ev: [sys::CUevent; 4],
    pub hits: u64, pub misses: u64, pub pf_issued: u64, pub ctx_len: usize, pub pos: usize,
}

unsafe impl Send for Gpu {}

impl Gpu {
    pub fn bind(&self) { self._ctx.bind_to_thread().expect("bind"); }
    pub fn new(m: Arc<Model>, ctx_len: usize, vram_margin_mib: usize) -> Gpu {
        let ctx = CudaContext::new(0).expect("cuda");
        let st = ctx.default_stream();
        let inc = ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include/cuda"].iter().find(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists()).map(|s| s.to_string());
        let mut src = String::from("#include <cuda_fp16.h>\n__constant__ unsigned int IQ3_G32[256] = {");
        for g in crate::quant::iq3_grid().iter() { src.push_str(&format!("{}u,", u32::from_le_bytes(*g))); }
        src.push_str("};\n");
        src.push_str(include_str!("cuda/kernels.cu"));
        let opts = CompileOptions { arch: Some("sm_86"), include_paths: inc.into_iter().collect(), ..Default::default() };
        let ptx = compile_ptx_with_opts(src, opts).unwrap_or_else(|e| panic!("nvrtc: {e:?}"));
        let module = ctx.load_module(ptx).expect("module");
        let mut f = HashMap::new();
        for n in ["matvecf", "quant_x", "matvec_2", "matvec_8", "matvec_11", "matvec_12", "matvec_13", "matvec_14", "matvec_18", "mmq_2", "mmq_8", "mmq_11", "mmq_12", "mmq_13", "mmq_14", "mmq_18", "rmsnorm", "add_inplace", "swiglu",
                  "router_topk", "moe_combine", "attn_prep", "attn_scores", "attn_softmax", "attn_pv", "delta_conv", "delta_step",
                  "mm_2", "mm_8", "mm_11", "mm_12", "mm_13", "mm_14", "mm_18", "mmf", "mmf2", "rmsnorm_b", "router_topk_b", "moe_combine_b", "attn_prep_b", "attn_scores_b", "attn_softmax_b", "attn_pv_b", "delta_conv_b", "delta_step_b"] {
            f.insert(n, module.load_function(n).unwrap_or_else(|e| panic!("{n}: {e:?}")));
        }
        let c = &m.cfg;
        let vec1 = |name: &str| -> Dp { let (ty, cols, rows) = m.dims2(name); let mut o = vec![0f32; cols * rows]; crate::quant::dequant_row(ty, m.raw(name), &mut o); dup(&o) };
        let mat = |name: &str| -> W {
            let (ty, cols, rows) = m.dims2(name);
            let d = dup(m.raw(name));
            W { cell: dup(&[d]), ty, rows, cols, raw: d }
        };
        let mut lay = Vec::new();
        for l in 0..c.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let full = (l + 1) % c.interval == 0;
            let z = W { cell: 0, ty: 0, rows: 0, cols: 0, raw: 0 };
            let sg = { let (ty, cols, _) = m.dims2(&p("ffn_gate_inp_shexp.weight")); let mut o = vec![0f32; cols]; crate::quant::dequant_row(ty, m.raw(&p("ffn_gate_inp_shexp.weight")), &mut o); let d = dup(&o); W { cell: dup(&[d]), ty: 0, rows: 1, cols, raw: d } };
            let mut ly = Lay {
                attn_norm: vec1(&p("attn_norm.weight")), post_norm: vec1(&p("post_attention_norm.weight")), router: { let d = vec1(&p("ffn_gate_inp.weight")); W { cell: dup(&[d]), ty: 0, rows: c.n_expert, cols: c.n_embd, raw: d } }, shg_in: sg,
                sh_gate: mat(&p("ffn_gate_shexp.weight")), sh_up: mat(&p("ffn_up_shexp.weight")), sh_down: mat(&p("ffn_down_shexp.weight")),
                full, q: None, k: None, v: None, o: None, qn: 0, kn: 0, qkv: None, gate: None, beta: None, alpha: None, ssm_out: None, conv_w: 0, a_log: 0, dt_b: 0, ssm_norm: 0,
                ety: m.expert_types(l), ebytes: m.expert_bytes(l),
            };
            let _ = z;
            if full {
                ly.q = Some(mat(&p("attn_q.weight"))); ly.k = Some(mat(&p("attn_k.weight"))); ly.v = Some(mat(&p("attn_v.weight"))); ly.o = Some(mat(&p("attn_output.weight")));
                ly.qn = vec1(&p("attn_q_norm.weight")); ly.kn = vec1(&p("attn_k_norm.weight"));
            } else {
                ly.qkv = Some(mat(&p("attn_qkv.weight"))); ly.gate = Some(mat(&p("attn_gate.weight")));
                let f32mat = |name: &str| -> W { let (ty, cols, rows) = m.dims2(name); let mut o = vec![0f32; cols * rows]; crate::quant::dequant_row(ty, m.raw(name), &mut o); let d = dup(&o); W { cell: dup(&[d]), ty: 0, rows, cols, raw: d } };
                ly.beta = Some(f32mat(&p("ssm_beta.weight"))); ly.alpha = Some(f32mat(&p("ssm_alpha.weight")));
                ly.ssm_out = Some(mat(&p("ssm_out.weight")));
                ly.conv_w = vec1(&p("ssm_conv1d.weight")); ly.a_log = vec1(&p("ssm_a")); ly.dt_b = vec1(&p("ssm_dt.bias")); ly.ssm_norm = vec1(&p("ssm_norm.weight"));
            }
            lay.push(ly);
        }
        let out_norm = vec1("output_norm.weight");
        let out_w = mat("output.weight");
        let (hd, nkv, nh) = (c.head_dim, c.n_kv, c.n_head);
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        let (mut conv, mut sdelta, mut kc, mut vc) = (vec![], vec![], vec![], vec![]);
        for l in 0..c.n_layer {
            if (l + 1) % c.interval == 0 {
                conv.push(0); sdelta.push(0);
                let (a, b) = (dalloc(ctx_len * nkv * hd * 2), dalloc(ctx_len * nkv * hd * 2)); kc.push(a); vc.push(b);
            } else {
                let (a, b) = (dalloc(conv_ch * (c.conv_k - 1) * 4), dalloc(c.n_vhead * c.d_state * c.d_state * 4));
                dzero(a, conv_ch * (c.conv_k - 1) * 4); dzero(b, c.n_vhead * c.d_state * c.d_state * 4);
                conv.push(a); sdelta.push(b); kc.push(0); vc.push(0);
            }
        }
        let mut slot_bytes = 0;
        for l in 0..c.n_layer { slot_bytes = slot_bytes.max(m.expert_bytes(l).iter().sum::<usize>()); }
        slot_bytes = slot_bytes.div_ceil(256) * 256;
        let pf = prefill::Pf::new(c, ctx_len);
        let (free, _tot) = result::mem_get_info().expect("meminfo");
        let scratch = 64usize << 20;
        let budget = free.saturating_sub(scratch + (vram_margin_mib << 20));
        let n_slots = (budget / slot_bytes).min(c.n_layer * c.n_expert).max(c.n_used);
        eprintln!("gpu: free {} MiB, slot {} KiB, {} expert slots ({} MiB)", free >> 20, slot_bytes >> 10, n_slots, (n_slots * slot_bytes) >> 20);
        let arena = dalloc(n_slots * slot_bytes);
        let big = c.n_embd.max(conv_ch).max(nh * hd * 2);
        let a = |n: usize| dalloc(n * 4);
        let (mut cs, mut ev) = (std::ptr::null_mut(), std::ptr::null_mut());
        let mut stg: *mut std::ffi::c_void = std::ptr::null_mut();
        let pf_extra: usize = std::env::var("LE_PF_EXTRA").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        unsafe {
            sys::cuStreamCreate(&mut cs, 1).result().unwrap();
            sys::cuEventCreate(&mut ev, 2).result().unwrap();
            sys::cuMemHostAlloc(&mut stg, (c.n_used + pf_extra).max(1) * slot_bytes, 0).result().expect("pinned staging");
        }
        let mut store_off = vec![0usize; c.n_layer + 1];
        for l in 0..c.n_layer { store_off[l + 1] = store_off[l] + m.expert_bytes(l).iter().sum::<usize>() * c.n_expert; }
        let mut store_p: *mut std::ffi::c_void = std::ptr::null_mut();
        let store_ok = std::env::var("LE_NOSTORE").is_err() && unsafe { sys::cuMemHostAlloc(&mut store_p, store_off[c.n_layer], 0).result().is_ok() };
        if store_ok {
            let t0 = std::time::Instant::now();
            let sp = store_p as usize;
            use rayon::prelude::*;
            (0..c.n_layer).into_par_iter().for_each(|l| {
                let eb: usize = m.expert_bytes(l).iter().sum();
                for e in 0..c.n_expert {
                    let mut o = sp + store_off[l] + e * eb;
                    for part in m.expert_parts(l, e) { unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), o as *mut u8, part.len()) }; o += part.len(); }
                }
            });
            eprintln!("pinned expert store {} MiB in {:.1}s", store_off[c.n_layer] >> 20, t0.elapsed().as_secs_f64());
        } else { store_p = std::ptr::null_mut(); }
        Gpu {
            store: store_p as *mut u8, store_off, pf: Some(Box::new(pf)), mmq: std::env::var("LE_MMQ").map_or(true, |v| v != "0"), dbg: std::env::var("LE_DBG").is_ok(), dbg_pos: std::env::var("LE_DBG").ok().and_then(|v| v.parse().ok()).unwrap_or(0), wave: 0, comp_ev: mk_events(), cp_ev: mk_events(),
            cs, ev, stg: stg as *mut u8, pf_pending: false, ph: a(c.n_embd), plog: a(c.n_expert), pidx: dalloc(64), prw: a(16), pf_extra, pf_issued: 0,
            x: a(c.n_embd), h: a(c.n_embd), t0: a(big), t1: a(big), t2: a(big), t3: a(big), y: a(big), logits: a(m.vocab.max(c.n_expert)), scores: a(nh * ctx_len),
            idx: dalloc(64), rw: a(16), shg: a(4),
            eg: a(c.n_used * c.ff_exp), eu: a(c.n_used * c.ff_exp), ea: a(c.n_used * c.ff_exp), ed: a(c.n_used * c.n_embd),
            shg_t: a(c.ff_shexp), shu_t: a(c.ff_shexp), sha_t: a(c.ff_shexp), shd_t: a(c.n_embd),
            ptr_g: dalloc(64), ptr_u: dalloc(64), ptr_d: dalloc(64),
            xqb: [dalloc(16384), dalloc(16384)], xdb: [dalloc(2048), dalloc(2048)], xsb: [dalloc(2048), dalloc(2048)],
            conv, sdelta, kc, vc, lay, out_norm, out_w,
            arena, slot_bytes, n_slots, slot_of: HashMap::new(), key_of: vec![None; n_slots], last_use: vec![0; n_slots], tick: 0, hits: 0, misses: 0, ctx_len, pos: 0,
            f, st, _ctx: ctx, m,
        }
    }

    fn launch(&self, name: &str, grid: (u32, u32), block: u32, smem: u32, args: &[A]) { self.launch3(name, (grid.0, grid.1, 1), block, smem, args) }
    fn launch3(&self, name: &str, grid: (u32, u32, u32), block: u32, smem: u32, args: &[A]) {
        let mut b = self.st.launch_builder(&self.f[name]);
        for a in args { match a { P(p) => { b.arg(p); } I(i) => { b.arg(i); } F(x) => { b.arg(x); } } }
        unsafe { b.launch(LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: smem }).expect("launch"); }
    }
    fn qx(&self, x: Dp, n: usize, set: usize) {
        let units = n / 32;
        self.launch("quant_x", (units.div_ceil(8) as u32, 1), 256, 0, &[P(x), P(self.xqb[set]), P(self.xdb[set]), P(self.xsb[set]), I(units as i32)]);
    }
    /// Quantised matvec (activations already quantised into `set`), or F32 matvec reading `x` directly when ty == 0.
    fn mv_ptrs(&self, ty: u32, ptrs: Dp, x: Dp, set: usize, out: Dp, rows: usize, cols: usize, xstride: usize, slots: usize) {
        if ty == 0 {
            self.launch("matvecf", (rows.div_ceil(4) as u32, slots as u32), 128, (cols * 4) as u32, &[P(ptrs), P(x), P(out), I(rows as i32), I(cols as i32), I(xstride as i32)]);
            return;
        }
        let name = match ty { 2 => "matvec_2", 8 => "matvec_8", 11 => "matvec_11", 12 => "matvec_12", 13 => "matvec_13", 14 => "matvec_14", 18 => "matvec_18", t => panic!("no kernel for type {t}") };
        let units = cols / 32; let lpr = units.min(32); let rpb = 8 * (32 / lpr);
        self.launch(name, (rows.div_ceil(rpb) as u32, slots as u32), 256, (units * 36 + units * 8) as u32,
            &[P(ptrs), P(self.xqb[set]), P(self.xdb[set]), P(self.xsb[set]), P(out), I(rows as i32), I(cols as i32), I(xstride as i32)]);
    }
    pub fn mv(&self, w: W, x: Dp, set: usize, out: Dp) { self.mv_ptrs(w.ty, w.cell, x, set, out, w.rows, w.cols, 0, 1); }
    fn rms(&self, x: Dp, w: Dp, o: Dp, n: usize) { self.launch("rmsnorm", (1, 1), 256, 0, &[P(x), P(w), P(o), I(n as i32), F(self.m.cfg.eps)]); }
    fn dump(&self, label: &str, p: Dp, n: usize) { let mut v = vec![0f32; n]; unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut v, p).unwrap() }; eprintln!("  {label} {:.4} {}", v.iter().map(|a| a * a).sum::<f32>().sqrt(), v[..4].iter().map(|a| format!("{a:.3}")).collect::<Vec<_>>().join(",")); }
    fn add(&self, x: Dp, y: Dp, n: usize) { self.launch("add_inplace", (n.div_ceil(256) as u32, 1), 256, 0, &[P(x), P(y), I(n as i32)]); }
    fn swiglu(&self, g: Dp, u: Dp, o: Dp, n: usize) { self.launch("swiglu", (n.div_ceil(256) as u32, 1), 256, 0, &[P(g), P(u), P(o), I(n as i32)]); }

    pub fn pos(&self) -> usize { self.pos }
    /// Copy only the recurrent (DeltaNet) state to host; KV is append-only so a position is enough for it.
    pub fn snap_recurrent(&mut self) -> Vec<u8> {
        self.bind();
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap() };
        let c = self.m.cfg.clone();
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        let mut out = Vec::new();
        for l in 0..c.n_layer { if !self.lay[l].full {
            for (p, n) in [(self.conv[l], conv_ch * (c.conv_k - 1) * 4), (self.sdelta[l], c.n_vhead * c.d_state * c.d_state * 4)] {
                let o = out.len(); out.resize(o + n, 0); unsafe { result::memcpy_dtoh_sync(&mut out[o..], p).unwrap() }
            }
        } }
        out
    }
    pub fn restore_recurrent(&mut self, b: &[u8], pos: usize) {
        self.bind();
        let c = self.m.cfg.clone();
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        let mut o = 0;
        for l in 0..c.n_layer { if !self.lay[l].full {
            for (p, n) in [(self.conv[l], conv_ch * (c.conv_k - 1) * 4), (self.sdelta[l], c.n_vhead * c.d_state * c.d_state * 4)] {
                unsafe { result::memcpy_htod_sync(p, &b[o..o + n]).unwrap() }; o += n;
            }
        } }
        self.pos = pos;
    }
    /// Serialise the model state (recurrent state + KV up to pos) into bytes.
    pub fn export_state(&mut self) -> Vec<u8> {
        self.bind();
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap() };
        let c = self.m.cfg.clone();
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        let mut out = Vec::new();
        out.extend_from_slice(&(self.pos as u64).to_le_bytes());
        let rd = |p: Dp, n: usize, out: &mut Vec<u8>| { let o = out.len(); out.resize(o + n, 0); unsafe { result::memcpy_dtoh_sync(&mut out[o..], p).unwrap() } };
        for l in 0..c.n_layer {
            if self.lay[l].full { let n = self.pos * c.n_kv * c.head_dim * 2; rd(self.kc[l], n, &mut out); rd(self.vc[l], n, &mut out); }
            else { rd(self.conv[l], conv_ch * (c.conv_k - 1) * 4, &mut out); rd(self.sdelta[l], c.n_vhead * c.d_state * c.d_state * 4, &mut out); }
        }
        out
    }
    pub fn import_state(&mut self, b: &[u8]) -> Result<(), String> {
        self.bind();
        let c = self.m.cfg.clone();
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        if b.len() < 8 { return Err("short".into()); }
        let pos = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        if pos >= self.ctx_len { return Err("state longer than context".into()); }
        let mut o = 8usize;
        let mut wr = |p: Dp, n: usize| -> Result<(), String> { if o + n > b.len() { return Err("truncated".into()) } unsafe { result::memcpy_htod_sync(p, &b[o..o + n]).unwrap() }; o += n; Ok(()) };
        for l in 0..c.n_layer {
            if self.lay[l].full { let n = pos * c.n_kv * c.head_dim * 2; wr(self.kc[l], n)?; wr(self.vc[l], n)?; }
            else { wr(self.conv[l], conv_ch * (c.conv_k - 1) * 4)?; wr(self.sdelta[l], c.n_vhead * c.d_state * c.d_state * 4)?; }
        }
        self.pos = pos;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.bind();
        let c = &self.m.cfg;
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        for l in 0..c.n_layer { if !self.lay[l].full { dzero(self.conv[l], conv_ch * (c.conv_k - 1) * 4); dzero(self.sdelta[l], c.n_vhead * c.d_state * c.d_state * 4); } }
        self.pos = 0;
    }

    /// Claim a slot for (layer, e); returns (slot, needs_copy).
    fn pick_slot(&mut self, layer: usize, e: usize, pinned: &[usize]) -> (usize, bool) {
        self.tick += 1;
        if let Some(&s) = self.slot_of.get(&(layer, e)) { self.last_use[s] = self.tick; return (s, false); }
        let (mut best, mut bt) = (usize::MAX, u64::MAX);
        for s in 0..self.n_slots {
            if pinned.contains(&s) { continue; }
            let t = if self.key_of[s].is_none() { 0 } else { self.last_use[s] };
            if t < bt { bt = t; best = s; }
        }
        if let Some(k) = self.key_of[best] { self.slot_of.remove(&k); }
        self.key_of[best] = Some((layer, e)); self.slot_of.insert((layer, e), best); self.last_use[best] = self.tick;
        (best, true)
    }
    fn slot_for(&mut self, layer: usize, e: usize, pinned: &[usize]) -> usize {
        let (s, copy) = self.pick_slot(layer, e, pinned);
        if !copy { self.hits += 1; return s; }
        self.misses += 1;
        let mut off = (s * self.slot_bytes) as u64;
        if !self.store.is_null() {
            let eb: usize = self.lay[layer].ebytes.iter().sum();
            unsafe { sys::cuMemcpyHtoDAsync_v2(self.arena + off, self.store.add(self.store_off[layer] + e * eb) as *const _, eb, self.st.cu_stream()).result().unwrap() };
            return s;
        }
        for part in self.m.expert_parts(layer, e) {
            unsafe { result::memcpy_htod_sync(self.arena + off, part).unwrap() };
            off += part.len() as u64;
        }
        s
    }
    /// Asynchronously stage predicted experts of `layer` on the copy stream through pinned memory.
    fn prefetch(&mut self, layer: usize, idxs: &[i32], pinned: &[usize]) {
        unsafe { sys::cuStreamSynchronize(self.cs).result().unwrap() };
        let mut pin: Vec<usize> = pinned.to_vec();
        let mut n = 0;
        for &e in idxs {
            let (s, copy) = self.pick_slot(layer, e as usize, &pin);
            pin.push(s);
            if !copy { continue; }
            n += 1; self.pf_issued += 1;
            if !self.store.is_null() {
                let eb: usize = self.lay[layer].ebytes.iter().sum();
                unsafe { sys::cuMemcpyHtoDAsync_v2(self.arena + (s * self.slot_bytes) as u64, self.store.add(self.store_off[layer] + e as usize * eb) as *const _, eb, self.cs).result().unwrap() };
                continue;
            }
            let mut so = 0usize;
            let base = self.stg as usize + (n - 1) * self.slot_bytes;
            for part in self.m.expert_parts(layer, e as usize) {
                unsafe {
                    std::ptr::copy_nonoverlapping(part.as_ptr(), (base + so) as *mut u8, part.len());
                    sys::cuMemcpyHtoDAsync_v2(self.arena + (s * self.slot_bytes + so) as u64, (base + so) as *const _, part.len(), self.cs).result().unwrap();
                }
                so += part.len();
            }
        }
        if n > 0 { unsafe { sys::cuEventRecord(self.ev, self.cs).result().unwrap() }; self.pf_pending = true; }
    }

    /// Decode one token at self.pos; returns device pointer of logits (vocab f32) after synchronisation.
    pub fn prefill(&mut self, ids: &[u32]) -> Vec<f32> {
        self.bind();
        let mut out = vec![];
        if self.pf.is_some() && ids.len() >= 8 && std::env::var("LE_BATCH").map_or(true, |v| v != "0") {
            let mut i = 0;
            while i < ids.len() { let e = (i + prefill::tmax()).min(ids.len()); out = self.prefill_chunk(&ids[i..e], e == ids.len()); i = e; }
            return out;
        }
        for (i, &t) in ids.iter().enumerate() { out = self.forward_n(t, i + 1 == ids.len()); }
        out
    }
    pub fn forward(&mut self, token: u32) -> Vec<f32> { self.bind(); self.forward_n(token, true) }
    fn forward_n(&mut self, token: u32, want: bool) -> Vec<f32> {
        let c = self.m.cfg.clone();
        let pos = self.pos;
        assert!(pos < self.ctx_len, "context full");
        unsafe { result::memcpy_htod_sync(self.x, &self.m.embed(token)).unwrap() };
        for l in 0..c.n_layer {
            let ly = &self.lay[l] as *const Lay; let ly = unsafe { &*ly };
            self.rms(self.x, ly.attn_norm, self.h, c.n_embd);
            if ly.full {
                let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
                self.qx(self.h, c.n_embd, 0); self.mv(ly.q.unwrap(), self.h, 0, self.t0); self.mv(ly.k.unwrap(), self.h, 0, self.t1); self.mv(ly.v.unwrap(), self.h, 0, self.t2);
                self.launch("attn_prep", ((nh + nkv) as u32, 1), 256, 0, &[P(self.t0), P(self.t1), P(self.t2), P(ly.qn), P(ly.kn), P(self.y), P(self.t3), P(self.kc[l]), P(self.vc[l]),
                    I(nh as i32), I(nkv as i32), I(hd as i32), I(c.rope_dim as i32), F(c.rope_base), I(pos as i32), F(c.eps)]);
                let n = pos + 1;
                self.launch("attn_scores", (n.div_ceil(8) as u32, nh as u32), 256, 0, &[P(self.y), P(self.kc[l]), P(self.scores), I(nkv as i32), I(nh as i32), I(hd as i32), I(n as i32), I(self.ctx_len as i32)]);
                self.launch("attn_softmax", (nh as u32, 1), 256, 0, &[P(self.scores), I(n as i32), I(self.ctx_len as i32)]);
                self.launch("attn_pv", (nh as u32, 1), hd as u32, 0, &[P(self.scores), P(self.vc[l]), P(self.t3), P(self.t0), I(nkv as i32), I(nh as i32), I(hd as i32), I(n as i32), I(self.ctx_len as i32)]);
                self.qx(self.t0, nh * hd, 1); self.mv(ly.o.unwrap(), self.t0, 1, self.t1);
                self.add(self.x, self.t1, c.n_embd);
            } else {
                let ch = c.d_inner + 2 * c.n_group * c.d_state;
                self.qx(self.h, c.n_embd, 0); self.mv(ly.qkv.unwrap(), self.h, 0, self.t0); self.mv(ly.gate.unwrap(), self.h, 0, self.t1);
                self.mv(ly.beta.unwrap(), self.h, 0, self.t2); self.mv(ly.alpha.unwrap(), self.h, 0, self.t3);
                if self.dbg && pos == self.dbg_pos && l == 0 { self.dump("qkv", self.t0, ch); self.dump("gate", self.t1, 4096); self.dump("beta", self.t2, 32); self.dump("alpha", self.t3, 32); }
                self.launch("delta_conv", (ch.div_ceil(256) as u32, 1), 256, 0, &[P(self.t0), P(ly.conv_w), P(self.conv[l]), P(self.y), I(ch as i32), I(c.conv_k as i32)]);
                self.launch("delta_step", (c.n_vhead as u32, 1), 128, 0, &[P(self.y), P(self.t3), P(self.t2), P(ly.a_log), P(ly.dt_b), P(self.t1), P(ly.ssm_norm), P(self.sdelta[l]), P(self.t0),
                    I(c.n_group as i32), I(c.n_vhead as i32), I(c.d_state as i32), F(c.eps)]);
                if self.dbg && pos == self.dbg_pos && l == 0 { self.dump("conv", self.y, ch); self.dump("dout", self.t0, 4096); }
                self.qx(self.t0, c.d_inner, 1); self.mv(ly.ssm_out.unwrap(), self.t0, 1, self.t1);
                self.add(self.x, self.t1, c.n_embd);
            }
            // MoE
            if self.dbg && pos == self.dbg_pos { self.dump("mix", self.x, c.n_embd); }
            self.rms(self.x, ly.post_norm, self.h, c.n_embd);
            self.mv(ly.router, self.h, 0, self.logits);
            self.qx(self.h, c.n_embd, 0);
            self.launch("router_topk", (1, 1), 32, 0, &[P(self.logits), I(c.n_expert as i32), I(c.n_used as i32), P(self.idx), P(self.rw)]);
            let mut idx = [0i32; 8];
            unsafe { result::memcpy_dtoh_sync(&mut idx[..c.n_used], self.idx).unwrap() };
            let mut slots: Vec<usize> = Vec::new();
            for i in 0..c.n_used { let s = self.slot_for(l, idx[i] as usize, &slots); slots.push(s); }
            if self.pf_pending { unsafe { sys::cuStreamWaitEvent(self.st.cu_stream(), self.ev, 0).result().unwrap() }; self.pf_pending = false; }
            if l + 1 < c.n_layer {
                let nl = &self.lay[l + 1] as *const Lay; let nl = unsafe { &*nl };
                self.rms(self.x, nl.post_norm, self.ph, c.n_embd);
                self.mv(nl.router, self.ph, 0, self.plog);
                let kk = (c.n_used + self.pf_extra).min(16);
                self.launch("router_topk", (1, 1), 32, 0, &[P(self.plog), I(c.n_expert as i32), I(kk as i32), P(self.pidx), P(self.prw)]);
                let mut pi = [0i32; 16];
                unsafe { result::memcpy_dtoh_sync(&mut pi[..kk], self.pidx).unwrap() };
                self.prefetch(l + 1, &pi[..kk], &slots);
            }
            let base = |s: usize, part: usize| self.arena + (s * self.slot_bytes) as u64 + ly.ebytes[..part].iter().sum::<usize>() as u64;
            let pg: Vec<u64> = slots.iter().map(|&s| base(s, 0)).collect();
            let pu: Vec<u64> = slots.iter().map(|&s| base(s, 1)).collect();
            let pd: Vec<u64> = slots.iter().map(|&s| base(s, 2)).collect();
            unsafe { result::memcpy_htod_sync(self.ptr_g, &pg).unwrap(); result::memcpy_htod_sync(self.ptr_u, &pu).unwrap(); result::memcpy_htod_sync(self.ptr_d, &pd).unwrap(); }
            let k = c.n_used;
            self.mv_ptrs(ly.ety[0], self.ptr_g, self.h, 0, self.eg, c.ff_exp, c.n_embd, 0, k);
            self.mv_ptrs(ly.ety[1], self.ptr_u, self.h, 0, self.eu, c.ff_exp, c.n_embd, 0, k);
            self.swiglu(self.eg, self.eu, self.ea, k * c.ff_exp);
            self.qx(self.ea, k * c.ff_exp, 1);
            self.mv_ptrs(ly.ety[2], self.ptr_d, self.ea, 1, self.ed, c.n_embd, c.ff_exp, c.ff_exp, k);
            self.mv(ly.sh_gate, self.h, 0, self.shg_t); self.mv(ly.sh_up, self.h, 0, self.shu_t);
            self.swiglu(self.shg_t, self.shu_t, self.sha_t, c.ff_shexp);
            self.qx(self.sha_t, c.ff_shexp, 1); self.mv(ly.sh_down, self.sha_t, 1, self.shd_t);
            self.mv(ly.shg_in, self.h, 0, self.shg);
            self.launch("moe_combine", (c.n_embd.div_ceil(256) as u32, 1), 256, 0, &[P(self.x), P(self.ed), P(self.rw), I(k as i32), P(self.shd_t), P(self.shg), I(c.n_embd as i32)]);
            if self.dbg && pos == self.dbg_pos { let mut v = vec![0f32; c.n_embd]; unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut v, self.x).unwrap() }; eprintln!("L{l} {:.4} {:.4}", v.iter().map(|a| a * a).sum::<f32>().sqrt(), v[..4].iter().map(|a| format!("{a:.3}")).collect::<Vec<_>>().join(",")); }
        }
        self.pos += 1;
        if !want { self.st.synchronize().unwrap(); return vec![]; }
        self.rms(self.x, self.out_norm, self.h, c.n_embd);
        self.qx(self.h, c.n_embd, 0); self.mv(self.out_w, self.h, 0, self.logits);
        let mut out = vec![0f32; self.m.vocab];
        unsafe { result::memcpy_dtoh_sync(&mut out, self.logits).unwrap() };
        out
    }
}

/// Diagnostic: H2D bandwidth from pageable, pinned, and registered-mmap sources.
pub fn h2d_bench(path: &str) {
    use cudarc::driver::sys;
    let _ctx = CudaContext::new(0).unwrap();
    let n = 256usize << 20;
    let dev = dalloc(n);
    let t = |label: &str, p: *const u8| {
        let t0 = std::time::Instant::now();
        for _ in 0..4 { unsafe { sys::cuMemcpyHtoD_v2(dev, p as *const _, n).result().unwrap() }; }
        let dt = t0.elapsed().as_secs_f64() / 4.0;
        println!("{label}: {:.2} GB/s", n as f64 / dt / 1e9);
    };
    let pageable = vec![1u8; n];
    t("pageable", pageable.as_ptr());
    let mut pp: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe { sys::cuMemHostAlloc(&mut pp, n, 0).result().unwrap() };
    unsafe { std::ptr::write_bytes(pp as *mut u8, 1, n) };
    t("pinned", pp as *const u8);
    let f = std::fs::File::open(path).unwrap();
    let mm = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let off = 100usize << 20;
    let r = unsafe { sys::cuMemHostRegister_v2(mm.as_ptr().add(off) as *mut _, n, 1 | 8).result() };
    println!("register mmap (portable|readonly): {:?}", r.is_ok());
    if r.is_ok() { t("registered mmap", unsafe { mm.as_ptr().add(off) }); }
}
