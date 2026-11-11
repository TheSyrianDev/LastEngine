//! CUDA backend for qwen35moe: non-expert weights resident in VRAM, routed experts through a VRAM slot cache fed from host memory.
use crate::cpu_ref::Model;
use crate::quant_ik_tables::*;
use cudarc::driver::{result, sys, sys::CUdeviceptr, CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::{collections::HashMap, sync::Arc};

type Dp = CUdeviceptr;
mod prefill;
pub mod dense;
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
    qkv: Option<W>, gate: Option<W>, beta: Option<W>, alpha: Option<W>, ssm_out: Option<W>, conv_w: Dp, a_log: Dp, dt_b: Dp, ssm_norm: Dp, wr: Dp, bd_ls: Dp, bd_hb: Dp, bd_hc: Dp,
    ety: [u32; 3], ebytes: [usize; 3],
}
pub struct Mtp { eh: W, enorm: Dp, hnorm: Dp, shn: Dp }
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
    pv_part: Dp, kscr: Dp, pub(crate) pheat: Vec<u32>, rope_f: Dp, rope_ms: f32, pub kvq: i32, tab_h: *mut i32, tab_d: Dp, prot: Vec<bool>, nall: usize, mtp: Option<Mtp>, hprev: Dp, mcat: Dp, snap: Vec<(Dp, Dp)>, pub rows_out: Vec<Vec<f32>>, pub want_rows: bool, pub preview: bool, pub draft_on: bool, pub drafts_iters: u64, ksc: Vec<Dp>, vsc: Vec<Dp>, pub hits: u64, pub misses: u64, pub pf_issued: u64, pub ctx_len: usize, pub pos: usize,
}

unsafe impl Send for Gpu {}

impl Gpu {
    pub fn bind(&self) { self._ctx.bind_to_thread().expect("bind"); }
    pub fn new(m: Arc<Model>, ctx_len: usize, vram_margin_mib: usize) -> Gpu {
        let ctx = CudaContext::new(0).unwrap_or_else(|e| { eprintln!("lastengine: no usable CUDA device: {e:?}"); std::process::exit(1) });
        let arch = cuda_arch(&ctx).unwrap_or_else(|e| { eprintln!("lastengine: {e}"); std::process::exit(1) });
        let st = ctx.default_stream();
        // Auto-tune the knobs the user left unset, from the free VRAM and the requested context.
        {
            let (free0, _) = result::mem_get_info().expect("meminfo");
            let tkey = format!("{}|{}-{}-{}|{}", crate::tune::gpu_name(), m.cfg.n_layer, m.cfg.n_embd, m.cfg.n_expert, ctx_len);
            eprintln!("tunekey {tkey}");
            if std::env::var("LE_NOTUNE").is_err() { if let Some(e) = crate::tune::lookup(&tkey) {
                if std::env::var("LE_KV").is_err() { std::env::set_var("LE_KV", &e.kv); }
                if std::env::var("LE_TMAX").is_err() { std::env::set_var("LE_TMAX", e.chunk.to_string()); }
                eprintln!("gpu: using tuned settings kv {} chunk {}", e.kv, e.chunk);
            } }
            let kv_auto = match std::env::var("LE_KV").ok().as_deref() { None | Some("") | Some("auto") => true, _ => false };
            if kv_auto { std::env::set_var("LE_KV", if ctx_len <= 8192 { "f16" } else { "q8" }); }
            if std::env::var("LE_TMAX").is_err() { std::env::set_var("LE_TMAX", if free0 >= (3200usize << 20) { "2048" } else { "1024" }); }
            eprintln!("gpu: auto-tune free {} MiB ctx {} -> kv {} chunk {}", free0 >> 20, ctx_len, std::env::var("LE_KV").unwrap(), std::env::var("LE_TMAX").unwrap());
        }
        let inc = ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include/cuda"].iter().find(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists()).map(|s| s.to_string());
        let src = kernel_source();
        let opts = CompileOptions { arch: Some(arch), include_paths: inc.into_iter().collect(), ..Default::default() };
        let ptx = compile_ptx_with_opts(src, opts).unwrap_or_else(|e| panic!("nvrtc: {e:?}"));
        let module = ctx.load_module(ptx).expect("module");
        let mut f = HashMap::new();
        for n in ["matvecf", "quant_x", "matvec_2", "matvec_8", "matvec_11", "matvec_12", "matvec_13", "matvec_14", "matvec_16", "matvec_17", "matvec_18", "matvec_19", "matvec_20", "matvec_21", "matvec_22", "matvec_23", "matvec_29", "matvec_134", "matvec_135", "matvec_137", "matvec_138", "matvec_139", "matvec_140", "matvec_141", "matvec_144", "matvec_145", "matvec_146", "matvec_152", "matvec_153", "matvec_154", "matvec_155", "matvec_156", "matvec_157", "matvec_158", "mmq_2", "mmq_8", "mmq_11", "mmq_12", "mmq_13", "mmq_14", "mmq_16", "mmq_17", "mmq_18", "mmq_19", "mmq_20", "mmq_21", "mmq_22", "mmq_23", "mmq_29", "mmq_134", "mmq_135", "mmq_137", "mmq_138", "mmq_139", "mmq_140", "mmq_141", "mmq_144", "mmq_145", "mmq_146", "mmq_152", "mmq_153", "mmq_154", "mmq_155", "mmq_156", "mmq_157", "mmq_158", "add_rmsnorm", "build_ptrs", "rmsnorm", "add_inplace", "swiglu",
                  "router_topk", "moe_combine", "attn_prep", "attn_scores", "attn_softmax", "attn_pv", "attn_pv_split", "attn_pv_reduce", "tcq_encode", "tcq_rot", "delta_conv", "delta_step",
                  "mm_2", "mm_8", "mm_11", "mm_12", "mm_13", "mm_14", "mm_16", "mm_17", "mm_18", "mm_19", "mm_20", "mm_21", "mm_22", "mm_23", "mm_29", "mm_134", "mm_135", "mm_137", "mm_138", "mm_139", "mm_140", "mm_141", "mm_144", "mm_145", "mm_146", "mm_152", "mm_153", "mm_154", "mm_155", "mm_156", "mm_157", "mm_158", "mmf", "mmf2", "rmsnorm_b", "router_topk_b", "moe_combine_b", "attn_prep_b", "attn_scores_b", "attn_softmax_b", "attn_pv_b", "attn_flash_256", "attn_flash_256_nc", "delta_conv_pv", "delta_pv", "delta_conv_b", "delta_step_b"] {
            f.insert(n, module.load_function(n).unwrap_or_else(|e| panic!("{n}: {e:?}")));
        }
        { use cudarc::driver::sys::CUfunction_attribute as A; f.get("attn_flash_256").unwrap().set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 72 * 1024).unwrap(); f.get("attn_flash_256_nc").unwrap().set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 72 * 1024).unwrap(); }
        let c = &m.cfg;
        let vec1 = |name: &str| -> Dp { let (ty, cols, rows) = m.dims2(name); let mut o = vec![0f32; cols * rows]; crate::quant::dequant_row(ty, m.raw(name), &mut o); dup(&o) };
        let mat = |name: &str| -> W {
            let (ty, cols, rows) = m.dims2(name);
            let d = dup(m.raw(name));
            W { cell: dup(&[d]), ty, rows, cols, raw: d }
        };
        let mtp_on = std::env::var("LE_MTP").map_or(false, |v| v != "0") && m.has(&format!("blk.{}.nextn.eh_proj.weight", c.n_layer)) && m.has(&format!("blk.{}.ffn_gate_exps.weight", c.n_layer));
        let draft_on = std::env::var("LE_CLOVER_DRAFT").map_or(false, |v| v != "0") && m.has("blk.0.ssm_conv1d.weight_right");
        let nall = c.n_layer + mtp_on as usize;
        let mut lay = Vec::new();
        for l in 0..nall {
            let p = |s: &str| format!("blk.{l}.{s}");
            let full = (l + 1) % c.interval == 0 || l >= c.n_layer;
            let z = W { cell: 0, ty: 0, rows: 0, cols: 0, raw: 0 };
            let sg = { let (ty, cols, _) = m.dims2(&p("ffn_gate_inp_shexp.weight")); let mut o = vec![0f32; cols]; crate::quant::dequant_row(ty, m.raw(&p("ffn_gate_inp_shexp.weight")), &mut o); let d = dup(&o); W { cell: dup(&[d]), ty: 0, rows: 1, cols, raw: d } };
            let mut ly = Lay {
                attn_norm: vec1(&p("attn_norm.weight")), post_norm: vec1(&p("post_attention_norm.weight")), router: { let d = vec1(&p("ffn_gate_inp.weight")); W { cell: dup(&[d]), ty: 0, rows: c.n_expert, cols: c.n_embd, raw: d } }, shg_in: sg,
                sh_gate: mat(&p("ffn_gate_shexp.weight")), sh_up: mat(&p("ffn_up_shexp.weight")), sh_down: mat(&p("ffn_down_shexp.weight")),
                full, q: None, k: None, v: None, o: None, qn: 0, kn: 0, qkv: None, gate: None, beta: None, alpha: None, ssm_out: None, conv_w: 0, a_log: 0, dt_b: 0, ssm_norm: 0, wr: 0, bd_ls: 0, bd_hb: 0, bd_hc: 0,
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
                ly.conv_w = vec1(&p("ssm_conv1d.weight")); ly.a_log = { let mut a = vec![0f32; c.n_vhead]; crate::quant::dequant_row(m.dims2(&p("ssm_a")).0, m.raw(&p("ssm_a")), &mut a); if m.ssm_a_raw { for v in a.iter_mut() { *v = -v.exp(); } } dup(&a) }; ly.dt_b = vec1(&p("ssm_dt.bias")); ly.ssm_norm = vec1(&p("ssm_norm.weight"));
                if draft_on { ly.wr = vec1(&p("ssm_conv1d.weight_right")); ly.bd_ls = vec1(&p("ssm_bd_logit_scale.weight")); ly.bd_hb = vec1(&p("ssm_bd_history.bias")); ly.bd_hc = vec1(&p("ssm_bd_history.weight")); }
            }
            lay.push(ly);
        }
        let out_norm = vec1("output_norm.weight");
        let out_w = mat("output.weight");
        let (hd, nkv, nh) = (c.head_dim, c.n_kv, c.n_head);
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        let (rope_tab, rope_ms) = {
            let half = c.rope_dim / 2;
            let scale: f32 = std::env::var("LE_ROPE_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            let yarn = std::env::var("LE_ROPE").map_or(false, |v| v == "yarn") && scale > 1.0;
            let orig: f32 = std::env::var("LE_YARN_ORIG").ok().and_then(|v| v.parse().ok()).unwrap_or(32768.0);
            let corr = |nrot: f32| c.rope_dim as f32 * (orig / (nrot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * c.rope_base.ln());
            let (lo, hi) = ((corr(32.0)).floor().max(0.0), (corr(1.0)).ceil().min(c.rope_dim as f32 - 1.0));
            let fs = 1.0 / scale;
            let tab: Vec<f32> = (0..half).map(|i| {
                let ex = c.rope_base.powf(-2.0 * i as f32 / c.rope_dim as f32);
                if !yarn { return ex; }
                let ramp = 1.0 - ((i as f32 - lo) / (hi - lo).max(0.001)).clamp(0.0, 1.0);
                ex * (fs * (1.0 - ramp) + ramp)
            }).collect();
            (tab, if yarn { 1.0 + 0.1 * (1.0 / fs).ln() } else { 1.0 })
        };
        let rope_f = dup(&rope_tab);
        let pbase = dalloc(192);
        let kvq: i32 = match std::env::var("LE_KV").unwrap_or_default().to_lowercase().as_str() { "" | "f16" | "fp16" => 0, s if s.contains("tcq3") || s == "tcq" => 3, s if s.contains('4') || s.starts_with("tcq") => 2, _ => 1 };
        let (mut ksc, mut vsc) = (vec![], vec![]);
        let (mut conv, mut sdelta, mut kc, mut vc) = (vec![], vec![], vec![], vec![]);
        for l in 0..nall {
            if (l + 1) % c.interval == 0 || l >= c.n_layer {
                conv.push(0); sdelta.push(0);
                let eb = match kvq { 0 => ctx_len * nkv * hd * 2, 1 => ctx_len * nkv * hd, 3 => ctx_len * nkv * (hd / 128) * 52, _ => ctx_len * nkv * hd / 2 };
                let (a, b) = (dalloc(eb), dalloc(eb)); kc.push(a); vc.push(b);
                let sb = if kvq == 0 || kvq == 3 { 16 } else { ctx_len * nkv * (hd / 32) * 2 }; ksc.push(dalloc(sb)); vsc.push(dalloc(sb));
            } else {
                let (a, b) = (dalloc(conv_ch * (c.conv_k - 1) * 4), dalloc(c.n_vhead * c.d_state * c.d_state * 4));
                dzero(a, conv_ch * (c.conv_k - 1) * 4); dzero(b, c.n_vhead * c.d_state * c.d_state * 4);
                conv.push(a); sdelta.push(b); kc.push(0); vc.push(0); ksc.push(0); vsc.push(0);
            }
        }
        let mut slot_bytes = 0;
        for l in 0..nall { slot_bytes = slot_bytes.max(m.expert_bytes(l).iter().sum::<usize>()); }
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
        let mut store_off = vec![0usize; nall + 1];
        for l in 0..nall { store_off[l + 1] = store_off[l] + m.expert_bytes(l).iter().sum::<usize>() * c.n_expert; }
        let mut store_p: *mut std::ffi::c_void = std::ptr::null_mut();
        let store_ok = std::env::var("LE_NOSTORE").is_err() && unsafe { sys::cuMemHostAlloc(&mut store_p, store_off[nall], 0).result().is_ok() };
        if store_ok {
            let t0 = std::time::Instant::now();
            let sp = store_p as usize;
            use rayon::prelude::*;
            (0..nall).into_par_iter().for_each(|l| {
                let eb: usize = m.expert_bytes(l).iter().sum();
                for e in 0..c.n_expert {
                    let mut o = sp + store_off[l] + e * eb;
                    for part in m.expert_parts(l, e) { unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), o as *mut u8, part.len()) }; o += part.len(); }
                }
            });
            eprintln!("pinned expert store {} MiB in {:.1}s", store_off[nall] >> 20, t0.elapsed().as_secs_f64());
        } else { store_p = std::ptr::null_mut(); }
        let mtp = if mtp_on { Some(Mtp { eh: mat(&format!("blk.{}.nextn.eh_proj.weight", c.n_layer)), enorm: vec1(&format!("blk.{}.nextn.enorm.weight", c.n_layer)), hnorm: vec1(&format!("blk.{}.nextn.hnorm.weight", c.n_layer)), shn: vec1(&format!("blk.{}.nextn.shared_head_norm.weight", c.n_layer)) }) } else { None };
        let snap: Vec<(Dp, Dp)> = if mtp_on || draft_on { (0..c.n_layer).map(|l| if conv[l] != 0 { (dalloc(conv_ch * (c.conv_k - 1) * 4), dalloc(c.n_vhead * c.d_state * c.d_state * 4)) } else { (0, 0) }).collect() } else { vec![] };
        if mtp_on { let l = c.n_layer; let _ = l; dzero(kc[c.n_layer], if kvq == 0 { ctx_len * nkv * hd * 2 } else if kvq == 1 { ctx_len * nkv * hd } else if kvq == 3 { ctx_len * nkv * (hd / 128) * 52 } else { ctx_len * nkv * hd / 2 }); }
        let ib = dalloc(256);
        let (tab_h, tab_d) = unsafe {
            let mut hp: *mut std::ffi::c_void = std::ptr::null_mut();
            sys::cuMemHostAlloc(&mut hp, nall * c.n_expert * 4, 2).result().expect("mapped slot table");
            let mut dp: u64 = 0; sys::cuMemHostGetDevicePointer_v2(&mut dp, hp, 0).result().unwrap();
            std::slice::from_raw_parts_mut(hp as *mut i32, nall * c.n_expert).fill(-1);
            (hp as *mut i32, dp)
        };
        let egb = a(16 * c.ff_exp);
        Gpu {
            tab_h, tab_d,
            prot: vec![false; n_slots],
            nall, mtp, hprev: dalloc(c.n_embd * 4), mcat: dalloc(2 * c.n_embd * 4), snap, rows_out: vec![], want_rows: false, preview: false, draft_on, drafts_iters: 0,
            store: store_p as *mut u8, store_off, pf: Some(Box::new(pf)), mmq: std::env::var("LE_MMQ").map_or(true, |v| v != "0"), dbg: std::env::var("LE_DBG").is_ok(), dbg_pos: std::env::var("LE_DBG").ok().and_then(|v| v.parse().ok()).unwrap_or(0), wave: 0, comp_ev: mk_events(), cp_ev: mk_events(),
            cs, ev, stg: stg as *mut u8, pf_pending: false, ph: a(c.n_embd), plog: a(c.n_expert), pidx: ib + 64, prw: a(16), pf_extra, pf_issued: 0,
            x: a(c.n_embd), h: a(c.n_embd), t0: a(big), t1: a(big), t2: a(big), t3: a(big), y: a(big), logits: a(m.vocab.max(c.n_expert)), scores: a(nh * ctx_len),
            idx: ib, rw: a(16), shg: a(4),
            eg: egb, eu: egb + (8 * c.ff_exp * 4) as u64, ea: a(c.n_used * c.ff_exp), ed: a(c.n_used * c.n_embd),
            shg_t: a(c.ff_shexp), shu_t: a(c.ff_shexp), sha_t: a(c.ff_shexp), shd_t: a(c.n_embd),
            ptr_g: pbase, ptr_u: pbase + 64, ptr_d: pbase + 128,
            xqb: [dalloc(16384), dalloc(16384)], xdb: [dalloc(2048), dalloc(2048)], xsb: [dalloc(2048), dalloc(2048)],
            conv, sdelta, kc, vc, lay, out_norm, out_w,
            arena, slot_bytes, n_slots, slot_of: HashMap::new(), key_of: vec![None; n_slots], last_use: vec![0; n_slots], tick: 0, hits: 0, misses: 0, ctx_len, pos: 0, kscr: dalloc(prefill::tmax().max(1) * nkv * hd * 4), pv_part: dalloc((ctx_len / 256 + 2) * nh * hd * 4), pheat: vec![0; c.n_layer * c.n_expert], rope_f, rope_ms, kvq, ksc, vsc,
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
        let name = match ty { 2 => "matvec_2", 8 => "matvec_8", 11 => "matvec_11", 12 => "matvec_12", 13 => "matvec_13", 14 => "matvec_14", 16 => "matvec_16", 17 => "matvec_17", 19 => "matvec_19", 20 => "matvec_20", 21 => "matvec_21", 23 => "matvec_23", 18 => "matvec_18", 22 => "matvec_22", 29 => "matvec_29", 134 => "matvec_134", 135 => "matvec_135", 137 => "matvec_137", 138 => "matvec_138", 139 => "matvec_139", 140 => "matvec_140", 141 => "matvec_141", 144 => "matvec_144", 145 => "matvec_145", 146 => "matvec_146", 152 => "matvec_152", 153 => "matvec_153", 154 => "matvec_154", 155 => "matvec_155", 156 => "matvec_156", 157 => "matvec_157", 158 => "matvec_158", t => panic!("no kernel for type {t}") };
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
            if self.lay[l].full { let (n, ns) = self.kv_bytes(self.pos); rd(self.kc[l], n, &mut out); rd(self.vc[l], n, &mut out); if ns > 0 { rd(self.ksc[l], ns, &mut out); rd(self.vsc[l], ns, &mut out); } }
            else { rd(self.conv[l], conv_ch * (c.conv_k - 1) * 4, &mut out); rd(self.sdelta[l], c.n_vhead * c.d_state * c.d_state * 4, &mut out); }
        }
        out
    }
    /// After a long prefill, fill the expert arena with the experts the prompt routed to most (all layers), so decode starts warm.
    pub fn warm_cache(&mut self) {
        if self.store.is_null() { return; }
        let ne = self.m.cfg.n_expert;
        let mut order: Vec<usize> = (0..self.pheat.len()).filter(|&i| self.pheat[i] > 0).collect();
        order.sort_by(|&a, &b| self.pheat[b].cmp(&self.pheat[a]));
        order.truncate(self.n_slots);
        let pin_frac: f32 = std::env::var("LE_PIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let npin = (order.len() as f32 * pin_frac) as usize;
        let want: std::collections::HashSet<(usize, usize)> = order.iter().map(|&i| (i / ne, i % ne)).collect();
        let mut free: Vec<usize> = (0..self.n_slots).filter(|&s| match self.key_of[s] { None => true, Some(k) => !want.contains(&k) }).collect();
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap() };
        for &(l, e) in order.iter().map(|&i| (i / ne, i % ne)).collect::<Vec<_>>().iter() {
            if self.slot_of.contains_key(&(l, e)) { continue; }
            let Some(s) = free.pop() else { break };
            if let Some(k) = self.key_of[s] { self.slot_of.remove(&k); unsafe { *self.tab_h.add(k.0 * ne + k.1) = -1 }; }
            self.key_of[s] = Some((l, e)); self.slot_of.insert((l, e), s); unsafe { *self.tab_h.add(l * ne + e) = s as i32 }; self.tick += 1; self.last_use[s] = self.tick;
            let eb: usize = self.lay[l].ebytes.iter().sum();
            unsafe { sys::cuMemcpyHtoDAsync_v2(self.arena + (s * self.slot_bytes) as u64, self.store.add(self.store_off[l] + e * eb) as *const _, eb, self.st.cu_stream()).result().unwrap() };
        }
        unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap() };
        self.prot.iter_mut().for_each(|p| *p = false);
        for &i in order.iter().take(npin) { if let Some(&sl) = self.slot_of.get(&(i / ne, i % ne)) { self.prot[sl] = true; } }
        self.pheat.iter_mut().for_each(|h| *h = 0);
    }
    /// TCQ: encode this step's K (post-rope, in kscr) and V (raw) for `t` tokens at pos0, then rotate the queries `q` the same way.
    pub(crate) fn tcq_after_prep(&self, l: usize, t: usize, pos0: usize, vsrc: Dp, q: Dp) {
        let c = &self.m.cfg; let (nkv, hd, nh) = (c.n_kv, c.head_dim, c.n_head); let hdb = hd / 128;
        let groups = (t * nkv * hdb) as u32;
        self.launch("tcq_encode", (groups, 1), 512, 0, &[P(self.kscr), P(self.kc[l]), I(0), I(nkv as i32), I(hdb as i32), I(pos0 as i32), F(1.0)]);
        self.launch("tcq_encode", (groups, 1), 512, 0, &[P(vsrc), P(self.vc[l]), I(1), I(nkv as i32), I(hdb as i32), I(pos0 as i32), F(1.04)]);
        self.launch("tcq_rot", ((t * nh * hd / 128) as u32, 1), 128, 0, &[P(q), P(0), I(0)]);
    }
    /// (data bytes per K or V buffer, scale bytes per buffer) for `pos` cached positions.
    fn kv_bytes(&self, pos: usize) -> (usize, usize) {
        let c = &self.m.cfg; let v = pos * c.n_kv * c.head_dim;
        match self.kvq { 0 => (v * 2, 0), 3 => (pos * c.n_kv * (c.head_dim / 128) * 52, 0), 1 => (v, pos * c.n_kv * (c.head_dim / 32) * 2), _ => (v / 2, pos * c.n_kv * (c.head_dim / 32) * 2) }
    }
    pub fn import_state(&mut self, b: &[u8]) -> Result<(), String> {
        self.bind();
        let c = self.m.cfg.clone();
        let conv_ch = c.d_inner + 2 * c.n_group * c.d_state;
        if b.len() < 8 { return Err("short".into()); }
        let pos = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        if pos >= self.ctx_len { return Err("state longer than context".into()); }
        let kvb = |p: usize| self.kv_bytes(p);
        let mut o = 8usize;
        let mut wr = |p: Dp, n: usize| -> Result<(), String> { if o + n > b.len() { return Err("truncated".into()) } unsafe { result::memcpy_htod_sync(p, &b[o..o + n]).unwrap() }; o += n; Ok(()) };
        for l in 0..c.n_layer {
            if self.lay[l].full { let (n, ns) = kvb(pos); wr(self.kc[l], n)?; wr(self.vc[l], n)?; if ns > 0 { wr(self.ksc[l], ns)?; wr(self.vsc[l], ns)?; } }
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
            if pinned.contains(&s) || self.prot[s] { continue; }
            let t = if self.key_of[s].is_none() { 0 } else { self.last_use[s] };
            if t < bt { bt = t; best = s; }
        }
        let ne = self.m.cfg.n_expert;
        if let Some(k) = self.key_of[best] { self.slot_of.remove(&k); unsafe { *self.tab_h.add(k.0 * ne + k.1) = -1 }; }
        self.key_of[best] = Some((layer, e)); self.slot_of.insert((layer, e), best); unsafe { *self.tab_h.add(layer * ne + e) = best as i32 }; self.last_use[best] = self.tick;
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
            if ids.len() >= 256 && std::env::var("LE_WARM").map_or(true, |v| v != "0") { self.warm_cache(); } else { self.pheat.iter_mut().for_each(|h| *h = 0); }
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
        self.run_layers(0, c.n_layer, pos);
        self.pos += 1;
        if self.mtp.is_some() { unsafe { result::memcpy_dtod_async(self.hprev, self.x, c.n_embd * 4, self.st.cu_stream()).unwrap() }; }
        if !want { self.st.synchronize().unwrap(); return vec![]; }
        self.head_logits()
    }
    fn head_logits(&mut self) -> Vec<f32> {
        let c = self.m.cfg.clone();
        self.rms(self.x, self.out_norm, self.h, c.n_embd);
        self.qx(self.h, c.n_embd, 0); self.mv(self.out_w, self.h, 0, self.logits);
        let mut out = vec![0f32; self.m.vocab];
        unsafe { result::memcpy_dtoh_sync(&mut out, self.logits).unwrap() };
        out
    }
    pub fn mtp_enabled(&self) -> bool { self.mtp.is_some() }
    /// Draft the token after `tok` with the MTP head; `hprev` holds the trunk hidden state of the position before `tok`.
    pub fn mtp_draft(&mut self, tok: u32) -> u32 {
        let c = self.m.cfg.clone();
        let pos = self.pos;
        let (enorm, hnorm, shn, eh) = { let m = self.mtp.as_ref().unwrap(); (m.enorm, m.hnorm, m.shn, m.eh) };
        unsafe { result::memcpy_htod_sync(self.t0, &self.m.embed(tok)).unwrap() };
        self.rms(self.t0, enorm, self.mcat, c.n_embd);
        self.rms(self.hprev, hnorm, self.mcat + (c.n_embd * 4) as u64, c.n_embd);
        self.qx(self.mcat, 2 * c.n_embd, 0); self.mv(eh, self.mcat, 0, self.x);
        self.run_layers(c.n_layer, c.n_layer + 1, pos);
        self.rms(self.x, shn, self.h, c.n_embd);
        self.qx(self.h, c.n_embd, 0); self.mv(self.out_w, self.h, 0, self.logits);
        let mut out = vec![0f32; self.m.vocab];
        unsafe { result::memcpy_dtoh_sync(&mut out, self.logits).unwrap() };
        out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
    }
    // ---- Clover block-diffusion self-speculation ----
    pub const BLOCK: usize = 32;
    /// Draft the 31 tokens after `cur` by iteratively denoising a block of mask tokens (read-only against the committed state).
    pub fn clover_draft(&mut self, cur: u32, thr: f32) -> Vec<u32> {
        const MASK: u32 = 151669;
        let n = Self::BLOCK.min(self.ctx_len - self.pos);
        let mut blk = vec![MASK; n]; blk[0] = cur;
        let mut pending: Vec<usize> = (1..n).collect();
        let mut iters = 0;
        while !pending.is_empty() && iters < n {
            iters += 1;
            self.bind(); self.preview = true; self.want_rows = true;
            self.prefill_chunk(&blk, false);
            self.preview = false; self.want_rows = false;
            let rows = std::mem::take(&mut self.rows_out);
            let mut cand: Vec<(usize, u32, f32)> = vec![];
            for &i in &pending {
                let r = &rows[i];
                let (mut bi, mut bv) = (0usize, f32::MIN);
                for (j, v) in r.iter().enumerate() { if j as u32 != MASK && *v > bv { bv = *v; bi = j; } }
                let z: f32 = r.iter().enumerate().filter(|(j, _)| *j as u32 != MASK).map(|(_, v)| (v - bv).exp()).sum();
                cand.push((i, bi as u32, 1.0 / z));
            }
            let mut take: Vec<(usize, u32)> = cand.iter().filter(|c| c.2 > thr).map(|c| (c.0, c.1)).collect();
            if take.is_empty() { let b = cand.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap(); take.push((b.0, b.1)); }
            for (i, t) in take { blk[i] = t; pending.retain(|p| *p != i); }
        }
        self.drafts_iters += iters as u64;
        blk[1..].to_vec()
    }
    pub fn snapshot_state(&self, restore: bool) { self.snapshot(restore) }
    /// Feed `toks` causally and return the logits after each one.
    pub fn feed_rows(&mut self, toks: &[u32]) -> Vec<Vec<f32>> {
        self.bind(); self.want_rows = true;
        self.prefill_chunk(toks, false);
        self.want_rows = false;
        std::mem::take(&mut self.rows_out)
    }
    pub fn feed(&mut self, toks: &[u32]) { self.bind(); self.prefill_chunk(toks, false); }
    fn snapshot(&self, restore: bool) {
        let c = &self.m.cfg;
        let (cb, sb) = (c.d_inner + 2 * c.n_group * c.d_state, c.n_vhead * c.d_state * c.d_state * 4);
        let cb = cb * (c.conv_k - 1) * 4;
        for l in 0..c.n_layer { if self.snap[l].0 != 0 {
            let (a, b) = if restore { (self.conv[l], self.snap[l].0) } else { (self.snap[l].0, self.conv[l]) };
            let (a2, b2) = if restore { (self.sdelta[l], self.snap[l].1) } else { (self.snap[l].1, self.sdelta[l]) };
            unsafe { result::memcpy_dtod_async(a, b, cb, self.st.cu_stream()).unwrap(); result::memcpy_dtod_async(a2, b2, sb, self.st.cu_stream()).unwrap(); }
        } }
    }
    /// One speculative step. `cur` is the next token to feed (not yet processed). Returns (tokens committed, next cur).
    pub fn spec_step(&mut self, cur: u32) -> (Vec<u32>, u32) {
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        let d = self.mtp_draft(cur);
        self.snapshot(false);
        let p0 = self.pos;
        self.want_rows = true;
        self.prefill_chunk_pub(&[cur, d]);
        self.want_rows = false;
        let rows = std::mem::take(&mut self.rows_out);
        let n0 = argmax(&rows[0]);
        let (n_embd, pfx) = (self.m.cfg.n_embd, self.pf_x());
        if n0 == d {
            let n1 = argmax(&rows[1]);
            unsafe { result::memcpy_dtod_async(self.hprev, pfx + (n_embd * 4) as u64, n_embd * 4, self.st.cu_stream()).unwrap() };
            (vec![cur, d], n1)
        } else {
            self.snapshot(true); self.pos = p0;
            self.forward_n(cur, false);
            (vec![cur], n0)
        }
    }
    fn run_layers(&mut self, lo: usize, hi: usize, pos: usize) {
        let c = self.m.cfg.clone();
        for l in lo..hi {
            let ly = &self.lay[l] as *const Lay; let ly = unsafe { &*ly };
            self.rms(self.x, ly.attn_norm, self.h, c.n_embd);
            if ly.full {
                let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
                self.qx(self.h, c.n_embd, 0); self.mv(ly.q.unwrap(), self.h, 0, self.t0); self.mv(ly.k.unwrap(), self.h, 0, self.t1); self.mv(ly.v.unwrap(), self.h, 0, self.t2);
                self.launch("attn_prep", ((nh + nkv) as u32, 1), 256, 0, &[P(self.t0), P(self.t1), P(self.t2), P(ly.qn), P(ly.kn), P(self.y), P(self.t3), P(self.kc[l]), P(self.vc[l]),
                    I(nh as i32), I(nkv as i32), I(hd as i32), I(c.rope_dim as i32), F(self.rope_ms), I(pos as i32), F(c.eps), P(if self.kvq == 3 { self.kscr } else { self.ksc[l] }), P(self.vsc[l]), I(self.kvq), P(self.rope_f)]);
                if self.kvq == 3 { self.tcq_after_prep(l, 1, pos, self.t2, self.y); }
                let n = pos + 1;
                self.launch3("attn_scores_b", (n.div_ceil(8) as u32, nkv as u32, 1), 256, 0, &[P(self.y), P(self.kc[l]), P(self.scores), I(nkv as i32), I(nh as i32), I(hd as i32), I(pos as i32), I(self.ctx_len as i32), I(0), P(self.ksc[l]), I(self.kvq)]);
                self.launch("attn_softmax_b", (nh as u32, 1), 256, 0, &[P(self.scores), I(nh as i32), I(self.ctx_len as i32), I(pos as i32)]);
                let nsp = n.div_ceil(256);
                self.launch3("attn_pv_split", (nkv as u32, nsp as u32, 1), hd as u32, 0, &[P(self.scores), P(self.vc[l]), P(self.pv_part), I(nkv as i32), I(nh as i32), I(hd as i32), I(n as i32), I(self.ctx_len as i32), P(self.vsc[l]), I(self.kvq), I(256)]);
                self.launch("attn_pv_reduce", (nh as u32, 1), hd as u32, 0, &[P(self.pv_part), P(self.t3), P(self.t0), I(nh as i32), I(hd as i32), I(nsp as i32), I(self.kvq)]);
                if self.kvq == 3 { self.launch("tcq_rot", ((nh * hd / 128) as u32, 1), 128, 0, &[P(self.t0), P(self.t3), I(1)]); }
                self.qx(self.t0, nh * hd, 1); self.mv(ly.o.unwrap(), self.t0, 1, self.t1);
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
            }
            // MoE
            if self.dbg && pos == self.dbg_pos { self.dump("mix", self.x, c.n_embd); }
            self.launch("add_rmsnorm", (1, 1), 256, 0, &[P(self.x), P(self.t1), P(ly.post_norm), P(self.h), I(c.n_embd as i32), F(c.eps)]);
            self.mv(ly.router, self.h, 0, self.logits);
            self.qx(self.h, c.n_embd, 0);
            self.launch("router_topk", (1, 1), 32, 0, &[P(self.logits), I(c.n_expert as i32), I(c.n_used as i32), P(self.idx), P(self.rw)]);
            let kk = (c.n_used + self.pf_extra).min(16);
            let has_next = l + 1 < c.n_layer;
            if has_next {
                let nl = &self.lay[l + 1] as *const Lay; let nl = unsafe { &*nl };
                self.rms(self.x, nl.post_norm, self.ph, c.n_embd);
                self.mv(nl.router, self.ph, 0, self.plog);
                self.launch("router_topk", (1, 1), 32, 0, &[P(self.plog), I(c.n_expert as i32), I(kk as i32), P(self.pidx), P(self.prw)]);
            }
            self.mv(ly.sh_gate, self.h, 0, self.shg_t); self.mv(ly.sh_up, self.h, 0, self.shu_t);
            self.swiglu(self.shg_t, self.shu_t, self.sha_t, c.ff_shexp);
            self.qx(self.sha_t, c.ff_shexp, 1); self.mv(ly.sh_down, self.sha_t, 1, self.shd_t);
            self.mv(ly.shg_in, self.h, 0, self.shg);
            let mut both = [0i32; 32];
            unsafe { result::memcpy_dtoh_sync(&mut both, self.idx).unwrap() };
            let idx = &both[..16]; let pi = &both[16..32];
            let mut slots: Vec<usize> = Vec::new();
            for i in 0..c.n_used { let s = self.slot_for(l, idx[i] as usize, &slots); slots.push(s); }
            if self.pf_pending { unsafe { sys::cuStreamWaitEvent(self.st.cu_stream(), self.ev, 0).result().unwrap() }; self.pf_pending = false; }
            if has_next { self.prefetch(l + 1, &pi[..kk], &slots); }
            let base = |s: usize, part: usize| self.arena + (s * self.slot_bytes) as u64 + ly.ebytes[..part].iter().sum::<usize>() as u64;
            let (o1, o2) = (ly.ebytes[0] as u64, (ly.ebytes[0] + ly.ebytes[1]) as u64);
            self.launch("build_ptrs", (1, 1), 32, 0, &[P(self.idx), P(self.tab_d), I((l * c.n_expert) as i32), I(c.n_used as i32), P(self.arena), P(self.slot_bytes as u64), P(o1), P(o2), P(self.ptr_g), P(self.idx + 128)]);
            let k = c.n_used;
            if ly.ety[0] == ly.ety[1] && k == 8 { self.mv_ptrs(ly.ety[0], self.ptr_g, self.h, 0, self.eg, c.ff_exp, c.n_embd, 0, 16); }
            else {
                self.mv_ptrs(ly.ety[0], self.ptr_g, self.h, 0, self.eg, c.ff_exp, c.n_embd, 0, k);
                self.mv_ptrs(ly.ety[1], self.ptr_u, self.h, 0, self.eu, c.ff_exp, c.n_embd, 0, k);
            }
            self.swiglu(self.eg, self.eu, self.ea, k * c.ff_exp);
            self.qx(self.ea, k * c.ff_exp, 1);
            self.mv_ptrs(ly.ety[2], self.ptr_d, self.ea, 1, self.ed, c.n_embd, c.ff_exp, c.ff_exp, k);
            self.launch("moe_combine", (c.n_embd.div_ceil(256) as u32, 1), 256, 0, &[P(self.x), P(self.ed), P(self.rw), I(k as i32), P(self.shd_t), P(self.shg), I(c.n_embd as i32)]);
            if self.dbg && pos == self.dbg_pos { let mut v = vec![0f32; c.n_embd]; unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut v, self.x).unwrap() }; eprintln!("L{l} {:.4} {:.4}", v.iter().map(|a| a * a).sum::<f32>().sqrt(), v[..4].iter().map(|a| format!("{a:.3}")).collect::<Vec<_>>().join(",")); }
        }
    }
}

/// Minimum CUDA compute capability: the prefill kernels use int8 and f16 tensor-core `mma.sync` (Ampere, sm_80).
pub const MIN_CC: (i32, i32) = (8, 0);
/// NVRTC target for the device behind `ctx`, or a plain-language error when the GPU is below the minimum.
pub fn cuda_arch(ctx: &CudaContext) -> Result<&'static str, String> {
    let (maj, min) = ctx.compute_capability().map_err(|e| format!("cannot query the GPU compute capability: {e:?}"))?;
    arch_for(&ctx.name().unwrap_or_else(|_| "this GPU".into()), maj, min)
}
fn arch_for(name: &str, maj: i32, min: i32) -> Result<&'static str, String> {
    if (maj, min) < MIN_CC {
        return Err(format!("{name} has compute capability {maj}.{min}, but LastEngine needs {}.{} or newer (NVIDIA Ampere or later). Use a newer GPU, or run the model with llama.cpp.", MIN_CC.0, MIN_CC.1));
    }
    Ok(Box::leak(format!("sm_{maj}{min}").into_boxed_str()))
}
#[cfg(test)]
mod arch_tests {
    #[test]
    fn arch_selection() {
        assert_eq!(super::arch_for("x", 8, 6).unwrap(), "sm_86");
        assert_eq!(super::arch_for("x", 12, 0).unwrap(), "sm_120");
        let e = super::arch_for("GTX 1060", 6, 1).unwrap_err();
        assert!(e.contains("GTX 1060") && e.contains("6.1") && e.contains("8.0"));
    }
}
/// True when device 0 exists and meets `MIN_CC` (used to pick GPU or CPU without panicking).
pub fn cuda_usable() -> bool {
    std::panic::catch_unwind(|| CudaContext::new(0).ok().map_or(false, |c| cuda_arch(&c).is_ok())).unwrap_or(false)
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

/// NVRTC source: lookup tables followed by src/cuda/kernels.cu.
pub fn kernel_source() -> String {
        let mut src = String::from("#include <cuda_fp16.h>\n__device__ const unsigned int IQ3_G32[256] = {");
        for g in crate::quant::iq3_grid().iter() { src.push_str(&format!("{}u,", u32::from_le_bytes(*g))); }
        src.push_str("};\n");
        for (nm, g2) in [("IQ2XXS_G", crate::quant::iq2xxs_grid().iter().map(|g| *g).collect::<Vec<_>>()), ("IQ2S_G", crate::quant::iq2s_grid().iter().map(|g| *g).collect::<Vec<_>>())] {
            src.push_str(&format!("__device__ const unsigned int {nm}[{}] = {{", g2.len() * 2));
            for g in g2 { src.push_str(&format!("{}u,{}u,", u32::from_le_bytes(g[0..4].try_into().unwrap()), u32::from_le_bytes(g[4..8].try_into().unwrap()))); }
            src.push_str("};\n");
        }
        src.push_str("__device__ const unsigned int IQ2XS_G[1024] = {");
        for g in IQ2XS_GRID.iter() { src.push_str(&format!("{}u,{}u,", u32::from_le_bytes(g[0..4].try_into().unwrap()), u32::from_le_bytes(g[4..8].try_into().unwrap()))); }
        src.push_str("};\n__device__ const unsigned int IQ3S_G[512] = {");
        for g in IQ3S_GRID.iter() { src.push_str(&format!("{}u,", u32::from_le_bytes(*g))); }
        src.push_str("};\n__device__ const unsigned int IQ1S_G[4096] = {");
        for g in IQ1S_GRID.iter() { let u: Vec<u8> = g.iter().map(|&x| x as u8).collect(); src.push_str(&format!("{}u,{}u,", u32::from_le_bytes(u[0..4].try_into().unwrap()), u32::from_le_bytes(u[4..8].try_into().unwrap()))); }
        src.push_str("};\n");
        for (nm, bytes, n) in [("TCQ_CB_K", &include_bytes!("../assets/tcq3_cb_k.bin")[..], 512), ("TCQ_CB_V", &include_bytes!("../assets/tcq3_cb_v.bin")[..], 512),
                               ("TCQ_S1", &include_bytes!("../assets/tcq3_signs1.bin")[..], 128), ("TCQ_S2", &include_bytes!("../assets/tcq3_signs2.bin")[..], 128)] {
            src.push_str(&format!("__device__ const float {nm}[{n}] = {{"));
            for ch in bytes.chunks(4) { src.push_str(&format!("{:e}f,", f32::from_le_bytes(ch.try_into().unwrap()))); }
            src.push_str("};\n");
        }
        for (nm, ty, v) in [("IQ2NL_V", "signed char", IQ2NL_VALUES.iter().map(|&x| x as i32).collect::<Vec<_>>()), ("IQ3NL_V", "signed char", IQ3NL_VALUES.iter().map(|&x| x as i32).collect()),
                            ("IQ4K_V", "signed char", IQ4K_VALUES.iter().map(|&x| x as i32).collect()), ("IQ5NL_V", "signed char", IQ5NL_VALUES.iter().map(|&x| x as i32).collect()),
                            ("IQ6NL_V", "signed char", IQ6NL_VALUES.iter().map(|&x| x as i32).collect()), ("IQ2KL_V", "unsigned short", IQ2KL_VALUES.iter().map(|&x| x as i32).collect())] {
            src.push_str(&format!("__device__ const {ty} {nm}[{}] = {{", v.len()));
            for x in v { src.push_str(&format!("{x},")); }
            src.push_str("};\n");
        }
        src.push_str(include_str!("cuda/kernels.cu"));
        src.push_str(include_str!("cuda/dense.cu"));
    src
}

#[cfg(test)]
mod quant_tests {
    use super::*;
    /// For every type listed in GPU_TYPES, runs the CUDA matvec on the oracle rows (ORACLE_DIR/or_<id>.q, 8 rows x 1024) and compares with the CPU dequant dot product.
    #[test]
    fn gpu_matvec_matches_cpu_dequant() {
        let Ok(dir) = std::env::var("ORACLE_DIR") else { return };
        let ctx = CudaContext::new(0).expect("cuda"); let st = ctx.default_stream();
        let inc = ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include/cuda"].iter().find(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists()).map(|s| s.to_string());
        let ptx = compile_ptx_with_opts(kernel_source(), CompileOptions { arch: Some(cuda_arch(&ctx).unwrap()), include_paths: inc.into_iter().collect(), ..Default::default() }).expect("nvrtc");
        let module = ctx.load_module(ptx).expect("module");
        let cols = 1024usize; let rows = 8usize;
        let x: Vec<f32> = (0..cols).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.8).collect();
        let xd = st.memcpy_stod(&x).unwrap();
        let units = cols / 32;
        let (xq, xdv, xs) = (st.alloc_zeros::<i8>(cols).unwrap(), st.alloc_zeros::<f32>(units).unwrap(), st.alloc_zeros::<i32>(units).unwrap());
        let fq = module.load_function("quant_x").unwrap(); let mut b = st.launch_builder(&fq);
        let nu = units as i32;
        unsafe { b.arg(&xd).arg(&xq).arg(&xdv).arg(&xs).arg(&nu).launch(LaunchConfig { grid_dim: (units.div_ceil(8) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }).unwrap(); }
        let mut bad = vec![];
        for ty in [16u32, 17, 18, 19, 20, 21, 22, 23, 29, 134, 135, 137, 138, 139, 140, 141, 144, 145, 146, 152, 153, 154, 155, 156, 157, 158] {
            let Ok(q) = std::fs::read(format!("{dir}/or_{ty}.q")) else { continue };
            if ty == 11 { continue; }
            let rb = crate::quant::row_bytes(ty, cols);
            let w = st.memcpy_stod(&q).unwrap();
            let (wp, _g) = { use cudarc::driver::DevicePtr; let (p, g) = w.device_ptr(&st); (p, g) };
            let ptrs = st.memcpy_stod(&[wp]).unwrap();
            let out = st.alloc_zeros::<f32>(rows).unwrap();
            let lpr = units.min(32); let rpb = 8 * (32 / lpr);
            let name = format!("matvec_{ty}");
            let fm = module.load_function(&name).unwrap(); let mut b = st.launch_builder(&fm);
            let (r_, c_, xs_) = (rows as i32, cols as i32, 0i32);
            unsafe { b.arg(&ptrs).arg(&xq).arg(&xdv).arg(&xs).arg(&out).arg(&r_).arg(&c_).arg(&xs_).launch(LaunchConfig { grid_dim: (rows.div_ceil(rpb) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (units * 36 + units * 8) as u32 }).unwrap(); }
            let got = st.memcpy_dtov(&out).unwrap();
            let mut worst = 0f32;
            for r in 0..rows {
                let mut wv = vec![0f32; cols]; crate::quant::dequant_row(ty, &q[r * rb..(r + 1) * rb], &mut wv);
                let want: f32 = wv.iter().zip(&x).map(|(a, b)| a * b).sum();
                let scale: f32 = wv.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
                worst = worst.max((got[r] - want).abs() / (scale * 0.05 + 1e-4));
            }
            if worst > 1.0 { bad.push(format!("{ty}: err {worst}")); }
            eprintln!("type {ty}: worst err {worst:.3} of tolerance");
        }
        assert!(bad.is_empty(), "GPU mismatch: {bad:?}");
    }
}

// No-GPU check: NVRTC + ptxas must accept every kernel for sm_86 (shared-memory limits, syntax). Runs without a device.
#[cfg(test)]
mod nvrtc_compile {
    use super::*;
    #[test]
    fn kernels_compile_for_sm86() {
        let inc = ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include/cuda"].iter().find(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists()).map(|s| s.to_string());
        let opts = CompileOptions { arch: Some("sm_86"), include_paths: inc.into_iter().collect(), ..Default::default() };
        compile_ptx_with_opts(kernel_source(), opts).expect("kernel source must compile for sm_86");
    }
}
