//! Dense transformer engines on the CPU: arch `llama` (MiniCPM5: rope NORM, plain SwiGLU) and `lfm3`
//! (hybrid: gated attention with qk-norm + sliding window, and short causal convolutions).
//! Weights stay quantized in the mmap and are dequantized per row inside the matvec.
use crate::{gguf::{Gguf, Value}, quant};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind { Llama, Lfm3 }

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum LayerKind { Attn { swa: bool }, Conv }

pub(crate) struct T { pub(crate) ty: u32, off: usize, pub(crate) dims: Vec<usize> }

pub struct Dense {
    pub kind: Kind, mmap: Mmap, pub(crate) t: HashMap<String, T>,
    pub n_layer: usize, pub n_embd: usize, pub n_head: usize, pub n_kv: usize, pub head_dim: usize, pub rope_dim: usize,
    pub eps: f32, pub rope_base: f32, pub rope_base_swa: f32, pub window: usize, pub l_cache: usize, pub vocab: usize,
    pub(crate) layers: Vec<LayerKind>, pub(crate) neox: bool,
}

#[derive(Clone)]
pub struct State { kc: Vec<Vec<f32>>, vc: Vec<Vec<f32>>, conv: Vec<Vec<f32>> }

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * b).collect()
}
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Clear error for architectures no engine here implements (e.g. `dflash` draft models are speculative drafters, not standalone LMs).
pub fn check_arch(g: &Gguf) -> Result<(), String> {
    match g.arch() {
        Some("qwen35moe" | "clover_moe" | "llama" | "lfm3") => Ok(()),
        Some(a) => Err(format!("unsupported model architecture '{a}' (supported: qwen35moe, clover_moe, llama, lfm3){}", if a == "dflash" { "; dflash is a draft model for speculative decoding, not a standalone model" } else { "" })),
        None => Err("GGUF has no general.architecture".into()),
    }
}

impl Dense {
    pub fn supports(g: &Gguf) -> Option<Kind> {
        match g.arch()? { "llama" => Some(Kind::Llama), "lfm3" => Some(Kind::Lfm3), _ => None }
    }
    pub fn load(path: &str, g: &Gguf) -> Dense {
        let kind = Self::supports(g).expect("dense arch");
        let a = g.arch().unwrap().to_string();
        let f = std::fs::File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&f).unwrap() };
        let u = |k: &str| g.kv.get(&format!("{a}.{k}")).and_then(Value::as_u64).unwrap_or_else(|| panic!("{k}")) as usize;
        let fl = |k: &str| g.kv.get(&format!("{a}.{k}")).and_then(Value::as_f32);
        let arr = |k: &str| -> Option<Vec<u64>> {
            match g.kv.get(&format!("{a}.{k}"))? { Value::Arr(v) => Some(v.iter().map(|x| x.as_u64().unwrap_or(0)).collect()), _ => None }
        };
        let n_layer = u("block_count");
        let n_head = u("attention.head_count");
        let kvs = arr("attention.head_count_kv");
        let swa = arr("attention.sliding_window_pattern");
        let layers = (0..n_layer).map(|l| {
            let nkv = match &kvs { Some(v) => v[l] as usize, None => u("attention.head_count_kv") };
            if nkv == 0 { LayerKind::Conv } else { LayerKind::Attn { swa: swa.as_ref().map_or(false, |v| v[l] == 1) } }
        }).collect::<Vec<_>>();
        let n_kv = match &kvs { Some(v) => v.iter().copied().find(|&x| x > 0).unwrap_or(0) as usize, None => u("attention.head_count_kv") };
        let t = g.tensors.iter().map(|t| (t.name.clone(), T { ty: t.ggml_type, off: (g.data_start + t.offset) as usize,
            dims: t.dims.iter().map(|d| *d as usize).collect() })).collect();
        let vocab = g.tensors.iter().find(|t| t.name == "token_embd.weight").unwrap().dims[1] as usize;
        let head_dim = g.kv.get(&format!("{a}.attention.key_length")).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(u("embedding_length") / n_head);
        Dense {
            kind, mmap, t, n_layer, n_embd: u("embedding_length"), n_head, n_kv, head_dim,
            rope_dim: g.kv.get(&format!("{a}.rope.dimension_count")).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(head_dim),
            eps: fl("attention.layer_norm_rms_epsilon").unwrap_or(1e-5), rope_base: fl("rope.freq_base").unwrap_or(10000.0),
            rope_base_swa: fl("rope.freq_base_swa").unwrap_or(10000.0),
            window: g.kv.get(&format!("{a}.attention.sliding_window")).and_then(Value::as_u64).unwrap_or(0) as usize,
            l_cache: g.kv.get(&format!("{a}.shortconv.l_cache")).and_then(Value::as_u64).unwrap_or(0) as usize,
            vocab, layers, neox: kind == Kind::Lfm3,
        }
    }
    pub fn new_state(&self) -> State {
        State { kc: vec![Vec::new(); self.n_layer], vc: vec![Vec::new(); self.n_layer],
                conv: (0..self.n_layer).map(|l| if self.layers[l] == LayerKind::Conv { vec![0.0; self.n_embd * (self.l_cache - 1)] } else { Vec::new() }).collect() }
    }
    pub(crate) fn bytes(&self, name: &str) -> (&T, &[u8]) {
        let t = self.t.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let n: usize = t.dims.iter().product();
        (t, &self.mmap[t.off..t.off + quant::row_bytes(t.ty, n)])
    }
    pub(crate) fn vec_f32(&self, name: &str) -> Vec<f32> {
        let (t, b) = self.bytes(name);
        let mut o = vec![0.0; t.dims.iter().product()];
        quant::dequant_row(t.ty, b, &mut o); o
    }
    fn matvec(&self, name: &str, x: &[f32]) -> Vec<f32> {
        let (t, b) = self.bytes(name);
        let (cols, rows) = (t.dims[0], t.dims[1]);
        let rb = quant::row_bytes(t.ty, cols);
        (0..rows).into_par_iter().with_min_len(16).map_init(|| vec![0f32; cols], |buf, r| {
            quant::dequant_row(t.ty, &b[r * rb..(r + 1) * rb], buf);
            buf.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    }
    fn rope(&self, v: &mut [f32], pos: usize, base: f32, ff: Option<&[f32]>) {
        let half = self.rope_dim / 2;
        for i in 0..half {
            let th = pos as f32 * base.powf(-(2.0 * i as f32) / self.rope_dim as f32) / ff.map_or(1.0, |f| f[i]);
            let (s, c) = th.sin_cos();
            let (ia, ib) = if self.neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
            let (a, b) = (v[ia], v[ib]);
            v[ia] = a * c - b * s; v[ib] = a * s + b * c;
        }
    }
    fn attn(&self, st: &mut State, l: usize, h: &[f32], pos: usize, swa: bool) -> Vec<f32> {
        let p = |s: &str| format!("blk.{l}.{s}");
        let (hd, nh, nkv) = (self.head_dim, self.n_head, self.n_kv);
        let mut q = self.matvec(&p("attn_q.weight"), h);
        let mut k = self.matvec(&p("attn_k.weight"), h);
        let v = self.matvec(&p("attn_v.weight"), h);
        if self.kind == Kind::Lfm3 {
            let (qn, kn) = (self.vec_f32(&p("attn_q_norm.weight")), self.vec_f32(&p("attn_k_norm.weight")));
            for i in 0..nh { let r = rms(&q[i * hd..(i + 1) * hd], &qn, self.eps); q[i * hd..(i + 1) * hd].copy_from_slice(&r); }
            for i in 0..nkv { let r = rms(&k[i * hd..(i + 1) * hd], &kn, self.eps); k[i * hd..(i + 1) * hd].copy_from_slice(&r); }
        }
        let base = if swa { self.rope_base_swa } else { self.rope_base };
        let ffv = if !swa && self.t.contains_key("rope_freqs.weight") { Some(self.vec_f32("rope_freqs.weight")) } else { None };
        for i in 0..nh { self.rope(&mut q[i * hd..(i + 1) * hd], pos, base, ffv.as_deref()); }
        for i in 0..nkv { self.rope(&mut k[i * hd..(i + 1) * hd], pos, base, ffv.as_deref()); }
        st.kc[l].extend_from_slice(&k); st.vc[l].extend_from_slice(&v);
        let n = st.kc[l].len() / (nkv * hd);
        let t0 = if swa && self.window > 0 && n > self.window { n - self.window } else { 0 };
        let mut o = vec![0f32; nh * hd];
        for hh in 0..nh {
            let kvh = hh / (nh / nkv);
            let mut sc: Vec<f32> = (t0..n).map(|t| {
                let kk = &st.kc[l][(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                kk.iter().zip(&q[hh * hd..(hh + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
            }).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut s = 0.0;
            for v in sc.iter_mut() { *v = (*v - m).exp(); s += *v; }
            for (i, t) in (t0..n).enumerate() {
                let vv = &st.vc[l][(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                let w = sc[i] / s;
                for d in 0..hd { o[hh * hd + d] += w * vv[d]; }
            }
        }
        if self.kind == Kind::Lfm3 {
            let g = self.matvec(&p("attn_gate.weight"), h);
            for hh in 0..nh { let s = sigmoid(g[hh]); for d in 0..hd { o[hh * hd + d] *= s; } }
        }
        self.matvec(&p("attn_output.weight"), &o)
    }
    fn shortconv(&self, st: &mut State, l: usize, h: &[f32]) -> Vec<f32> {
        let p = |s: &str| format!("blk.{l}.{s}");
        let (n, lc) = (self.n_embd, self.l_cache);
        let bcx = self.matvec(&p("shortconv.in_proj.weight"), h);
        let w = self.vec_f32(&p("shortconv.conv.weight"));
        let cs = &mut st.conv[l];
        let mut y = vec![0f32; n];
        for ch in 0..n {
            let bx = bcx[ch] * bcx[2 * n + ch];
            let s = &mut cs[ch * (lc - 1)..(ch + 1) * (lc - 1)];
            let mut acc = bx * w[ch * lc + lc - 1];
            for k in 0..lc - 1 { acc += s[k] * w[ch * lc + k]; }
            for k in 0..lc - 2 { s[k] = s[k + 1]; }
            s[lc - 2] = bx;
            y[ch] = bcx[n + ch] * acc;
        }
        self.matvec(&p("shortconv.out_proj.weight"), &y)
    }
    pub fn forward(&self, st: &mut State, token: u32, pos: usize) -> Vec<f32> {
        let mut x = vec![0f32; self.n_embd];
        {
            let (t, b) = self.bytes("token_embd.weight");
            let rb = quant::row_bytes(t.ty, self.n_embd);
            quant::dequant_row(t.ty, &b[token as usize * rb..(token as usize + 1) * rb], &mut x);
        }
        for l in 0..self.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let h = rms(&x, &self.vec_f32(&p("attn_norm.weight")), self.eps);
            let mut out = match self.layers[l] {
                LayerKind::Attn { swa } => self.attn(st, l, &h, pos, swa),
                LayerKind::Conv => self.shortconv(st, l, &h),
            };
            if self.kind == Kind::Lfm3 { out = rms(&out, &self.vec_f32(&p("post_attention_norm.weight")), self.eps); }
            for i in 0..x.len() { x[i] += out[i]; }
            let h2 = rms(&x, &self.vec_f32(&p("ffn_norm.weight")), self.eps);
            let g = self.matvec(&p("ffn_gate.weight"), &h2);
            let u = self.matvec(&p("ffn_up.weight"), &h2);
            let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            let mut f = self.matvec(&p("ffn_down.weight"), &a);
            if self.kind == Kind::Lfm3 { f = rms(&f, &self.vec_f32(&p("post_ffw_norm.weight")), self.eps); }
            for i in 0..x.len() { x[i] += f[i]; }
        }
        let xn = rms(&x, &self.vec_f32("output_norm.weight"), self.eps);
        self.matvec(if self.t.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" }, &xn)
    }
}

/// Stateful engine with the same surface `Local` uses for the Qwen GPU engine: CUDA when available, else the CPU path.
pub enum DenseEng { Cpu { m: Dense, st: State, pos: usize, snap: Option<State>, ctx: usize }, Cuda(Box<crate::gpu::dense::DenseGpu>) }
impl DenseEng {
    pub fn new(m: Dense, ctx: usize) -> DenseEng {
        if std::env::var("LE_CPU").is_err() && crate::gpu::cuda_usable() {
            return DenseEng::Cuda(Box::new(crate::gpu::dense::DenseGpu::new(m, ctx)));
        }
        let st = m.new_state(); DenseEng::Cpu { m, st, pos: 0, snap: None, ctx }
    }
    pub fn pos(&self) -> usize { match self { DenseEng::Cpu { pos, .. } => *pos, DenseEng::Cuda(g) => g.pos() } }
    pub fn reset(&mut self) { match self { DenseEng::Cpu { m, st, pos, .. } => { *st = m.new_state(); *pos = 0; } DenseEng::Cuda(g) => g.reset() } }
    pub fn export_state(&mut self) -> Result<Vec<u8>, String> { match self { DenseEng::Cuda(g) => Ok(g.export_state()), _ => Err("slots need the GPU engine for dense models".into()) } }
    pub fn import_state(&mut self, b: &[u8]) -> Result<(), String> { match self { DenseEng::Cuda(g) => g.import_state(b), _ => Err("slots need the GPU engine for dense models".into()) } }
    pub fn forward(&mut self, t: u32) -> Vec<f32> {
        match self { DenseEng::Cpu { m, st, pos, .. } => { let l = m.forward(st, t, *pos); *pos += 1; l } DenseEng::Cuda(g) => g.forward(t) }
    }
    pub fn prefill(&mut self, ids: &[u32]) -> Vec<f32> {
        match self {
            DenseEng::Cpu { ctx, pos, .. } => { assert!(*pos + ids.len() <= (*ctx).max(1) + 1); let mut l = vec![]; for &t in ids { l = self.forward(t); } l }
            DenseEng::Cuda(g) => g.prefill(ids),
        }
    }
    pub fn snap_recurrent(&mut self) -> Vec<u8> {
        match self { DenseEng::Cpu { st, snap, .. } => { *snap = Some(st.clone()); Vec::new() } DenseEng::Cuda(g) => g.snap_recurrent() }
    }
    pub fn restore_recurrent(&mut self, b: &[u8], p: usize) {
        match self { DenseEng::Cpu { st, snap, pos, .. } => { if let Some(s) = snap { *st = s.clone(); *pos = p; } } DenseEng::Cuda(g) => g.restore_recurrent(b, p) }
    }
}
