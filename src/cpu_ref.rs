//! Slow, simple f32 CPU reference forward pass for qwen35moe (Gated-DeltaNet + gated attention + shared-expert MoE).
//! Purpose: correctness oracle for the fast backends. Dequantizes weight rows on the fly.
use crate::{gguf::{Gguf, Value}, quant};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct Cfg {
    pub n_layer: usize, pub n_embd: usize, pub n_head: usize, pub n_kv: usize, pub head_dim: usize,
    pub rope_dim: usize, pub rope_base: f32, pub eps: f32, pub n_expert: usize, pub n_used: usize,
    pub ff_exp: usize, pub ff_shexp: usize, pub interval: usize,
    pub conv_k: usize, pub d_inner: usize, pub d_state: usize, pub n_group: usize, pub n_vhead: usize,
}

struct T { ty: u32, off: usize, dims: Vec<usize> }

pub struct Model {
    pub cfg: Cfg, mmap: Mmap, t: HashMap<String, T>, pub vocab: usize,
    pub kv_repeat_interleave: bool,
}

pub struct State {
    conv: Vec<Vec<f32>>,       // per layer [channels][k-1]
    s: Vec<Vec<f32>>,          // per layer [vhead][dk][dv]
    kc: Vec<Vec<f32>>, vc: Vec<Vec<f32>>, // per layer [pos][kv][hd]
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * b).collect()
}
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

impl Model {
    pub fn load(path: &str, g: &Gguf) -> Model {
        let f = std::fs::File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&f).unwrap() };
        let u = |k: &str| g.kv.get(&format!("qwen35moe.{k}")).and_then(Value::as_u64).unwrap_or_else(|| panic!("{k}")) as usize;
        let fl = |k: &str| g.kv.get(&format!("qwen35moe.{k}")).and_then(Value::as_f32).unwrap();
        let cfg = Cfg {
            n_layer: u("block_count") - u("nextn_predict_layers"), n_embd: u("embedding_length"), n_head: u("attention.head_count"),
            n_kv: u("attention.head_count_kv"), head_dim: u("attention.key_length"), rope_dim: u("rope.dimension_count"),
            rope_base: fl("rope.freq_base"), eps: fl("attention.layer_norm_rms_epsilon"), n_expert: u("expert_count"),
            n_used: u("expert_used_count"), ff_exp: u("expert_feed_forward_length"), ff_shexp: u("expert_shared_feed_forward_length"),
            interval: u("full_attention_interval"), conv_k: u("ssm.conv_kernel"), d_inner: u("ssm.inner_size"),
            d_state: u("ssm.state_size"), n_group: u("ssm.group_count"), n_vhead: u("ssm.time_step_rank"),
        };
        let t = g.tensors.iter().map(|t| (t.name.clone(), T { ty: t.ggml_type, off: (g.data_start + t.offset) as usize,
            dims: t.dims.iter().map(|d| *d as usize).collect() })).collect();
        let vocab = g.tensors.iter().find(|t| t.name == "token_embd.weight").unwrap().dims[1] as usize;
        Model { cfg, mmap, t, vocab, kv_repeat_interleave: std::env::var("KV_INTERLEAVE").is_ok() }
    }

    pub fn embed(&self, token: u32) -> Vec<f32> {
        let n = self.cfg.n_embd; let mut x = vec![0f32; n];
        let (t, b) = self.bytes("token_embd.weight"); let rb = quant::row_bytes(t.ty, n);
        quant::dequant_row(t.ty, &b[token as usize * rb..(token as usize + 1) * rb], &mut x); x
    }
    pub fn has(&self, name: &str) -> bool { self.t.contains_key(name) }
    pub fn raw(&self, name: &str) -> &[u8] { self.bytes(name).1 }
    /// (ggml type, cols = dims[0], rows = dims[1] or 1)
    pub fn dims2(&self, name: &str) -> (u32, usize, usize) { let t = &self.t[name]; (t.ty, t.dims[0], *t.dims.get(1).unwrap_or(&1)) }
    /// bytes of one expert's (gate, up, down) weights in layer l
    pub fn expert_bytes(&self, l: usize) -> [usize; 3] {
        ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"].map(|n| {
            let t = &self.t[&format!("blk.{l}.{n}")];
            quant::row_bytes(t.ty, t.dims[0]) * t.dims[1]
        })
    }
    pub fn expert_types(&self, l: usize) -> [u32; 3] {
        ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"].map(|n| self.t[&format!("blk.{l}.{n}")].ty)
    }
    pub fn expert_parts(&self, l: usize, e: usize) -> [&[u8]; 3] {
        let sizes = self.expert_bytes(l);
        let names = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
        let mut i = 0;
        names.map(|n| { let t = &self.t[&format!("blk.{l}.{n}")]; let o = t.off + e * sizes[i]; let r = &self.mmap[o..o + sizes[i]]; i += 1; r })
    }
    pub fn new_state(&self) -> State {
        let c = &self.cfg;
        State {
            conv: (0..c.n_layer).map(|_| vec![0.0; (c.d_inner * 2 + 2 * c.n_group * c.d_state) * (c.conv_k - 1)]).collect(),
            s: (0..c.n_layer).map(|_| vec![0.0; c.n_vhead * c.d_state * c.d_state]).collect(),
            kc: vec![Vec::new(); c.n_layer], vc: vec![Vec::new(); c.n_layer],
        }
    }

    fn bytes(&self, name: &str) -> (&T, &[u8]) {
        let t = self.t.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let n: usize = t.dims.iter().product();
        let len = quant::row_bytes(t.ty, n);
        (t, &self.mmap[t.off..t.off + len])
    }
    fn vec_f32(&self, name: &str) -> Vec<f32> {
        let (t, b) = self.bytes(name);
        let mut o = vec![0.0; t.dims.iter().product()];
        quant::dequant_row(t.ty, b, &mut o);
        o
    }
    /// y[r] = W[r,:]·x for rows [r0, r0+rows) of a 2D tensor (row length = dims[0]); `expert` selects a slab of a 3D tensor.
    pub fn matvec(&self, name: &str, expert: usize, x: &[f32]) -> Vec<f32> {
        let (t, b) = self.bytes(name);
        let (cols, rows) = (t.dims[0], t.dims[1]);
        let rb = quant::row_bytes(t.ty, cols);
        let base = expert * rows * rb;
        (0..rows).into_par_iter().map_init(|| vec![0f32; cols], |buf, r| {
            quant::dequant_row(t.ty, &b[base + r * rb..base + (r + 1) * rb], buf);
            buf.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    }

    pub fn forward(&self, st: &mut State, token: u32, pos: usize) -> Vec<f32> {
        let c = &self.cfg;
        let mut x = vec![0f32; c.n_embd];
        {
            let (t, b) = self.bytes("token_embd.weight");
            let rb = quant::row_bytes(t.ty, c.n_embd);
            quant::dequant_row(t.ty, &b[token as usize * rb..(token as usize + 1) * rb], &mut x);
        }
        for l in 0..c.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let h = rms(&x, &self.vec_f32(&p("attn_norm.weight")), c.eps);
            let out = if (l + 1) % c.interval == 0 { self.attn(st, l, &h, pos) } else { self.delta(st, l, &h) };
            for i in 0..x.len() { x[i] += out[i]; }
            let h2 = rms(&x, &self.vec_f32(&p("post_attention_norm.weight")), c.eps);
            let m = self.moe(l, &h2);
            for i in 0..x.len() { x[i] += m[i]; }
        }
        let xf = rms(&x, &self.vec_f32("output_norm.weight"), c.eps);
        self.matvec("output.weight", 0, &xf)
    }

    fn attn(&self, st: &mut State, l: usize, h: &[f32], pos: usize) -> Vec<f32> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{l}.{s}");
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_kv);
        let qf = self.matvec(&p("attn_q.weight"), 0, h);
        let mut k = self.matvec(&p("attn_k.weight"), 0, h);
        let v = self.matvec(&p("attn_v.weight"), 0, h);
        let (qn, kn) = (self.vec_f32(&p("attn_q_norm.weight")), self.vec_f32(&p("attn_k_norm.weight")));
        let mut q = vec![0f32; nh * hd];
        let mut gate = vec![0f32; nh * hd];
        for i in 0..nh {
            q[i * hd..(i + 1) * hd].copy_from_slice(&rms(&qf[i * 2 * hd..i * 2 * hd + hd], &qn, c.eps));
            gate[i * hd..(i + 1) * hd].copy_from_slice(&qf[i * 2 * hd + hd..(i + 1) * 2 * hd]);
        }
        for i in 0..nkv { let r = rms(&k[i * hd..(i + 1) * hd], &kn, c.eps); k[i * hd..(i + 1) * hd].copy_from_slice(&r); }
        let rope = |v: &mut [f32]| {
            let half = c.rope_dim / 2;
            for i in 0..half {
                let th = pos as f32 * c.rope_base.powf(-(2.0 * i as f32) / c.rope_dim as f32);
                let (s, co) = th.sin_cos();
                let (a, b) = (v[i], v[i + half]);
                v[i] = a * co - b * s; v[i + half] = a * s + b * co;
            }
        };
        for i in 0..nh { rope(&mut q[i * hd..(i + 1) * hd]); }
        for i in 0..nkv { rope(&mut k[i * hd..(i + 1) * hd]); }
        st.kc[l].extend_from_slice(&k);
        st.vc[l].extend_from_slice(&v);
        let n = st.kc[l].len() / (nkv * hd);
        let mut o = vec![0f32; nh * hd];
        for hh in 0..nh {
            let kvh = hh / (nh / nkv);
            let mut sc: Vec<f32> = (0..n).map(|t| {
                let kk = &st.kc[l][(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                kk.iter().zip(&q[hh * hd..(hh + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
            }).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut s = 0.0;
            for v in sc.iter_mut() { *v = (*v - m).exp(); s += *v; }
            for t in 0..n {
                let vv = &st.vc[l][(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                for d in 0..hd { o[hh * hd + d] += sc[t] / s * vv[d]; }
            }
        }
        for i in 0..o.len() { o[i] *= sigmoid(gate[i]); }
        self.matvec(&p("attn_output.weight"), 0, &o)
    }

    fn delta(&self, st: &mut State, l: usize, h: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{l}.{s}");
        let (dk, nv, ng) = (c.d_state, c.n_vhead, c.n_group);
        let qkv = self.matvec(&p("attn_qkv.weight"), 0, h);
        let z = self.matvec(&p("attn_gate.weight"), 0, h);
        let beta: Vec<f32> = self.matvec(&p("ssm_beta.weight"), 0, h).iter().map(|v| sigmoid(*v)).collect();
        let alpha = self.matvec(&p("ssm_alpha.weight"), 0, h);
        let (a_log, dtb) = (self.vec_f32(&p("ssm_a")), self.vec_f32(&p("ssm_dt.bias")));
        let cw = self.vec_f32(&p("ssm_conv1d.weight"));
        let ch = qkv.len();
        let km1 = c.conv_k - 1;
        let mut y = vec![0f32; ch];
        for i in 0..ch {
            let stt = &mut st.conv[l][i * km1..(i + 1) * km1];
            let mut acc = 0.0;
            for k in 0..km1 { acc += cw[i * c.conv_k + k] * stt[k]; }
            acc += cw[i * c.conv_k + km1] * qkv[i];
            for k in 0..km1 - 1 { stt[k] = stt[k + 1]; }
            stt[km1 - 1] = qkv[i];
            y[i] = silu(acc);
        }
        let (qs, rest) = y.split_at_mut(ng * dk);
        let (ks, vs) = rest.split_at_mut(ng * dk);
        for hh in 0..ng {
            for part in [&mut qs[hh * dk..(hh + 1) * dk], &mut ks[hh * dk..(hh + 1) * dk]] {
                let n = part.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
                for v in part.iter_mut() { *v /= n; }
            }
        }
        let scale = 1.0 / (dk as f32).sqrt();
        let nw = self.vec_f32(&p("ssm_norm.weight"));
        let mut o = vec![0f32; nv * dk];
        for vh in 0..nv {
            let kh = if self.kv_repeat_interleave { vh / (nv / ng) } else { vh % ng };
            let g = a_log[vh] * { let a = alpha[vh] + dtb[vh]; if a > 20.0 { a } else { (1.0 + a.exp()).ln() } };
            let s = &mut st.s[l][vh * dk * dk..(vh + 1) * dk * dk];
            let (kv, qv, vv) = (&ks[kh * dk..(kh + 1) * dk], &qs[kh * dk..(kh + 1) * dk], &vs[vh * dk..(vh + 1) * dk]);
            let eg = g.exp();
            let mut kmem = vec![0f32; dk];
            for i in 0..dk { for j in 0..dk { s[i * dk + j] *= eg; kmem[j] += s[i * dk + j] * kv[i]; } }
            let d: Vec<f32> = (0..dk).map(|j| (vv[j] - kmem[j]) * beta[vh]).collect();
            let oo = &mut o[vh * dk..(vh + 1) * dk];
            for i in 0..dk { for j in 0..dk { s[i * dk + j] += kv[i] * d[j]; oo[j] += s[i * dk + j] * qv[i] * scale; } }
            let nrm = rms(oo, &nw, c.eps);
            for j in 0..dk { oo[j] = nrm[j] * silu(z[vh * dk + j]); }
        }
        self.matvec(&p("ssm_out.weight"), 0, &o)
    }

    fn moe(&self, l: usize, h: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{l}.{s}");
        let logits = self.matvec(&p("ffn_gate_inp.weight"), 0, h);
        let m = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut pr: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = pr.iter().sum();
        for v in pr.iter_mut() { *v /= s; }
        let mut idx: Vec<usize> = (0..pr.len()).collect();
        idx.sort_by(|a, b| pr[*b].partial_cmp(&pr[*a]).unwrap());
        idx.truncate(c.n_used);
        let tot: f32 = idx.iter().map(|i| pr[*i]).sum();
        let mut out = vec![0f32; c.n_embd];
        let ffn = |g: &str, u: &str, d: &str, e: usize, w: f32, out: &mut [f32]| {
            let gg = self.matvec(g, e, h);
            let uu = self.matvec(u, e, h);
            let a: Vec<f32> = gg.iter().zip(&uu).map(|(g, u)| silu(*g) * u).collect();
            let r = self.matvec(d, e, &a);
            for i in 0..out.len() { out[i] += w * r[i]; }
        };
        for &e in &idx { ffn(&p("ffn_gate_exps.weight"), &p("ffn_up_exps.weight"), &p("ffn_down_exps.weight"), e, pr[e] / tot, &mut out); }
        let sg = self.vec_f32(&p("ffn_gate_inp_shexp.weight")).iter().zip(h).map(|(a, b)| a * b).sum::<f32>();
        ffn(&p("ffn_gate_shexp.weight"), &p("ffn_up_shexp.weight"), &p("ffn_down_shexp.weight"), 0, sigmoid(sg), &mut out);
        out
    }
}
