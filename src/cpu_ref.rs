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
    /// clover_moe GGUFs store the raw A_log in `ssm_a`; the decay rate is `-exp(A_log)`.
    pub ssm_a_raw: bool,
}

#[derive(Clone)]
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
        let arch = g.arch().unwrap_or("qwen35moe").to_string();
        let f = std::fs::File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&f).unwrap() };
        let u = |k: &str| g.kv.get(&format!("{arch}.{k}")).and_then(Value::as_u64).unwrap_or_else(|| panic!("{k}")) as usize;
        let fl = |k: &str| g.kv.get(&format!("{arch}.{k}")).and_then(Value::as_f32).unwrap();
        let cfg = Cfg {
            n_layer: u("block_count") - g.kv.get(&format!("{arch}.nextn_predict_layers")).and_then(Value::as_u64).unwrap_or(0) as usize, n_embd: u("embedding_length"), n_head: u("attention.head_count"),
            n_kv: u("attention.head_count_kv"), head_dim: u("attention.key_length"), rope_dim: u("rope.dimension_count"),
            rope_base: fl("rope.freq_base"), eps: fl("attention.layer_norm_rms_epsilon"), n_expert: u("expert_count"),
            n_used: u("expert_used_count"), ff_exp: u("expert_feed_forward_length"), ff_shexp: u("expert_shared_feed_forward_length"),
            interval: u("full_attention_interval"), conv_k: u("ssm.conv_kernel"), d_inner: u("ssm.inner_size"),
            d_state: u("ssm.state_size"), n_group: u("ssm.group_count"), n_vhead: u("ssm.time_step_rank"),
        };
        let t = g.tensors.iter().map(|t| (t.name.clone(), T { ty: t.ggml_type, off: (g.data_start + t.offset) as usize,
            dims: t.dims.iter().map(|d| *d as usize).collect() })).collect();
        let vocab = g.tensors.iter().find(|t| t.name == "token_embd.weight").unwrap().dims[1] as usize;
        Model { cfg, mmap, t, vocab, ssm_a_raw: arch == "clover_moe", kv_repeat_interleave: std::env::var("KV_INTERLEAVE").is_ok() }
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
        if self.ssm_a_raw && name.ends_with(".ssm_a") { for v in o.iter_mut() { *v = -v.exp(); } }
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

    /// Row-major batched matvec: out[n][r] = W[r,:]·xs[n]; each weight row is dequantised once for all inputs.
    pub fn matmat(&self, name: &str, expert: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let (t, b) = self.bytes(name);
        let (cols, rows) = (t.dims[0], t.dims[1]);
        let rb = quant::row_bytes(t.ty, cols);
        let base = expert * rows * rb;
        let cols_out: Vec<Vec<f32>> = (0..rows).into_par_iter().map_init(|| vec![0f32; cols], |buf, r| {
            quant::dequant_row(t.ty, &b[base + r * rb..base + (r + 1) * rb], buf);
            xs.iter().map(|x| buf.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()).collect()
        }).collect();
        (0..xs.len()).map(|n| cols_out.iter().map(|c| c[n]).collect()).collect()
    }

    /// Clover block-diffusion "preview": one bidirectional pass over a block of (mostly mask) tokens at
    /// positions pos0.., reading the committed state in `st` without changing it. Returns logits per position.
    pub fn preview(&self, st: &State, toks: &[u32], pos0: usize) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let nb = toks.len();
        let mut xs: Vec<Vec<f32>> = toks.iter().map(|&tk| {
            let mut x = vec![0f32; c.n_embd];
            let (t, b) = self.bytes("token_embd.weight");
            let rb = quant::row_bytes(t.ty, c.n_embd);
            quant::dequant_row(t.ty, &b[tk as usize * rb..(tk as usize + 1) * rb], &mut x);
            x
        }).collect();
        for l in 0..c.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let an = self.vec_f32(&p("attn_norm.weight"));
            let hs: Vec<Vec<f32>> = xs.iter().map(|x| rms(x, &an, c.eps)).collect();
            let outs = if (l + 1) % c.interval == 0 { self.attn_preview(st, l, &hs, pos0) } else { self.delta_preview(st, l, &hs) };
            for (x, o) in xs.iter_mut().zip(&outs) { for i in 0..x.len() { x[i] += o[i]; } }
            let pn = self.vec_f32(&p("post_attention_norm.weight"));
            let h2: Vec<Vec<f32>> = xs.iter().map(|x| rms(x, &pn, c.eps)).collect();
            let ms = self.moe_preview(l, &h2);
            for (x, o) in xs.iter_mut().zip(&ms) { for i in 0..x.len() { x[i] += o[i]; } }
        }
        let on = self.vec_f32("output_norm.weight");
        let xf: Vec<Vec<f32>> = xs.iter().map(|x| rms(x, &on, c.eps)).collect();
        debug_assert_eq!(xf.len(), nb);
        self.matmat("output.weight", 0, &xf)
    }

    fn attn_preview(&self, st: &State, l: usize, hs: &[Vec<f32>], pos0: usize) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{l}.{s}");
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_kv);
        let nb = hs.len();
        let qf = self.matmat(&p("attn_q.weight"), 0, hs);
        let mut ks = self.matmat(&p("attn_k.weight"), 0, hs);
        let vs = self.matmat(&p("attn_v.weight"), 0, hs);
        let (qn, kn) = (self.vec_f32(&p("attn_q_norm.weight")), self.vec_f32(&p("attn_k_norm.weight")));
        let rope = |v: &mut [f32], pos: usize| {
            let half = c.rope_dim / 2;
            for i in 0..half {
                let th = pos as f32 * c.rope_base.powf(-(2.0 * i as f32) / c.rope_dim as f32);
                let (s, co) = th.sin_cos();
                let (a, b) = (v[i], v[i + half]);
                v[i] = a * co - b * s; v[i + half] = a * s + b * co;
            }
        };
        let mut qs = vec![vec![0f32; nh * hd]; nb];
        let mut gates = vec![vec![0f32; nh * hd]; nb];
        for t in 0..nb {
            for i in 0..nh {
                qs[t][i * hd..(i + 1) * hd].copy_from_slice(&rms(&qf[t][i * 2 * hd..i * 2 * hd + hd], &qn, c.eps));
                gates[t][i * hd..(i + 1) * hd].copy_from_slice(&qf[t][i * 2 * hd + hd..(i + 1) * 2 * hd]);
                rope(&mut qs[t][i * hd..(i + 1) * hd], pos0 + t);
            }
            for i in 0..nkv {
                let r = rms(&ks[t][i * hd..(i + 1) * hd], &kn, c.eps);
                ks[t][i * hd..(i + 1) * hd].copy_from_slice(&r);
                rope(&mut ks[t][i * hd..(i + 1) * hd], pos0 + t);
            }
        }
        let np = st.kc[l].len() / (nkv * hd);
        let outs: Vec<Vec<f32>> = (0..nb).into_par_iter().map(|t| {
            let mut o = vec![0f32; nh * hd];
            for hh in 0..nh {
                let kvh = hh / (nh / nkv);
                let q = &qs[t][hh * hd..(hh + 1) * hd];
                let dot = |kk: &[f32]| kk.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt();
                let mut sc: Vec<f32> = (0..np).map(|j| dot(&st.kc[l][(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd]))
                    .chain((0..nb).map(|j| dot(&ks[j][kvh * hd..(kvh + 1) * hd]))).collect();
                let m = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.0;
                for v in sc.iter_mut() { *v = (*v - m).exp(); sum += *v; }
                for j in 0..np + nb {
                    let vv = if j < np { &st.vc[l][(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd] } else { &vs[j - np][kvh * hd..(kvh + 1) * hd] };
                    for d in 0..hd { o[hh * hd + d] += sc[j] / sum * vv[d]; }
                }
            }
            for i in 0..o.len() { o[i] *= sigmoid(gates[t][i]); }
            o
        }).collect();
        self.matmat(&p("attn_output.weight"), 0, &outs)
    }

    fn delta_preview(&self, st: &State, l: usize, hs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{l}.{s}");
        let (dk, nv, ng) = (c.d_state, c.n_vhead, c.n_group);
        let nb = hs.len();
        let qkv = self.matmat(&p("attn_qkv.weight"), 0, hs);
        let zs = self.matmat(&p("attn_gate.weight"), 0, hs);
        let alphas = self.matmat(&p("ssm_alpha.weight"), 0, hs);
        let (a_log, dtb) = (self.vec_f32(&p("ssm_a")), self.vec_f32(&p("ssm_dt.bias")));
        let cw = self.vec_f32(&p("ssm_conv1d.weight"));
        let wr = self.vec_f32(&p("ssm_conv1d.weight_right"));
        let (lscale, hbias, hc) = (self.vec_f32(&p("ssm_bd_logit_scale.weight")), self.vec_f32(&p("ssm_bd_history.bias")), self.vec_f32(&p("ssm_bd_history.weight")));
        let ch = qkv[0].len();
        let (km1, kw) = (c.conv_k - 1, c.conv_k);
        // conv over [3 committed columns ; block], plus the independent right-hand taps
        let mut ys = vec![vec![0f32; ch]; nb];
        for t in 0..nb {
            for i in 0..ch {
                let hist = &st.conv[l][i * km1..(i + 1) * km1];
                let at = |k: usize| -> f32 { let j = t + k; if j < km1 { hist[j] } else { qkv[j - km1][i] } };
                let mut acc = 0.0;
                for k in 0..kw { acc += cw[i * kw + k] * at(k); }
                for d in 1..kw.min(nb) { if t + d < nb { acc += qkv[t + d][i] * wr[i * km1 + d - 1]; } }
                ys[t][i] = silu(acc);
            }
        }
        let l2 = |v: &mut [f32]| { let r = 1.0 / (v.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt(); for a in v.iter_mut() { *a *= r; } };
        for y in ys.iter_mut() { for i in 0..2 * ng { l2(&mut y[i * dk..(i + 1) * dk]); } }
        let scale = 1.0 / (dk as f32).sqrt();
        let nw = self.vec_f32(&p("ssm_norm.weight"));
        let kh_of = |vh: usize| if self.kv_repeat_interleave { vh / (nv / ng) } else { vh % ng };
        let mut os = vec![vec![0f32; nv * dk]; nb];
        for vh in 0..nv {
            let kh = kh_of(vh);
            let s = &st.s[l][vh * dk * dk..(vh + 1) * dk * dk];
            let q = |t: usize| &ys[t][kh * dk..(kh + 1) * dk];
            let k = |t: usize| &ys[t][ng * dk + kh * dk..ng * dk + (kh + 1) * dk];
            let v = |t: usize| &ys[t][2 * ng * dk + vh * dk..2 * ng * dk + (vh + 1) * dk];
            for t in 0..nb {
                let g = a_log[vh] * { let a = alphas[t][vh] + dtb[vh]; if a > 20.0 { a } else { (1.0 + a.exp()).ln() } };
                let eg = g.exp();
                let mut hist = vec![0f32; dk];
                for i in 0..dk { let qi = q(t)[i] * scale; for j in 0..dk { hist[j] += qi * s[i * dk + j]; } }
                let mut lg: Vec<f32> = (0..nb).map(|u| q(t).iter().zip(k(u)).map(|(a, b)| a * b).sum::<f32>() * lscale[vh]).collect();
                lg.push(hbias[vh] + hc[vh] * (nb as f32 / 32.0).ln());
                let m = lg.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.0;
                for a in lg.iter_mut() { *a = (*a - m).exp(); sum += *a; }
                let oo = &mut os[t][vh * dk..(vh + 1) * dk];
                for j in 0..dk { oo[j] = lg[nb] / sum * hist[j] * eg; }
                for u in 0..nb { let w = lg[u] / sum; for j in 0..dk { oo[j] += w * v(u)[j]; } }
                let nrm = rms(oo, &nw, c.eps);
                for j in 0..dk { oo[j] = nrm[j] * silu(zs[t][vh * dk + j]); }
            }
        }
        self.matmat(&p("ssm_out.weight"), 0, &os)
    }

    fn moe_preview(&self, l: usize, hs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{l}.{s}");
        let nb = hs.len();
        let softmax = |v: &[f32]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); let mut o: Vec<f32> = v.iter().map(|a| (a - m).exp()).collect(); let s: f32 = o.iter().sum(); for a in o.iter_mut() { *a /= s; } o };
        let mut logits = self.matmat(&p("ffn_gate_inp.weight"), 0, hs);
        // coreset: experts the whole block routes to, kept until their cumulative share reaches top_p
        let mut totals = vec![0f32; c.n_expert];
        for lg in &logits {
            let pr = softmax(lg);
            let mut idx: Vec<usize> = (0..pr.len()).collect();
            idx.sort_by(|a, b| pr[*b].partial_cmp(&pr[*a]).unwrap());
            for &e in idx.iter().take(c.n_used) { totals[e] += pr[e]; }
        }
        let sum: f32 = totals.iter().sum::<f32>().max(1e-9);
        let mut order: Vec<usize> = (0..c.n_expert).collect();
        order.sort_by(|a, b| totals[*b].partial_cmp(&totals[*a]).unwrap());
        let mut allowed = vec![false; c.n_expert];
        let mut cum = 0.0;
        for (r, &e) in order.iter().enumerate() { let pe = totals[e] / sum; if r == 0 || cum < 0.9 { allowed[e] = true; } cum += pe; }
        for lg in logits.iter_mut() { for e in 0..c.n_expert { if !allowed[e] { lg[e] = f32::NEG_INFINITY; } } }
        let mut route: Vec<Vec<(usize, f32)>> = vec![vec![]; c.n_expert];
        for (t, lg) in logits.iter().enumerate() {
            let pr = softmax(lg);
            let mut idx: Vec<usize> = (0..pr.len()).collect();
            idx.sort_by(|a, b| pr[*b].partial_cmp(&pr[*a]).unwrap());
            idx.truncate(c.n_used);
            let tot: f32 = idx.iter().map(|i| pr[*i]).sum();
            for &e in &idx { route[e].push((t, pr[e] / tot)); }
        }
        let mut out = vec![vec![0f32; c.n_embd]; nb];
        let ffn = |g: &str, u: &str, d: &str, e: usize, items: &[(usize, f32)], out: &mut Vec<Vec<f32>>| {
            let xs: Vec<Vec<f32>> = items.iter().map(|(t, _)| hs[*t].clone()).collect();
            let gg = self.matmat(g, e, &xs);
            let uu = self.matmat(u, e, &xs);
            let a: Vec<Vec<f32>> = gg.iter().zip(&uu).map(|(g, u)| g.iter().zip(u).map(|(g, u)| silu(*g) * u).collect()).collect();
            let r = self.matmat(d, e, &a);
            for ((t, w), r) in items.iter().zip(&r) { for i in 0..r.len() { out[*t][i] += w * r[i]; } }
        };
        for e in 0..c.n_expert {
            if route[e].is_empty() { continue; }
            ffn(&p("ffn_gate_exps.weight"), &p("ffn_up_exps.weight"), &p("ffn_down_exps.weight"), e, &route[e], &mut out);
        }
        let sgw = self.vec_f32(&p("ffn_gate_inp_shexp.weight"));
        let items: Vec<(usize, f32)> = (0..nb).map(|t| (t, sigmoid(sgw.iter().zip(&hs[t]).map(|(a, b)| a * b).sum::<f32>()))).collect();
        ffn(&p("ffn_gate_shexp.weight"), &p("ffn_up_shexp.weight"), &p("ffn_down_shexp.weight"), 0, &items, &mut out);
        out
    }
}
