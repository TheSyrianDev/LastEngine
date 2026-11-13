//! Batched prefill: dense layers as multi-token GEMMs, MoE experts grouped by expert and streamed in waves
//! (host->device copies of wave w+1 overlap compute of wave w).
use super::*;
use crate::cpu_ref::Cfg as Config;

pub fn tmax() -> usize { static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *T.get_or_init(|| std::env::var("LE_TMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048)) }
pub const WV: usize = 32;

pub struct Pf {
    x: Dp, h: Dp, b0: Dp, b1: Dp, b2: Dp, b3: Dp, y: Dp, o: Dp, lg: Dp, idx: Dp, rw: Dp, sg: Dp, su: Dp, sa: Dp, sd: Dp, shg: Dp,
    eg: Dp, eu: Dp, ea: Dp, ed: Dp,
    xq: [Dp; 2], xd: [Dp; 2], xs: [Dp; 2], scores: Dp, g: usize,
    d_ptrs: [Dp; 4], d_off: [Dp; 4], h_ptrs: [*mut u64; 4], h_off: [*mut i32; 4], d_in: Dp, d_out: Dp, h_in: *mut i32, h_out: *mut i32, d_dense_off: Dp,
}
unsafe impl Send for Pf {}

fn hostalloc<T>(n: usize) -> *mut T { let mut p: *mut std::ffi::c_void = std::ptr::null_mut(); unsafe { sys::cuMemHostAlloc(&mut p, n * std::mem::size_of::<T>(), 0).result().unwrap() }; p as *mut T }

impl Pf {
    pub fn new(c: &Config, ctx_len: usize) -> Pf {
        let t = tmax(); let p = t * c.n_used;
        let a = |n: usize| dalloc(n * 4);
        let g = ((32usize << 20) / (c.n_head * ctx_len * 4)).clamp(1, t);
        let d_off = dup(&[0i32, t as i32]);
        Pf {
            x: a(t * c.n_embd), h: a(t * c.n_embd), b0: a(t * 8192), b1: a(t * 4096), b2: a(t * 512), b3: a(t * 4096), y: a(t * 8192), o: a(t * 4096),
            lg: a(t * c.n_expert), idx: a(p), rw: a(p), sg: a(t * c.ff_shexp), su: a(t * c.ff_shexp), sa: a(t * c.ff_shexp), sd: a(t * c.n_embd), shg: a(t),
            eg: a(p * c.ff_exp), eu: a(p * c.ff_exp), ea: a(p * c.ff_exp), ed: a(p * c.n_embd),
            xq: [dalloc(t * c.n_embd), dalloc(t * 4096)], xd: [a(t * c.n_embd / 32), a(t * 128)], xs: [a(t * c.n_embd / 32), a(t * 128)],
            scores: a(g * c.n_head * ctx_len), g,
            d_ptrs: [dalloc(3 * WV * 8), dalloc(3 * WV * 8), dalloc(3 * WV * 8), dalloc(3 * WV * 8)], d_off: [dalloc(WV * 4 + 8), dalloc(WV * 4 + 8), dalloc(WV * 4 + 8), dalloc(WV * 4 + 8)],
            h_ptrs: [hostalloc(3 * WV), hostalloc(3 * WV), hostalloc(3 * WV), hostalloc(3 * WV)], h_off: [hostalloc(WV + 2), hostalloc(WV + 2), hostalloc(WV + 2), hostalloc(WV + 2)],
            d_in: dalloc(p * 4), d_out: dalloc(p * 4), h_in: hostalloc(p), h_out: hostalloc(p), d_dense_off: d_off,
        }
    }
}

impl Gpu {
    fn pqx(&self, x: Dp, n_units: usize, set: usize, row0_units: usize) {
        let pf = self.pf.as_ref().unwrap();
        self.launch("quant_x", (n_units.div_ceil(8) as u32, 1), 256, 0, &[P(x), P(pf.xq[set] + (row0_units * 32) as u64), P(pf.xd[set] + (row0_units * 4) as u64), P(pf.xs[set] + (row0_units * 4) as u64), I(n_units as i32)]);
    }
    /// Grouped matmul launch. `ptrs`/`off` are device arrays for `slots` weight slots.
    fn mm(&self, ty: u32, ptrs: Dp, off: Dp, ii: Dp, oi: Dp, set: usize, out: Dp, rows: usize, cols: usize, slots: usize, max_n: usize) {
        let pf = self.pf.as_ref().unwrap();
        let units = cols / 32; let lpr = units.min(32); let rpb = 8 * (32 / lpr);
        if self.mmq {
            let name = match ty { 2 => "mmq_2", 8 => "mmq_8", 11 => "mmq_11", 12 => "mmq_12", 13 => "mmq_13", 14 => "mmq_14", 16 => "mmq_16", 17 => "mmq_17", 18 => "mmq_18", 19 => "mmq_19", 20 => "mmq_20", 21 => "mmq_21", 23 => "mmq_23", 22 => "mmq_22", 29 => "mmq_29", 134 => "mmq_134", 135 => "mmq_135", 137 => "mmq_137", 138 => "mmq_138", 139 => "mmq_139", 140 => "mmq_140", 141 => "mmq_141", 144 => "mmq_144", 145 => "mmq_145", 146 => "mmq_146", 152 => "mmq_152", 153 => "mmq_153", 154 => "mmq_154", 155 => "mmq_155", 156 => "mmq_156", 157 => "mmq_157", 158 => "mmq_158", t => panic!("no mmq kernel for type {t}") };
            self.launch3(name, (rows.div_ceil(64) as u32, max_n.div_ceil(64) as u32, slots as u32), 256, 0,
                &[P(ptrs), P(off), P(ii), P(oi), P(pf.xq[set]), P(pf.xd[set]), P(pf.xs[set]), P(out), I(rows as i32), I(cols as i32)]);
            return;
        }
        let name = match ty { 2 => "mm_2", 8 => "mm_8", 11 => "mm_11", 12 => "mm_12", 13 => "mm_13", 14 => "mm_14", 16 => "mm_16", 17 => "mm_17", 18 => "mm_18", 19 => "mm_19", 20 => "mm_20", 21 => "mm_21", 23 => "mm_23", 22 => "mm_22", 29 => "mm_29", 134 => "mm_134", 135 => "mm_135", 137 => "mm_137", 138 => "mm_138", 139 => "mm_139", 140 => "mm_140", 141 => "mm_141", 144 => "mm_144", 145 => "mm_145", 146 => "mm_146", 152 => "mm_152", 153 => "mm_153", 154 => "mm_154", 155 => "mm_155", 156 => "mm_156", 157 => "mm_157", 158 => "mm_158", t => panic!("no mm kernel for type {t}") };
        self.launch3(name, (rows.div_ceil(rpb) as u32, max_n.div_ceil(8) as u32, slots as u32), 256, (8 * units * 44) as u32,
            &[P(ptrs), P(off), P(ii), P(oi), P(pf.xq[set]), P(pf.xd[set]), P(pf.xs[set]), P(out), I(rows as i32), I(cols as i32)]);
    }
    fn dense(&self, w: W, set: usize, out: Dp, t: usize) {
        let pf = self.pf.as_ref().unwrap();
        self.mm(w.ty, w.cell, pf.d_dense_off, 0, 0, set, out, w.rows, w.cols, 1, t);
    }
    fn fmm(&self, w: W, x: Dp, out: Dp, t: usize) {
        self.launch("mmf2", (w.rows.div_ceil(64) as u32, t.div_ceil(64) as u32), 256, 0, &[P(w.raw), P(x), P(out), I(w.rows as i32), I(w.cols as i32), I(t as i32)]);
    }
    fn prms(&self, x: Dp, w: Dp, o: Dp, t: usize) { self.launch("rmsnorm_b", (t as u32, 1), 256, 0, &[P(x), P(w), P(o), I(self.m.cfg.n_embd as i32), F(self.m.cfg.eps)]); }

    pub(super) fn pf_x(&self) -> Dp { self.pf.as_ref().unwrap().x }
    pub(super) fn prefill_chunk_pub(&mut self, ids: &[u32]) { self.bind(); self.prefill_chunk(ids, false); }
    /// Process `ids` (<= TMAX) as one batch; returns logits of the last token.
    pub(super) fn prefill_chunk(&mut self, ids: &[u32], want: bool) -> Vec<f32> {
        let c = self.m.cfg.clone();
        let t = ids.len(); let pos0 = self.pos;
        assert!(t <= tmax() && pos0 + t <= self.ctx_len, "context full");
        let pf = &**self.pf.as_ref().unwrap() as *const Pf; let pf = unsafe { &*pf };
        let mut emb = vec![0f32; t * c.n_embd];
        for (i, &id) in ids.iter().enumerate() { emb[i * c.n_embd..(i + 1) * c.n_embd].copy_from_slice(&self.m.embed(id)); }
        unsafe { sys::cuStreamSynchronize(self.cs).result().unwrap(); result::memcpy_htod_sync(pf.x, &emb).unwrap() };
        let nu = |n: usize| (t * n) / 32;
        for l in 0..c.n_layer {
            let ly = &self.lay[l] as *const Lay; let ly = unsafe { &*ly };
            self.prms(pf.x, ly.attn_norm, pf.h, t);
            self.pqx(pf.h, nu(c.n_embd), 0, 0);
            if ly.full {
                let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
                self.dense(ly.q.unwrap(), 0, pf.b0, t); self.dense(ly.k.unwrap(), 0, pf.b1, t); self.dense(ly.v.unwrap(), 0, pf.b2, t);
                self.launch("attn_prep_b", ((nh + nkv) as u32, t as u32), 256, 0, &[P(pf.b0), P(pf.b1), P(pf.b2), P(ly.qn), P(ly.kn), P(pf.y), P(pf.b3), P(self.kc[l]), P(self.vc[l]),
                    I(nh as i32), I(nkv as i32), I(hd as i32), I(c.rope_dim as i32), F(self.rope_ms), I(pos0 as i32), F(c.eps), P(if self.kvq == 3 { self.kscr } else { self.ksc[l] }), P(self.vsc[l]), I(self.kvq), P(self.rope_f)]);
                if self.kvq == 3 { self.tcq_after_prep(l, t, pos0, pf.b2, pf.y); }
                if hd == 256 && std::env::var("LE_FLASH").map_or(true, |v| v != "0") {
                    let smem = ((64 + 32) * (hd + 8) + hd * 40) * 2;
                    self.launch(if self.preview { "attn_flash_256_nc" } else { "attn_flash_256" }, (t.div_ceil(64) as u32, nh as u32), 128, smem as u32, &[P(pf.y), P(self.kc[l]), P(self.vc[l]), P(pf.b3), P(pf.o), I(pos0 as i32), I(t as i32), I(nh as i32), I(nkv as i32), P(self.ksc[l]), P(self.vsc[l]), I(self.kvq)]);
                } else {
                let mut tb = 0;
                    while tb < t {
                        let g = pf.g.min(t - tb); let pos_first = pos0 + tb; let nmax = pos_first + g;
                        self.launch3("attn_scores_b", (nmax.div_ceil(8) as u32, nkv as u32, g as u32), 256, 0, &[P(pf.y), P(self.kc[l]), P(pf.scores), I(nkv as i32), I(nh as i32), I(hd as i32), I(pos_first as i32), I(self.ctx_len as i32), I(tb as i32), P(self.ksc[l]), I(self.kvq)]);
                        self.launch("attn_softmax_b", (nh as u32, g as u32), 256, 0, &[P(pf.scores), I(nh as i32), I(self.ctx_len as i32), I(pos_first as i32)]);
                        self.launch("attn_pv_b", (nkv as u32, g as u32), hd as u32, 0, &[P(pf.scores), P(self.vc[l]), P(pf.b3), P(pf.o), I(nkv as i32), I(nh as i32), I(hd as i32), I(pos_first as i32), I(self.ctx_len as i32), I(tb as i32), P(self.vsc[l]), I(self.kvq)]);
                        tb += g;
                    }
    }
                if self.kvq == 3 { self.launch("tcq_rot", ((t * nh * hd / 128) as u32, 1), 128, 0, &[P(pf.o), P(pf.b3), I(1)]); }
                self.pqx(pf.o, nu(nh * hd), 1, 0); self.dense(ly.o.unwrap(), 1, pf.b1, t);
                self.launch("add_inplace", ((t * c.n_embd).div_ceil(256) as u32, 1), 256, 0, &[P(pf.x), P(pf.b1), I((t * c.n_embd) as i32)]);
            } else {
                let ch = c.d_inner + 2 * c.n_group * c.d_state;
                self.dense(ly.qkv.unwrap(), 0, pf.b0, t); self.dense(ly.gate.unwrap(), 0, pf.b1, t);
                self.fmm(ly.beta.unwrap(), pf.h, pf.b2, t); self.fmm(ly.alpha.unwrap(), pf.h, pf.b3, t);
                let dr = |s: &Gpu, nm: &str, p: Dp, w: usize| if s.dbg && l == 0 && s.dbg_pos >= pos0 && s.dbg_pos < pos0 + t { s.dump(nm, p + ((s.dbg_pos - pos0) * w * 4) as u64, w) };
                dr(self, "qkv", pf.b0, ch); dr(self, "gate", pf.b1, 4096); dr(self, "beta", pf.b2, 32); dr(self, "alpha", pf.b3, 32);
                if self.preview {
                    self.launch("delta_conv_pv", (ch.div_ceil(256) as u32, 1), 256, 0, &[P(pf.b0), P(ly.conv_w), P(ly.wr), P(self.conv[l]), P(pf.y), I(ch as i32), I(c.conv_k as i32), I(t as i32)]);
                    self.launch("delta_pv", (c.n_vhead as u32, 1), 128, 0, &[P(pf.y), P(pf.b3), P(ly.a_log), P(ly.dt_b), P(pf.b1), P(ly.ssm_norm), P(self.sdelta[l]), P(ly.bd_ls), P(ly.bd_hb), P(ly.bd_hc), P(pf.o),
                        I(c.n_group as i32), I(c.n_vhead as i32), I(c.d_state as i32), F(c.eps), I(t as i32), I(ch as i32)]);
                } else {
                self.launch("delta_conv_b", (ch.div_ceil(256) as u32, 1), 256, 0, &[P(pf.b0), P(ly.conv_w), P(self.conv[l]), P(pf.y), I(ch as i32), I(c.conv_k as i32), I(t as i32)]);
                self.launch("delta_step_b", (c.n_vhead as u32, 1), 128, 0, &[P(pf.y), P(pf.b3), P(pf.b2), P(ly.a_log), P(ly.dt_b), P(pf.b1), P(ly.ssm_norm), P(self.sdelta[l]), P(pf.o),
                    I(c.n_group as i32), I(c.n_vhead as i32), I(c.d_state as i32), F(c.eps), I(t as i32), I(ch as i32)]);
                }
                dr(self, "conv", pf.y, ch); dr(self, "dout", pf.o, 4096);
                self.pqx(pf.o, nu(c.d_inner), 1, 0); self.dense(ly.ssm_out.unwrap(), 1, pf.b1, t);
                self.launch("add_inplace", ((t * c.n_embd).div_ceil(256) as u32, 1), 256, 0, &[P(pf.x), P(pf.b1), I((t * c.n_embd) as i32)]);
            }
            // ---- MoE ----
            if self.dbg && self.dbg_pos >= pos0 && self.dbg_pos < pos0 + t { self.dump("mix", pf.x + ((self.dbg_pos - pos0) * c.n_embd * 4) as u64, c.n_embd); }
            self.prms(pf.x, ly.post_norm, pf.h, t);
            self.fmm(ly.router, pf.h, pf.lg, t);
            if self.preview {
                // block coreset: every token of the block may only route to the experts the block as a whole prefers
                let ne = c.n_expert; let mut lg = vec![0f32; t * ne];
                unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut lg, pf.lg).unwrap(); }
                let mut totals = vec![0f32; ne];
                for r in 0..t {
                    let row = &lg[r * ne..(r + 1) * ne];
                    let m = row.iter().cloned().fold(f32::MIN, f32::max); let ex: Vec<f32> = row.iter().map(|a| (a - m).exp()).collect(); let sm: f32 = ex.iter().sum();
                    let mut ord: Vec<usize> = (0..ne).collect(); ord.sort_by(|a, b| ex[*b].partial_cmp(&ex[*a]).unwrap().then(a.cmp(b)));
                    for &e in ord.iter().take(c.n_used) { totals[e] += ex[e] / sm; }
                }
                let sum: f32 = totals.iter().sum::<f32>().max(1e-9);
                let mut ord: Vec<usize> = (0..ne).collect(); ord.sort_by(|a, b| totals[*b].partial_cmp(&totals[*a]).unwrap().then(a.cmp(b)));
                let mut allowed = vec![false; ne]; let mut cum = 0f32;
                for (i, &e) in ord.iter().enumerate() { let p = totals[e] / sum; if i == 0 || cum < 0.9 { allowed[e] = true; } cum += p; }
                for r in 0..t { for e in 0..ne { if !allowed[e] { lg[r * ne + e] = -1e30; } } }
                unsafe { result::memcpy_htod_sync(pf.lg, &lg).unwrap(); }
            }
            self.pqx(pf.h, nu(c.n_embd), 0, 0);
            let k = c.n_used;
            self.launch("router_topk_b", (t as u32, 1), 256, 0, &[P(pf.lg), I(c.n_expert as i32), I(k as i32), P(pf.idx), P(pf.rw)]);
            // shared expert (dense) while the host reads the routing
            self.dense(ly.sh_gate, 0, pf.sg, t); self.dense(ly.sh_up, 0, pf.su, t);
            self.launch("swiglu", ((t * c.ff_shexp).div_ceil(256) as u32, 1), 256, 0, &[P(pf.sg), P(pf.su), P(pf.sa), I((t * c.ff_shexp) as i32)]);
            self.pqx(pf.sa, nu(c.ff_shexp), 1, 0); self.dense(ly.sh_down, 1, pf.sd, t);
            self.fmm(ly.shg_in, pf.h, pf.shg, t);
            let mut idx = vec![0i32; t * k];
            unsafe { result::memcpy_dtoh_sync(&mut idx, pf.idx).unwrap() };
            // group pairs by expert
            let mut lists: Vec<Vec<(i32, i32)>> = vec![vec![]; c.n_expert];
            for tk in 0..t { for s in 0..k { lists[idx[tk * k + s] as usize].push((tk as i32, (tk * k + s) as i32)); } }
            for e in 0..c.n_expert { self.pheat[l * c.n_expert + e] += lists[e].len() as u32; }
            let active: Vec<usize> = (0..c.n_expert).filter(|&e| !lists[e].is_empty()).collect();
            let mut ins: Vec<i32> = Vec::with_capacity(t * k); let mut outs: Vec<i32> = Vec::with_capacity(t * k);
            let mut starts: Vec<i32> = vec![];
            for &e in &active { starts.push(ins.len() as i32); for &(a, b) in &lists[e] { ins.push(a); outs.push(b); } }
            starts.push(ins.len() as i32);
            unsafe {
                std::ptr::copy_nonoverlapping(ins.as_ptr(), pf.h_in, ins.len()); std::ptr::copy_nonoverlapping(outs.as_ptr(), pf.h_out, outs.len());
                sys::cuMemcpyHtoDAsync_v2(pf.d_in, pf.h_in as *const _, ins.len() * 4, self.st.cu_stream()).result().unwrap();
                sys::cuMemcpyHtoDAsync_v2(pf.d_out, pf.h_out as *const _, outs.len() * 4, self.st.cu_stream()).result().unwrap();
            }
            let eb: usize = ly.ebytes.iter().sum();
            let wv = WV.min(self.n_slots / 2).max(1);
            let mut prev_slots: Vec<usize> = vec![];
            for wave in active.chunks(wv).enumerate() {
                let (wi, exps) = wave;
                let g = self.wave; self.wave += 1; let r = (g % 4) as usize;
                if g >= 4 { unsafe { sys::cuEventSynchronize(self.comp_ev[r]).result().unwrap() }; }
                let mut pin = prev_slots.clone(); let mut slots = vec![]; let mut any = false;
                if g >= 2 { unsafe { sys::cuStreamWaitEvent(self.cs, self.comp_ev[((g - 2) % 4) as usize], 0).result().unwrap() }; }
                for &e in exps {
                    let (sl, copy) = self.pick_slot(l, e, &pin);
                    pin.push(sl); slots.push(sl);
                    if copy {
                        any = true; self.misses += 1;
                        let dst = self.arena + (sl * self.slot_bytes) as u64;
                        if !self.store.is_null() { unsafe { sys::cuMemcpyHtoDAsync_v2(dst, self.store.add(self.store_off[l] + e * eb) as *const _, eb, self.cs).result().unwrap() }; }
                        else { let mut off = 0u64; for part in self.m.expert_parts(l, e) { unsafe { sys::cuMemcpyHtoDAsync_v2(dst + off, part.as_ptr() as *const _, part.len(), self.cs).result().unwrap() }; off += part.len() as u64; } }
                    } else { self.hits += 1; }
                }
                if any { unsafe { sys::cuEventRecord(self.cp_ev[r], self.cs).result().unwrap(); sys::cuStreamWaitEvent(self.st.cu_stream(), self.cp_ev[r], 0).result().unwrap(); } }
                let n = exps.len(); let base_i = wi * wv;
                let (i0, i1) = (starts[base_i] as usize, starts[base_i + n] as usize);
                unsafe {
                    for z in 0..n {
                        let b = self.arena + (slots[z] * self.slot_bytes) as u64;
                        *pf.h_ptrs[r].add(z) = b;
                        *pf.h_ptrs[r].add(WV + z) = b + ly.ebytes[0] as u64;
                        *pf.h_ptrs[r].add(2 * WV + z) = b + (ly.ebytes[0] + ly.ebytes[1]) as u64;
                        *pf.h_off[r].add(z) = starts[base_i + z];
                    }
                    *pf.h_off[r].add(n) = starts[base_i + n];
                    sys::cuMemcpyHtoDAsync_v2(pf.d_ptrs[r], pf.h_ptrs[r] as *const _, 3 * WV * 8, self.st.cu_stream()).result().unwrap();
                    sys::cuMemcpyHtoDAsync_v2(pf.d_off[r], pf.h_off[r] as *const _, (n + 1) * 4, self.st.cu_stream()).result().unwrap();
                }
                let (dpg, dof) = (pf.d_ptrs[r], pf.d_off[r]);
                let (ff, ne) = (c.ff_exp, c.n_embd);
                let nn = i1 - i0; let maxn = (0..n).map(|z| (starts[base_i + z + 1] - starts[base_i + z]) as usize).max().unwrap();
                let (tg, tu, td) = (ly.ety[0], ly.ety[1], ly.ety[2]);
                self.mm(tg, dpg, dof, pf.d_in, 0, 0, pf.eg, ff, ne, n, maxn);
                self.mm(tu, dpg + (WV * 8) as u64, dof, pf.d_in, 0, 0, pf.eu, ff, ne, n, maxn);
                self.launch("swiglu", ((nn * ff).div_ceil(256) as u32, 1), 256, 0, &[P(pf.eg + (i0 * ff * 4) as u64), P(pf.eu + (i0 * ff * 4) as u64), P(pf.ea + (i0 * ff * 4) as u64), I((nn * ff) as i32)]);
                self.pqx(pf.ea + (i0 * ff * 4) as u64, nn * ff / 32, 1, i0 * ff / 32);
                self.mm(td, dpg + (2 * WV * 8) as u64, dof, 0, pf.d_out, 1, pf.ed, ne, ff, n, maxn);
                unsafe { sys::cuEventRecord(self.comp_ev[r], self.st.cu_stream()).result().unwrap(); }
                prev_slots = slots;
            }
            self.launch3("moe_combine_b", ((c.n_embd.div_ceil(256)) as u32, t as u32, 1), 256, 0, &[P(pf.x), P(pf.ed), P(pf.rw), I(k as i32), P(pf.sd), P(pf.shg), I(c.n_embd as i32)]);
            if self.dbg && self.dbg_pos >= pos0 && self.dbg_pos < pos0 + t { let r = self.dbg_pos - pos0; let mut v = vec![0f32; c.n_embd]; unsafe { sys::cuStreamSynchronize(self.st.cu_stream()).result().unwrap(); result::memcpy_dtoh_sync(&mut v, pf.x + (r * c.n_embd * 4) as u64).unwrap() }; eprintln!("L{l} {:.4} {:.4}", v.iter().map(|a| a * a).sum::<f32>().sqrt(), v[..4].iter().map(|a| format!("{a:.3}")).collect::<Vec<_>>().join(",")); }
        }
        if !self.preview { self.pos += t; }
        if self.want_rows {
            self.rows_out.clear();
            for r in 0..t {
                unsafe { result::memcpy_dtod_async(self.x, pf.x + (r * c.n_embd * 4) as u64, c.n_embd * 4, self.st.cu_stream()).unwrap() };
                let lg = self.head_logits(); self.rows_out.push(lg);
            }
            return vec![];
        }
        if !want { if self.mtp.is_some() { unsafe { result::memcpy_dtod_async(self.hprev, pf.x + ((t - 1) * c.n_embd * 4) as u64, c.n_embd * 4, self.st.cu_stream()).unwrap() }; } return vec![]; }
        if self.mtp.is_some() { unsafe { result::memcpy_dtod_async(self.hprev, pf.x + ((t - 1) * c.n_embd * 4) as u64, c.n_embd * 4, self.st.cu_stream()).unwrap() }; }
        unsafe { result::memcpy_dtod_async(self.x, pf.x + ((t - 1) * c.n_embd * 4) as u64, c.n_embd * 4, self.st.cu_stream()).unwrap() };
        self.rms(self.x, self.out_norm, self.h, c.n_embd);
        self.qx(self.h, c.n_embd, 0); self.mv(self.out_w, self.h, 0, self.logits);
        let mut out = vec![0f32; self.m.vocab];
        unsafe { result::memcpy_dtoh_sync(&mut out, self.logits).unwrap() };
        out
    }
}
