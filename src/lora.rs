//! LoRA adapters in GGUF form (`general.type = adapter`, `adapter.type = lora`).
//!
//! An adapter is a set of low-rank deltas, one A (rank x cols) and one B (rows x rank) per
//! targeted tensor. The engine mounts them at run time: the quantised base weight stays as it
//! is and the delta is added to each matvec result, `y = W x + (alpha/rank) B (A x)`.
//! That costs two small extra dot products per projection and keeps full f32 precision for the
//! delta, instead of folding it into the base weight's quantisation.
use crate::{
    gguf::{Gguf, Value},
    quant,
};
use memmap2::Mmap;
use std::collections::HashMap;

/// One targeted tensor: `a` is row-major [rank][cols], `b` is row-major [rows][rank].
pub struct Pair {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
}

pub struct Lora {
    /// Keyed by the base weight's full tensor name, e.g. `blk.3.attn_q.weight`.
    pub pairs: HashMap<String, Pair>,
    /// `adapter.lora.alpha`; the multiplier is `alpha / rank`.
    pub alpha: f32,
    pub path: String,
}

fn split(name: &str) -> Option<(&str, bool)> {
    if let Some(b) = name.strip_suffix(".lora_a") { return Some((b, true)); }
    if let Some(b) = name.strip_suffix(".lora_b") { return Some((b, false)); }
    None
}

impl Lora {
    pub fn load(path: &str) -> Result<Lora, String> {
        let g = Gguf::open(path).map_err(|e| format!("{path}: {e}"))?;
        match g.kv.get("general.type").and_then(Value::as_str) {
            Some("adapter") => {}
            other => return Err(format!("{path}: not an adapter GGUF (general.type = {other:?})")),
        }
        match g.kv.get("adapter.type").and_then(Value::as_str) {
            None | Some("lora") => {}
            Some(t) => return Err(format!("{path}: adapter type {t:?} is not supported, only lora")),
        }
        let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&f).map_err(|e| format!("{path}: {e}"))? };
        let alpha = g.kv.get("adapter.lora.alpha").and_then(Value::as_f32).unwrap_or(0.0);

        // Dequantise every A and B slab once. The whole file is a few tens of MB, so f32 in RAM
        // is cheaper than re-reading and re-scaling them per token.
        let mut a: HashMap<String, (Vec<f32>, usize, usize)> = HashMap::new();
        let mut b: HashMap<String, (Vec<f32>, usize, usize)> = HashMap::new();
        for t in &g.tensors {
            let Some((base, is_a)) = split(&t.name) else { continue };
            let d0 = t.dims[0] as usize;
            let d1 = *t.dims.get(1).unwrap_or(&1) as usize;
            let n = d0 * d1;
            let src = &mmap[(g.data_start + t.offset) as usize..];
            let mut v = vec![0f32; n];
            quant::dequant_row(t.ggml_type, &src[..quant::row_bytes(t.ggml_type, n)], &mut v);
            // File order is row-major with row length dims[0]: A is [cols][rank], B is [rank][rows].
            // A is what a matvec reads as rows=rank, cols=cols; B is rows=rows, cols=rank.
            if is_a { a.insert(base.to_string(), (v, d0, d1)); } else { b.insert(base.to_string(), (v, d0, d1)); }
        }
        let mut pairs = HashMap::new();
        for (base, (av, cols, rank)) in a {
            let Some((bv, rank_b, rows)) = b.remove(&base) else { continue };
            if rank_b != rank {
                return Err(format!("{path}: {base}: lora_a rank {rank} != lora_b rank {rank_b}"));
            }
            // The base already carries `.weight` (the file names are `<tensor>.weight.lora_a`).
            pairs.insert(base, Pair { a: av, b: bv, rank, rows, cols });
        }
        if pairs.is_empty() {
            return Err(format!("{path}: adapter has no lora_a/lora_b pairs"));
        }
        if alpha <= 0.0 {
            return Err(format!("{path}: missing adapter.lora.alpha, cannot scale the delta"));
        }
        Ok(Lora { pairs, alpha, path: path.to_string() })
    }

    /// `y += (alpha/rank) * B (A x)` for one tensor. `None` when this tensor is not targeted.
    pub fn delta(&self, name: &str, x: &[f32]) -> Option<Vec<f32>> {
        let p = self.pairs.get(name)?;
        let scale = self.alpha / p.rank as f32;
        let mut t = vec![0f32; p.rank];
        for (r, tr) in t.iter_mut().enumerate() {
            let row = &p.a[r * p.cols..(r + 1) * p.cols];
            *tr = row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>();
        }
        let mut y = vec![0f32; p.rows];
        for (o, yo) in y.iter_mut().enumerate() {
            let row = &p.b[o * p.rank..(o + 1) * p.rank];
            *yo = scale * row.iter().zip(&t).map(|(w, v)| w * v).sum::<f32>();
        }
        Some(y)
    }

    /// Number of targeted tensors, for the startup log line.
    pub fn len(&self) -> usize { self.pairs.len() }
    pub fn is_empty(&self) -> bool { self.pairs.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_matches_the_definition() {
        // rank 2, cols 3, rows 2; alpha 4 -> scale 2.
        let p = Pair { a: vec![1., 0., 0., 0., 1., 0.], b: vec![1., 2., 3., 4.], rank: 2, rows: 2, cols: 3 };
        let mut pairs = HashMap::new();
        pairs.insert("blk.0.x.weight".to_string(), p);
        let l = Lora { pairs, alpha: 4.0, path: "test".into() };
        // A x = [x0, x1]; B t = [t0 + 2 t1, 3 t0 + 4 t1]; scaled by 2.
        let d = l.delta("blk.0.x.weight", &[2., 5., 7.]).unwrap();
        assert_eq!(d, vec![2.0 * (2.0 + 2.0 * 5.0), 2.0 * (3.0 * 2.0 + 4.0 * 5.0)]);
        assert!(l.delta("blk.0.y.weight", &[1., 2., 3.]).is_none());
    }
}
