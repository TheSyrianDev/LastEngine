//! CPU reference for the step-3 small-N weight GEMV (batched decode). Spec for the GPU kernel:
//! out[s][r] = dot(dequant(W[r,:]), x[s]) for every slot s, where each weight row is read once and reused for all N slots.
//! Used as the oracle for the GPU kernel gates; not on any serving path.
use crate::quant;

/// W: row-major quantized rows of `cols` values each (row_bytes(ty, cols) bytes per row). xs: N activation vectors of length cols.
pub fn gemv_batch_ref(ty: u32, w: &[u8], rows: usize, cols: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rb = quant::row_bytes(ty, cols);
    assert_eq!(w.len(), rows * rb, "weight bytes");
    let mut out = vec![vec![0f32; rows]; xs.len()];
    let mut row = vec![0f32; cols];
    for r in 0..rows {
        quant::dequant_row(ty, &w[r * rb..(r + 1) * rb], &mut row);
        for (s, x) in xs.iter().enumerate() {
            assert_eq!(x.len(), cols, "activation length");
            out[s][r] = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slots_are_independent() {
        // f32 weights (type 0) keep the test independent of quant tables.
        let (rows, cols) = (3usize, 4usize);
        let w: Vec<u8> = (0..rows * cols).flat_map(|i| (i as f32 * 0.5 - 1.0).to_le_bytes()).collect();
        let xa = vec![1.0f32, 2.0, 3.0, 4.0];
        let xb = vec![-1.0f32, 0.5, 0.0, 2.0];
        let both = gemv_batch_ref(0, &w, rows, cols, &[xa.clone(), xb.clone()]);
        let a = gemv_batch_ref(0, &w, rows, cols, &[xa]);
        let b = gemv_batch_ref(0, &w, rows, cols, &[xb]);
        assert_eq!(both[0], a[0]);
        assert_eq!(both[1], b[0]);
    }
}
