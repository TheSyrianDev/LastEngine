//! Dequantization of GGML block formats, written from the public format descriptions.
//! `dequant_row` expands one row (a whole number of blocks) into f32.
#[path = "quant_iq3xxs_grid.rs"]
mod iq3xxs_grid;
use iq3xxs_grid::IQ3XXS_GRID;
#[path = "quant_iq2_grids.rs"]
mod iq2_grids;
use iq2_grids::{IQ2S_GRID, IQ2XXS_GRID};

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q8_0: u32 = 8;
pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const IQ3_XXS: u32 = 18;
pub const IQ2_XXS: u32 = 16;
pub const IQ2_S: u32 = 22;
pub const BF16: u32 = 30;

/// (values per block, bytes per block)
pub fn block_info(ty: u32) -> Option<(usize, usize)> {
    Some(match ty {
        F32 => (1, 4), F16 | BF16 => (1, 2),
        Q4_0 => (32, 18), Q8_0 => (32, 34),
        Q3_K => (256, 110), Q4_K => (256, 144), Q5_K => (256, 176), Q6_K => (256, 210), IQ3_XXS => (256, 98), IQ2_XXS => (256, 66), IQ2_S => (256, 82),
        _ => { let (v, b, _) = crate::quant_ik::info(ty)?; (v, b) }
    })
}

pub fn row_header(ty: u32) -> usize { crate::quant_ik::info(ty).map_or(0, |i| i.2) }

pub fn row_bytes(ty: u32, n: usize) -> usize { let (v, b) = block_info(ty).expect("quant type"); n / v * b + row_header(ty) }

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            let mut e2 = 113u32; let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            (s << 31) | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 { (s << 31) | (0xff << 23) | (m << 13) } else { (s << 31) | ((e + 112) << 23) | (m << 13) };
    f32::from_bits(bits)
}
fn h(b: &[u8], o: usize) -> f32 { f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]])) }

fn scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 { (s[j] & 63, s[j + 4] & 63) }
    else { ((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)) }
}

pub fn ksign(i: usize) -> u8 { (i as u8 & 0x7f) | (((i as u32 & 0x7f).count_ones() & 1) as u8) << 7 }

pub fn dequant_row(ty: u32, src: &[u8], out: &mut [f32]) {
    match ty {
        F32 => for (i, o) in out.iter_mut().enumerate() { *o = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()); },
        F16 => for (i, o) in out.iter_mut().enumerate() { *o = h(src, i * 2); },
        BF16 => for (i, o) in out.iter_mut().enumerate() { *o = f32::from_bits((u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]) as u32) << 16); },
        Q4_0 => for (b, o) in src.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
            let d = h(b, 0);
            for j in 0..16 { o[j] = d * ((b[2 + j] & 15) as f32 - 8.0); o[j + 16] = d * ((b[2 + j] >> 4) as f32 - 8.0); }
        },
        Q8_0 => for (b, o) in src.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
            let d = h(b, 0);
            for j in 0..32 { o[j] = d * (b[2 + j] as i8) as f32; }
        },
        Q4_K => for (b, o) in src.chunks_exact(144).zip(out.chunks_exact_mut(256)) {
            let (d, dmin, sc, qs) = (h(b, 0), h(b, 2), &b[4..16], &b[16..144]);
            for g in 0..4 {
                let (s1, m1) = scale_min_k4(2 * g, sc);
                let (s2, m2) = scale_min_k4(2 * g + 1, sc);
                for l in 0..32 {
                    o[g * 64 + l] = d * s1 as f32 * (qs[g * 32 + l] & 15) as f32 - dmin * m1 as f32;
                    o[g * 64 + 32 + l] = d * s2 as f32 * (qs[g * 32 + l] >> 4) as f32 - dmin * m2 as f32;
                }
            }
        },
        Q5_K => for (b, o) in src.chunks_exact(176).zip(out.chunks_exact_mut(256)) {
            let (d, dmin, sc, qh, qs) = (h(b, 0), h(b, 2), &b[4..16], &b[16..48], &b[48..176]);
            for g in 0..4 {
                let (s1, m1) = scale_min_k4(2 * g, sc);
                let (s2, m2) = scale_min_k4(2 * g + 1, sc);
                for l in 0..32 {
                    let lo = (qs[g * 32 + l] & 15) as f32 + if qh[l] >> (2 * g) & 1 != 0 { 16.0 } else { 0.0 };
                    let hi = (qs[g * 32 + l] >> 4) as f32 + if qh[l] >> (2 * g + 1) & 1 != 0 { 16.0 } else { 0.0 };
                    o[g * 64 + l] = d * s1 as f32 * lo - dmin * m1 as f32;
                    o[g * 64 + 32 + l] = d * s2 as f32 * hi - dmin * m2 as f32;
                }
            }
        },
        Q6_K => for (b, o) in src.chunks_exact(210).zip(out.chunks_exact_mut(256)) {
            let (ql, qh, sc, d) = (&b[0..128], &b[128..192], &b[192..208], h(b, 208));
            for n in 0..2 {
                let (ql, qh, sc, o) = (&ql[n * 64..], &qh[n * 32..], &sc[n * 8..], &mut o[n * 128..]);
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 15) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        },
        Q3_K => for (b, o) in src.chunks_exact(110).zip(out.chunks_exact_mut(256)) {
            let (hm, qs, sc, d) = (&b[0..32], &b[32..96], &b[96..108], h(b, 108));
            let mut scales = [0i32; 16];
            for i in 0..16 {
                let lo = if i < 8 { sc[i] & 15 } else { sc[i - 8] >> 4 };
                let hi = (sc[8 + i % 4] >> (2 * (i / 4))) & 3;
                scales[i] = (lo | (hi << 4)) as i32 - 32;
            }
            for e in 0..256 {
                let (c, s, l) = (e / 128, (e % 128) / 32, e % 32);
                let q = ((qs[32 * c + l] >> (2 * s)) & 3) as i32 - if hm[l] >> (e / 32) & 1 == 0 { 4 } else { 0 };
                o[e] = d * scales[e / 16] as f32 * q as f32;
            }
        },
        IQ3_XXS => for (b, o) in src.chunks_exact(98).zip(out.chunks_exact_mut(256)) {
            let (d, qs, sg) = (h(b, 0), &b[2..66], &b[66..98]);
            for ib in 0..8 {
                let aux = u32::from_le_bytes(sg[ib * 4..ib * 4 + 4].try_into().unwrap());
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let sgn = ksign(((aux >> (7 * l)) & 127) as usize);
                    let (g1, g2) = (&IQ3XXS_GRID[qs[ib * 8 + 2 * l] as usize], &IQ3XXS_GRID[qs[ib * 8 + 2 * l + 1] as usize]);
                    for j in 0..4 {
                        o[ib * 32 + l * 8 + j] = db * g1[j] as f32 * if sgn >> j & 1 != 0 { -1.0 } else { 1.0 };
                        o[ib * 32 + l * 8 + j + 4] = db * g2[j] as f32 * if sgn >> (j + 4) & 1 != 0 { -1.0 } else { 1.0 };
                    }
                }
            }
        },
        IQ2_XXS => for (b, o) in src.chunks_exact(66).zip(out.chunks_exact_mut(256)) {
            let (d, qs) = (h(b, 0), &b[2..66]);
            for ib in 0..8 {
                let q = &qs[ib * 8..ib * 8 + 8];
                let aux = u32::from_le_bytes(q[4..8].try_into().unwrap());
                let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let sgn = ksign(((aux >> (7 * l)) & 127) as usize);
                    let g = &IQ2XXS_GRID[q[l] as usize];
                    for j in 0..8 { o[ib * 32 + l * 8 + j] = db * g[j] as f32 * if sgn >> j & 1 != 0 { -1.0 } else { 1.0 }; }
                }
            }
        },
        IQ2_S => for (b, o) in src.chunks_exact(82).zip(out.chunks_exact_mut(256)) {
            let (d, qs, qh, sc) = (h(b, 0), &b[2..34], &b[66..74], &b[74..82]);
            let signs = &b[34..66];
            for ib in 0..8 {
                let db = [d * (0.5 + (sc[ib] & 15) as f32) * 0.25, d * (0.5 + (sc[ib] >> 4) as f32) * 0.25];
                for l in 0..4 {
                    let gi = qs[ib * 4 + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300);
                    let g = &IQ2S_GRID[gi];
                    let sgn = signs[ib * 4 + l];
                    for j in 0..8 { o[ib * 32 + l * 8 + j] = db[l / 2] * g[j] as f32 * if sgn >> j & 1 != 0 { -1.0 } else { 1.0 }; }
                }
            }
        },
        _ => if !crate::quant_ik::dequant(ty, src, out) { panic!("unsupported quant type {ty}") },
    }
}

pub fn iq3_grid() -> &'static [[u8; 4]; 256] { &IQ3XXS_GRID }
pub fn iq2xxs_grid() -> &'static [[u8; 8]; 256] { &IQ2XXS_GRID }
pub fn iq2s_grid() -> &'static [[u8; 8]; 1024] { &IQ2S_GRID }
