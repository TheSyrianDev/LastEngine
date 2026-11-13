//! Dequantization of the remaining IQ formats, translated from ik_llama.cpp / ggml (MIT; see NOTICE).
//! Every function takes one full row (including the per-row header for types that have one) and fills `out`.
use crate::quant::{f16_to_f32, ksign};
use crate::quant_ik_tables::*;

fn h(b: &[u8], o: usize) -> f32 { f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]])) }
fn u16le(b: &[u8], o: usize) -> u16 { u16::from_le_bytes([b[o], b[o + 1]]) }
const KV_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const K_MULT: [u8; 5] = [81, 27, 9, 3, 1];


/// Trellis generator shared by the *_KT formats: `n` values from a 16-bit code (is_int variant, ik_llama QuantizerIQKT::set_values).
fn kt_values(code: u32, offset: u32, scale: f32, abs: bool, out: &mut [f32]) {
    let mut x = code.wrapping_add(offset);
    for v in out.iter_mut() {
        x = 0xCBAC1FEDu32.wrapping_mul(x);
        let s = x & 0x3f3f3f3f;
        let t = (s & 0xff) as i8 as i32 + ((s >> 8) & 0xff) as i8 as i32 + ((s >> 16) & 0xff) as i8 as i32 + ((s >> 24) & 0xff) as i8 as i32;
        let f = t as f32 - 126.0;
        *v = scale * if abs { f.abs() } else { f };
    }
}

/// (values per block, bytes per block, bytes of per-row header)
pub fn info(ty: u32) -> Option<(usize, usize, usize)> {
    Some(match ty {
        17 => (256, 74, 0), 19 => (256, 50, 0), 20 => (32, 18, 0), 21 => (256, 110, 0), 23 => (256, 136, 0), 29 => (256, 56, 0),
        134 => (64, 13, 2), 135 => (64, 16, 4),
        137 => (256, 76, 0), 138 => (256, 110, 0), 139 => (256, 144, 0), 140 => (256, 176, 0), 141 => (256, 212, 0),
        144 => (256, 136, 4), 145 => (256, 70, 2), 146 => (256, 128, 4), 152 => (256, 168, 4),
        156 => (256, 102, 2), 157 => (256, 86, 2),
        153 => (256, 68, 4), 154 => (256, 100, 4), 155 => (256, 128, 4), 158 => (256, 56, 4),
        _ => return None,
    })
}

pub fn dequant(ty: u32, src: &[u8], out: &mut [f32]) -> bool {
    let Some((vb, bb, meta)) = info(ty) else { return false };
    let (hdr, body) = src.split_at(meta);
    let rd = if meta == 4 { f32::from_le_bytes(hdr[..4].try_into().unwrap()) } else if meta == 2 { h(hdr, 0) } else { 0.0 };
    for (b, o) in body.chunks_exact(bb).zip(out.chunks_exact_mut(vb)) {
        match ty {
            17 => { // IQ2_XS
                let d = h(b, 0);
                for ib in 0..8 {
                    let sc = b[66 + ib];
                    let db = [d * (0.5 + (sc & 15) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
                    for l in 0..4 {
                        let q = u16le(b, 2 + 2 * (4 * ib + l));
                        let g = &IQ2XS_GRID[(q & 511) as usize]; let sg = ksign((q >> 9) as usize);
                        for j in 0..8 { o[ib * 32 + l * 8 + j] = db[l / 2] * g[j] as f32 * if sg >> j & 1 != 0 { -1.0 } else { 1.0 }; }
                    }
                }
            }
            19 => { // IQ1_S
                let d = h(b, 0);
                for ib in 0..8 {
                    let qh = u16le(b, 34 + 2 * ib);
                    let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                    let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
                    for l in 0..4 {
                        let g = &IQ1S_GRID[b[2 + ib * 4 + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8)];
                        for j in 0..8 { o[ib * 32 + l * 8 + j] = dl * (g[j] as f32 + delta); }
                    }
                }
            }
            20 => { let d = h(b, 0); for j in 0..16 { o[j] = d * KV_IQ4NL[(b[2 + j] & 15) as usize] as f32; o[j + 16] = d * KV_IQ4NL[(b[2 + j] >> 4) as usize] as f32; } }
            21 => { // IQ3_S
                let d = h(b, 0);
                for p in 0..4 {
                    let (qs, qh, sg) = (&b[2 + p * 16..], &b[66 + p * 2..], &b[74 + p * 8..]);
                    let sc = b[106 + p];
                    let db = [d * (1 + 2 * (sc & 15) as i32) as f32, d * (1 + 2 * (sc >> 4) as i32) as f32];
                    for half in 0..2 {
                        let (qs, sg, qh) = (&qs[half * 8..], &sg[half * 4..], qh[half] as usize);
                        for l in 0..4 {
                            let g1 = &IQ3S_GRID[qs[2 * l] as usize | ((qh << (8 - 2 * l)) & 256)];
                            let g2 = &IQ3S_GRID[qs[2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256)];
                            for j in 0..4 {
                                o[p * 64 + half * 32 + l * 8 + j] = db[half] * g1[j] as f32 * if sg[l] >> j & 1 != 0 { -1.0 } else { 1.0 };
                                o[p * 64 + half * 32 + l * 8 + 4 + j] = db[half] * g2[j] as f32 * if sg[l] >> (j + 4) & 1 != 0 { -1.0 } else { 1.0 };
                            }
                        }
                    }
                }
            }
            23 => { // IQ4_XS
                let d = h(b, 0); let sh = u16le(b, 2);
                for ib in 0..8 {
                    let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
                    let dl = d * (ls - 32) as f32;
                    for j in 0..16 { let q = b[8 + ib * 16 + j]; o[ib * 32 + j] = dl * KV_IQ4NL[(q & 15) as usize] as f32; o[ib * 32 + j + 16] = dl * KV_IQ4NL[(q >> 4) as usize] as f32; }
                }
            }
            29 => { // IQ1_M
                let sc = |i: usize| u16le(b, 48 + 2 * i);
                let dbits = (sc(0) >> 12) | ((sc(1) >> 8) & 0xf0) | ((sc(2) >> 4) & 0xf00) | (sc(3) & 0xf000);
                let d = f16_to_f32(dbits);
                for ib in 0..8 {
                    let s = sc(ib / 2); let sh = 6 * (ib % 2);
                    let dl = [d * (2 * ((s >> sh) & 7) + 1) as f32, d * (2 * ((s >> (sh + 3)) & 7) + 1) as f32];
                    let (qs, qh) = (&b[ib * 4..], &b[32 + ib * 2..]);
                    for l in 0..4 {
                        let hb = qh[l / 2] as usize;
                        let idx = qs[l] as usize | if l % 2 == 0 { (hb << 8) & 0x700 } else { (hb << 4) & 0x700 };
                        let neg = hb & if l % 2 == 0 { 0x08 } else { 0x80 } != 0;
                        let delta = if neg { -0.125 } else { 0.125 };
                        let g = &IQ1S_GRID[idx];
                        for j in 0..8 { o[ib * 32 + l * 8 + j] = dl[l / 2] * (g[j] as f32 + delta); }
                    }
                }
            }
            134 => { // IQ1_BN (row f16 scale)
                let mut y = 0;
                for i16 in 0..4 {
                    for k in 0..3 { for j in 0..5 { let v = b[i16 * 3 + k].wrapping_mul(K_MULT[j]); o[y] = rd * ((((v as u32 + (v as u32 >> 1)) >> 7) as u8 as i8) as f32 - 1.0); y += 1; } }
                    let v = b[12].wrapping_mul(K_MULT[i16]); o[y] = rd * ((((v as u32 + (v as u32 >> 1)) >> 7) as u8 as i8) as f32 - 1.0); y += 1;
                }
            }
            135 => { // IQ2_BN (row f32 scale)
                for j in 0..16 { for i in 0..4 { o[j + 16 * i] = rd * (((b[j] >> (2 * i)) & 3) as f32 - 1.0); } }
            }
            137 => { // IQ2_K
                let d = h(b, 0); let mut extra = u16le(b, 2); let (sc, qs) = (&b[4..12], &b[12..]);
                let (mut shift, mut q0) = (0, 0);
                for ib in 0..8 {
                    let dl = [d * ((sc[ib] & 15) as i32 - 8) as f32, d * ((sc[ib] >> 4) as i32 - 8) as f32];
                    let v = [if extra & 1 != 0 { 4 } else { 0 }, if extra & 2 != 0 { 4 } else { 0 }]; extra >>= 2;
                    for j in 0..16 {
                        o[ib * 32 + j] = dl[0] * IQ2NL_VALUES[v[0] + ((qs[q0 + j] >> shift) & 3) as usize] as f32;
                        o[ib * 32 + j + 16] = dl[1] * IQ2NL_VALUES[v[1] + ((qs[q0 + j + 16] >> shift) & 3) as usize] as f32;
                    }
                    shift += 2; if shift == 8 { q0 += 32; shift = 0; }
                }
            }
            145 => { // IQ2_KS
                let mut extra = u16le(b, 0); let (sc, qs) = (&b[2..6], &b[6..]);
                let (mut shift, mut q0) = (0, 0);
                for ib in 0..4 {
                    let dl1 = rd * (((sc[ib] & 15) as i32 | ((extra as i32 >> 4) & 0x10)) - 16) as f32;
                    let dl2 = rd * (((sc[ib] >> 4) as i32 | ((extra as i32 >> 5) & 0x10)) - 16) as f32;
                    let v1 = if extra & 1 != 0 { 4 } else { 0 }; let v2 = if extra & 2 != 0 { 4 } else { 0 }; extra >>= 2;
                    for j in 0..32 {
                        o[ib * 64 + j] = dl1 * IQ2NL_VALUES[v1 + ((qs[q0 + j] >> shift) & 3) as usize] as f32;
                        o[ib * 64 + j + 32] = dl2 * IQ2NL_VALUES[v2 + ((qs[q0 + j] >> (shift + 2)) & 3) as usize] as f32;
                    }
                    shift += 4; if shift == 8 { q0 += 32; shift = 0; }
                }
            }
            157 => { // IQ2_KL
                let (sh, sl, qs, qh) = (u16le(b, 0) as i32, &b[2..6], &b[6..70], &b[70..86]);
                for ib in 0..4 {
                    let dl1 = rd * ((((sl[(2 * ib) % 4] >> (4 * (ib / 2))) & 15) as i32 | (((sh >> (4 * ib)) & 3) << 4)) - 32) as f32;
                    let dl2 = rd * ((((sl[(2 * ib + 1) % 4] >> (4 * (ib / 2))) & 15) as i32 | (((sh >> (4 * ib + 2)) & 3) << 4)) - 32) as f32;
                    for j in 0..16 {
                        let q = qs[ib * 16 + j];
                        let i1 = ((q & 15) as usize) | ((((qh[j] >> (2 * ib)) & 1) as usize) << 4);
                        let i2 = ((q >> 4) as usize) | ((((qh[j] >> (2 * ib + 1)) & 1) as usize) << 4);
                        let (v1, v2) = (IQ2KL_VALUES[i1], IQ2KL_VALUES[i2]);
                        o[ib * 64 + 2 * j] = dl1 * (v1 as u8 as i8) as f32; o[ib * 64 + 2 * j + 1] = dl1 * ((v1 >> 8) as u8 as i8) as f32;
                        o[ib * 64 + 2 * j + 32] = dl2 * (v2 as u8 as i8) as f32; o[ib * 64 + 2 * j + 33] = dl2 * ((v2 >> 8) as u8 as i8) as f32;
                    }
                }
            }
            138 => { // IQ3_K
                let d = h(b, 0); let mut extra = u16le(b, 2); let mut sh = u16le(b, 4);
                let (sl, qs, qh) = (&b[6..14], &b[14..78], &b[78..110]);
                let mut q0 = 0;
                for ib in 0..8 {
                    let dl1 = d * ((2 * (sl[ib] & 15) as i32 + 1) * if sh & 1 != 0 { -1 } else { 1 }) as f32;
                    let dl2 = d * ((2 * (sl[ib] >> 4) as i32 + 1) * if sh & 2 != 0 { -1 } else { 1 }) as f32;
                    sh >>= 2;
                    let v1 = if extra & 1 != 0 { 8 } else { 0 }; let v2 = if extra & 2 != 0 { 8 } else { 0 }; extra >>= 2;
                    let (sl_, sh_) = (2 * (ib % 4), ib % 8);
                    for j in 0..16 {
                        o[ib * 32 + j] = dl1 * IQ3NL_VALUES[v1 + (((qs[q0 + j] >> sl_) & 3) | (((qh[j] >> sh_) & 1) << 2)) as usize] as f32;
                        o[ib * 32 + j + 16] = dl2 * IQ3NL_VALUES[v2 + (((qs[q0 + j + 16] >> sl_) & 3) | (((qh[j + 16] >> sh_) & 1) << 2)) as usize] as f32;
                    }
                    if sl_ == 6 { q0 += 32; }
                }
            }
            156 => { // IQ3_KS
                let extra = u16le(b, 0) as i32; let (sc, qs, qh) = (&b[2..6], &b[6..70], &b[70..102]);
                let mut dl = [0f32; 8];
                for j in 0..4 {
                    dl[j] = rd * ((((sc[j] & 15) as i32) | (((extra >> j) & 1) << 4)) - 16) as f32;
                    dl[j + 4] = rd * ((((sc[j] >> 4) as i32) | (((extra >> (j + 4)) & 1) << 4)) - 16) as f32;
                }
                for i128 in 0..2 { for ib in 0..4 {
                    let vo = (((extra >> (8 + 4 * i128 + ib)) & 1) << 3) as usize;
                    for j in 0..32 {
                        o[i128 * 128 + ib * 32 + j] = dl[4 * i128 + ib] * IQ3NL_VALUES[vo + (((qs[i128 * 32 + j] >> (2 * ib)) & 3) | (((qh[j] >> (4 * i128 + ib)) & 1) << 2)) as usize] as f32;
                    }
                } }
            }
            139 => { // IQ4_K
                let d = h(b, 0); let mut extra = u16le(b, 2); let (sh_, sl, qs) = (&b[4..8], &b[8..16], &b[16..144]);
                for ib in 0..8 {
                    let sh = sh_[ib / 2] >> (4 * (ib % 2));
                    let dl1 = d * (((sl[ib] & 15) as i32 | ((sh as i32) << 4 & 0x30)) - 32) as f32;
                    let dl2 = d * (((sl[ib] >> 4) as i32 | ((sh as i32) << 2 & 0x30)) - 32) as f32;
                    let v1 = if extra & 1 != 0 { 16 } else { 0 }; let v2 = if extra & 2 != 0 { 16 } else { 0 }; extra >>= 2;
                    for j in 0..16 { let q = qs[ib * 16 + j]; o[ib * 32 + j] = dl1 * IQ4K_VALUES[v1 + (q & 15) as usize] as f32; o[ib * 32 + j + 16] = dl2 * IQ4K_VALUES[v2 + (q >> 4) as usize] as f32; }
                }
            }
            140 => { // IQ5_K
                let d = h(b, 0); let mut extra = u16le(b, 2); let (sh, sl, qs, qh0) = (&b[4..8], &b[8..16], &b[16..144], &b[144..176]);
                let (mut shift, mut qhoff) = (0, 0);
                for ib in 0..4 {
                    let s = sh[ib] as i32;
                    let dl = [d * (((sl[2 * ib] & 15) as i32 | ((s << 4) & 0x30)) - 32) as f32, d * (((sl[2 * ib] >> 4) as i32 | ((s << 2) & 0x30)) - 32) as f32,
                              d * (((sl[2 * ib + 1] & 15) as i32 | (s & 0x30)) - 32) as f32, d * (((sl[2 * ib + 1] >> 4) as i32 | ((s >> 2) & 0x30)) - 32) as f32];
                    let vo = [((extra & 1) << 5) as usize, ((extra & 2) << 4) as usize, ((extra & 4) << 3) as usize, ((extra & 8) << 2) as usize];
                    let qs = &qs[ib * 32..]; let qh = &qh0[qhoff..];
                    for j in 0..16 {
                        o[ib * 64 + j] = dl[0] * IQ5NL_VALUES[vo[0] + ((qs[j] & 15) | (((qh[j] >> shift) & 1) << 4)) as usize] as f32;
                        o[ib * 64 + j + 16] = dl[1] * IQ5NL_VALUES[vo[1] + ((qs[j + 16] & 15) | (((qh[j + 16] >> shift) & 1) << 4)) as usize] as f32;
                        o[ib * 64 + j + 32] = dl[2] * IQ5NL_VALUES[vo[2] + ((qs[j] >> 4) | (((qh[j] >> shift) & 2) << 3)) as usize] as f32;
                        o[ib * 64 + j + 48] = dl[3] * IQ5NL_VALUES[vo[3] + ((qs[j + 16] >> 4) | (((qh[j + 16] >> shift) & 2) << 3)) as usize] as f32;
                    }
                    extra >>= 4; shift += 2; if shift == 8 { qhoff += 32; shift = 0; }
                }
            }
            141 => { // IQ6_K
                let d = h(b, 0); let mut extra = u16le(b, 2); let (sl, qs, qh0) = (&b[4..20], &b[20..148], &b[148..212]);
                let (mut shift, mut qhoff) = (0, 0);
                let poly = |q: f32| -127.0 + q * (6.2568 + q * (-0.11218 + q * 0.0011972));
                for ib in 0..4 {
                    let qs = &qs[ib * 32..]; let qh = &qh0[qhoff..];
                    for j in 0..16 {
                        let q = [(qs[j] & 15) | (((qh[j] >> shift) & 3) << 4), (qs[j + 16] & 15) | (((qh[j + 16] >> shift) & 3) << 4),
                                 (qs[j] >> 4) | (((qh[j] >> shift) & 12) << 2), (qs[j + 16] >> 4) | (((qh[j + 16] >> shift) & 12) << 2)];
                        for k in 0..4 { let m = if extra >> k & 1 != 0 { 1.0 } else { 0.0 }; o[ib * 64 + j + 16 * k] = d * (sl[4 * ib + k] as i8) as f32 * (poly(q[k] as f32) + m); }
                    }
                    extra >>= 4; shift += 4; if shift == 8 { qhoff += 32; shift = 0; }
                }
            }
            144 => { // IQ4_KS
                let (sc, qs) = (&b[0..8], &b[8..136]);
                for ib in 0..8 {
                    let dl = rd * ((sc[ib] & 254) as i32 - 127) as f32; let vo = ((sc[ib] & 1) as usize) << 4;
                    for j in 0..16 { let q = qs[ib * 16 + j]; o[ib * 32 + j] = dl * IQ4K_VALUES[vo + (q & 15) as usize] as f32; o[ib * 32 + j + 16] = dl * IQ4K_VALUES[vo + (q >> 4) as usize] as f32; }
                }
            }
            152 => { // IQ5_KS
                let (sc, qs, qh) = (&b[0..8], &b[8..136], &b[136..168]);
                for ib in 0..4 {
                    let d1 = rd * ((sc[2 * ib] & 254) as i32 - 127) as f32; let d2 = rd * ((sc[2 * ib + 1] & 254) as i32 - 127) as f32;
                    let v1 = ((sc[2 * ib] & 1) as usize) << 5; let v2 = ((sc[2 * ib + 1] & 1) as usize) << 5;
                    for j in 0..32 {
                        let q = qs[ib * 32 + j];
                        o[ib * 64 + j] = d1 * IQ5NL_VALUES[v1 + ((q & 15) | (((qh[j] >> (2 * ib)) & 1) << 4)) as usize] as f32;
                        o[ib * 64 + j + 32] = d2 * IQ5NL_VALUES[v2 + ((q >> 4) | (((qh[j] >> (2 * ib + 1)) & 1) << 4)) as usize] as f32;
                    }
                }
            }
            146 => { // IQ4_KSS
                for ib in 0..8 {
                    let mut a = [0u16; 8]; let mut ls = 0i32;
                    for k in 0..8 { let q = u16le(b, ib * 16 + 2 * k); a[k] = q & 0xfffe; a[k] ^= a[k] >> 1; ls |= ((q & 1) as i32) << k; }
                    let vo = ((ls & 1) as usize) << 4; let dl = rd * ((ls & 254) - 127) as f32;
                    let mut aux8 = [0u8; 16]; for k in 0..8 { aux8[2 * k] = a[k] as u8; aux8[2 * k + 1] = (a[k] >> 8) as u8; }
                    for j in 0..16 { o[ib * 32 + j] = dl * IQ4K_VALUES[vo + (aux8[j] & 15) as usize] as f32; o[ib * 32 + j + 16] = dl * IQ4K_VALUES[vo + (aux8[j] >> 4) as usize] as f32; }
                }
            }
            153 => { // IQ2_KT: scales[4], ql[64] (u16 codes: first half of the block, then second half)
                let d = rd;
                for ib in 0..4 {
                    let (sl, sh) = (d * IQ4K_VALUES[(b[ib] & 15) as usize] as f32, d * IQ4K_VALUES[(b[ib] >> 4) as usize] as f32);
                    for ig in 0..4 {
                        let ql = u16le(b, 4 + 2 * (ib * 4 + ig)) as u32; let qh = u16le(b, 4 + 2 * (16 + ib * 4 + ig)) as u32;
                        kt_values(ql, 4096, sl, false, &mut o[(ib * 4 + ig) * 8..][..8]);
                        kt_values(qh, 4096, sh, false, &mut o[128 + (ib * 4 + ig) * 8..][..8]);
                    }
                }
            }
            154 => { // IQ3_KT: scales[4], ql[64], qh[32] (sign bits)
                let d = rd * 1.01;
                for ib in 0..4 {
                    let (sl, sh) = (d * (b[ib] & 15) as f32, d * (b[ib] >> 4) as f32);
                    let (lm, hm) = (1u8 << ib, 1u8 << (ib + 4));
                    for ig in 0..4 {
                        let jj = ib * 4 + ig;
                        let ql = u16le(b, 4 + 2 * jj) as u32; let qh = u16le(b, 4 + 2 * (16 + jj)) as u32;
                        let (yl, yh) = (jj * 8, 128 + jj * 8);
                        kt_values(ql, 4096, sl, true, &mut o[yl..yl + 8]);
                        kt_values(qh, 4096, sh, true, &mut o[yh..yh + 8]);
                        for j in 0..8 {
                            let sg = b[68 + ig * 8 + j];
                            if sg & lm != 0 { o[yl + j] = -o[yl + j]; }
                            if sg & hm != 0 { o[yh + j] = -o[yh + j]; }
                        }
                    }
                }
            }
            155 => { // IQ4_KT: shb u32[8], ql[64], qh[32]; groups of 4
                for ib in 0..8 {
                    let shb = u32::from_le_bytes(b[ib * 4..ib * 4 + 4].try_into().unwrap());
                    let offset = if shb & 1 != 0 { 32768 + 4096 } else { 4096 };
                    let ls = ((shb & 0xff) >> 1) as i32 - 64; let sl = rd * ls as f32;
                    for ig in 0..8 {
                        let jj = ib * 8 + ig;
                        let idx = b[32 + jj] as u32 | (((b[96 + jj % 32] as u32) << (8 - 4 * (jj / 32))) & 0xf00) | (((shb >> (8 + 3 * ig)) & 7) << 12);
                        kt_values(idx, offset, sl, false, &mut o[jj * 4..jj * 4 + 4]);
                    }
                }
            }
            158 => { // IQ1_KT: sh[8], ql[32], qh[16]
                for ib in 0..8 {
                    let sl = rd * IQ4K_VALUES[(b[ib] & 15) as usize] as f32;
                    for ig in 0..4 {
                        let mut idx = b[8 + ib * 4 + ig] as u32 | (((b[40 + (ib % 4) * 4 + ig] as u32) << (8 - 4 * (ib / 4))) & 0xf00);
                        idx |= ((b[ib] as u32) << (8 - ig)) & 0x1000;
                        kt_values(idx, 4096, sl, false, &mut o[ib * 32 + ig * 8..][..8]);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    true
}

#[cfg(test)]
mod tests {
    /// Compares every type's dequant to dumps from ik_llama's libggml (tools/oracle/oracle.c): ORACLE_DIR/or_<id>.{q,f}, 8 rows x 1024 cols.
    #[test]
    fn matches_ik_oracle() {
        let Ok(dir) = std::env::var("ORACLE_DIR") else { return };
        let mut bad = vec![];
        for ty in [16u32, 17, 18, 19, 20, 21, 22, 23, 29, 134, 135, 137, 138, 139, 140, 141, 144, 145, 146, 152, 153, 154, 155, 156, 157, 158] {
            let (Ok(q), Ok(f)) = (std::fs::read(format!("{dir}/or_{ty}.q")), std::fs::read(format!("{dir}/or_{ty}.f"))) else { continue };
            if crate::quant::block_info(ty).is_none() { bad.push(format!("{ty}: not implemented")); continue; }
            let rb = crate::quant::row_bytes(ty, 1024);
            let mut worst = 0f32;
            let mut ok = true;
            for r in 0..8 {
                let mut out = vec![0f32; 1024];
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::quant::dequant_row(ty, &q[r * rb..(r + 1) * rb], &mut out))).is_err() { ok = false; break; }
                for i in 0..1024 {
                    let e = f32::from_le_bytes(f[(r * 1024 + i) * 4..(r * 1024 + i) * 4 + 4].try_into().unwrap());
                    worst = worst.max((e - out[i]).abs() / (e.abs() + 1e-3));
                }
            }
            if !ok || worst > 1e-4 { bad.push(format!("{ty}: {}", if ok { format!("rel err {worst}") } else { "unsupported".into() })); }
        }
        assert!(bad.is_empty(), "mismatch: {bad:?}");
    }
}
