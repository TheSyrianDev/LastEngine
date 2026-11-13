// LastEngine CUDA kernels (clean-room, written from the GGML block-format descriptions).
// Host prepends: <cuda_fp16.h>, `__constant__ unsigned int IQ3_G32[256]` (IQ3_XXS grid as 4 packed bytes).
typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32; typedef signed char i8;
#define FULL 0xffffffffu

__device__ __forceinline__ float h2f(const u8* p) { return __half2float(*(const __half*)p); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(FULL, v, o); return v; }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigm_f(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ int ld32(const u8* p) { return (int)*(const u16*)p | ((int)*(const u16*)(p + 2) << 16); }
__device__ __forceinline__ int ld32a(const u8* p) { return *(const int*)p; }
#define DP(a, b, c) __dp4a((int)(a), (int)(b), (int)(c))

// x -> int8 units of 32 (scale xd, int sum xs). One warp per unit.
extern "C" __global__ void quant_x(const float* x, signed char* xq, float* xd, int* xs, int n_units) {
  int u = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (u >= n_units) return;
  float v = x[u * 32 + lane], a = fabsf(v);
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
  float d = a / 127.f, id = d > 0.f ? 1.f / d : 0.f;
  int q = __float2int_rn(v * id);
  xq[u * 32 + lane] = (signed char)q;
  int s = q; for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
  if (lane == 0) { xd[u] = d; xs[u] = s; }
}

__device__ __forceinline__ void sm_k4(int j, const u8* s, int& sc, int& m) {
  if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
  else { sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}
// Contribution of one 32-value unit (unit j of block b) dotted with int8 activations xq[8 ints], scale dx, int sum sx.
template <int TY> __device__ float udot(const u8* b, int j, const int* xq, float dx, int sx);
template <> __device__ float udot<2>(const u8* b, int, const int* xq, float dx, int sx) {
  int a = 0;
  for (int i = 0; i < 4; i++) { int v = ld32(b + 2 + 4 * i); a = DP(v & 0x0F0F0F0F, xq[i], a); a = DP((v >> 4) & 0x0F0F0F0F, xq[4 + i], a); }
  return h2f(b) * dx * (float)(a - 8 * sx);
}
template <> __device__ float udot<8>(const u8* b, int, const int* xq, float dx, int) {
  int a = 0; for (int i = 0; i < 8; i++) a = DP(ld32(b + 2 + 4 * i), xq[i], a);
  return h2f(b) * dx * (float)a;
}
template <> __device__ float udot<12>(const u8* b, int j, const int* xq, float dx, int sx) {
  float d = h2f(b), dm = h2f(b + 2); int sc, m; sm_k4(j, b + 4, sc, m);
  const u8* qs = b + 16 + (j >> 1) * 32; int sh = (j & 1) * 4, a = 0;
  for (int i = 0; i < 8; i++) a = DP((ld32a(qs + 4 * i) >> sh) & 0x0F0F0F0F, xq[i], a);
  return dx * (d * sc * (float)a - dm * m * (float)sx);
}
template <> __device__ float udot<13>(const u8* b, int j, const int* xq, float dx, int sx) {
  float d = h2f(b), dm = h2f(b + 2); int sc, m; sm_k4(j, b + 4, sc, m);
  const u8* qh = b + 16; const u8* qs = b + 48 + (j >> 1) * 32; int sh = (j & 1) * 4, a = 0;
  for (int i = 0; i < 8; i++) a = DP(((ld32a(qs + 4 * i) >> sh) & 0x0F0F0F0F) | (((ld32a(qh + 4 * i) >> j) & 0x01010101) << 4), xq[i], a);
  return dx * (d * sc * (float)a - dm * m * (float)sx);
}
template <> __device__ float udot<11>(const u8* b, int j, const int* xq, float dx, int) {
  const u8* hm = b; const u8* qs = b + 32 + (j >> 2) * 32; const u8* sc = b + 96; int s2 = 2 * (j & 3), a0 = 0, a1 = 0;
  for (int i = 0; i < 8; i++) {
    int v = (ld32(qs + 4 * i) >> s2) & 0x03030303, h = (~ld32(hm + 4 * i) >> j) & 0x01010101;
    int q = __vsubss4(v, h << 2);
    if (i < 4) a0 = DP(q, xq[i], a0); else a1 = DP(q, xq[i], a1);
  }
  float r = 0.f;
  for (int h = 0; h < 2; h++) {
    int i = 2 * j + h; int lo = i < 8 ? (sc[i] & 15) : (sc[i - 8] >> 4); int hi = (sc[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
    r += (float)(((lo | (hi << 4)) - 32) * (h ? a1 : a0));
  }
  return h2f(b + 108) * dx * r;
}
template <> __device__ float udot<14>(const u8* b, int j, const int* xq, float dx, int) {
  int half = j >> 2, s = j & 3; const u8* ql = b + half * 64 + (s & 1) * 32; const u8* qh = b + 128 + half * 32;
  const i8* sc = (const i8*)(b + 192) + half * 8 + s * 2; int a0 = 0, a1 = 0, sh = (s >> 1) * 4;
  for (int i = 0; i < 8; i++) {
    int q = __vsubss4(((ld32(ql + 4 * i) >> sh) & 0x0F0F0F0F) | (((ld32(qh + 4 * i) >> (2 * s)) & 0x03030303) << 4), 0x20202020);
    if (i < 4) a0 = DP(q, xq[i], a0); else a1 = DP(q, xq[i], a1);
  }
  return h2f(b + 208) * dx * (float)(sc[0] * a0 + sc[1] * a1);
}
__device__ __forceinline__ u32 sgmask4(u32 nib) { return (nib & 1 ? 0xFFu : 0) | (nib & 2 ? 0xFF00u : 0) | (nib & 4 ? 0xFF0000u : 0) | (nib & 8 ? 0xFF000000u : 0); }
template <> __device__ float udot<16>(const u8* b, int j, const int* xq, float dx, int) {
  const u8* q = b + 2 + j * 8; u32 aux = (u32)*(const u16*)(q + 4) | ((u32)*(const u16*)(q + 6) << 16);
  int a = 0;
  for (int l = 0; l < 4; l++) {
    u32 si = (aux >> (7 * l)) & 127; u32 sg = si | ((__popc(si) & 1) << 7);
    for (int h = 0; h < 2; h++) { u32 m = sgmask4((sg >> (4 * h)) & 15); a = DP((int)__vsub4(IQ2XXS_G[q[l] * 2 + h] ^ m, m), xq[2 * l + h], a); }
  }
  return h2f(b) * (0.5f + (float)(aux >> 28)) * 0.25f * dx * (float)a;
}
template <> __device__ float udot<22>(const u8* b, int j, const int* xq, float dx, int) {
  const u8* qs = b + 2 + j * 4; const u8* sn = b + 34 + j * 4; u32 qh = b[66 + j]; u32 sc = b[74 + j];
  int a0 = 0, a1 = 0;
  for (int l = 0; l < 4; l++) {
    u32 gi = qs[l] | ((qh << (8 - 2 * l)) & 0x300); u32 sb = sn[l];
    for (int h = 0; h < 2; h++) { u32 m = sgmask4((sb >> (4 * h)) & 15); int g = (int)__vsub4(IQ2S_G[gi * 2 + h] ^ m, m); if (l < 2) a0 = DP(g, xq[2 * l + h], a0); else a1 = DP(g, xq[2 * l + h], a1); }
  }
  return h2f(b) * 0.25f * dx * ((0.5f + (float)(sc & 15)) * (float)a0 + (0.5f + (float)(sc >> 4)) * (float)a1);
}
template <> __device__ float udot<18>(const u8* b, int j, const int* xq, float dx, int) {
  const u8* qs = b + 2 + j * 8; u32 aux = (u32)*(const u16*)(b + 66 + j * 4) | ((u32)*(const u16*)(b + 68 + j * 4) << 16);
  int a = 0;
  for (int l = 0; l < 4; l++) {
    u32 si = (aux >> (7 * l)) & 127; u32 sg = si | ((__popc(si) & 1) << 7);
    for (int h = 0; h < 2; h++) {
      u32 nib = (sg >> (4 * h)) & 15;
      u32 m = (nib & 1 ? 0xFFu : 0) | (nib & 2 ? 0xFF00u : 0) | (nib & 4 ? 0xFF0000u : 0) | (nib & 8 ? 0xFF000000u : 0);
      int g = (int)__vsub4(IQ3_G32[qs[2 * l + h]] ^ m, m);
      a = DP(g, xq[2 * l + h], a);
    }
  }
  return h2f(b) * (0.5f + (float)(aux >> 28)) * 0.5f * dx * (float)a;
}

// Unpack one 32-value unit into 8 int32 (4 int8 each) + scales: result = dx * (s0*dot(w[0..3],x[0..3]) + s1*dot(w[4..7],x[4..7]) - mn*sx)
template <int TY> __device__ void uload(const u8* b, int j, int* w, float& s0, float& s1, float& mn);
template <> __device__ void uload<2>(const u8* b, int, int* w, float& s0, float& s1, float& mn) {
  for (int i = 0; i < 4; i++) { int v = ld32(b + 2 + 4 * i); w[i] = v & 0x0F0F0F0F; w[4 + i] = (v >> 4) & 0x0F0F0F0F; }
  s0 = s1 = h2f(b); mn = 8.f * s0;
}
template <> __device__ void uload<8>(const u8* b, int, int* w, float& s0, float& s1, float& mn) {
  for (int i = 0; i < 8; i++) w[i] = ld32(b + 2 + 4 * i);
  s0 = s1 = h2f(b); mn = 0.f;
}
template <> __device__ void uload<12>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  float d = h2f(b), dm = h2f(b + 2); int sc, m; sm_k4(j, b + 4, sc, m);
  const u8* qs = b + 16 + (j >> 1) * 32; int sh = (j & 1) * 4;
  for (int i = 0; i < 8; i++) w[i] = (ld32a(qs + 4 * i) >> sh) & 0x0F0F0F0F;
  s0 = s1 = d * sc; mn = dm * m;
}
template <> __device__ void uload<13>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  float d = h2f(b), dm = h2f(b + 2); int sc, m; sm_k4(j, b + 4, sc, m);
  const u8* qh = b + 16; const u8* qs = b + 48 + (j >> 1) * 32; int sh = (j & 1) * 4;
  for (int i = 0; i < 8; i++) w[i] = ((ld32a(qs + 4 * i) >> sh) & 0x0F0F0F0F) | (((ld32a(qh + 4 * i) >> j) & 0x01010101) << 4);
  s0 = s1 = d * sc; mn = dm * m;
}
template <> __device__ void uload<11>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  const u8* hm = b; const u8* qs = b + 32 + (j >> 2) * 32; const u8* sc = b + 96; int s2 = 2 * (j & 3);
  for (int i = 0; i < 8; i++) { int v = (ld32(qs + 4 * i) >> s2) & 0x03030303, h = (~ld32(hm + 4 * i) >> j) & 0x01010101; w[i] = __vsubss4(v, h << 2); }
  float d = h2f(b + 108); float r[2];
  for (int h = 0; h < 2; h++) { int i = 2 * j + h; int lo = i < 8 ? (sc[i] & 15) : (sc[i - 8] >> 4); int hi = (sc[8 + (i & 3)] >> (2 * (i >> 2))) & 3; r[h] = (float)((lo | (hi << 4)) - 32); }
  s0 = d * r[0]; s1 = d * r[1]; mn = 0.f;
}
template <> __device__ void uload<14>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  int half = j >> 2, s = j & 3; const u8* ql = b + half * 64 + (s & 1) * 32; const u8* qh = b + 128 + half * 32;
  const i8* sc = (const i8*)(b + 192) + half * 8 + s * 2; int sh = (s >> 1) * 4;
  for (int i = 0; i < 8; i++) w[i] = __vsubss4(((ld32(ql + 4 * i) >> sh) & 0x0F0F0F0F) | (((ld32(qh + 4 * i) >> (2 * s)) & 0x03030303) << 4), 0x20202020);
  float d = h2f(b + 208); s0 = d * sc[0]; s1 = d * sc[1]; mn = 0.f;
}
template <> __device__ void uload<18>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  const u8* qs = b + 2 + j * 8; u32 aux = (u32)*(const u16*)(b + 66 + j * 4) | ((u32)*(const u16*)(b + 68 + j * 4) << 16);
  for (int l = 0; l < 4; l++) {
    u32 si = (aux >> (7 * l)) & 127; u32 sg = si | ((__popc(si) & 1) << 7);
    for (int h = 0; h < 2; h++) {
      u32 nib = (sg >> (4 * h)) & 15;
      u32 m = (nib & 1 ? 0xFFu : 0) | (nib & 2 ? 0xFF00u : 0) | (nib & 4 ? 0xFF0000u : 0) | (nib & 8 ? 0xFF000000u : 0);
      w[2 * l + h] = (int)__vsub4(IQ3_G32[qs[2 * l + h]] ^ m, m);
    }
  }
  s0 = s1 = h2f(b) * (0.5f + (float)(aux >> 28)) * 0.5f; mn = 0.f;
}
template <> __device__ void uload<16>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  const u8* q = b + 2 + j * 8; u32 aux = (u32)*(const u16*)(q + 4) | ((u32)*(const u16*)(q + 6) << 16);
  for (int l = 0; l < 4; l++) {
    u32 si = (aux >> (7 * l)) & 127; u32 sg = si | ((__popc(si) & 1) << 7);
    for (int h = 0; h < 2; h++) { u32 m = sgmask4((sg >> (4 * h)) & 15); w[2 * l + h] = (int)__vsub4(IQ2XXS_G[q[l] * 2 + h] ^ m, m); }
  }
  s0 = s1 = h2f(b) * (0.5f + (float)(aux >> 28)) * 0.25f; mn = 0.f;
}
template <> __device__ void uload<22>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  const u8* qs = b + 2 + j * 4; const u8* sn = b + 34 + j * 4; u32 qh = b[66 + j]; u32 sc = b[74 + j];
  for (int l = 0; l < 4; l++) {
    u32 gi = qs[l] | ((qh << (8 - 2 * l)) & 0x300); u32 sb = sn[l];
    for (int h = 0; h < 2; h++) { u32 m = sgmask4((sb >> (4 * h)) & 15); w[2 * l + h] = (int)__vsub4(IQ2S_G[gi * 2 + h] ^ m, m); }
  }
  float d = h2f(b) * 0.25f; s0 = d * (0.5f + (float)(sc & 15)); s1 = d * (0.5f + (float)(sc >> 4)); mn = 0.f;
}
template <int TY> struct Q { static const int UB = 8, BYTES = 0, HDR = 0, NEW = 0; };
#define QN(ty, ub, by, hdr) template <> struct Q<ty> { static const int UB = ub, BYTES = by, HDR = hdr, NEW = 1; };
QN(29, 8, 56, 0) QN(134, 2, 13, 2) QN(135, 2, 16, 4) QN(137, 8, 76, 0) QN(138, 8, 110, 0) QN(139, 8, 144, 0) QN(140, 8, 176, 0) QN(141, 8, 212, 0)
QN(144, 8, 136, 4) QN(145, 8, 70, 2) QN(146, 8, 128, 4) QN(152, 8, 168, 4) QN(153, 8, 68, 4) QN(154, 8, 100, 4) QN(155, 8, 128, 4) QN(156, 8, 102, 2) QN(157, 8, 86, 2) QN(158, 8, 56, 4)
template <> struct Q<2>  { static const int UB = 1, BYTES = 18, HDR = 0, NEW = 0; };
template <> struct Q<8>  { static const int UB = 1, BYTES = 34, HDR = 0, NEW = 0; };
template <> struct Q<11> { static const int UB = 8, BYTES = 110, HDR = 0, NEW = 0; };
template <> struct Q<12> { static const int UB = 8, BYTES = 144, HDR = 0, NEW = 0; };
template <> struct Q<13> { static const int UB = 8, BYTES = 176, HDR = 0, NEW = 0; };
template <> struct Q<14> { static const int UB = 8, BYTES = 210, HDR = 0, NEW = 0; };
template <> struct Q<18> { static const int UB = 8, BYTES = 98, HDR = 0, NEW = 0; };
template <> struct Q<16> { static const int UB = 8, BYTES = 66, HDR = 0, NEW = 0; };
template <> struct Q<22> { static const int UB = 8, BYTES = 82, HDR = 0, NEW = 0; };

// ---- IQ formats translated from ik_llama.cpp / ggml (MIT; see NOTICE): unpack one 32-value unit ----
__device__ const signed char KVN[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ __forceinline__ int kv4(u32 nib4) { // four nibbles (one per byte) -> four iq4_nl values packed as int8x4
  return (int)((u32)(u8)KVN[nib4 & 15] | ((u32)(u8)KVN[(nib4 >> 8) & 15] << 8) | ((u32)(u8)KVN[(nib4 >> 16) & 15] << 16) | ((u32)(u8)KVN[(nib4 >> 24) & 15] << 24));
}
__device__ __forceinline__ void nl_unit(const u8* q, int* w) {
  for (int i = 0; i < 4; i++) { u32 v = (u32)ld32(q + 4 * i); w[i] = kv4(v & 0x0F0F0F0F); w[4 + i] = kv4((v >> 4) & 0x0F0F0F0F); }
}
template <> __device__ void uload<20>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) { nl_unit(b + 2, w); s0 = s1 = h2f(b); mn = 0.f; }
template <> __device__ void uload<23>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  int sh = (int)*(const u16*)(b + 2); int ls = ((b[4 + (j >> 1)] >> (4 * (j & 1))) & 15) | (((sh >> (2 * j)) & 3) << 4);
  nl_unit(b + 8 + 16 * j, w); s0 = s1 = h2f(b) * (float)(ls - 32); mn = 0.f;
}
template <> __device__ void uload<17>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  for (int l = 0; l < 4; l++) {
    u32 q = *(const u16*)(b + 2 + 2 * (4 * j + l)); u32 si = q >> 9; u32 sg = si | ((__popc(si) & 1) << 7);
    for (int h = 0; h < 2; h++) { u32 m = sgmask4((sg >> (4 * h)) & 15); w[2 * l + h] = (int)__vsub4(IQ2XS_G[(q & 511) * 2 + h] ^ m, m); }
  }
  u32 sc = b[66 + j]; float d = h2f(b) * 0.25f; s0 = d * (0.5f + (float)(sc & 15)); s1 = d * (0.5f + (float)(sc >> 4)); mn = 0.f;
}
template <> __device__ void uload<21>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  int p = j >> 1, half = j & 1; const u8* qs = b + 2 + p * 16 + half * 8; u32 qh = b[66 + p * 2 + half]; const u8* sg = b + 74 + p * 8 + half * 4; u32 sc = b[106 + p];
  for (int l = 0; l < 4; l++) {
    u32 g1 = qs[2 * l] | ((qh << (8 - 2 * l)) & 256), g2 = qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256); u32 sb = sg[l];
    w[2 * l] = (int)__vsub4(IQ3S_G[g1] ^ sgmask4(sb & 15), sgmask4(sb & 15));
    w[2 * l + 1] = (int)__vsub4(IQ3S_G[g2] ^ sgmask4(sb >> 4), sgmask4(sb >> 4));
  }
  s0 = s1 = h2f(b) * (float)(1 + 2 * (half ? (sc >> 4) : (sc & 15))); mn = 0.f;
}
template <> __device__ void uload<19>(const u8* b, int j, int* w, float& s0, float& s1, float& mn) {
  u32 qh = *(const u16*)(b + 34 + 2 * j);
  for (int l = 0; l < 4; l++) { u32 gi = b[2 + 4 * j + l] | (((qh >> (3 * l)) & 7) << 8); w[2 * l] = (int)IQ1S_G[gi * 2]; w[2 * l + 1] = (int)IQ1S_G[gi * 2 + 1]; }
  float dl = h2f(b) * (float)(2 * ((qh >> 12) & 7) + 1); s0 = s1 = dl; mn = -dl * ((qh & 0x8000) ? -0.125f : 0.125f);
}
template <int TY> __device__ float udot_gen(const u8* b, int j, const int* xq, float dx, int sx) {
  int w[8]; float s0, s1, mn; uload<TY>(b, j, w, s0, s1, mn); int a0 = 0, a1 = 0;
  for (int i = 0; i < 4; i++) { a0 = DP(w[i], xq[i], a0); a1 = DP(w[4 + i], xq[4 + i], a1); }
  return dx * (s0 * (float)a0 + s1 * (float)a1 - mn * (float)sx);
}
template <> __device__ float udot<17>(const u8* b, int j, const int* xq, float dx, int sx) { return udot_gen<17>(b, j, xq, dx, sx); }
template <> __device__ float udot<19>(const u8* b, int j, const int* xq, float dx, int sx) { return udot_gen<19>(b, j, xq, dx, sx); }
template <> __device__ float udot<20>(const u8* b, int j, const int* xq, float dx, int sx) { return udot_gen<20>(b, j, xq, dx, sx); }
template <> __device__ float udot<21>(const u8* b, int j, const int* xq, float dx, int sx) { return udot_gen<21>(b, j, xq, dx, sx); }
template <> __device__ float udot<23>(const u8* b, int j, const int* xq, float dx, int sx) { return udot_gen<23>(b, j, xq, dx, sx); }

// ---- ik_llama IQ formats: one unit = 32 values -> int8 weights + two scales; `row` is the row start (per-row scale header lives there) ----
__device__ __forceinline__ float rowscale(const u8* row, int hdr) { return hdr == 4 ? *(const float*)row : hdr == 2 ? h2f(row) : 0.f; }
__device__ __forceinline__ u32 rd16(const u8* p) { return (u32)p[0] | ((u32)p[1] << 8); }
__device__ __forceinline__ void kt_gen(u32 code, u32 off, int n, bool ab, signed char* out) {
  u32 x = code + off;
  for (int i = 0; i < n; i++) {
    x *= 0xCBAC1FEDu; u32 s = x & 0x3f3f3f3fu;
    int t = (int)(s & 0xff) + (int)((s >> 8) & 0xff) + (int)((s >> 16) & 0xff) + (int)(s >> 24) - 126;
    out[i] = (signed char)(ab ? (t < 0 ? -t : t) : t);
  }
}
template <int TY> __device__ void uloadr(const u8* b, int j, int* w, float& s0, float& s1, float& mn, const u8* row) {
  if constexpr (!Q<TY>::NEW) { uload<TY>(b, j, w, s0, s1, mn); }
  else {
    __align__(4) signed char v[32]; float rd = rowscale(row, Q<TY>::HDR); mn = 0.f; float d0 = 0.f, d1 = 0.f;
    if constexpr (TY == 29) {
      u32 sc[4]; for (int i = 0; i < 4; i++) sc[i] = rd16(b + 48 + 2 * i);
      u16 db = (u16)((sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000)); float d = __half2float(*(const __half*)&db);
      u32 s = sc[j / 2]; int sh = 6 * (j % 2);
      d0 = d * (2 * ((s >> sh) & 7) + 1) * 0.125f; d1 = d * (2 * ((s >> (sh + 3)) & 7) + 1) * 0.125f;
      const u8* qs = b + j * 4; const u8* qh = b + 32 + j * 2;
      for (int l = 0; l < 4; l++) {
        u32 hb = qh[l / 2]; u32 idx = qs[l] | ((l % 2 == 0 ? (hb << 8) : (hb << 4)) & 0x700); int neg = (hb & (l % 2 == 0 ? 0x08 : 0x80)) != 0;
        const signed char* g = (const signed char*)&IQ1S_G[idx * 2];
        for (int i = 0; i < 8; i++) v[l * 8 + i] = (signed char)(8 * g[i] + (neg ? -1 : 1));
      }
    } else if constexpr (TY == 134) {
      d0 = d1 = rd;
      for (int k = 0; k < 32; k++) {
        int n = j * 32 + k, i16 = n / 16, r = n % 16; u32 vv = r < 15 ? ((u32)b[i16 * 3 + r / 5] * (u32)(81 / (r % 5 == 0 ? 1 : r % 5 == 1 ? 3 : r % 5 == 2 ? 9 : r % 5 == 3 ? 27 : 81))) : 0;
        static_assert(true, "");
        const int km[5] = {81, 27, 9, 3, 1};
        vv = (u32)(u8)((u32)b[r < 15 ? i16 * 3 + r / 5 : 12] * (u32)(r < 15 ? km[r % 5] : km[i16]));
        v[k] = (signed char)(((vv + (vv >> 1)) >> 7) - 1);
      }
    } else if constexpr (TY == 135) {
      d0 = d1 = rd;
      for (int k = 0; k < 32; k++) { int n = j * 32 + k; v[k] = (signed char)(((b[n % 16] >> (2 * (n / 16))) & 3) - 1); }
    } else if constexpr (TY == 137) {
      float d = h2f(b); u32 ex = rd16(b + 2) >> (2 * j); u32 sc = b[4 + j]; int sh = 2 * (j % 4); const u8* qs = b + 12 + 32 * (j / 4);
      d0 = d * (float)((int)(sc & 15) - 8); d1 = d * (float)((int)(sc >> 4) - 8); int v0 = (ex & 1) ? 4 : 0, v1 = (ex & 2) ? 4 : 0;
      for (int k = 0; k < 32; k++) v[k] = IQ2NL_V[(k < 16 ? v0 : v1) + ((qs[k] >> sh) & 3)];
    } else if constexpr (TY == 145) {
      int ib = j / 2, hf = j % 2; u32 ex = rd16(b) >> (2 * ib); u32 sc = b[2 + ib]; int sh = 4 * (ib % 2) + 2 * hf; const u8* qs = b + 6 + 32 * (ib / 2);
      d0 = d1 = rd * (float)((int)(hf ? ((sc >> 4) | ((ex >> 5) & 0x10)) : ((sc & 15) | ((ex >> 4) & 0x10))) - 16); int vo = (ex >> hf) & 1 ? 4 : 0;
      for (int k = 0; k < 32; k++) v[k] = IQ2NL_V[vo + ((qs[k] >> sh) & 3)];
    } else if constexpr (TY == 157) {
      int ib = j / 2, hf = j % 2; int shh = (int)rd16(b); const u8* sl = b + 2; const u8* qs = b + 6 + ib * 16; const u8* qh = b + 70;
      int lo = (sl[(2 * ib + hf) % 4] >> (4 * (ib / 2))) & 15, hi = (shh >> (4 * ib + 2 * hf)) & 3;
      d0 = d1 = rd * (float)((lo | (hi << 4)) - 32);
      for (int jj = 0; jj < 16; jj++) {
        u32 q = qs[jj]; u32 idx = (hf ? (q >> 4) : (q & 15)) | (((qh[jj] >> (2 * ib + hf)) & 1) << 4); u32 vv = IQ2KL_V[idx];
        v[2 * jj] = (signed char)(vv & 0xff); v[2 * jj + 1] = (signed char)(vv >> 8);
      }
    } else if constexpr (TY == 138) {
      float d = h2f(b); u32 ex = rd16(b + 2) >> (2 * j), shs = rd16(b + 4) >> (2 * j); u32 sl = b[6 + j];
      d0 = d * (float)((2 * (int)(sl & 15) + 1) * ((shs & 1) ? -1 : 1)); d1 = d * (float)((2 * (int)(sl >> 4) + 1) * ((shs & 2) ? -1 : 1));
      int v0 = (ex & 1) ? 8 : 0, v1 = (ex & 2) ? 8 : 0; int sh = 2 * (j % 4); const u8* qs = b + 14 + 32 * (j / 4); const u8* qh = b + 78;
      for (int k = 0; k < 32; k++) v[k] = IQ3NL_V[(k < 16 ? v0 : v1) + (((qs[k] >> sh) & 3) | (((qh[k] >> j) & 1) << 2))];
    } else if constexpr (TY == 156) {
      int ex = (int)rd16(b); int i128 = j / 4, ib = j % 4; const u8* sc = b + 2; const u8* qs = b + 6 + i128 * 32; const u8* qh = b + 70;
      int lowp = j < 4 ? (sc[j] & 15) : (sc[j - 4] >> 4);
      d0 = d1 = rd * (float)((lowp | (((ex >> j) & 1) << 4)) - 16); int vo = ((ex >> (8 + j)) & 1) << 3;
      for (int k = 0; k < 32; k++) v[k] = IQ3NL_V[vo + (((qs[k] >> (2 * ib)) & 3) | (((qh[k] >> (4 * i128 + ib)) & 1) << 2))];
    } else if constexpr (TY == 139) {
      float d = h2f(b); u32 ex = rd16(b + 2) >> (2 * j); u32 sh = b[4 + j / 2] >> (4 * (j % 2)); u32 sl = b[8 + j];
      d0 = d * (float)((int)((sl & 15) | ((sh << 4) & 0x30)) - 32); d1 = d * (float)((int)((sl >> 4) | ((sh << 2) & 0x30)) - 32);
      int v0 = (ex & 1) ? 16 : 0, v1 = (ex & 2) ? 16 : 0; const u8* qs = b + 16 + j * 16;
      for (int k = 0; k < 32; k++) v[k] = k < 16 ? IQ4K_V[v0 + (qs[k] & 15)] : IQ4K_V[v1 + (qs[k - 16] >> 4)];
    } else if constexpr (TY == 140) {
      float d = h2f(b); int ib = j / 2, hf = j % 2; u32 ex = rd16(b + 2) >> (4 * ib); u32 s = b[4 + ib]; const u8* sl = b + 8; int shf = 2 * (ib % 4);
      const u8* qs = b + 16 + ib * 32; const u8* qh = b + 144 + 32 * (ib / 4);
      float dl[4]; dl[0] = d * (float)((int)((sl[2 * ib] & 15) | ((s << 4) & 0x30)) - 32); dl[1] = d * (float)((int)((sl[2 * ib] >> 4) | ((s << 2) & 0x30)) - 32);
      dl[2] = d * (float)((int)((sl[2 * ib + 1] & 15) | (s & 0x30)) - 32); dl[3] = d * (float)((int)((sl[2 * ib + 1] >> 4) | ((s >> 2) & 0x30)) - 32);
      d0 = dl[2 * hf]; d1 = dl[2 * hf + 1]; int vo0 = hf ? ((ex & 4) << 3) : ((ex & 1) << 5), vo1 = hf ? ((ex & 8) << 2) : ((ex & 2) << 4);
      for (int k = 0; k < 32; k++) {
        u32 lowb = hf ? ((qs[k] >> 4) | (((qh[k] >> shf) & 2) << 3)) : ((qs[k] & 15) | (((qh[k] >> shf) & 1) << 4));
        v[k] = IQ5NL_V[(k < 16 ? vo0 : vo1) + lowb];
      }
    } else if constexpr (TY == 141) {
      float d = h2f(b); int ib = j / 2, hf = j % 2; u32 ex = rd16(b + 2) >> (4 * ib); const signed char* sl = (const signed char*)(b + 4);
      const u8* qs = b + 20 + ib * 32; const u8* qh = b + 148 + 32 * (ib / 2); int shf = 4 * (ib % 2);
      d0 = d * (float)sl[4 * ib + 2 * hf]; d1 = d * (float)sl[4 * ib + 2 * hf + 1];
      for (int k = 0; k < 32; k++) {
        int n = 2 * hf + (k >> 4); u32 q = hf ? ((qs[k] >> 4) | (((qh[k] >> shf) & 12) << 2)) : ((qs[k] & 15) | (((qh[k] >> shf) & 3) << 4));
        v[k] = IQ6NL_V[q + (((ex >> n) & 1) ? 64 : 0)];
      }
    } else if constexpr (TY == 144) {
      u32 sc = b[j]; d0 = d1 = rd * (float)((int)(sc & 254) - 127); int vo = (sc & 1) << 4; const u8* qs = b + 8 + j * 16;
      for (int k = 0; k < 32; k++) v[k] = k < 16 ? IQ4K_V[vo + (qs[k] & 15)] : IQ4K_V[vo + (qs[k - 16] >> 4)];
    } else if constexpr (TY == 152) {
      int ib = j / 2, hf = j % 2; u32 sc = b[2 * ib + hf]; d0 = d1 = rd * (float)((int)(sc & 254) - 127); int vo = (sc & 1) << 5;
      const u8* qs = b + 8 + ib * 32; const u8* qh = b + 136;
      for (int k = 0; k < 32; k++) v[k] = IQ5NL_V[vo + ((hf ? (qs[k] >> 4) : (qs[k] & 15)) | (((qh[k] >> (2 * ib + hf)) & 1) << 4))];
    } else if constexpr (TY == 146) {
      u32 a[8]; int ls = 0; for (int k = 0; k < 8; k++) { u32 q = rd16(b + j * 16 + 2 * k); a[k] = q & 0xfffe; a[k] ^= a[k] >> 1; ls |= (int)(q & 1) << k; }
      d0 = d1 = rd * (float)((ls & 254) - 127); int vo = (ls & 1) << 4; u8 ax[16]; for (int k = 0; k < 8; k++) { ax[2 * k] = (u8)a[k]; ax[2 * k + 1] = (u8)(a[k] >> 8); }
      for (int k = 0; k < 32; k++) v[k] = k < 16 ? IQ4K_V[vo + (ax[k] & 15)] : IQ4K_V[vo + (ax[k - 16] >> 4)];
    } else if constexpr (TY == 153 || TY == 154) {
      int ib = j % 4, hf = j / 4; float sc = rd * (TY == 154 ? 1.01f : 1.f); u32 nb = b[ib];
      float s = (float)IQ4K_V[(nb >> (4 * hf)) & 15]; d0 = d1 = sc * (TY == 154 ? (float)(nb >> (4 * hf) & 15) : s);
      for (int ig = 0; ig < 4; ig++) {
        u32 code = rd16(b + 4 + 2 * (hf * 16 + ib * 4 + ig)); kt_gen(code, 4096, 8, TY == 154, v + ig * 8);
        if (TY == 154) for (int i = 0; i < 8; i++) if (b[68 + ig * 8 + i] & (1u << (ib + 4 * hf))) v[ig * 8 + i] = -v[ig * 8 + i];
      }
    } else if constexpr (TY == 155) {
      u32 shb = *(const u32*)(b + j * 4); u32 off = (shb & 1) ? 32768 + 4096 : 4096; d0 = d1 = rd * (float)((int)((shb & 0xff) >> 1) - 64);
      for (int ig = 0; ig < 8; ig++) {
        int jj = j * 8 + ig; u32 idx = b[32 + jj] | (((u32)b[96 + jj % 32] << (8 - 4 * (jj / 32))) & 0xf00) | (((shb >> (8 + 3 * ig)) & 7) << 12);
        kt_gen(idx, off, 4, false, v + ig * 4);
      }
    } else if constexpr (TY == 158) {
      d0 = d1 = rd * (float)IQ4K_V[b[j] & 15];
      for (int ig = 0; ig < 4; ig++) {
        u32 idx = b[8 + j * 4 + ig] | (((u32)b[40 + (j % 4) * 4 + ig] << (8 - 4 * (j / 4))) & 0xf00); idx |= ((u32)b[j] << (8 - ig)) & 0x1000;
        kt_gen(idx, 4096, 8, false, v + ig * 8);
      }
    }
    for (int i = 0; i < 8; i++) w[i] = *(const int*)&v[4 * i];
    s0 = d0; s1 = d1;
  }
}
template <int TY> __device__ float udotr(const u8* b, int j, const int* xq, float dx, int sx, const u8* row) {
  if constexpr (!Q<TY>::NEW) { return udot<TY>(b, j, xq, dx, sx); }
  else {
    int w[8]; float s0, s1, mn; uloadr<TY>(b, j, w, s0, s1, mn, row); int a0 = 0, a1 = 0;
    for (int i = 0; i < 4; i++) { a0 = DP(w[i], xq[i], a0); a1 = DP(w[4 + i], xq[4 + i], a1); }
    return dx * (s0 * (float)a0 + s1 * (float)a1);
  }
}
template <> struct Q<17> { static const int UB = 8, BYTES = 74, HDR = 0, NEW = 0; };
template <> struct Q<19> { static const int UB = 8, BYTES = 50, HDR = 0, NEW = 0; };
template <> struct Q<20> { static const int UB = 1, BYTES = 18, HDR = 0, NEW = 0; };
template <> struct Q<21> { static const int UB = 8, BYTES = 110, HDR = 0, NEW = 0; };
template <> struct Q<23> { static const int UB = 8, BYTES = 136, HDR = 0, NEW = 0; };

// out[slot*rows + r] = W[slot][r,:] . x[slot]. Quantised activations: xq bytes, xd scales, xs sums, slot stride xstride elements.
// 256 threads = 8 warps; a warp covers 32/lpr rows, lpr = min(32, cols/32) lanes per row.
template <int TY>
__device__ void mv_impl(const u8* const* ptrs, const signed char* xq, const float* xd, const int* xs, float* out, int rows, int cols, int xstride) {
  extern __shared__ int sm[];
  int units = cols >> 5, slot = blockIdx.y;
  int* sxq = sm; float* sxd = (float*)(sm + units * 9); int* sxs = (int*)(sxd + units);
  const int* gq = (const int*)(xq + (size_t)slot * xstride);
  for (int i = threadIdx.x; i < units * 8; i += 256) sxq[(i >> 3) * 9 + (i & 7)] = gq[i];
  for (int i = threadIdx.x; i < units; i += 256) { sxd[i] = xd[(size_t)slot * (xstride >> 5) + i]; sxs[i] = xs[(size_t)slot * (xstride >> 5) + i]; }
  __syncthreads();
  const int UB = Q<TY>::UB, BY = Q<TY>::BYTES;
  int lpr = units < 32 ? units : 32, rpw = 32 / lpr;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = (blockIdx.x * 8 + warp) * rpw + lane / lpr, li = lane % lpr;
  bool ok = r < rows;
  const u8* row = ptrs[slot] + (size_t)(ok ? r : 0) * ((cols / (UB * 32)) * BY + Q<TY>::HDR);
  float acc = 0.f;
  if (ok) for (int u = li; u < units; u += lpr) acc += udotr<TY>(row + Q<TY>::HDR + (u / UB) * BY, u % UB, sxq + u * 9, sxd[u], sxs[u], row);
  for (int o = lpr >> 1; o; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
  if (ok && li == 0) out[(size_t)slot * rows + r] = acc;
}
#define MV(TY) extern "C" __global__ void matvec_##TY(const u8* const* p, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols, int xstride) { mv_impl<TY>(p, xq, xd, xs, o, rows, cols, xstride); }
MV(2) MV(8) MV(11) MV(12) MV(13) MV(14) MV(16) MV(17) MV(18) MV(29) MV(134) MV(135) MV(137) MV(138) MV(139) MV(140) MV(141) MV(144) MV(145) MV(146) MV(152) MV(153) MV(154) MV(155) MV(156) MV(157) MV(158) MV(19) MV(20) MV(21) MV(22) MV(23)

// F32 matvec (router, tiny projections): same shape
extern "C" __global__ void matvecf(const u8* const* p, const float* x, float* o, int rows, int cols, int xstride) {
  extern __shared__ float xs[];
  int slot = blockIdx.y; const float* xp = x + (size_t)slot * xstride;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = xp[i];
  __syncthreads();
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const float* w = (const float*)p[slot] + (size_t)r * cols; float a = 0.f;
  for (int i = lane; i < cols; i += 32) a += w[i] * xs[i];
  a = warp_sum(a);
  if (lane == 0) o[(size_t)slot * rows + r] = a;
}

// ---- elementwise / norms ----
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* o, int n, float eps) {
  __shared__ float red[32];
  float s = 0.f; for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
  s = warp_sum(s); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; v = warp_sum(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float r = rsqrtf(red[0] / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = x[i] * r * w[i];
}
extern "C" __global__ void add_rmsnorm(float* x, const float* y, const float* w, float* o, int n, float eps) {
  __shared__ float red[32];
  float s = 0.f; for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = x[i] + y[i]; x[i] = v; s += v * v; }
  s = warp_sum(s); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; v = warp_sum(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float r = rsqrtf(red[0] / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = x[i] * r * w[i];
}
extern "C" __global__ void add_inplace(float* x, const float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += y[i]; }
extern "C" __global__ void swiglu(const float* g, const float* u, float* o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = silu_f(g[i]) * u[i]; }
// Fused gate/up: gu[t][2*ffn] holds gate in [0,ffn) and up in [ffn,2*ffn) per token. o[t][ffn] = silu(gate) * up.
extern "C" __global__ void swiglu_gu(const float* gu, float* o, int ffn, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int tok = i / ffn, j = i - tok * ffn; const float* r = gu + (size_t)tok * 2 * ffn;
  o[i] = silu_f(r[j]) * r[ffn + j];
}

// Router: softmax over n experts, top-k, renormalise. One block of 256 threads, n <= 1024.
extern "C" __global__ void router_topk(const float* logits, int n, int k, int* idx_out, float* w_out) {
  // one warp; lane owns experts lane, lane+32, ...
  __shared__ float p[1024]; int lane = threadIdx.x & 31;
  if (threadIdx.x >= 32) return;
  float m = -1e30f; for (int i = lane; i < n; i += 32) { p[i] = logits[i]; m = fmaxf(m, p[i]); }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
  float s = 0.f; for (int i = lane; i < n; i += 32) { p[i] = __expf(p[i] - m); s += p[i]; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
  __syncwarp();
  float tot = 0.f; float selw[16]; int sel[16];
  for (int t = 0; t < k; t++) {
    float bv = -1.f; int bi = 0x7fffffff;
    for (int i = lane; i < n; i += 32) if (p[i] > bv) { bv = p[i]; bi = i; }
    for (int o = 16; o; o >>= 1) { float ov = __shfl_xor_sync(FULL, bv, o); int oi = __shfl_xor_sync(FULL, bi, o); if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
    sel[t] = bi; selw[t] = bv / s; tot += bv / s;
    if (lane == (bi & 31)) p[bi] = -2.f;
    __syncwarp();
  }
  if (lane == 0) for (int t = 0; t < k; t++) { idx_out[t] = sel[t]; w_out[t] = selw[t] / tot; }
}
// Device-side expert pointer build: ptrs[t], ptrs[8+t], ptrs[16+t] from the slot table; flag = 1 if any expert is not resident.
extern "C" __global__ void build_ptrs(const int* idx, const int* tab, int base, int k, unsigned long long arena, unsigned long long slot_bytes,
                                      unsigned long long off1, unsigned long long off2, unsigned long long* ptrs, int* flag) {
  __shared__ int m; int t = threadIdx.x;
  if (t == 0) m = 0; __syncthreads();
  if (t < k) {
    int sl = tab[base + idx[t]]; if (sl < 0) { atomicOr(&m, 1); sl = 0; }
    unsigned long long b = arena + (unsigned long long)sl * slot_bytes;
    ptrs[t] = b; ptrs[8 + t] = b + off1; ptrs[16 + t] = b + off2;
  }
  __syncthreads(); if (t == 0) *flag = m;
}
// moe combine: out[i] += sum_s w[s]*down[s*n+i] + sigmoid(gate)*sh[i]
extern "C" __global__ void moe_combine(float* out, const float* down, const float* w, int k, const float* sh, const float* shgate, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float a = 0.f; for (int s = 0; s < k; s++) a += w[s] * down[(size_t)s * n + i];
  a += sigm_f(shgate[0]) * sh[i];
  out[i] += a;
}


// ---- quantised KV cache: kvq 0 = f16, 1 = int8, 2 = int4; one f16 scale per 32 values (data and scales in separate arrays) ----
__device__ __forceinline__ float kv_wmax(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o)); return v; }
// Collective over the warp: element i of vector `vec` (hd values) gets value v. All 32 lanes must call it together.
__device__ __forceinline__ void kv_store(void* c, __half* sc, int kvq, size_t vec, int i, float v, int hd) {
  if (kvq == 0) { ((__half*)c)[vec * hd + i] = __float2half(v); return; }
  float a = kv_wmax(fabsf(v)), lim = kvq == 1 ? 127.f : 7.f, s = a / lim;
  __half sh = __float2half(s); float sr = __half2float(sh); float inv = sr > 0.f ? 1.f / sr : 0.f;
  int q = (int)rintf(v * inv); q = max(-(int)lim - (kvq == 2), min((int)lim, q));
  if ((i & 31) == 0) sc[vec * (hd / 32) + i / 32] = sh;
  if (kvq == 1) ((signed char*)c)[vec * hd + i] = (signed char)q;
  else {
    int nb = __shfl_xor_sync(FULL, q, 1);
    if ((i & 1) == 0) ((unsigned char*)c)[vec * (hd / 2) + i / 2] = (unsigned char)((q & 15) | ((nb & 15) << 4));
  }
}
__device__ __forceinline__ float tcq_dec(const u8* b, int i, int isv) {
  int bp = (i & 127) * 3; const u8* q = b + 2 + (bp >> 3);
  int st = (((int)q[0] | ((int)q[1] << 8)) >> (bp & 7)) & 0x1FF;
  return (isv ? TCQ_CB_V[st] : TCQ_CB_K[st]) * h2f(b);
}
__device__ __forceinline__ float kv_ld(const void* c, const __half* sc, int kvq, size_t vec, int i, int hd, int isv = 0) {
  if (kvq == 0) return __half2float(((const __half*)c)[vec * hd + i]);
  if (kvq == 3) return tcq_dec((const u8*)c + (vec * (hd >> 7) + (i >> 7)) * 52, i, isv);
  float s = __half2float(sc[vec * (hd / 32) + i / 32]);
  if (kvq == 1) return (float)((const signed char*)c)[vec * hd + i] * s;
  int b = ((const unsigned char*)c)[vec * (hd / 2) + i / 2]; int n = (i & 1) ? (b >> 4) : (b & 15);
  return (float)((n ^ 8) - 8) * s;
}
// Trellis-coded 3-bit KV (TCQ): encoder after spiritbuun's design (buun-llama-cpp, MIT). One block per 128-value group, thread = trellis state.
// src group g of [T][nkv*hd] floats -> 52-byte block at ((pos0+tok)*nkv*(hd/128)+rem).
extern "C" __global__ void __launch_bounds__(512) tcq_encode(const float* src, unsigned char* dst, int isv, int nkv, int hdb, int pos0, float alpha) {
  __shared__ float x[128]; __shared__ float cost[2][512]; __shared__ unsigned char bt[128 * 64]; __shared__ float pmin[64];
  __shared__ unsigned char outs[128]; __shared__ float red[16]; __shared__ int s_init;
  const int sid = threadIdx.x; const int per = nkv * hdb; const long g = blockIdx.x;
  const long tok = g / per, rem = g % per;
  const float* sp = src + (size_t)tok * per * 128 + rem * 128;
  unsigned char* blk = dst + ((size_t)(pos0 + tok) * per + rem) * 52;
  const float* cb = isv ? TCQ_CB_V : TCQ_CB_K;
  if (sid < 128) x[sid] = sp[sid];
  __syncthreads();
  if (sid < 128) { float v = warp_sum(x[sid] * x[sid]); if ((sid & 31) == 0) red[sid >> 5] = v; }
  __syncthreads();
  const float gn = sqrtf(red[0] + red[1] + red[2] + red[3]);
  __syncthreads();
  if (sid < 128) x[sid] = (gn > 1e-20f ? x[sid] / gn : 0.f) * TCQ_S1[sid];
  __syncthreads();
  for (int h = 1; h < 128; h <<= 1) {
    if (sid < 64) { int i = (sid / h) * 2 * h + (sid % h); float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
    __syncthreads();
  }
  if (sid < 128) x[sid] *= 0.08838834764831845f * TCQ_S2[sid];
  cost[0][sid] = 0.f;
  __syncthreads();
  const float cbs = cb[sid];
  for (int t = 0; t < 128; t++) {
    const float* rd = cost[t & 1]; float* wr = cost[(t + 1) & 1];
    if (sid < 64) {
      float best = 3.0e38f; int bp = 0;
      #pragma unroll
      for (int p = 0; p < 8; p++) { float cc = rd[(sid << 3) | p]; if (cc < best) { best = cc; bp = p; } }
      pmin[sid] = best; bt[t * 64 + sid] = (unsigned char)bp;
    }
    __syncthreads();
    float d = x[t] - cbs; wr[sid] = pmin[sid & 63] + d * d;
    __syncthreads();
  }
  if (sid == 0) {
    const float* fc = cost[0]; float best = 3.0e38f; int st = 0;
    for (int s2 = 0; s2 < 512; s2++) if (fc[s2] < best) { best = fc[s2]; st = s2; }
    for (int t = 127; t >= 0; t--) { outs[t] = (unsigned char)(st >> 6); int p = bt[t * 64 + (st & 63)]; st = ((st & 63) << 3) | p; }
    s_init = st;
  }
  __syncthreads();
  const int prefix = (s_init >> 3) & 63;
  if (sid < 128) {
    int a = sid >= 2 ? outs[sid - 2] : (sid == 0 ? (prefix & 7) : ((prefix >> 3) & 7));
    int b = sid >= 1 ? outs[sid - 1] : ((prefix >> 3) & 7);
    int stt = a | (b << 3) | ((int)outs[sid] << 6);
    float r = cb[stt]; float v = warp_sum(r * r); if ((sid & 31) == 0) red[4 + (sid >> 5)] = v;
  }
  __syncthreads();
  if (sid == 0) {
    float rn = sqrtf(red[4] + red[5] + red[6] + red[7]);
    float cn = (rn > 1e-10f ? gn / rn : gn) * alpha;
    *(__half*)blk = __float2half(cn);
    unsigned char* qs = blk + 2;
    for (int i = 0; i < 49; i++) qs[i] = 0;
    for (int k = 0; k < 6; k++) qs[k >> 3] |= ((prefix >> k) & 1) << (k & 7);
    for (int i = 0; i < 128; i++) { int bp = 6 + 3 * i; int v = outs[i]; for (int k = 0; k < 3; k++) qs[(bp + k) >> 3] |= ((v >> k) & 1) << ((bp + k) & 7); }
  }
}
// Rotate (inv=0: x*S1, FWHT, *1/sqrt128*S2) or unrotate (inv=1: x*S2, FWHT, *1/sqrt128*S1, then * sigmoid(gate)) 128-groups in place. grid = groups, block 128.
extern "C" __global__ void tcq_rot(float* x, const float* gate, int inv) {
  __shared__ float b[128]; int t = threadIdx.x; size_t o = (size_t)blockIdx.x * 128 + t;
  b[t] = x[o] * (inv ? TCQ_S2[t] : TCQ_S1[t]);
  __syncthreads();
  for (int h = 1; h < 128; h <<= 1) {
    if (t < 64) { int i = (t / h) * 2 * h + (t % h); float a = b[i], c = b[i + h]; b[i] = a + c; b[i + h] = a - c; }
    __syncthreads();
  }
  float v = b[t] * 0.08838834764831845f * (inv ? TCQ_S1[t] : TCQ_S2[t]);
  x[o] = inv ? v * sigm_f(gate[o]) : v;
}
// ---- full attention ----
// split qf [nh][2*hd] -> q (rmsnorm*w, rope) and gate; k rmsnorm+rope; write k,v into f16 cache at pos. One block per head (nh q heads then nkv k heads).
extern "C" __global__ void attn_prep(const float* qf, const float* kf, const float* vf, const float* qn, const float* kn,
    float* q, float* gate, void* kc, void* vc, int nh, int nkv, int hd, int rope_dim, float mscale, int pos, float eps, __half* ksc, __half* vsc, int kvq, const float* invf) {
  __shared__ float buf[512]; __shared__ float red[32];
  int h = blockIdx.x; bool isq = h < nh; int hh = isq ? h : h - nh;
  const float* src = isq ? qf + (size_t)hh * 2 * hd : kf + (size_t)hh * hd;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) buf[i] = src[i];
  __syncthreads();
  float s = 0.f; for (int i = threadIdx.x; i < hd; i += blockDim.x) s += buf[i] * buf[i];
  s = warp_sum(s); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; v = warp_sum(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float r = rsqrtf(red[0] / hd + eps); const float* w = isq ? qn : kn;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) buf[i] = buf[i] * r * w[i];
  __syncthreads();
  int half = rope_dim / 2;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) {
    float v = buf[i];
    if (i < half) { float th = pos * invf[i]; float sn, cs; sincosf(th, &sn, &cs); sn *= mscale; cs *= mscale; v = buf[i] * cs - buf[i + half] * sn; }
    else if (i < rope_dim) { int j = i - half; float th = pos * invf[j]; float sn, cs; sincosf(th, &sn, &cs); sn *= mscale; cs *= mscale; v = buf[j] * sn + buf[i] * cs; }
    if (isq) { q[(size_t)hh * hd + i] = v; gate[(size_t)hh * hd + i] = qf[(size_t)hh * 2 * hd + hd + i]; }
    else if (kvq == 3) { ((float*)ksc)[(size_t)hh * hd + i] = v; }
    else { size_t vec = (size_t)pos * nkv + hh; kv_store(kc, ksc, kvq, vec, i, v, hd); kv_store(vc, vsc, kvq, vec, i, vf[(size_t)hh * hd + i], hd); }
  }
}
// scores[h][t] = q_h . k_t / sqrt(hd). grid (ceil(n/8), nh); each warp one position.
extern "C" __global__ void attn_scores(const float* q, const void* kc, float* scores, int nkv, int nh, int hd, int n, int stride, const __half* ksc, int kvq) {
  int h = blockIdx.y, kvh = h / (nh / nkv); int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= n) return;
  size_t vec = (size_t)t * nkv + kvh; float a = 0.f;
  for (int i = lane; i < hd; i += 32) a += q[h * hd + i] * kv_ld(kc, ksc, kvq, vec, i, hd);
  a = warp_sum(a); if (lane == 0) scores[(size_t)h * stride + t] = a * rsqrtf((float)hd);
}
extern "C" __global__ void attn_softmax(float* scores, int n, int stride) {
  __shared__ float red[32]; float* s = scores + (size_t)blockIdx.x * stride;
  float m = -1e30f; for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, s[i]);
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -1e30f; for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); m = red[0]; __syncthreads();
  float sum = 0.f; for (int i = threadIdx.x; i < n; i += blockDim.x) { float e = __expf(s[i] - m); s[i] = e; sum += e; }
  sum = warp_sum(sum); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; v = warp_sum(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); float inv = 1.f / red[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] *= inv;
}
// o[h][d] = sigmoid(gate) * sum_t p[t] v[t][d]. grid nh, block hd.
extern "C" __global__ void attn_pv(const float* scores, const void* vc, const float* gate, float* o, int nkv, int nh, int hd, int n, int stride, const __half* vsc, int kvq) {
  int h = blockIdx.x, kvh = h / (nh / nkv), d = threadIdx.x; float a = 0.f;
  for (int t = 0; t < n; t++) a += scores[(size_t)h * stride + t] * kv_ld(vc, vsc, kvq, (size_t)t * nkv + kvh, d, hd, 1);
  o[h * hd + d] = kvq == 3 ? a : a * sigm_f(gate[h * hd + d]);
}

// ---- Gated DeltaNet ----
// conv1d (kernel K) + silu over channels; state [ch][K-1] updated in place.
extern "C" __global__ void delta_conv(const float* qkv, const float* w, float* state, float* y, int ch, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ch) return;
  float* st = state + (size_t)c * (K - 1); float a = 0.f;
  for (int k = 0; k < K - 1; k++) a += w[c * K + k] * st[k];
  a += w[c * K + K - 1] * qkv[c];
  for (int k = 0; k < K - 2; k++) st[k] = st[k + 1];
  st[K - 2] = qkv[c];
  y[c] = silu_f(a);
}
// One block per v-head (128 threads, thread j owns column j of S[i][j]). y layout: q[ng*dk] k[ng*dk] v[nv*dk].
extern "C" __global__ void delta_step(const float* y, const float* alpha, const float* betaraw, const float* a_log, const float* dtb,
    const float* z, const float* nw, float* S, float* o, int ng, int nv, int dk, float eps) {
  __shared__ float qs[128], ks[128], red[8];
  int vh = blockIdx.x, j = threadIdx.x, kh = vh % ng;
  const float* q = y + kh * dk; const float* k = y + ng * dk + kh * dk; const float* v = y + 2 * ng * dk + vh * dk;
  qs[j] = q[j]; ks[j] = k[j]; __syncthreads();
  float sq = qs[j] * qs[j], sk = ks[j] * ks[j];
  sq = warp_sum(sq); sk = warp_sum(sk);
  __shared__ float r2[8];
  if ((j & 31) == 0) { red[j >> 5] = sq; r2[j >> 5] = sk; } __syncthreads();
  float nq = fmaxf(sqrtf(red[0] + red[1] + red[2] + red[3]), 1e-6f), nk = fmaxf(sqrtf(r2[0] + r2[1] + r2[2] + r2[3]), 1e-6f);
  __syncthreads();
  qs[j] = qs[j] / nq * rsqrtf((float)dk); ks[j] = ks[j] / nk; __syncthreads();
  float a = alpha[vh] + dtb[vh]; float sp = a > 20.f ? a : logf(1.f + __expf(a));
  float eg = __expf(a_log[vh] * sp), beta = sigm_f(betaraw[vh]);
  float* Sh = S + (size_t)vh * dk * dk;
  float col[128];
  float kmem = 0.f;
  #pragma unroll
  for (int i = 0; i < 128; i++) { float s = Sh[i * dk + j] * eg; col[i] = s; kmem += s * ks[i]; }
  float d = (v[j] - kmem) * beta, oo = 0.f;
  #pragma unroll
  for (int i = 0; i < 128; i++) { float s = col[i] + ks[i] * d; Sh[i * dk + j] = s; oo += s * qs[i]; }
  // gated rmsnorm over the head's dk outputs
  float s2 = warp_sum(oo * oo); if ((j & 31) == 0) red[j >> 5] = s2; __syncthreads();
  float rr = rsqrtf((red[0] + red[1] + red[2] + red[3]) / dk + eps);
  o[vh * dk + j] = oo * rr * nw[j] * silu_f(z[vh * dk + j]);
}

// =================== batched (prefill) kernels ===================
// Grouped matmul: slot z has weights ptrs[z] [rows][cols] and tokens off[z]..off[z+1] (sorted-pair index i). Input row of pair i is
// in_idx ? in_idx[i] : i; output row is out_idx ? out_idx[i] : i. Warp-per-row, 8 tokens at once.
template <int TY>
__device__ void mm_impl(const u8* const* ptrs, const int* off, const int* in_idx, const int* out_idx, const signed char* xq, const float* xd, const int* xs,
                        float* out, int rows, int cols) {
  extern __shared__ int sm[];
  int units = cols >> 5, z = blockIdx.z;
  int n = off[z + 1] - off[z], t0 = blockIdx.y * 8;
  if (t0 >= n) return;
  int nt = min(8, n - t0), base = off[z] + t0;
  int* sxq = sm; float* sxd = (float*)(sm + 8 * units * 9); int* sxs = (int*)(sxd + 8 * units);
  for (int j = 0; j < nt; j++) {
    int xr = in_idx ? in_idx[base + j] : base + j;
    const int* gq = (const int*)(xq + (size_t)xr * cols);
    for (int i = threadIdx.x; i < units * 8; i += 256) sxq[(j * units + (i >> 3)) * 9 + (i & 7)] = gq[i];
    for (int i = threadIdx.x; i < units; i += 256) { sxd[j * units + i] = xd[(size_t)xr * units + i]; sxs[j * units + i] = xs[(size_t)xr * units + i]; }
  }
  __syncthreads();
  const int UB = Q<TY>::UB, BY = Q<TY>::BYTES;
  int lpr = units < 32 ? units : 32, rpw = 32 / lpr;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = (blockIdx.x * 8 + warp) * rpw + lane / lpr, li = lane % lpr;
  bool ok = r < rows;
  const u8* row = ptrs[z] + (size_t)(ok ? r : 0) * ((cols / (UB * 32)) * BY + Q<TY>::HDR);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ok) for (int u = li; u < units; u += lpr) {
    const u8* wb = row + Q<TY>::HDR + (u / UB) * BY; int uj = u % UB;
    #pragma unroll
    int w[8]; float s0, s1, mn; uloadr<TY>(wb, uj, w, s0, s1, mn, row);
    #pragma unroll
    for (int j = 0; j < 8; j++) if (j < nt) {
      const int* xq = sxq + (j * units + u) * 9; int a0 = 0, a1 = 0;
      #pragma unroll
      for (int i = 0; i < 4; i++) { a0 = DP(w[i], xq[i], a0); a1 = DP(w[4 + i], xq[4 + i], a1); }
      acc[j] += sxd[j * units + u] * (s0 * (float)a0 + s1 * (float)a1 - mn * (float)sxs[j * units + u]);
    }
  }
  #pragma unroll
  for (int j = 0; j < 8; j++) for (int o = lpr >> 1; o; o >>= 1) acc[j] += __shfl_xor_sync(FULL, acc[j], o);
  if (ok && li == 0) for (int j = 0; j < nt; j++) { int orow = out_idx ? out_idx[base + j] : base + j; out[(size_t)orow * rows + r] = acc[j]; }
}
#define MM(TY) extern "C" __global__ void mm_##TY(const u8* const* p, const int* off, const int* ii, const int* oi, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols) { mm_impl<TY>(p, off, ii, oi, xq, xd, xs, o, rows, cols); }
MM(2) MM(8) MM(11) MM(12) MM(13) MM(14) MM(16) MM(17) MM(18) MM(19) MM(20) MM(21) MM(22) MM(23) MM(29) MM(134) MM(135) MM(137) MM(138) MM(139) MM(140) MM(141) MM(144) MM(145) MM(146) MM(152) MM(153) MM(154) MM(155) MM(156) MM(157) MM(158)

// F32 matmul O[t][r] = W[r].X[t] for T tokens. grid (ceil(rows/8), ceil(T/32)), 256 threads.
extern "C" __global__ void mmf(const float* W, const float* X, float* O, int rows, int cols, int T) {
  int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  int tb = blockIdx.y * 32, te = min(T, tb + 32);
  const float* w = W + (size_t)r * cols;
  for (int t = tb; t < te; t++) {
    const float* x = X + (size_t)t * cols; float a = 0.f;
    for (int i = lane; i < cols; i += 32) a += w[i] * x[i];
    a = warp_sum(a); if (lane == 0) O[(size_t)t * rows + r] = a;
  }
}
extern "C" __global__ void rmsnorm_b(const float* x, const float* w, float* o, int n, float eps) {
  __shared__ float red[32]; x += (size_t)blockIdx.x * n; o += (size_t)blockIdx.x * n;
  float s = 0.f; for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
  s = warp_sum(s); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; v = warp_sum(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float r = rsqrtf(red[0] / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = x[i] * r * w[i];
}
extern "C" __global__ void router_topk_b(const float* logits, int n, int k, int* idx_out, float* w_out) {
  __shared__ float p[1024]; int t = blockIdx.x;
  logits += (size_t)t * n; idx_out += t * k; w_out += t * k;
  for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] = logits[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -1e30f; for (int i = 0; i < n; i++) m = fmaxf(m, p[i]);
    float s = 0.f; for (int i = 0; i < n; i++) { p[i] = __expf(p[i] - m); s += p[i]; }
    float tot = 0.f; int sel[16]; float selw[16];
    for (int q = 0; q < k; q++) { int b = 0; float bv = -1.f; for (int i = 0; i < n; i++) if (p[i] > bv) { bv = p[i]; b = i; } sel[q] = b; selw[q] = bv / s; tot += bv / s; p[b] = -2.f; }
    for (int q = 0; q < k; q++) { idx_out[q] = sel[q]; w_out[q] = selw[q] / tot; }
  }
}
// out[t] += sum_s w[t*k+s]*down[(t*k+s)*n+i] + sigmoid(shgate[t])*sh[t*n+i]. grid (ceil(n/256), T)
extern "C" __global__ void moe_combine_b(float* out, const float* down, const float* w, int k, const float* sh, const float* shgate, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y; if (i >= n) return;
  float a = 0.f; for (int s = 0; s < k; s++) a += w[t * k + s] * down[((size_t)t * k + s) * n + i];
  a += sigm_f(shgate[t]) * sh[(size_t)t * n + i];
  out[(size_t)t * n + i] += a;
}
// ---- batched full attention ----
// grid (nh + nkv, T). qf [T][nh*2hd], kf/vf [T][nkv*hd]; q/gate out [T][nh*hd]. position = pos0 + token.
extern "C" __global__ void attn_prep_b(const float* qf, const float* kf, const float* vf, const float* qn, const float* kn,
    float* q, float* gate, void* kc, void* vc, int nh, int nkv, int hd, int rope_dim, float mscale, int pos0, float eps, __half* ksc, __half* vsc, int kvq, const float* invf) {
  __shared__ float buf[512]; __shared__ float red[32];
  int t = blockIdx.y, pos = pos0 + t;
  qf += (size_t)t * nh * 2 * hd; kf += (size_t)t * nkv * hd; vf += (size_t)t * nkv * hd; q += (size_t)t * nh * hd; gate += (size_t)t * nh * hd;
  int h = blockIdx.x; bool isq = h < nh; int hh = isq ? h : h - nh;
  const float* src = isq ? qf + (size_t)hh * 2 * hd : kf + (size_t)hh * hd;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) buf[i] = src[i];
  __syncthreads();
  float s = 0.f; for (int i = threadIdx.x; i < hd; i += blockDim.x) s += buf[i] * buf[i];
  s = warp_sum(s); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; v = warp_sum(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float r = rsqrtf(red[0] / hd + eps); const float* w = isq ? qn : kn;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) buf[i] = buf[i] * r * w[i];
  __syncthreads();
  int half = rope_dim / 2;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) {
    float v = buf[i];
    if (i < half) { float th = pos * invf[i]; float sn, cs; sincosf(th, &sn, &cs); sn *= mscale; cs *= mscale; v = buf[i] * cs - buf[i + half] * sn; }
    else if (i < rope_dim) { int j = i - half; float th = pos * invf[j]; float sn, cs; sincosf(th, &sn, &cs); sn *= mscale; cs *= mscale; v = buf[j] * sn + buf[i] * cs; }
    if (isq) { q[(size_t)hh * hd + i] = v; gate[(size_t)hh * hd + i] = qf[(size_t)hh * 2 * hd + hd + i]; }
    else if (kvq == 3) { ((float*)ksc)[((size_t)t * nkv + hh) * hd + i] = v; }
    else { size_t vec = (size_t)pos * nkv + hh; kv_store(kc, ksc, kvq, vec, i, v, hd); kv_store(vc, vsc, kvq, vec, i, vf[(size_t)hh * hd + i], hd); }
  }
}
// Query group: tokens tb..tb+G. scores[g][h][pos] for pos <= pos0+tb+g. grid (ceil(nmax/8), nh, G).
extern "C" __global__ void attn_scores_b(const float* q, const void* kc, float* scores, int nkv, int nh, int hd, int pos_first, int stride, int tb, const __half* ksc, int kvq) {
  int g = blockIdx.z, kvh = blockIdx.y, hpk = nh / nkv; int p = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  int n = pos_first + g + 1;
  if (p >= n) return;
  size_t kvec = (size_t)p * nkv + kvh; float kr[8];
  for (int j = 0; j < 8; j++) { int i = lane + 32 * j; kr[j] = i < hd ? kv_ld(kc, ksc, kvq, kvec, i, hd) : 0.f; }
  float sc = rsqrtf((float)hd);
  for (int hh = 0; hh < hpk; hh++) {
    int h = kvh * hpk + hh; const float* qq = q + ((size_t)(tb + g) * nh + h) * hd; float a = 0.f;
    for (int j = 0; j < 8; j++) { int i = lane + 32 * j; if (i < hd) a += qq[i] * kr[j]; }
    a = warp_sum(a); if (lane == 0) scores[((size_t)g * nh + h) * stride + p] = a * sc;
  }
}
extern "C" __global__ void attn_softmax_b(float* scores, int nh, int stride, int pos_first) {
  __shared__ float red[32]; int n = pos_first + blockIdx.y + 1;
  float* s = scores + ((size_t)blockIdx.y * nh + blockIdx.x) * stride;
  float m = -1e30f; for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, s[i]);
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -1e30f; for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); m = red[0]; __syncthreads();
  float sum = 0.f; for (int i = threadIdx.x; i < n; i += blockDim.x) { float e = __expf(s[i] - m); s[i] = e; sum += e; }
  sum = warp_sum(sum); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum; __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; v = warp_sum(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); float inv = 1.f / red[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] *= inv;
}
// grid (nkv, G), block hd.
extern "C" __global__ void attn_pv_b(const float* scores, const void* vc, const float* gate, float* o, int nkv, int nh, int hd, int pos_first, int stride, int tb, const __half* vsc, int kvq) {
  int kvh = blockIdx.x, g = blockIdx.y, hpk = nh / nkv, d = threadIdx.x, n = pos_first + g + 1; float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float* sc = scores + ((size_t)g * nh + kvh * hpk) * stride;
  for (int t = 0; t < n; t++) {
    float v = kv_ld(vc, vsc, kvq, (size_t)t * nkv + kvh, d, hd, 1);
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) if (hh < hpk) a[hh] += sc[(size_t)hh * stride + t] * v;
  }
  for (int hh = 0; hh < hpk; hh++) { size_t oi = ((size_t)(tb + g) * nh + kvh * hpk + hh) * hd + d; o[oi] = kvq == 3 ? a[hh] : a[hh] * sigm_f(gate[oi]); }
}
// Decode PV, split over positions. grid (nkv, nsplit), block hd. part[(split*nh + h)*hd + d] = sum over the split's positions.
extern "C" __global__ void attn_pv_split(const float* scores, const void* vc, float* part, int nkv, int nh, int hd, int n, int stride, const __half* vsc, int kvq, int chunk) {
  int kvh = blockIdx.x, sp = blockIdx.y, hpk = nh / nkv, d = threadIdx.x; float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int t0 = sp * chunk, t1 = min(n, t0 + chunk);
  const float* sc = scores + (size_t)(kvh * hpk) * stride;
  for (int t = t0; t < t1; t++) {
    float v = kv_ld(vc, vsc, kvq, (size_t)t * nkv + kvh, d, hd, 1);
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) if (hh < hpk) a[hh] += sc[(size_t)hh * stride + t] * v;
  }
  for (int hh = 0; hh < hpk; hh++) part[((size_t)sp * nh + kvh * hpk + hh) * hd + d] = a[hh];
}
// grid nh, block hd: o[h][d] = sigmoid(gate) * sum_split part.
extern "C" __global__ void attn_pv_reduce(const float* part, const float* gate, float* o, int nh, int hd, int nsplit, int kvq) {
  int h = blockIdx.x, d = threadIdx.x; float a = 0.f;
  for (int s = 0; s < nsplit; s++) a += part[((size_t)s * nh + h) * hd + d];
  o[h * hd + d] = kvq == 3 ? a : a * sigm_f(gate[h * hd + d]);
}
// ---- batched DeltaNet ----
// conv over T tokens sequentially per channel. qkv [T][ch] -> y [T][ch].
extern "C" __global__ void delta_conv_b(const float* qkv, const float* w, float* state, float* y, int ch, int K, int T) {
  int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ch) return;
  float st[4]; for (int k = 0; k < K - 1; k++) st[k] = state[(size_t)c * (K - 1) + k];
  for (int t = 0; t < T; t++) {
    float x = qkv[(size_t)t * ch + c], a = 0.f;
    for (int k = 0; k < K - 1; k++) a += w[c * K + k] * st[k];
    a += w[c * K + K - 1] * x;
    for (int k = 0; k < K - 2; k++) st[k] = st[k + 1];
    st[K - 2] = x;
    y[(size_t)t * ch + c] = silu_f(a);
  }
  for (int k = 0; k < K - 1; k++) state[(size_t)c * (K - 1) + k] = st[k];
}
// One block per v-head, thread j owns column j of S held in registers (dk == 128). Sequential over T tokens.
extern "C" __global__ void delta_step_b(const float* y, const float* alpha, const float* betaraw, const float* a_log, const float* dtb,
    const float* z, const float* nw, float* S, float* o, int ng, int nv, int dk, float eps, int T, int ych) {
  __shared__ float qs[128], ks[128], red[8], r2[8];
  int vh = blockIdx.x, j = threadIdx.x, kh = vh % ng;
  float* Sh = S + (size_t)vh * dk * dk;
  float s[128];
  #pragma unroll
  for (int i = 0; i < 128; i++) s[i] = Sh[i * 128 + j];
  for (int t = 0; t < T; t++) {
    const float* yt = y + (size_t)t * ych;
    const float* q = yt + kh * dk; const float* k = yt + ng * dk + kh * dk; const float* v = yt + 2 * ng * dk + vh * dk;
    qs[j] = q[j]; ks[j] = k[j]; __syncthreads();
    float sq = warp_sum(qs[j] * qs[j]), sk = warp_sum(ks[j] * ks[j]);
    if ((j & 31) == 0) { red[j >> 5] = sq; r2[j >> 5] = sk; } __syncthreads();
    float nq = fmaxf(sqrtf(red[0] + red[1] + red[2] + red[3]), 1e-6f), nk = fmaxf(sqrtf(r2[0] + r2[1] + r2[2] + r2[3]), 1e-6f);
    __syncthreads();
    qs[j] = qs[j] / nq * rsqrtf((float)dk); ks[j] = ks[j] / nk; __syncthreads();
    float a = alpha[(size_t)t * nv + vh] + dtb[vh]; float sp = a > 20.f ? a : logf(1.f + __expf(a));
    float eg = __expf(a_log[vh] * sp), beta = sigm_f(betaraw[(size_t)t * nv + vh]);
    float kmem = 0.f;
    #pragma unroll
    for (int i = 0; i < 128; i++) { s[i] *= eg; kmem += s[i] * ks[i]; }
    float d = (v[j] - kmem) * beta, oo = 0.f;
    #pragma unroll
    for (int i = 0; i < 128; i++) { s[i] += ks[i] * d; oo += s[i] * qs[i]; }
    float s2 = warp_sum(oo * oo); if ((j & 31) == 0) red[j >> 5] = s2; __syncthreads();
    float rr = rsqrtf((red[0] + red[1] + red[2] + red[3]) / dk + eps);
    o[((size_t)t * nv + vh) * dk + j] = oo * rr * nw[j] * silu_f(z[((size_t)t * nv + vh) * dk + j]);
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 128; i++) Sh[i * 128 + j] = s[i];
}


// ---- Clover block-diffusion preview (read-only recurrent state) ----
extern "C" __global__ void delta_conv_pv(const float* qkv, const float* w, const float* wr, const float* state, float* y, int ch, int K, int T) {
  int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ch) return;
  for (int t = 0; t < T; t++) {
    float a = 0.f;
    for (int k = 0; k < K; k++) { int p = t - (K - 1) + k; float x = p < 0 ? state[(size_t)c * (K - 1) + p + K - 1] : qkv[(size_t)p * ch + c]; a += w[c * K + k] * x; }
    for (int d = 1; d < K; d++) if (t + d < T) a += qkv[(size_t)(t + d) * ch + c] * wr[c * (K - 1) + d - 1];
    y[(size_t)t * ch + c] = silu_f(a);
  }
}
extern "C" __global__ void delta_pv(const float* y, const float* alpha, const float* a_log, const float* dtb, const float* z, const float* nw,
    const float* S, const float* lscale, const float* hbias, const float* hc, float* o, int ng, int nv, int dk, float eps, int T, int ych) {
  __shared__ float qn[32][128], kn[32][128];
  __shared__ float lg[32][33], red[4], egs[32];
  int vh = blockIdx.x, j = threadIdx.x, kh = vh % ng;
  for (int t = 0; t < T; t++) {
    const float* yt = y + (size_t)t * ych;
    float q = yt[kh * dk + j], k = yt[ng * dk + kh * dk + j];
    float sq = warp_sum(q * q); if ((j & 31) == 0) red[j >> 5] = sq; __syncthreads();
    float rq = rsqrtf(red[0] + red[1] + red[2] + red[3] + 1e-6f); __syncthreads();
    float sk = warp_sum(k * k); if ((j & 31) == 0) red[j >> 5] = sk; __syncthreads();
    float rk = rsqrtf(red[0] + red[1] + red[2] + red[3] + 1e-6f); __syncthreads();
    qn[t][j] = q * rq; kn[t][j] = k * rk;
    if (j == 0) { float a = alpha[(size_t)t * nv + vh] + dtb[vh]; float sp = a > 20.f ? a : logf(1.f + __expf(a)); egs[t] = __expf(a_log[vh] * sp); }
  }
  __syncthreads();
  const float* Sh = S + (size_t)vh * dk * dk;
  float h[32];
  #pragma unroll
  for (int t = 0; t < 32; t++) h[t] = 0.f;
  for (int i = 0; i < dk; i++) { float sv = Sh[i * 128 + j];
    #pragma unroll
    for (int t = 0; t < 32; t++) h[t] += qn[t < T ? t : 0][i] * sv; }
  float sc = rsqrtf((float)dk), ls = lscale[vh];
  for (int p = j; p < T * T; p += 128) { int t = p / T, u = p % T; float a = 0.f; for (int i = 0; i < dk; i++) a += qn[t][i] * kn[u][i]; lg[t][u] = a * ls; }
  __syncthreads();
  if (j < T) {
    lg[j][T] = hbias[vh] + hc[vh] * logf((float)T / 32.f);
    float m = -1e30f; for (int u = 0; u <= T; u++) m = fmaxf(m, lg[j][u]);
    float s = 0.f; for (int u = 0; u <= T; u++) { lg[j][u] = __expf(lg[j][u] - m); s += lg[j][u]; }
    for (int u = 0; u <= T; u++) lg[j][u] /= s;
  }
  __syncthreads();
  for (int t = 0; t < T; t++) {
    float oo = lg[t][T] * h[t] * sc * egs[t];
    for (int u = 0; u < T; u++) oo += lg[t][u] * y[(size_t)u * ych + 2 * ng * dk + vh * dk + j];
    float s2 = warp_sum(oo * oo); if ((j & 31) == 0) red[j >> 5] = s2; __syncthreads();
    float rr = rsqrtf((red[0] + red[1] + red[2] + red[3]) / dk + eps);
    o[((size_t)t * nv + vh) * dk + j] = oo * rr * nw[j] * silu_f(z[((size_t)t * nv + vh) * dk + j]);
    __syncthreads();
  }
}

// =================== tensor-core (int8 mma) grouped GEMM for prefill ===================
// Tile: 64 weight rows x 64 tokens, 4 units (128 columns) per step, 8 warps (4 along rows x 2 along tokens, each 16 x 32).
// Weights are unpacked to int8 in shared memory; per-unit int32 products are scaled in f32.
__device__ __forceinline__ void mma32(int* c, int a0, int a1, int a2, int a3, int b0, int b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16(int* c, int a0, int a1, int b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
#ifndef NODEQ
#define NODEQ 0
#endif
template <int TY, bool HALF>
__device__ void mmq_impl(const u8* const* ptrs, const int* off, const int* in_idx, const int* out_idx, const signed char* xq, const float* xd, const int* xs,
                         float* out, int rows, int cols) {
  __shared__ int As[4][64][9], Bs[4][64][9];
  __shared__ float Sa0[4][64], Sa1[4][64], Sam[4][64], Sbd[4][64], Sbs[4][64];
  int z = blockIdx.z, n = off[z + 1] - off[z], t0 = blockIdx.y * 64;
  if (t0 >= n) return;
  int r0 = blockIdx.x * 64, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  int units = cols >> 5;
  const int UB = Q<TY>::UB, BY = Q<TY>::BYTES;
  // loader roles
  int la_row = tid >> 2, la_u = tid & 3;                 // weight (row, unit-in-step)
  int lb_tok = tid >> 2, lb_u = tid & 3;                 // activation (token, unit-in-step)
  int wr = r0 + la_row; bool wok = wr < rows;
  const u8* wrow = ptrs[z] + (size_t)(wok ? wr : 0) * ((cols / (UB * 32)) * BY + Q<TY>::HDR);
  int tk = t0 + lb_tok; bool tok_ok = tk < n;
  int xr = tok_ok ? (in_idx ? in_idx[off[z] + tk] : off[z] + tk) : 0;
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  int wm = (warp & 3) * 16, wn = (warp >> 2) * 32;
  // Register prefetch: step u0+4's global loads are issued before this step's mma, so memory latency overlaps tensor-core work.
  int w[8]; float s0 = 0.f, s1 = 0.f, mn = 0.f; int4 q0 = make_int4(0, 0, 0, 0), q1 = q0; float bd = 0.f, bs = 0.f;
  auto fetch = [&](int u0) {
    int u = u0 + la_u;
    if (wok && !NODEQ) { uloadr<TY>(wrow + Q<TY>::HDR + (u / UB) * BY, u % UB, w, s0, s1, mn, wrow); }
    else { for (int i = 0; i < 8; i++) w[i] = 0; s0 = s1 = mn = 0.f; }
    int ub = u0 + lb_u;
    if (tok_ok) {
      const int4* gq = (const int4*)(xq + (size_t)xr * cols + ub * 32);
      q0 = gq[0]; q1 = gq[1];
      bd = xd[(size_t)xr * units + ub]; bs = (float)xs[(size_t)xr * units + ub];
    } else { q0 = make_int4(0, 0, 0, 0); q1 = q0; bd = 0.f; bs = 0.f; }
  };
  fetch(0);
  for (int u0 = 0; u0 < units; u0 += 4) {
    #pragma unroll
    for (int i = 0; i < 8; i++) As[la_u][la_row][i] = w[i];
    Sa0[la_u][la_row] = s0; Sa1[la_u][la_row] = s1; Sam[la_u][la_row] = mn;
    Bs[lb_u][lb_tok][0] = q0.x; Bs[lb_u][lb_tok][1] = q0.y; Bs[lb_u][lb_tok][2] = q0.z; Bs[lb_u][lb_tok][3] = q0.w;
    Bs[lb_u][lb_tok][4] = q1.x; Bs[lb_u][lb_tok][5] = q1.y; Bs[lb_u][lb_tok][6] = q1.z; Bs[lb_u][lb_tok][7] = q1.w;
    Sbd[lb_u][lb_tok] = bd; Sbs[lb_u][lb_tok] = bs;
    __syncthreads();
    if (u0 + 4 < units) fetch(u0 + 4);
    #pragma unroll
    for (int uu = 0; uu < 4; uu++) {
      int ra = wm + g, rb = wm + g + 8;
      int a0 = As[uu][ra][t], a1 = As[uu][rb][t], a2 = As[uu][ra][t + 4], a3 = As[uu][rb][t + 4];
      float sA0 = Sa0[uu][ra], sA1 = Sa0[uu][rb], mA0 = Sam[uu][ra], mA1 = Sam[uu][rb];
      float sB0 = Sa1[uu][ra], sB1 = Sa1[uu][rb];
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        int tb = wn + nt * 8 + g;
        int b0 = Bs[uu][tb][t], b1 = Bs[uu][tb][t + 4];
        int cc[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0};
        float d0 = Sbd[uu][wn + nt * 8 + 2 * t], d1 = Sbd[uu][wn + nt * 8 + 2 * t + 1];
        float x0 = Sbs[uu][wn + nt * 8 + 2 * t], x1 = Sbs[uu][wn + nt * 8 + 2 * t + 1];
        if (HALF) {
          mma16(cc, a0, a1, b0); mma16(c2, a2, a3, b1);
          acc[nt][0] += d0 * (sA0 * cc[0] + sB0 * c2[0]); acc[nt][1] += d1 * (sA0 * cc[1] + sB0 * c2[1]);
          acc[nt][2] += d0 * (sA1 * cc[2] + sB1 * c2[2]); acc[nt][3] += d1 * (sA1 * cc[3] + sB1 * c2[3]);
        } else {
          mma32(cc, a0, a1, a2, a3, b0, b1);
          acc[nt][0] += d0 * (sA0 * cc[0] - mA0 * x0); acc[nt][1] += d1 * (sA0 * cc[1] - mA0 * x1);
          acc[nt][2] += d0 * (sA1 * cc[2] - mA1 * x0); acc[nt][3] += d1 * (sA1 * cc[3] - mA1 * x1);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int nt = 0; nt < 4; nt++)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int r = r0 + wm + g + ((i >> 1) ? 8 : 0), tkn = t0 + wn + nt * 8 + 2 * t + (i & 1);
      if (r < rows && tkn < n) { int ob = off[z] + tkn; int orow = out_idx ? out_idx[ob] : ob; out[(size_t)orow * rows + r] = acc[nt][i]; }
    }
}

template <int TY, bool HALF>
__device__ void mmq128_impl(const u8* const* ptrs, const int* off, const int* in_idx, const int* out_idx, const signed char* xq, const float* xd, const int* xs,
                         float* out, int rows, int cols) {
  __shared__ int As[4][64][9], Bs[4][128][9];
  __shared__ float Sa0[4][64], Sa1[4][64], Sam[4][64], Sbd[4][128], Sbs[4][128];
  int z = blockIdx.z, n = off[z + 1] - off[z], t0 = blockIdx.y * 128;
  if (t0 >= n) return;
  int r0 = blockIdx.x * 64, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  int units = cols >> 5;
  const int UB = Q<TY>::UB, BY = Q<TY>::BYTES;
  // loader roles
  int la_row = tid >> 2, la_u = tid & 3;                 // weight (row, unit-in-step)
  int lb_u = tid & 3; int lb_tok[2] = {tid >> 2, (tid >> 2) + 64};   // activations: two tokens per thread (128 x 4 per step)
  int wr = r0 + la_row; bool wok = wr < rows;
  const u8* wrow = ptrs[z] + (size_t)(wok ? wr : 0) * ((cols / (UB * 32)) * BY + Q<TY>::HDR);
  int xr[2]; bool tok_ok[2];
  for (int j = 0; j < 2; j++) { int tk = t0 + lb_tok[j]; tok_ok[j] = tk < n; xr[j] = tok_ok[j] ? (in_idx ? in_idx[off[z] + tk] : off[z] + tk) : 0; }
  float acc[8][4];
  #pragma unroll
  for (int i = 0; i < 8; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  int wm = (warp & 3) * 16, wn = (warp >> 2) * 64;
  // Register prefetch: step u0+4's global loads are issued before this step's mma, so memory latency overlaps tensor-core work.
  int w[8]; float s0 = 0.f, s1 = 0.f, mn = 0.f; int4 q0[2], q1[2]; float bd[2], bs[2];
  for (int j = 0; j < 2; j++) { q0[j] = make_int4(0, 0, 0, 0); q1[j] = q0[j]; bd[j] = bs[j] = 0.f; }
  auto fetch = [&](int u0) {
    int u = u0 + la_u;
    if (wok && !NODEQ) { uloadr<TY>(wrow + Q<TY>::HDR + (u / UB) * BY, u % UB, w, s0, s1, mn, wrow); }
    else { for (int i = 0; i < 8; i++) w[i] = 0; s0 = s1 = mn = 0.f; }
    for (int j = 0; j < 2; j++) {
      int ub = u0 + lb_u;
      if (tok_ok[j]) {
        const int4* gq = (const int4*)(xq + (size_t)xr[j] * cols + ub * 32);
        q0[j] = gq[0]; q1[j] = gq[1];
        bd[j] = xd[(size_t)xr[j] * units + ub]; bs[j] = (float)xs[(size_t)xr[j] * units + ub];
      } else { q0[j] = make_int4(0, 0, 0, 0); q1[j] = q0[j]; bd[j] = 0.f; bs[j] = 0.f; }
    }
  };
  fetch(0);
  for (int u0 = 0; u0 < units; u0 += 4) {
    #pragma unroll
    for (int i = 0; i < 8; i++) As[la_u][la_row][i] = w[i];
    Sa0[la_u][la_row] = s0; Sa1[la_u][la_row] = s1; Sam[la_u][la_row] = mn;
    for (int j = 0; j < 2; j++) {
      Bs[lb_u][lb_tok[j]][0] = q0[j].x; Bs[lb_u][lb_tok[j]][1] = q0[j].y; Bs[lb_u][lb_tok[j]][2] = q0[j].z; Bs[lb_u][lb_tok[j]][3] = q0[j].w;
      Bs[lb_u][lb_tok[j]][4] = q1[j].x; Bs[lb_u][lb_tok[j]][5] = q1[j].y; Bs[lb_u][lb_tok[j]][6] = q1[j].z; Bs[lb_u][lb_tok[j]][7] = q1[j].w;
      Sbd[lb_u][lb_tok[j]] = bd[j]; Sbs[lb_u][lb_tok[j]] = bs[j];
    }
    __syncthreads();
    if (u0 + 4 < units) fetch(u0 + 4);
    #pragma unroll
    for (int uu = 0; uu < 4; uu++) {
      int ra = wm + g, rb = wm + g + 8;
      int a0 = As[uu][ra][t], a1 = As[uu][rb][t], a2 = As[uu][ra][t + 4], a3 = As[uu][rb][t + 4];
      float sA0 = Sa0[uu][ra], sA1 = Sa0[uu][rb], mA0 = Sam[uu][ra], mA1 = Sam[uu][rb];
      float sB0 = Sa1[uu][ra], sB1 = Sa1[uu][rb];
      #pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        int tb = wn + nt * 8 + g;
        int b0 = Bs[uu][tb][t], b1 = Bs[uu][tb][t + 4];
        int cc[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0};
        float d0 = Sbd[uu][wn + nt * 8 + 2 * t], d1 = Sbd[uu][wn + nt * 8 + 2 * t + 1];
        float x0 = Sbs[uu][wn + nt * 8 + 2 * t], x1 = Sbs[uu][wn + nt * 8 + 2 * t + 1];
        if (HALF) {
          mma16(cc, a0, a1, b0); mma16(c2, a2, a3, b1);
          acc[nt][0] += d0 * (sA0 * cc[0] + sB0 * c2[0]); acc[nt][1] += d1 * (sA0 * cc[1] + sB0 * c2[1]);
          acc[nt][2] += d0 * (sA1 * cc[2] + sB1 * c2[2]); acc[nt][3] += d1 * (sA1 * cc[3] + sB1 * c2[3]);
        } else {
          mma32(cc, a0, a1, a2, a3, b0, b1);
          acc[nt][0] += d0 * (sA0 * cc[0] - mA0 * x0); acc[nt][1] += d1 * (sA0 * cc[1] - mA0 * x1);
          acc[nt][2] += d0 * (sA1 * cc[2] - mA1 * x0); acc[nt][3] += d1 * (sA1 * cc[3] - mA1 * x1);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int nt = 0; nt < 8; nt++)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int r = r0 + wm + g + ((i >> 1) ? 8 : 0), tkn = t0 + wn + nt * 8 + 2 * t + (i & 1);
      if (r < rows && tkn < n) { int ob = off[z] + tkn; int orow = out_idx ? out_idx[ob] : ob; out[(size_t)orow * rows + r] = acc[nt][i]; }
    }
}

// Zero a float buffer (used before split-K atomic accumulation).
extern "C" __global__ void zero_f(float* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = 0.f; }

// 128-row x 128-token block tile (llama.cpp-style reuse): 8 warps, each 16 rows x 128 tokens. Selected by LE_MMQR=1.
template <int TY, bool HALF>
__device__ void mmq128r_impl(const u8* const* ptrs, const int* off, const int* in_idx, const int* out_idx, const signed char* xq, const float* xd, const int* xs,
                          float* out, int rows, int cols) {
  __shared__ int As[4][128][9], Bs[4][128][9];
  __shared__ float Sa0[4][128], Sa1[4][128], Sam[4][128], Sbd[4][128], Sbs[4][128];
  int z = blockIdx.z, n = off[z + 1] - off[z], t0 = blockIdx.y * 128;
  if (t0 >= n) return;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  int units = cols >> 5;
  // Split-K over units (opt-in, LE_MMQSK=S): grid.x = row_blocks * S, block index = row_block * S + split. Requires units % 4 == 0.
  int rb_count = (rows + 127) / 128, S = gridDim.x / rb_count; int split = blockIdx.x % S; int r0 = (blockIdx.x / S) * 128;
  int steps = units / 4, s_lo = split * steps / S, s_hi = (split + 1) * steps / S;
  const int UB = Q<TY>::UB, BY = Q<TY>::BYTES;
  int la_row = tid >> 2, la_u = tid & 3;                     // weight rows la_row and la_row+64, unit la_u
  int lb_u = tid & 3; int lb_tok[2] = {tid >> 2, (tid >> 2) + 64};
  int wr[2]; bool wok[2]; const u8* wrow[2];
  #pragma unroll
  for (int k = 0; k < 2; k++) { wr[k] = r0 + la_row + 64 * k; wok[k] = wr[k] < rows; wrow[k] = ptrs[z] + (size_t)(wok[k] ? wr[k] : 0) * ((cols / (UB * 32)) * BY + Q<TY>::HDR); }
  int xr[2]; bool tok_ok[2];
  for (int j = 0; j < 2; j++) { int tk = t0 + lb_tok[j]; tok_ok[j] = tk < n; xr[j] = tok_ok[j] ? (in_idx ? in_idx[off[z] + tk] : off[z] + tk) : 0; }
  float acc[16][4];
  #pragma unroll
  for (int i = 0; i < 16; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  int wm = warp * 16;
  int w[2][8]; float s0[2] = {0.f, 0.f}, s1[2] = {0.f, 0.f}, mn[2] = {0.f, 0.f}; int4 q0[2], q1[2]; float bd[2], bs[2];
  for (int j = 0; j < 2; j++) { q0[j] = make_int4(0, 0, 0, 0); q1[j] = q0[j]; bd[j] = bs[j] = 0.f; }
  auto fetch = [&](int u0) {
    int u = u0 + la_u;
    #pragma unroll
    for (int k = 0; k < 2; k++) {
      if (wok[k] && !NODEQ) { uloadr<TY>(wrow[k] + Q<TY>::HDR + (u / UB) * BY, u % UB, w[k], s0[k], s1[k], mn[k], wrow[k]); }
      else { for (int i = 0; i < 8; i++) w[k][i] = 0; s0[k] = s1[k] = mn[k] = 0.f; }
    }
    for (int j = 0; j < 2; j++) {
      int ub = u0 + lb_u;
      if (tok_ok[j]) {
        const int4* gq = (const int4*)(xq + (size_t)xr[j] * cols + ub * 32);
        q0[j] = gq[0]; q1[j] = gq[1];
        bd[j] = xd[(size_t)xr[j] * units + ub]; bs[j] = (float)xs[(size_t)xr[j] * units + ub];
      } else { q0[j] = make_int4(0, 0, 0, 0); q1[j] = q0[j]; bd[j] = 0.f; bs[j] = 0.f; }
    }
  };
  fetch(s_lo * 4);
  for (int u0 = s_lo * 4; u0 < s_hi * 4; u0 += 4) {
    #pragma unroll
    for (int k = 0; k < 2; k++) {
      int rw = la_row + 64 * k;
      #pragma unroll
      for (int i = 0; i < 8; i++) As[la_u][rw][i] = w[k][i];
      Sa0[la_u][rw] = s0[k]; Sa1[la_u][rw] = s1[k]; Sam[la_u][rw] = mn[k];
    }
    for (int j = 0; j < 2; j++) {
      Bs[lb_u][lb_tok[j]][0] = q0[j].x; Bs[lb_u][lb_tok[j]][1] = q0[j].y; Bs[lb_u][lb_tok[j]][2] = q0[j].z; Bs[lb_u][lb_tok[j]][3] = q0[j].w;
      Bs[lb_u][lb_tok[j]][4] = q1[j].x; Bs[lb_u][lb_tok[j]][5] = q1[j].y; Bs[lb_u][lb_tok[j]][6] = q1[j].z; Bs[lb_u][lb_tok[j]][7] = q1[j].w;
      Sbd[lb_u][lb_tok[j]] = bd[j]; Sbs[lb_u][lb_tok[j]] = bs[j];
    }
    __syncthreads();
    if (u0 + 4 < s_hi * 4) fetch(u0 + 4);
    #pragma unroll
    for (int uu = 0; uu < 4; uu++) {
      int ra = wm + g, rb = wm + g + 8;
      int a0 = As[uu][ra][t], a1 = As[uu][rb][t], a2 = As[uu][ra][t + 4], a3 = As[uu][rb][t + 4];
      float sA0 = Sa0[uu][ra], sA1 = Sa0[uu][rb], mA0 = Sam[uu][ra], mA1 = Sam[uu][rb];
      float sB0 = Sa1[uu][ra], sB1 = Sa1[uu][rb];
      #pragma unroll
      for (int nt = 0; nt < 16; nt++) {
        int tb = nt * 8 + g;
        int b0 = Bs[uu][tb][t], b1 = Bs[uu][tb][t + 4];
        int cc[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0};
        float d0 = Sbd[uu][nt * 8 + 2 * t], d1 = Sbd[uu][nt * 8 + 2 * t + 1];
        float x0 = Sbs[uu][nt * 8 + 2 * t], x1 = Sbs[uu][nt * 8 + 2 * t + 1];
        if (HALF) {
          mma16(cc, a0, a1, b0); mma16(c2, a2, a3, b1);
          acc[nt][0] += d0 * (sA0 * cc[0] + sB0 * c2[0]); acc[nt][1] += d1 * (sA0 * cc[1] + sB0 * c2[1]);
          acc[nt][2] += d0 * (sA1 * cc[2] + sB1 * c2[2]); acc[nt][3] += d1 * (sA1 * cc[3] + sB1 * c2[3]);
        } else {
          mma32(cc, a0, a1, a2, a3, b0, b1);
          acc[nt][0] += d0 * (sA0 * cc[0] - mA0 * x0); acc[nt][1] += d1 * (sA0 * cc[1] - mA0 * x1);
          acc[nt][2] += d0 * (sA1 * cc[2] - mA1 * x0); acc[nt][3] += d1 * (sA1 * cc[3] - mA1 * x1);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int nt = 0; nt < 16; nt++)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int r = r0 + wm + g + ((i >> 1) ? 8 : 0), tkn = t0 + nt * 8 + 2 * t + (i & 1);
      if (r < rows && tkn < n) { int ob = off[z] + tkn; int orow = out_idx ? out_idx[ob] : ob; if (S == 1) out[(size_t)orow * rows + r] = acc[nt][i]; else atomicAdd(&out[(size_t)orow * rows + r], acc[nt][i]); }
    }
}

#define MMQ128R(TY, H) extern "C" __global__ void __launch_bounds__(256, 1) mmq128r_##TY(const u8* const* p, const int* off, const int* ii, const int* oi, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols) { mmq128r_impl<TY, H>(p, off, ii, oi, xq, xd, xs, o, rows, cols); }
MMQ128R(2, false) MMQ128R(8, false) MMQ128R(11, true) MMQ128R(12, false) MMQ128R(13, false) MMQ128R(14, true)

// Same 128-row kernel with a 2-blocks-per-SM register cap (opt-in LE_MMQR2=1). Register use is capped at 128 by launch_bounds(256,2).
#define MMQ128R2(TY, H) extern "C" __global__ void __launch_bounds__(256, 2) mmq128r2_##TY(const u8* const* p, const int* off, const int* ii, const int* oi, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols) { mmq128r_impl<TY, H>(p, off, ii, oi, xq, xd, xs, o, rows, cols); }
MMQ128R2(2, false) MMQ128R2(8, false) MMQ128R2(11, true) MMQ128R2(12, false) MMQ128R2(13, false) MMQ128R2(14, true)

#define MMQ128(TY, H) extern "C" __global__ void __launch_bounds__(256, 2) mmq128_##TY(const u8* const* p, const int* off, const int* ii, const int* oi, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols) { mmq128_impl<TY, H>(p, off, ii, oi, xq, xd, xs, o, rows, cols); }
MMQ128(2, false) MMQ128(8, false) MMQ128(11, true) MMQ128(12, false) MMQ128(13, false) MMQ128(14, true)
#define MMQ(TY, H) extern "C" __global__ void __launch_bounds__(256) mmq_##TY(const u8* const* p, const int* off, const int* ii, const int* oi, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols) { mmq_impl<TY, H>(p, off, ii, oi, xq, xd, xs, o, rows, cols); }
MMQ(2, false) MMQ(8, false) MMQ(11, true) MMQ(12, false) MMQ(13, false) MMQ(14, true) MMQ(16, false) MMQ(17, true) MMQ(18, false) MMQ(19, false) MMQ(20, false) MMQ(21, false) MMQ(22, true) MMQ(23, false) MMQ(29, false) MMQ(134, false) MMQ(135, false) MMQ(137, false) MMQ(138, false) MMQ(139, false) MMQ(140, false) MMQ(141, false) MMQ(144, false) MMQ(145, false) MMQ(146, false) MMQ(152, false) MMQ(153, false) MMQ(154, false) MMQ(155, false) MMQ(156, false) MMQ(157, false) MMQ(158, false)

// F32 tiled matmul O[t][r] = W[r].X[t]; block 16x16, each thread 4x4 outputs; tile 64 rows x 64 tokens, K step 16. grid (ceil(rows/64), ceil(T/64)).
extern "C" __global__ void mmf2(const float* W, const float* X, float* O, int rows, int cols, int T) {
  __shared__ float Ws[16][65], Xs[16][65];
  int tx = threadIdx.x & 15, ty = threadIdx.x >> 4, r0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < cols; k0 += 16) {
    for (int i = threadIdx.x; i < 64 * 16; i += 256) {
      int rr = i >> 4, kk = i & 15; int gr = r0 + rr, gt = t0 + rr;
      Ws[kk][rr] = gr < rows && k0 + kk < cols ? W[(size_t)gr * cols + k0 + kk] : 0.f;
      Xs[kk][rr] = gt < T && k0 + kk < cols ? X[(size_t)gt * cols + k0 + kk] : 0.f;
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 16; kk++) {
      float w[4], x[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) { w[i] = Ws[kk][tx + 16 * i]; x[i] = Xs[kk][ty + 16 * i]; }
      #pragma unroll
      for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] += w[i] * x[j];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 4; i++)
    #pragma unroll
    for (int j = 0; j < 4; j++) { int r = r0 + tx + 16 * i, t = t0 + ty + 16 * j; if (r < rows && t < T) O[(size_t)t * rows + r] = acc[i][j]; }
}

// ---- flash-style prefill attention (fp16 tensor cores, online softmax) ----
// grid (ceil(T/64), nh), block 128: warp w owns 16 query rows. K tile kept [key][d], V tile transposed [d][key] so that the
// P*V B-fragments are contiguous. Dynamic smem: Q 64*(HD+8) + K 32*(HD+8) + Vt HD*40 halves.
__device__ __forceinline__ unsigned pack2h(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *(unsigned*)&h; }
__device__ __forceinline__ void mma16816h(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int HD>
__device__ void flash_impl(const float* q, const void* kc, const void* vc, const float* gate, float* out, int pos0, int T, int nh, int nkv, const __half* ksc, const __half* vsc, int kvq, int nc, int win = 0, int gm = 0) {
  constexpr int PQ = HD + 8, VT = 40;
  extern __shared__ __align__(16) __half fsm[];
  __half* qs = fsm; __half* ks = qs + 64 * PQ; __half* vt = ks + 32 * PQ;
  const int q0 = blockIdx.x * 64, h = blockIdx.y, kvh = h / (nh / nkv), kvd = nkv * HD;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const float scale = rsqrtf((float)HD);
  for (int i = tid; i < 64 * HD / 4; i += 128) {
    int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
    float4 v = q0 + r < T ? *(const float4*)(q + (size_t)(q0 + r) * nh * HD + h * HD + c) : make_float4(0.f, 0.f, 0.f, 0.f);
    *(uint2*)(qs + r * PQ + c) = make_uint2(pack2h(v.x * scale, v.y * scale), pack2h(v.z * scale, v.w * scale));
  }
  const int last = nc ? pos0 + T - 1 : pos0 + min(q0 + 64, T) - 1;
  const int qp0 = nc ? pos0 + T - 1 : pos0 + q0 + warp * 16 + g, qp1 = nc ? qp0 : qp0 + 8;
  float o[HD / 8][4];
  #pragma unroll
  for (int d = 0; d < HD / 8; d++) { o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f; }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  const int kstart = win > 0 ? (max(0, pos0 + q0 - win + 1) & ~31) : 0;
  for (int k0 = kstart; k0 <= last; k0 += 32) {
    __syncthreads();
    for (int i = tid; i < 32 * HD / 8; i += 128) {
      int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
      if (kvq == 0) {
        if (k0 + r <= last) { size_t off = (size_t)(k0 + r) * kvd + kvh * HD + c; kv = *(const uint4*)((const __half*)kc + off); vv = *(const uint4*)((const __half*)vc + off); }
        *(uint4*)(ks + r * PQ + c) = kv;
        const __half* vh = (const __half*)&vv;
        #pragma unroll
        for (int j = 0; j < 8; j++) vt[(c + j) * VT + r] = vh[j];
      } else if (kvq == 3) {
        size_t vec = (size_t)(k0 + r) * nkv + kvh; bool ok = k0 + r <= last;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          float kf = 0.f, vf = 0.f;
          if (ok) { kf = kv_ld(kc, ksc, 3, vec, c + j, HD, 0); vf = kv_ld(vc, vsc, 3, vec, c + j, HD, 1); }
          ks[r * PQ + c + j] = __float2half(kf); vt[(c + j) * VT + r] = __float2half(vf);
        }
      } else {
        size_t vec = (size_t)(k0 + r) * nkv + kvh; bool ok = k0 + r <= last;
        float kf[8], vf[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) { kf[j] = 0.f; vf[j] = 0.f; }
        if (ok) {
          float sk = __half2float(ksc[vec * (HD / 32) + c / 32]), sv = __half2float(vsc[vec * (HD / 32) + c / 32]);
          if (kvq == 1) {
            uint2 kk = *(const uint2*)((const signed char*)kc + vec * HD + c), vq = *(const uint2*)((const signed char*)vc + vec * HD + c);
            const signed char* kb = (const signed char*)&kk; const signed char* vb = (const signed char*)&vq;
            #pragma unroll
            for (int j = 0; j < 8; j++) { kf[j] = kb[j] * sk; vf[j] = vb[j] * sv; }
          } else {
            unsigned kk = *(const unsigned*)((const unsigned char*)kc + vec * (HD / 2) + c / 2), vq = *(const unsigned*)((const unsigned char*)vc + vec * (HD / 2) + c / 2);
            #pragma unroll
            for (int j = 0; j < 8; j++) { int kn = (kk >> (4 * j)) & 15, vn = (vq >> (4 * j)) & 15; kf[j] = ((kn ^ 8) - 8) * sk; vf[j] = ((vn ^ 8) - 8) * sv; }
          }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) { ks[r * PQ + c + j] = __float2half(kf[j]); vt[(c + j) * VT + r] = __float2half(vf[j]); }
      }
    }
    __syncthreads();
    float s[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f; }
    #pragma unroll
    for (int kk = 0; kk < HD; kk += 16) {
      const __half* qa = qs + (warp * 16 + g) * PQ + kk + 2 * t4;
      unsigned a0 = *(const unsigned*)qa, a1 = *(const unsigned*)(qa + 8 * PQ), a2 = *(const unsigned*)(qa + 8), a3 = *(const unsigned*)(qa + 8 * PQ + 8);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const __half* kb = ks + (nt * 8 + g) * PQ + kk + 2 * t4;
        mma16816h(s[nt], a0, a1, a2, a3, *(const unsigned*)kb, *(const unsigned*)(kb + 8));
      }
    }
    float mx0 = m0, mx1 = m1;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      int key = k0 + nt * 8 + 2 * t4;
      if (key > qp0 || (win > 0 && key <= qp0 - win)) s[nt][0] = -1e30f; if (key + 1 > qp0 || (win > 0 && key + 1 <= qp0 - win)) s[nt][1] = -1e30f;
      if (key > qp1 || (win > 0 && key <= qp1 - win)) s[nt][2] = -1e30f; if (key + 1 > qp1 || (win > 0 && key + 1 <= qp1 - win)) s[nt][3] = -1e30f;
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1])); mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    float c0 = __expf(m0 - mx0), c1 = __expf(m1 - mx1); m0 = mx0; m1 = mx1; l0 *= c0; l1 *= c1;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      s[nt][0] = __expf(s[nt][0] - m0); s[nt][1] = __expf(s[nt][1] - m0); s[nt][2] = __expf(s[nt][2] - m1); s[nt][3] = __expf(s[nt][3] - m1);
      l0 += s[nt][0] + s[nt][1]; l1 += s[nt][2] + s[nt][3];
    }
    #pragma unroll
    for (int d = 0; d < HD / 8; d++) { o[d][0] *= c0; o[d][1] *= c0; o[d][2] *= c1; o[d][3] *= c1; }
    #pragma unroll
    for (int kb = 0; kb < 2; kb++) {
      unsigned a0 = pack2h(s[2 * kb][0], s[2 * kb][1]), a1 = pack2h(s[2 * kb][2], s[2 * kb][3]);
      unsigned a2 = pack2h(s[2 * kb + 1][0], s[2 * kb + 1][1]), a3 = pack2h(s[2 * kb + 1][2], s[2 * kb + 1][3]);
      #pragma unroll
      for (int d = 0; d < HD / 8; d++) {
        const __half* vb = vt + (d * 8 + g) * VT + kb * 16 + 2 * t4;
        mma16816h(o[d], a0, a1, a2, a3, *(const unsigned*)vb, *(const unsigned*)(vb + 8));
      }
    }
  }
  l0 += __shfl_xor_sync(FULL, l0, 1); l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1); l1 += __shfl_xor_sync(FULL, l1, 2);
  #pragma unroll
  for (int r = 0; r < 2; r++) {
    int t = q0 + warp * 16 + g + 8 * r; if (t >= T) continue;
    float inv = 1.f / (r ? l1 : l0); size_t base = ((size_t)t * nh + h) * HD;
    #pragma unroll
    for (int d = 0; d < HD / 8; d++) for (int e = 0; e < 2; e++) {
      int col = d * 8 + 2 * t4 + e; out[base + col] = o[d][2 * r + e] * inv * (gm == 1 ? 1.f : gm == 2 ? sigm_f(gate[(size_t)t * nh + h]) : (kvq == 3 ? 1.f : sigm_f(gate[base + col])));
    }
  }
}
extern "C" __global__ void __launch_bounds__(128) attn_flash_256(const float* q, const void* kc, const void* vc, const float* gate, float* out, int pos0, int T, int nh, int nkv, const __half* ksc, const __half* vsc, int kvq) {
  flash_impl<256>(q, kc, vc, gate, out, pos0, T, nh, nkv, ksc, vsc, kvq, 0);
}
// block-diffusion preview: every query sees the whole prefix and the whole block
extern "C" __global__ void __launch_bounds__(128) attn_flash_256_nc(const float* q, const void* kc, const void* vc, const float* gate, float* out, int pos0, int T, int nh, int nkv, const __half* ksc, const __half* vsc, int kvq) {
  flash_impl<256>(q, kc, vc, gate, out, pos0, T, nh, nkv, ksc, vsc, kvq, 1);
}

// Dense-model prefill attention (hd 128, f16 KV, optional sliding window, optional per-head sigmoid gate [T][nh]).
extern "C" __global__ void __launch_bounds__(128) dn_flash_128(const float* q, const void* kc, const void* vc, const float* gate, float* out, int pos0, int T, int nh, int nkv, int win) {
  flash_impl<128>(q, kc, vc, gate, out, pos0, T, nh, nkv, nullptr, nullptr, 0, 0, win, gate ? 2 : 1);
}

// Dense prefill attention v2 (hd 128, f16 KV). Differences from dn_flash_128:
//  - 64-key tiles (was 32) so each tile has 8 mma column blocks and half as many syncs/softmax rescales;
//  - Q kept in registers (A fragments loaded once from global), so smem is only K and V;
//  - V stored row-major like K, and its B fragments come from ldmatrix.trans (no scalar transpose stores).
// Smem: 2 * 64 * (128+8) halves = 34816 B (dynamic). Same causal/window mask and sigmoid-gate semantics.
__device__ __forceinline__ void ldsm_x4_t(const __half* p, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
extern "C" __global__ void __launch_bounds__(128) dn_flash_128b(const float* q, const void* kc, const void* vc, const float* gate, float* out, int pos0, int T, int nh, int nkv, int win) {
  constexpr int HDc = 128, PQ = HDc + 8, KT = 64;
  extern __shared__ __align__(16) __half fsm2[];
  __half* ks = fsm2; __half* vs = ks + KT * PQ;
  const int q0 = blockIdx.x * 64, h = blockIdx.y, kvh = h / (nh / nkv), kvd = nkv * HDc;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const float scale = rsqrtf((float)HDc);
  const int rowA = q0 + warp * 16 + g, rowB = rowA + 8;
  unsigned qf[8][4];
  #pragma unroll
  for (int kk = 0; kk < 8; kk++) {
    int c0 = kk * 16 + 2 * t4, c1 = c0 + 8;
    auto qpk = [&](int row, int c) -> unsigned { if (row >= T) return 0u; float2 v = *(const float2*)(q + (size_t)row * nh * HDc + h * HDc + c); return pack2h(v.x * scale, v.y * scale); };
    qf[kk][0] = qpk(rowA, c0); qf[kk][1] = qpk(rowB, c0); qf[kk][2] = qpk(rowA, c1); qf[kk][3] = qpk(rowB, c1);
  }
  const int last = pos0 + min(q0 + 64, T) - 1;
  const int qp0 = pos0 + q0 + warp * 16 + g, qp1 = qp0 + 8;
  float o[HDc / 8][4];
  #pragma unroll
  for (int d = 0; d < HDc / 8; d++) { o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f; }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  const int kstart = win > 0 ? (max(0, pos0 + q0 - win + 1) & ~(KT - 1)) : 0;
  for (int k0 = kstart; k0 <= last; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * HDc / 8; i += 128) {
      int r = i / (HDc / 8), c = (i % (HDc / 8)) * 8;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
      if (k0 + r <= last) { size_t off = (size_t)(k0 + r) * kvd + kvh * HDc + c; kv = *(const uint4*)((const __half*)kc + off); vv = *(const uint4*)((const __half*)vc + off); }
      *(uint4*)(ks + r * PQ + c) = kv;
      *(uint4*)(vs + r * PQ + c) = vv;
    }
    __syncthreads();
    float s[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f; }
    #pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      #pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        const __half* kb = ks + (nt * 8 + g) * PQ + kk * 16 + 2 * t4;
        mma16816h(s[nt], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], *(const unsigned*)kb, *(const unsigned*)(kb + 8));
      }
    }
    float mx0 = m0, mx1 = m1;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      int key = k0 + nt * 8 + 2 * t4;
      if (key > qp0 || (win > 0 && key <= qp0 - win)) s[nt][0] = -1e30f; if (key + 1 > qp0 || (win > 0 && key + 1 <= qp0 - win)) s[nt][1] = -1e30f;
      if (key > qp1 || (win > 0 && key <= qp1 - win)) s[nt][2] = -1e30f; if (key + 1 > qp1 || (win > 0 && key + 1 <= qp1 - win)) s[nt][3] = -1e30f;
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1])); mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    float c0 = __expf(m0 - mx0), c1 = __expf(m1 - mx1); m0 = mx0; m1 = mx1; l0 *= c0; l1 *= c1;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = __expf(s[nt][0] - m0); s[nt][1] = __expf(s[nt][1] - m0); s[nt][2] = __expf(s[nt][2] - m1); s[nt][3] = __expf(s[nt][3] - m1);
      l0 += s[nt][0] + s[nt][1]; l1 += s[nt][2] + s[nt][3];
    }
    #pragma unroll
    for (int d = 0; d < HDc / 8; d++) { o[d][0] *= c0; o[d][1] *= c0; o[d][2] *= c1; o[d][3] *= c1; }
    const int lk = (lane >> 3) & 1, ld = lane & 7, lh = lane >> 4;
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
      unsigned a0 = pack2h(s[2 * kb][0], s[2 * kb][1]), a1 = pack2h(s[2 * kb][2], s[2 * kb][3]);
      unsigned a2 = pack2h(s[2 * kb + 1][0], s[2 * kb + 1][1]), a3 = pack2h(s[2 * kb + 1][2], s[2 * kb + 1][3]);
      #pragma unroll
      for (int dp = 0; dp < HDc / 16; dp++) {
        unsigned r0, r1, r2, r3;
        ldsm_x4_t(vs + (kb * 16 + lk * 8 + ld) * PQ + dp * 16 + lh * 8, r0, r1, r2, r3);
        mma16816h(o[2 * dp], a0, a1, a2, a3, r0, r1);
        mma16816h(o[2 * dp + 1], a0, a1, a2, a3, r2, r3);
      }
    }
  }
  l0 += __shfl_xor_sync(FULL, l0, 1); l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1); l1 += __shfl_xor_sync(FULL, l1, 2);
  #pragma unroll
  for (int r = 0; r < 2; r++) {
    int t = q0 + warp * 16 + g + 8 * r; if (t >= T) continue;
    float inv = 1.f / (r ? l1 : l0); size_t base = ((size_t)t * nh + h) * HDc;
    float gm_ = gate ? sigm_f(gate[(size_t)t * nh + h]) : 1.f;
    #pragma unroll
    for (int d = 0; d < HDc / 8; d++) for (int e = 0; e < 2; e++) {
      int col = d * 8 + 2 * t4 + e; out[base + col] = o[d][2 * r + e] * inv * gm_;
    }
  }
}
