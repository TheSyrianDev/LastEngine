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
template <int TY> struct Q { static const int UB = 8, BYTES = 0; };
template <> struct Q<2>  { static const int UB = 1, BYTES = 18; };
template <> struct Q<8>  { static const int UB = 1, BYTES = 34; };
template <> struct Q<11> { static const int UB = 8, BYTES = 110; };
template <> struct Q<12> { static const int UB = 8, BYTES = 144; };
template <> struct Q<13> { static const int UB = 8, BYTES = 176; };
template <> struct Q<14> { static const int UB = 8, BYTES = 210; };
template <> struct Q<18> { static const int UB = 8, BYTES = 98; };

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
  const u8* row = ptrs[slot] + (size_t)(ok ? r : 0) * (cols / (UB * 32)) * BY;
  float acc = 0.f;
  if (ok) for (int u = li; u < units; u += lpr) acc += udot<TY>(row + (u / UB) * BY, u % UB, sxq + u * 9, sxd[u], sxs[u]);
  for (int o = lpr >> 1; o; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
  if (ok && li == 0) out[(size_t)slot * rows + r] = acc;
}
#define MV(TY) extern "C" __global__ void matvec_##TY(const u8* const* p, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols, int xstride) { mv_impl<TY>(p, xq, xd, xs, o, rows, cols, xstride); }
MV(2) MV(8) MV(11) MV(12) MV(13) MV(14) MV(18)

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
extern "C" __global__ void add_inplace(float* x, const float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += y[i]; }
extern "C" __global__ void swiglu(const float* g, const float* u, float* o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = silu_f(g[i]) * u[i]; }

// Router: softmax over n experts, top-k, renormalise. One block of 256 threads, n <= 1024.
extern "C" __global__ void router_topk(const float* logits, int n, int k, int* idx_out, float* w_out) {
  __shared__ float p[1024]; __shared__ float red[32]; __shared__ int sel[32]; __shared__ float selw[32];
  for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] = logits[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -1e30f; for (int i = 0; i < n; i++) m = fmaxf(m, p[i]);
    float s = 0.f; for (int i = 0; i < n; i++) { p[i] = __expf(p[i] - m); s += p[i]; }
    float tot = 0.f;
    for (int t = 0; t < k; t++) { int b = 0; float bv = -1.f; for (int i = 0; i < n; i++) if (p[i] > bv) { bv = p[i]; b = i; } sel[t] = b; selw[t] = bv / s; tot += bv / s; p[b] = -2.f; }
    for (int t = 0; t < k; t++) { idx_out[t] = sel[t]; w_out[t] = selw[t] / tot; }
  }
}
// moe combine: out[i] += sum_s w[s]*down[s*n+i] + sigmoid(gate)*sh[i]
extern "C" __global__ void moe_combine(float* out, const float* down, const float* w, int k, const float* sh, const float* shgate, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float a = 0.f; for (int s = 0; s < k; s++) a += w[s] * down[(size_t)s * n + i];
  a += sigm_f(shgate[0]) * sh[i];
  out[i] += a;
}

// ---- full attention ----
// split qf [nh][2*hd] -> q (rmsnorm*w, rope) and gate; k rmsnorm+rope; write k,v into f16 cache at pos. One block per head (nh q heads then nkv k heads).
extern "C" __global__ void attn_prep(const float* qf, const float* kf, const float* vf, const float* qn, const float* kn,
    float* q, float* gate, __half* kc, __half* vc, int nh, int nkv, int hd, int rope_dim, float base, int pos, float eps) {
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
    if (i < half) { float th = pos * powf(base, -2.f * i / rope_dim); float sn, cs; sincosf(th, &sn, &cs); v = buf[i] * cs - buf[i + half] * sn; }
    else if (i < rope_dim) { int j = i - half; float th = pos * powf(base, -2.f * j / rope_dim); float sn, cs; sincosf(th, &sn, &cs); v = buf[j] * sn + buf[i] * cs; }
    if (isq) { q[(size_t)hh * hd + i] = v; gate[(size_t)hh * hd + i] = qf[(size_t)hh * 2 * hd + hd + i]; }
    else { kc[((size_t)pos * nkv + hh) * hd + i] = __float2half(v); vc[((size_t)pos * nkv + hh) * hd + i] = __float2half(vf[(size_t)hh * hd + i]); }
  }
}
// scores[h][t] = q_h . k_t / sqrt(hd). grid (ceil(n/8), nh); each warp one position.
extern "C" __global__ void attn_scores(const float* q, const __half* kc, float* scores, int nkv, int nh, int hd, int n, int stride) {
  int h = blockIdx.y, kvh = h / (nh / nkv); int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= n) return;
  const __half* k = kc + ((size_t)t * nkv + kvh) * hd; float a = 0.f;
  for (int i = lane; i < hd; i += 32) a += q[h * hd + i] * __half2float(k[i]);
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
extern "C" __global__ void attn_pv(const float* scores, const __half* vc, const float* gate, float* o, int nkv, int nh, int hd, int n, int stride) {
  int h = blockIdx.x, kvh = h / (nh / nkv), d = threadIdx.x; float a = 0.f;
  for (int t = 0; t < n; t++) a += scores[(size_t)h * stride + t] * __half2float(vc[((size_t)t * nkv + kvh) * hd + d]);
  o[h * hd + d] = a * sigm_f(gate[h * hd + d]);
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
  float kmem = 0.f;
  for (int i = 0; i < dk; i++) { float s = Sh[i * dk + j] * eg; Sh[i * dk + j] = s; kmem += s * ks[i]; }
  float d = (v[j] - kmem) * beta, oo = 0.f;
  for (int i = 0; i < dk; i++) { float s = Sh[i * dk + j] + ks[i] * d; Sh[i * dk + j] = s; oo += s * qs[i]; }
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
  const u8* row = ptrs[z] + (size_t)(ok ? r : 0) * (cols / (UB * 32)) * BY;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ok) for (int u = li; u < units; u += lpr) {
    const u8* wb = row + (u / UB) * BY; int uj = u % UB;
    #pragma unroll
    int w[8]; float s0, s1, mn; uload<TY>(wb, uj, w, s0, s1, mn);
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
MM(2) MM(8) MM(11) MM(12) MM(13) MM(14) MM(18)

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
    float* q, float* gate, __half* kc, __half* vc, int nh, int nkv, int hd, int rope_dim, float base, int pos0, float eps) {
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
    if (i < half) { float th = pos * powf(base, -2.f * i / rope_dim); float sn, cs; sincosf(th, &sn, &cs); v = buf[i] * cs - buf[i + half] * sn; }
    else if (i < rope_dim) { int j = i - half; float th = pos * powf(base, -2.f * j / rope_dim); float sn, cs; sincosf(th, &sn, &cs); v = buf[j] * sn + buf[i] * cs; }
    if (isq) { q[(size_t)hh * hd + i] = v; gate[(size_t)hh * hd + i] = qf[(size_t)hh * 2 * hd + hd + i]; }
    else { kc[((size_t)pos * nkv + hh) * hd + i] = __float2half(v); vc[((size_t)pos * nkv + hh) * hd + i] = __float2half(vf[(size_t)hh * hd + i]); }
  }
}
// Query group: tokens tb..tb+G. scores[g][h][pos] for pos <= pos0+tb+g. grid (ceil(nmax/8), nh, G).
extern "C" __global__ void attn_scores_b(const float* q, const __half* kc, float* scores, int nkv, int nh, int hd, int pos_first, int stride, int tb) {
  int g = blockIdx.z, kvh = blockIdx.y, hpk = nh / nkv; int p = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  int n = pos_first + g + 1;
  if (p >= n) return;
  const __half* k = kc + ((size_t)p * nkv + kvh) * hd; float kr[8];
  for (int j = 0; j < 8; j++) { int i = lane + 32 * j; kr[j] = i < hd ? __half2float(k[i]) : 0.f; }
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
extern "C" __global__ void attn_pv_b(const float* scores, const __half* vc, const float* gate, float* o, int nkv, int nh, int hd, int pos_first, int stride, int tb) {
  int kvh = blockIdx.x, g = blockIdx.y, hpk = nh / nkv, d = threadIdx.x, n = pos_first + g + 1; float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float* sc = scores + ((size_t)g * nh + kvh * hpk) * stride;
  for (int t = 0; t < n; t++) {
    float v = __half2float(vc[((size_t)t * nkv + kvh) * hd + d]);
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) if (hh < hpk) a[hh] += sc[(size_t)hh * stride + t] * v;
  }
  for (int hh = 0; hh < hpk; hh++) { size_t oi = ((size_t)(tb + g) * nh + kvh * hpk + hh) * hd + d; o[oi] = a[hh] * sigm_f(gate[oi]); }
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
  const u8* wrow = ptrs[z] + (size_t)(wok ? wr : 0) * (cols / (UB * 32)) * BY;
  int tk = t0 + lb_tok; bool tok_ok = tk < n;
  int xr = tok_ok ? (in_idx ? in_idx[off[z] + tk] : off[z] + tk) : 0;
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  int wm = (warp & 3) * 16, wn = (warp >> 2) * 32;
  for (int u0 = 0; u0 < units; u0 += 4) {
    int u = u0 + la_u;
    {
      int w[8]; float s0 = 0.f, s1 = 0.f, mn = 0.f;
      if (wok && !NODEQ) { uload<TY>(wrow + (u / UB) * BY, u % UB, w, s0, s1, mn); } else { for (int i = 0; i < 8; i++) w[i] = 0; }
      #pragma unroll
      for (int i = 0; i < 8; i++) As[la_u][la_row][i] = w[i];
      Sa0[la_u][la_row] = s0; Sa1[la_u][la_row] = s1; Sam[la_u][la_row] = mn;
    }
    {
      int ub = u0 + lb_u;
      if (tok_ok) {
        const int4* gq = (const int4*)(xq + (size_t)xr * cols + ub * 32);
        int4 q0 = gq[0], q1 = gq[1];
        Bs[lb_u][lb_tok][0] = q0.x; Bs[lb_u][lb_tok][1] = q0.y; Bs[lb_u][lb_tok][2] = q0.z; Bs[lb_u][lb_tok][3] = q0.w;
        Bs[lb_u][lb_tok][4] = q1.x; Bs[lb_u][lb_tok][5] = q1.y; Bs[lb_u][lb_tok][6] = q1.z; Bs[lb_u][lb_tok][7] = q1.w;
        Sbd[lb_u][lb_tok] = xd[(size_t)xr * units + ub]; Sbs[lb_u][lb_tok] = (float)xs[(size_t)xr * units + ub];
      } else {
        for (int i = 0; i < 8; i++) Bs[lb_u][lb_tok][i] = 0;
        Sbd[lb_u][lb_tok] = 0.f; Sbs[lb_u][lb_tok] = 0.f;
      }
    }
    __syncthreads();
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
#define MMQ(TY, H) extern "C" __global__ void __launch_bounds__(256) mmq_##TY(const u8* const* p, const int* off, const int* ii, const int* oi, const signed char* xq, const float* xd, const int* xs, float* o, int rows, int cols) { mmq_impl<TY, H>(p, off, ii, oi, xq, xd, xs, o, rows, cols); }
MMQ(2, false) MMQ(8, false) MMQ(11, true) MMQ(12, false) MMQ(13, false) MMQ(14, true) MMQ(18, false)

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
__device__ void flash_impl(const float* q, const __half* kc, const __half* vc, const float* gate, float* out, int pos0, int T, int nh, int nkv) {
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
  const int last = pos0 + min(q0 + 64, T) - 1;
  const int qp0 = pos0 + q0 + warp * 16 + g, qp1 = qp0 + 8;
  float o[HD / 8][4];
  #pragma unroll
  for (int d = 0; d < HD / 8; d++) { o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f; }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  for (int k0 = 0; k0 <= last; k0 += 32) {
    __syncthreads();
    for (int i = tid; i < 32 * HD / 8; i += 128) {
      int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
      if (k0 + r <= last) { size_t off = (size_t)(k0 + r) * kvd + kvh * HD + c; kv = *(const uint4*)(kc + off); vv = *(const uint4*)(vc + off); }
      *(uint4*)(ks + r * PQ + c) = kv;
      const __half* vh = (const __half*)&vv;
      #pragma unroll
      for (int j = 0; j < 8; j++) vt[(c + j) * VT + r] = vh[j];
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
      if (key > qp0) s[nt][0] = -1e30f; if (key + 1 > qp0) s[nt][1] = -1e30f;
      if (key > qp1) s[nt][2] = -1e30f; if (key + 1 > qp1) s[nt][3] = -1e30f;
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
      int col = d * 8 + 2 * t4 + e; out[base + col] = o[d][2 * r + e] * inv * sigm_f(gate[base + col]);
    }
  }
}
extern "C" __global__ void __launch_bounds__(128) attn_flash_256(const float* q, const __half* kc, const __half* vc, const float* gate, float* out, int pos0, int T, int nh, int nkv) {
  flash_impl<256>(q, kc, vc, gate, out, pos0, T, nh, nkv);
}
