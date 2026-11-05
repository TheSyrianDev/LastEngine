// LastEngine CUDA kernels (clean-room, written from the GGML block-format descriptions).
// A `__constant__ unsigned char IQ3_GRID[1024]` definition is prepended at runtime.
typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32; typedef signed char i8;
#define FULL 0xffffffffu

__device__ __forceinline__ float h2f(const u8* p) { return __half2float(*(const __half*)p); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(FULL, v, o); return v; }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigm_f(float x) { return 1.f / (1.f + __expf(-x)); }

// ---- 32-value unit dot products: x points at the unit's 32 activations ----
__device__ __forceinline__ void sm_k4(int j, const u8* s, int& sc, int& m) {
  if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
  else { sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}
template <int TY> __device__ float unit_dot(const u8* b, int j, const float* x);

// Q4_0 (2): 18B / 32 values, one unit per block
template <> __device__ float unit_dot<2>(const u8* b, int, const float* x) {
  float d = h2f(b), a = 0.f;
  for (int l = 0; l < 16; l++) { a += ((b[2 + l] & 15) - 8) * x[l] + ((b[2 + l] >> 4) - 8) * x[l + 16]; }
  return a * d;
}
// Q8_0 (8): 34B / 32 values
template <> __device__ float unit_dot<8>(const u8* b, int, const float* x) {
  float d = h2f(b), a = 0.f;
  for (int l = 0; l < 32; l++) a += (float)(i8)b[2 + l] * x[l];
  return a * d;
}
// Q4_K (12): 144B / 256
template <> __device__ float unit_dot<12>(const u8* b, int j, const float* x) {
  float d = h2f(b), dm = h2f(b + 2); int sc, m; sm_k4(j, b + 4, sc, m);
  const u8* qs = b + 16 + (j >> 1) * 32; int hi = j & 1;
  float a = 0.f, sx = 0.f;
  for (int l = 0; l < 32; l++) { int q = hi ? (qs[l] >> 4) : (qs[l] & 15); a += q * x[l]; sx += x[l]; }
  return d * sc * a - dm * m * sx;
}
// Q5_K (13): 176B / 256
template <> __device__ float unit_dot<13>(const u8* b, int j, const float* x) {
  float d = h2f(b), dm = h2f(b + 2); int sc, m; sm_k4(j, b + 4, sc, m);
  const u8* qh = b + 16; const u8* qs = b + 48 + (j >> 1) * 32; int hi = j & 1;
  float a = 0.f, sx = 0.f;
  for (int l = 0; l < 32; l++) { int q = (hi ? (qs[l] >> 4) : (qs[l] & 15)) + (((qh[l] >> j) & 1) << 4); a += q * x[l]; sx += x[l]; }
  return d * sc * a - dm * m * sx;
}
// Q6_K (14): 210B / 256
template <> __device__ float unit_dot<14>(const u8* b, int j, const float* x) {
  const u8* ql = b + (j >> 2) * 64; const u8* qh = b + 128 + (j >> 2) * 32; const i8* sc = (const i8*)(b + 192) + (j >> 2) * 8;
  int s = j & 3; float d = h2f(b + 208), a0 = 0.f, a1 = 0.f;
  for (int l = 0; l < 32; l++) {
    int lo = (s & 1) ? ql[32 + l] : ql[l]; lo = (s & 2) ? (lo >> 4) : (lo & 15);
    int q = (lo | (((qh[l] >> (2 * s)) & 3) << 4)) - 32;
    if (l < 16) a0 += q * x[l]; else a1 += q * x[l];
  }
  return d * (sc[(s * 2)] * a0 + sc[s * 2 + 1] * a1);
}
// Q3_K (11): 110B / 256
template <> __device__ float unit_dot<11>(const u8* b, int j, const float* x) {
  const u8* hm = b; const u8* qs = b + 32 + (j >> 2) * 32; const u8* sc = b + 96; float d = h2f(b + 108);
  int s = j & 3; float a[2] = {0.f, 0.f};
  for (int l = 0; l < 32; l++) {
    int q = ((qs[l] >> (2 * s)) & 3) - (((hm[l] >> j) & 1) ? 0 : 4);
    a[l >> 4] += q * x[l];
  }
  float r = 0.f;
  for (int h = 0; h < 2; h++) {
    int i = 2 * j + h; int lo = i < 8 ? (sc[i] & 15) : (sc[i - 8] >> 4); int hi = (sc[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
    r += ((lo | (hi << 4)) - 32) * a[h];
  }
  return d * r;
}
// IQ3_XXS (18): 98B / 256
template <> __device__ float unit_dot<18>(const u8* b, int j, const float* x) {
  float d = h2f(b); const u8* qs = b + 2 + j * 8; u32 aux = (u32)*(const u16*)(b + 66 + j * 4) | ((u32)*(const u16*)(b + 68 + j * 4) << 16);
  float db = d * (0.5f + (aux >> 28)) * 0.5f, a = 0.f;
  for (int l = 0; l < 4; l++) {
    u32 si = (aux >> (7 * l)) & 127; u32 sg = si | ((__popc(si) & 1) << 7);
    for (int h = 0; h < 2; h++) {
      const u8* g = &IQ3_GRID[qs[2 * l + h] * 4];
      for (int k = 0; k < 4; k++) { float v = g[k] * ((sg >> (4 * h + k)) & 1 ? -1.f : 1.f); a += v * x[l * 8 + h * 4 + k]; }
    }
  }
  return a * db;
}

template <int TY> struct Q { static const int UB = 8, BYTES = 0; };
template <> struct Q<2>  { static const int UB = 1, BYTES = 18; };
template <> struct Q<8>  { static const int UB = 1, BYTES = 34; };
template <> struct Q<11> { static const int UB = 8, BYTES = 110; };
template <> struct Q<12> { static const int UB = 8, BYTES = 144; };
template <> struct Q<13> { static const int UB = 8, BYTES = 176; };
template <> struct Q<14> { static const int UB = 8, BYTES = 210; };
template <> struct Q<18> { static const int UB = 8, BYTES = 98; };

// out[slot*rows + r] = W[slot][r,:] . x[slot or 0]. One warp per row, 4 rows per block.
// ptrs: per-slot weight base. xstride: 0 = shared x, else per-slot stride (floats).
template <int TY>
__device__ void matvec_impl(const u8* const* ptrs, const float* x, float* out, int rows, int cols, int xstride) {
  extern __shared__ float xs[];
  int slot = blockIdx.y; const float* xp = x + (size_t)slot * xstride;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = xp[i];
  __syncthreads();
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const int UB = Q<TY>::UB, BY = Q<TY>::BYTES, UV = 256 / UB > 32 ? 32 : 256 / UB;
  int nunits = cols / UV;
  const u8* row = ptrs[slot] + (size_t)r * (cols / (UB * UV)) * BY;
  float acc = 0.f;
  for (int u = lane; u < nunits; u += 32) acc += unit_dot<TY>(row + (u / UB) * BY, u % UB, xs + u * UV);
  acc = warp_sum(acc);
  if (lane == 0) out[(size_t)slot * rows + r] = acc;
}
#define MV(TY) extern "C" __global__ void matvec_##TY(const u8* const* p, const float* x, float* o, int rows, int cols, int xs) { matvec_impl<TY>(p, x, o, rows, cols, xs); }
MV(2) MV(8) MV(11) MV(12) MV(13) MV(14) MV(18)

// F32 matvec (router, tiny projections): same shape
extern "C" __global__ void matvec_0(const u8* const* p, const float* x, float* o, int rows, int cols, int xstride) {
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
