// ---- dense transformer (llama / lfm3) decode kernels ----
// One block per head: optional per-head RMS norm, then RoPE (NORM = adjacent pairs, NEOX = split halves), written as f16 (K) or f32 (Q).
extern "C" __global__ void dn_prep(const float* in, void* out, int out_half, int hd, int rope_dim, int pos, float base, const float* ff, int neox, const float* nw, float eps) {
  __shared__ float v[256]; __shared__ float red[8];
  int h = blockIdx.x; in += (size_t)h * hd;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) v[i] = in[i];
  __syncthreads();
  float sc = 1.f;
  if (nw) {
    float s = 0.f; for (int i = threadIdx.x; i < hd; i += blockDim.x) s += v[i] * v[i];
    s = warp_sum(s); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s; __syncthreads();
    float t = 0.f; for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += red[w];
    sc = rsqrtf(t / hd + eps); __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) v[i] = v[i] * sc * nw[i];
    __syncthreads();
  }
  int half = rope_dim / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float th = pos * powf(base, -(2.f * i) / rope_dim); if (ff) th /= ff[i];
    float s, c; sincosf(th, &s, &c);
    int ia = neox ? i : 2 * i, ib = neox ? i + half : 2 * i + 1;
    float a = v[ia], b = v[ib]; v[ia] = a * c - b * s; v[ib] = a * s + b * c;
  }
  __syncthreads();
  if (out_half) { __half* o = (__half*)out + (size_t)h * hd; for (int i = threadIdx.x; i < hd; i += blockDim.x) o[i] = __float2half(v[i]); }
  else { float* o = (float*)out + (size_t)h * hd; for (int i = threadIdx.x; i < hd; i += blockDim.x) o[i] = v[i]; }
}
extern "C" __global__ void dn_cvt_half(const float* in, __half* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = __float2half(in[i]); }
// Decode attention, one block (8 warps) per query head; warps split the keys, online softmax, merged through shared memory.
extern "C" __global__ void dn_attn(const float* q, const __half* kc, const __half* vc, const float* gate, float* o, int t0, int n, int nh, int nkv, int hd) {
  __shared__ float mw[8], lw[8]; __shared__ float ow[8][256];
  int h = blockIdx.x, kvh = h / (nh / nkv), w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int per = hd / 32; float qv[8], acc[8];
  for (int j = 0; j < per; j++) { qv[j] = q[h * hd + lane + 32 * j]; acc[j] = 0.f; }
  float m = -1e30f, l = 0.f, scale = rsqrtf((float)hd);
  for (int t = t0 + w; t < n; t += 8) {
    const __half* kp = kc + ((size_t)t * nkv + kvh) * hd; const __half* vp = vc + ((size_t)t * nkv + kvh) * hd;
    float s = 0.f; for (int j = 0; j < per; j++) s += qv[j] * __half2float(kp[lane + 32 * j]);
    s = warp_sum(s) * scale;
    float mn = fmaxf(m, s), cr = __expf(m - mn), p = __expf(s - mn);
    l = l * cr + p;
    for (int j = 0; j < per; j++) acc[j] = acc[j] * cr + p * __half2float(vp[lane + 32 * j]);
    m = mn;
  }
  if (lane == 0) { mw[w] = m; lw[w] = l; }
  for (int j = 0; j < per; j++) ow[w][lane + 32 * j] = acc[j];
  __syncthreads();
  float M = -1e30f; for (int i = 0; i < 8; i++) M = fmaxf(M, mw[i]);
  float L = 0.f; for (int i = 0; i < 8; i++) L += lw[i] * __expf(mw[i] - M);
  float g = gate ? sigm_f(gate[h]) : 1.f;
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float a = 0.f; for (int i = 0; i < 8; i++) a += ow[i][d] * __expf(mw[i] - M);
    o[h * hd + d] = a / L * g;
  }
}
// LFM short convolution step: bcx = [b | c | x] (3n), state n*(lc-1) updated, y = c * (conv(b*x)).
extern "C" __global__ void dn_shortconv(const float* bcx, const float* w, float* st, float* y, int n, int lc) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x; if (ch >= n) return;
  float bx = bcx[ch] * bcx[2 * n + ch]; float* s = st + (size_t)ch * (lc - 1);
  float acc = bx * w[ch * lc + lc - 1];
  for (int k = 0; k < lc - 1; k++) acc += s[k] * w[ch * lc + k];
  for (int k = 0; k < lc - 2; k++) s[k] = s[k + 1];
  s[lc - 2] = bx;
  y[ch] = bcx[n + ch] * acc;
}
