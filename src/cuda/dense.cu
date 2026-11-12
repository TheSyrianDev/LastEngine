// ---- dense transformer (llama / lfm3) decode kernels ----
// One block per head: optional per-head RMS norm, then RoPE (NORM = adjacent pairs, NEOX = split halves), written as f16 (K) or f32 (Q).
extern "C" __global__ void dn_prep(const float* in, void* out, int out_half, int hd, int rope_dim, int pos, float base, const float* ff, int neox, const float* nw, float eps, const int* pdev, int rstride) {
  __shared__ float v[256]; __shared__ float red[8];
  int h = blockIdx.x; size_t tb = (size_t)blockIdx.y * gridDim.x * hd; size_t ob = pdev ? (size_t)pdev[0] * rstride : 0; if (pdev) pos = pdev[0]; pos += blockIdx.y; in += tb + (size_t)h * hd;
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
  if (out_half) { __half* o = (__half*)out + ob + tb + (size_t)h * hd; for (int i = threadIdx.x; i < hd; i += blockDim.x) o[i] = __float2half(v[i]); }
  else { float* o = (float*)out + ob + tb + (size_t)h * hd; for (int i = threadIdx.x; i < hd; i += blockDim.x) o[i] = v[i]; }
}
extern "C" __global__ void dn_cvt_half(const float* in, __half* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = __float2half(in[i]); }
// Decode variant: destination row is pdev[0] (device-resident position), so the launch args do not change per token.
extern "C" __global__ void dn_cvt_half_d(const float* in, __half* base, const int* pdev, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) base[(size_t)pdev[0] * n + i] = __float2half(in[i]); }
// RMSNorm + activation quantisation in one launch (block 256). The norm is computed exactly as rmsnorm; the q8_1-style unit quantiser is quant_x's.
// Writes the float normalised vector o (kept for residual/other uses) and xq/xd/xs for the next matvec. n % 32 == 0.
extern "C" __global__ void rms_quant(const float* x, const float* w, float* o, signed char* xq, float* xd, int* xs, int n, float eps) {
  // one block per 32-wide unit slice: each block recomputes the sum of squares over n (n <= 4096) with 1 warp, then quantises its own unit.
  // Wider n: sum loop is per-warp, so latency is one warp's pass over n; blocks run in parallel across the grid.
  __shared__ float r_s;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp == 0) {
    float s = 0.f; for (int i = lane; i < n; i += 32) s += x[i] * x[i];
    s = warp_sum(s); if (lane == 0) r_s = rsqrtf(s / n + eps);
  }
  __syncthreads();
  float r = r_s;
  int nu = n / 32;
  for (int u = blockIdx.x * (blockDim.x >> 5) + warp; u < nu; u += gridDim.x * (blockDim.x >> 5)) {
    float v = x[u * 32 + lane] * r * w[u * 32 + lane];
    o[u * 32 + lane] = v;
    float a = fabsf(v);
    for (int off = 16; off; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
    float d = a / 127.f, id = d > 0.f ? 1.f / d : 0.f;
    int q = __float2int_rn(v * id);
    xq[u * 32 + lane] = (signed char)q;
    int sm = q; for (int off = 16; off; off >>= 1) sm += __shfl_xor_sync(0xffffffff, sm, off);
    if (lane == 0) { xd[u] = d; xs[u] = sm; }
  }
}
// Fused decode prep for one token: blocks [0,nh) do q (norm+rope, in place); blocks [nh,nh+nkv) do k (norm+rope into the K cache row at pdev[0])
// and copy v into the V cache row at pdev[0]. Same math as dn_prep (q) + dn_prep (k) + dn_cvt_half_d (v), one launch. Block 128 threads, hd <= 256.
__device__ void dn_rot_load(const float* in, int hd, int rope_dim, int pos, float base, const float* ff, int neox, const float* nw, float eps, float* v, float* red) {
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
}
extern "C" __global__ void dn_qkv_dec(float* q, const float* k, const float* v, __half* kc, __half* vc, const int* pdev, int pos0, int nh, int nkv, int hd, int rope_dim, float base, const float* ff, int neox, const float* qn, const float* kn, float eps) {
  __shared__ float vb[256]; __shared__ float red[8];
  if (blockIdx.x < nh) {
    int h = blockIdx.x;
    dn_rot_load(q + h * hd, hd, rope_dim, pdev[0], base, ff, neox, qn, eps, vb, red);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) q[h * hd + i] = vb[i];
  } else {
    int kh = blockIdx.x - nh; int pos = pdev[0];
    dn_rot_load(k + kh * hd, hd, rope_dim, pos, base, ff, neox, kn, eps, vb, red);
    size_t ro = (size_t)pos * nkv * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) { kc[ro + kh * hd + i] = __float2half(vb[i]); vc[ro + kh * hd + i] = __float2half(v[kh * hd + i]); }
  }
}
// Decode attention, one block (8 warps) per query head; warps split the keys, online softmax, merged through shared memory.
extern "C" __global__ void dn_attn(const float* q, const __half* kc, const __half* vc, const float* gate, float* o, int win, int n, int nh, int nkv, int hd, const int* ndev) {
  if (ndev) n = ndev[0];
  __shared__ float mw[8], lw[8]; __shared__ float ow[8][256];
  int h = blockIdx.x, kvh = h / (nh / nkv), w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  n += blockIdx.y; int t0 = (win > 0 && n > win) ? n - win : 0;
  q += (size_t)blockIdx.y * nh * hd; o += (size_t)blockIdx.y * nh * hd; if (gate) gate += (size_t)blockIdx.y * nh;
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

// Batched LFM short convolution: bcx [T][3n] -> y [T][n], sequential over tokens per channel.
extern "C" __global__ void dn_shortconv_b(const float* bcx, const float* w, float* st, float* y, int n, int lc, int T) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x; if (ch >= n) return;
  float s[8]; for (int k = 0; k < lc - 1; k++) s[k] = st[(size_t)ch * (lc - 1) + k];
  for (int t = 0; t < T; t++) {
    const float* r = bcx + (size_t)t * 3 * n;
    float bx = r[ch] * r[2 * n + ch];
    float acc = bx * w[ch * lc + lc - 1];
    for (int k = 0; k < lc - 1; k++) acc += s[k] * w[ch * lc + k];
    for (int k = 0; k < lc - 2; k++) s[k] = s[k + 1];
    s[lc - 2] = bx;
    y[(size_t)t * n + ch] = r[n + ch] * acc;
  }
  for (int k = 0; k < lc - 1; k++) st[(size_t)ch * (lc - 1) + k] = s[k];
}

// Split-K decode attention for long contexts: grid (nh, S), each block covers a slice of the key range; partial (m, l, acc[hd]) go to
// part[(h*S+s)*(hd+2)], dn_attn_comb merges them (and applies the head gate).
extern "C" __global__ void dn_attn_split(const float* q, const __half* kc, const __half* vc, float* part, int win, int n, int nh, int nkv, int hd, int S, const int* ndev) {
  if (ndev) n = ndev[0];
  __shared__ float mw[8], lw[8]; __shared__ float ow[8][128];
  int h = blockIdx.x, s = blockIdx.y, kvh = h / (nh / nkv), w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int t0 = (win > 0 && n > win) ? n - win : 0; int len = n - t0; int per_s = (len + S - 1) / S;
  int a = t0 + s * per_s, b = min(n, a + per_s);
  int per = hd / 32; float qv[4], acc[4];
  for (int j = 0; j < per; j++) { qv[j] = q[h * hd + lane * 4 + j]; acc[j] = 0.f; }
  float m = -1e30f, l = 0.f, scale = rsqrtf((float)hd);
  for (int t = a + w; t < b; t += 8) {
    const __half* kp = kc + ((size_t)t * nkv + kvh) * hd; const __half* vp = vc + ((size_t)t * nkv + kvh) * hd;
    uint2 kr = *(const uint2*)(kp + lane * 4), vr = *(const uint2*)(vp + lane * 4);
    const __half* kh = (const __half*)&kr; const __half* vh = (const __half*)&vr;
    float sc = 0.f; for (int j = 0; j < 4; j++) sc += qv[j] * __half2float(kh[j]);
    sc = warp_sum(sc) * scale;
    float mn = fmaxf(m, sc), cr = __expf(m - mn), p = __expf(sc - mn);
    l = l * cr + p;
    for (int j = 0; j < 4; j++) acc[j] = acc[j] * cr + p * __half2float(vh[j]);
    m = mn;
  }
  if (lane == 0) { mw[w] = m; lw[w] = l; }
  for (int j = 0; j < per; j++) ow[w][lane * 4 + j] = acc[j];
  __syncthreads();
  float M = -1e30f; for (int i = 0; i < 8; i++) M = fmaxf(M, mw[i]);
  float* pp = part + ((size_t)h * S + s) * (hd + 2);
  if (threadIdx.x == 0) { float L = 0.f; for (int i = 0; i < 8; i++) L += lw[i] * __expf(mw[i] - M); pp[0] = M; pp[1] = L; }
  for (int d = threadIdx.x; d < hd; d += blockDim.x) { float x = 0.f; for (int i = 0; i < 8; i++) x += ow[i][d] * __expf(mw[i] - M); pp[2 + d] = x; }
}
// GQA split-K decode attention: grid (nkv, S), block 256 = 8 warps, warp w = query head w of the KV group (needs nh/nkv == 8, hd == 128).
// K/V tiles of 32 keys are staged once per KV head (fp16 copy). Lane L owns head dims 4L..4L+3. Each lane forms its partial dot for all 32 keys,
// then a butterfly reduce-scatter gives lane L the full score of key L: one reduce per tile, not per key.
// Output layout matches dn_attn_split (dn_attn_comb merges it).
extern "C" __global__ void dn_attn_gqa(const float* q, const __half* kc, const __half* vc, float* part, int win, int n, int nh, int nkv, int hd, int S, const int* ndev) {
  if (ndev) n = ndev[0];
  __shared__ float qs[8][128]; __shared__ __half ks[32][128]; __shared__ __half vs[32][128];
  const int G = nh / nkv; int kvh = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  int h = kvh * G + w;
  int t0 = (win > 0 && n > win) ? n - win : 0; int len = n - t0; int per_s = (len + S - 1) / S;
  int a = t0 + s * per_s, b = min(n, a + per_s);
  for (int i = tid; i < 8 * 128; i += 256) { int hh = i >> 7, d = i & 127; qs[hh][d] = q[(kvh * G + hh) * hd + d]; }
  float m = -1e30f, l = 0.f, scale = rsqrtf((float)hd);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int t = a; t < b; t += 32) {
    int cnt = min(32, b - t);
    __syncthreads();
    for (int i = tid; i < 32 * 128; i += 256) {
      int kk = i >> 7, d = i & 127;
      if (kk < cnt) { size_t row = (size_t)(t + kk) * nkv + kvh; ks[kk][d] = kc[row * hd + d]; vs[kk][d] = vc[row * hd + d]; }
    }
    __syncthreads();
    float qv[4]; for (int j = 0; j < 4; j++) qv[j] = qs[w][lane * 4 + j];
    float p[32];
#pragma unroll
    for (int kk = 0; kk < 32; kk++) {
      uint2 kr = *(const uint2*)&ks[kk][lane * 4]; const __half* kh = (const __half*)&kr;
      float dot = 0.f; for (int j = 0; j < 4; j++) dot += qv[j] * __half2float(kh[j]);
      p[kk] = dot;
    }
    // butterfly reduce-scatter: after the 5 steps, p[0] in lane L is the full dot of key L
#pragma unroll
    for (int i = 0; i < 16; i++) { float snd = (lane & 16) ? p[i] : p[i + 16], kep = (lane & 16) ? p[i + 16] : p[i]; p[i] = kep + __shfl_xor_sync(0xffffffff, snd, 16); }
#pragma unroll
    for (int i = 0; i < 8; i++) { float snd = (lane & 8) ? p[i] : p[i + 8], kep = (lane & 8) ? p[i + 8] : p[i]; p[i] = kep + __shfl_xor_sync(0xffffffff, snd, 8); }
#pragma unroll
    for (int i = 0; i < 4; i++) { float snd = (lane & 4) ? p[i] : p[i + 4], kep = (lane & 4) ? p[i + 4] : p[i]; p[i] = kep + __shfl_xor_sync(0xffffffff, snd, 4); }
#pragma unroll
    for (int i = 0; i < 2; i++) { float snd = (lane & 2) ? p[i] : p[i + 2], kep = (lane & 2) ? p[i + 2] : p[i]; p[i] = kep + __shfl_xor_sync(0xffffffff, snd, 2); }
    { float snd = (lane & 1) ? p[0] : p[1], kep = (lane & 1) ? p[1] : p[0]; p[0] = kep + __shfl_xor_sync(0xffffffff, snd, 1); }
    bool valid = lane < cnt;
    float sc = valid ? p[0] * scale : -1e30f;
    float tm = sc;
    for (int o = 16; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffff, tm, o));
    float mn = fmaxf(m, tm), cr = __expf(m - mn);
    float pe = valid ? __expf(sc - mn) : 0.f;
    l = l * cr + warp_sum(pe);
    for (int j = 0; j < 4; j++) acc[j] *= cr;
    for (int k = 0; k < cnt; k++) {
      float pk = __shfl_sync(0xffffffff, pe, k);
      for (int j = 0; j < 4; j++) acc[j] += pk * __half2float(vs[k][lane * 4 + j]);
    }
    m = mn;
  }
  float* pp = part + ((size_t)h * S + s) * (hd + 2);
  if (lane == 0) { pp[0] = m; pp[1] = l; }
  for (int j = 0; j < 4; j++) pp[2 + lane * 4 + j] = acc[j];
}
extern "C" __global__ void dn_attn_comb(const float* part, const float* gate, float* o, int hd, int S) {
  int h = blockIdx.x; const float* pp = part + (size_t)h * S * (hd + 2);
  float M = -1e30f; for (int s = 0; s < S; s++) M = fmaxf(M, pp[s * (hd + 2)]);
  float L = 0.f; for (int s = 0; s < S; s++) L += pp[s * (hd + 2) + 1] * __expf(pp[s * (hd + 2)] - M);
  float g = gate ? sigm_f(gate[h]) : 1.f;
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float a = 0.f; for (int s = 0; s < S; s++) a += pp[s * (hd + 2) + 2 + d] * __expf(pp[s * (hd + 2)] - M);
    o[h * hd + d] = a / L * g;
  }
}
