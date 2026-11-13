# Compares `lastengine dequant` against the gguf-py oracle for one tensor per quant type.
import sys, subprocess, numpy as np
sys.path.insert(0, sys.argv[1]); model = sys.argv[2]; exe = sys.argv[3]
from gguf import GGUFReader, quants
r = GGUFReader(model); seen = {}
for t in r.tensors:
    ty = int(t.tensor_type)
    if ty not in seen and len(t.shape) >= 1 and t.shape[0] % 256 == 0 or True: seen[ty] = t
for ty, t in sorted(seen.items()):
    if ty in (0, 1): continue
    n = 256 * 64
    sub = quants.dequantize(np.frombuffer(t.data, dtype=np.uint8)[: n // quants.GGML_QUANT_SIZES[t.tensor_type][0] * quants.GGML_QUANT_SIZES[t.tensor_type][1]].reshape(-1), t.tensor_type)
    subprocess.run([exe, 'dequant', model, t.name, str(n), '/tmp/claude-1000/dq.bin'], check=True)
    mine = np.fromfile('/tmp/claude-1000/dq.bin', dtype=np.float32)
    print(ty, t.tensor_type.name, t.name, 'max abs diff', float(np.abs(mine - sub.reshape(-1)[:n]).max()))
