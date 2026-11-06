#!/usr/bin/env python3
"""Teacher-forced comparison of LastEngine vs llama.cpp.
usage: tf_compare.py MODEL.gguf PROMPT N [--cpu-moe]
LastEngine greedy-decodes N tokens (lastengine tf-dump); llama-server then scores each prefix prompt+gen[:k]
(token-id prompts, n_probs). Reports top-1 agreement, near-tie disagreements (our token within 0.1 logprob of
llama's best) and mean |dlogprob| of our chosen token. Pass bar: top-1 >= 98% counting near-ties as agreement."""
import json, subprocess, sys, time, urllib.request, os
model, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
moe = "--cpu-moe" in sys.argv
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = open(os.environ["TF_DUMP"]).read().strip().splitlines()[-1] if os.environ.get("TF_DUMP") else subprocess.run([f"{root}/target/release/lastengine", "tf-dump", model, prompt, str(n)], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
d = json.loads(out); pids, gen, top = d["prompt_ids"], d["gen_ids"], d["top"]
port = int(os.environ.get("LLAMA_PORT", "8397"))
external = "LLAMA_PORT" in os.environ
cmd = [os.environ.get("LLAMA_SERVER") or os.path.expanduser("~/AI/engines/llama-mainline/build/bin/llama-server"), "-m", model, "--port", str(port), "-c", "8192", "-ngl", "99", "--no-webui"] + (["--cpu-moe"] if moe else [])
srv = None if external else subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))
try:
    for _ in range(300):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception: time.sleep(2)
    agree = near = 0; dsum = 0.0; cnt = 0; bad = []
    for k in range(n):
        r = post("/completion", {"prompt": pids + gen[:k], "n_predict": 1, "n_probs": 10, "temperature": 0, "cache_prompt": False})
        cp = r["completion_probabilities"][0]
        tl = cp.get("top_logprobs") or cp.get("top_probs") or []
        key = "logprob" if tl and "logprob" in tl[0] else "prob"
        lp = {t["id"]: (t[key] if key == "logprob" else __import__("math").log(max(t[key], 1e-12))) for t in tl}
        best_id = max(lp, key=lp.get)
        ours = gen[k]
        if best_id == ours: agree += 1
        elif ours in lp and lp[best_id] - lp[ours] < 0.1: near += 1
        else: bad.append((k, ours, best_id))
        if ours in lp: dsum += abs(lp[ours] - top[k][0][1]); cnt += 1
    print(f"{os.path.basename(model)}: top1 {agree}/{n}, near-tie {near}, other {len(bad)}; effective agreement {100*(agree+near)/n:.1f}%; mean |dlogprob| {dsum/max(cnt,1):.4f} over {cnt}")
    for b in bad[:5]: print("  mismatch step", b[0], "ours", b[1], "llama", b[2])
finally:
    if srv: srv.terminate(); srv.wait()
