#!/usr/bin/env python3
"""Matched speed benchmark. usage: bench_vs_llamacpp.py MODEL.gguf [--moe]
LastEngine: serve + one chat request with ~PP-token prompt (cache bypassed by a unique prefix) -> timings.
llama.cpp: llama-bench -p PP -n 128 -ngl 99 [-ncmoe 99 for --moe] -fa 1."""
import json, os, subprocess, sys, time, urllib.request
model, moe = sys.argv[1], "--moe" in sys.argv
PP = int(os.environ.get("PP", "2048")); root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
srv = subprocess.Popen([f"{root}/target/release/lastengine", "serve", "--model", model, "--addr", "127.0.0.1:8398", "--ctx", str(PP + 400)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def post(b):
    r = urllib.request.Request("http://127.0.0.1:8398/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json"}); return json.load(urllib.request.urlopen(r, timeout=900))
try:
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:8398/health", timeout=2)).get("status") == "ok": break
        except Exception: time.sleep(2)
    import random; random.seed(PP); W="the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split(); words = " ".join(random.choice(W) for _ in range(PP - 40))
    post({"messages": [{"role": "user", "content": "warm up"}], "max_tokens": 4, "temperature": 0})
    d = post({"messages": [{"role": "user", "content": words + " Summarize in one sentence."}], "max_tokens": 128, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    print("lastengine:", json.dumps({"usage": d["usage"], "timings": d.get("timings")}))
finally:
    srv.terminate(); srv.wait()
time.sleep(3)
cmd = [os.path.expanduser("~/AI/engines/llama-mainline/build/bin/llama-bench"), "-m", model, "-p", str(PP), "-n", "128", "-ngl", "99", "-fa", "1", "-r", "2"] + (["-ncmoe", "99"] if moe else [])
print("cmd:", " ".join(cmd)); print(subprocess.run(cmd, capture_output=True, text=True).stdout)
