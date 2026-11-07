#!/usr/bin/env python3
"""API conformance checks for what LastHarness/llama-swap use. usage: lhd_conformance.py MODEL.gguf [--keep]
Starts `lastengine serve` on a spare port and checks: /health 503->200, /v1/models, /props, usage/timings, stream usage,
stop strings, tool_calls, prefix-cache reuse, slot save/restore, cancel on client disconnect."""
import json, os, socket, subprocess, sys, tempfile, time, urllib.request, urllib.error
model = sys.argv[1]; port = 8399; base = f"http://127.0.0.1:{port}"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slots = tempfile.mkdtemp()
srv = subprocess.Popen([f"{root}/target/release/lastengine", "serve", "--model", model, "--addr", f"127.0.0.1:{port}", "--ctx", "4096", "--slot-save-path", slots], stdout=subprocess.DEVNULL, stderr=open("/tmp/claude-1000/conf_srv.log","w"))
res = []
def check(name, ok, info=""):
    ok = bool(ok); res.append(ok); print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
def req(path, body=None, timeout=300):
    r = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f: return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {}
def chat(msgs, **kw):
    return req("/v1/chat/completions", {"messages": msgs, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **kw})
try:
    saw503 = False
    for _ in range(300):
        try:
            c, j = req("/health", timeout=2)
            if c == 200: break
            if c == 503: saw503 = True
        except Exception: pass
        time.sleep(0.5)
    check("/health 200 after load", c == 200, f"saw 503 while loading: {saw503}")
    c, j = req("/v1/models"); check("/v1/models", c == 200 and j["data"][0]["id"], j["data"][0]["id"] if c == 200 else "")
    c, j = req("/props"); check("/props fields", c == 200 and "total_slots" in j and "n_ctx" in j["default_generation_settings"] and "chat_template_caps" in j)
    long_user = "Summarize in one word: " + " ".join(["alpha beta gamma delta"] * 60)
    c, a = chat([{"role": "user", "content": long_user}], max_tokens=8)
    check("chat usage/timings", c == 200 and a["usage"]["prompt_tokens"] > 100 and a["usage"]["total_tokens"] == a["usage"]["prompt_tokens"] + a["usage"]["completion_tokens"] and a["timings"]["predicted_per_second"] > 0, str(a.get("usage")))
    c, b = chat([{"role": "user", "content": long_user}], max_tokens=8)
    check("prefix-cache reuse on identical prompt", c == 200 and b["usage"]["prompt_tokens_details"]["cached_tokens"] > 50, f"cached {b['usage']['prompt_tokens_details']['cached_tokens']}")
    c, d = chat([{"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Repeat the list."}], max_tokens=400, stop=["5"])
    txt = d["choices"][0]["message"]["content"]; check("stop string", c == 200 and "5" not in txt and d["choices"][0]["finish_reason"] == "stop", repr(txt[-30:]))
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    c, t = chat([{"role": "user", "content": "What is the weather in Paris? Use the tool."}], max_tokens=400, tools=tools)
    tc = t["choices"][0]["message"].get("tool_calls") if c == 200 else None
    check("tool_calls (non-stream)", bool(tc) and tc[0]["function"]["name"] == "get_weather" and json.loads(tc[0]["function"]["arguments"]).get("city"), str(tc)[:100])
    # streaming
    r = urllib.request.Request(base + "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 10, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"content-type": "application/json"})
    lines = [l.decode().strip() for l in urllib.request.urlopen(r, timeout=120)]; data = [l[6:] for l in lines if l.startswith("data: ")]
    chunks = [json.loads(x) for x in data if x != "[DONE]"]
    check("stream: [DONE] + usage chunk + finish_reason", data[-1] == "[DONE]" and any(c.get("usage") for c in chunks) and any(c["choices"] and c["choices"][0]["finish_reason"] for c in chunks))
    # slots
    c, j = req("/slots/0?action=save", {"filename": "t.bin"}); check("slot save", c == 200, str(j)[:120])
    c, j = req("/slots/0?action=restore", {"filename": "t.bin"}); check("slot restore", c == 200, str(j)[:120])
    c, e = req("/slots/0?action=save", {"filename": "../x"}); check("slot path traversal rejected", c == 400)
    # cancel on disconnect (non-stream): the next request must not wait for a 4096-token generation
    s = socket.create_connection(("127.0.0.1", port)); body = json.dumps({"messages": [{"role": "user", "content": "Write a very long story about a dragon."}], "max_tokens": 3000, "temperature": 0.8}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body); time.sleep(2); s.close()
    t0 = time.time(); c, j = chat([{"role": "user", "content": "Say ok."}], max_tokens=4); dt = time.time() - t0
    check("non-stream cancel on disconnect", c == 200 and dt < 20, f"next request took {dt:.1f}s")
finally:
    srv.terminate(); srv.wait()
print(f"{sum(res)}/{len(res)} passed"); sys.exit(0 if all(res) else 1)
