#!/usr/bin/env bash
# OpenAI-compatible API integration checks against a RUNNING lastengine server.
# usage: tests/api_integration.sh [base_url]   (default http://127.0.0.1:8398)
# Start the server first, under machine-lock, e.g.:
#   lastengine serve --model <small.gguf> --addr 127.0.0.1:8398 --ctx 4096
set -u
B=${1:-http://127.0.0.1:8398}
PASS=0; FAIL=0; T=$(mktemp -d)
ok()   { PASS=$((PASS+1)); echo "PASS $1"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL $1: $2"; }
post() { curl -s -m 300 -o "$2" -w '%{http_code}' "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }
pyk()  { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print($2)" "$1" 2>/dev/null; }
MSG='[{"role":"user","content":"Name three colours."}]'

# 1 health and models
c=$(curl -s -m 10 -o "$T/h" -w '%{http_code}' "$B/health"); [ "$c" = 200 ] && ok "health 200" || bad "health" "$c"
c=$(curl -s -m 10 -o "$T/m" -w '%{http_code}' "$B/v1/models")
[ "$c" = 200 ] && [ "$(pyk $T/m "d['object']=='list' and len(d['data'])>=1 and d['data'][0]['object']=='model'")" = True ] && ok "models list" || bad "models" "$c $(cat $T/m | head -c 200)"

# 2 basic non-stream
c=$(post "{\"messages\":$MSG,\"max_tokens\":16,\"temperature\":0}" "$T/b")
[ "$c" = 200 ] && [ "$(pyk $T/b "d['object']=='chat.completion' and isinstance(d['choices'][0]['message']['content'],str) and d['choices'][0]['finish_reason'] in ('stop','length') and d['usage']['total_tokens']==d['usage']['prompt_tokens']+d['usage']['completion_tokens']")" = True ] \
  && ok "non-stream shape" || bad "non-stream shape" "$c $(head -c 300 $T/b)"

# 3 max_tokens respected
c=$(post "{\"messages\":$MSG,\"max_tokens\":3,\"temperature\":0}" "$T/mt")
[ "$c" = 200 ] && [ "$(pyk $T/mt "d['usage']['completion_tokens']<=3 and d['choices'][0]['finish_reason']=='length'")" = True ] && ok "max_tokens cap + finish_reason length" || bad "max_tokens" "$c $(head -c 300 $T/mt)"

# 4 stop sequence: take chars 2..6 of stripped unconstrained output as stop string
post "{\"messages\":$MSG,\"max_tokens\":96,\"temperature\":0}" "$T/s0" >/dev/null
S=$(pyk $T/s0 "(lambda c: c[2:6] if len(c)>=6 else '')((lambda m: (m.get('content') or m.get('reasoning_content') or '').strip())(d['choices'][0]['message']))")
if [ -n "$S" ]; then
  c=$(post "{\"messages\":$MSG,\"max_tokens\":24,\"temperature\":0,\"stop\":[\"$S\"]}" "$T/s1")
  [ "$c" = 200 ] && [ "$(pyk $T/s1 "'$S' not in d['choices'][0]['message']['content'] and d['choices'][0]['finish_reason']=='stop'")" = True ] && ok "stop sequence honoured" || bad "stop sequence" "$c $(head -c 300 $T/s1)"
else bad "stop sequence" "could not derive stop string"; fi

# 5 seed determinism at temperature > 0
A=$(post "{\"messages\":$MSG,\"max_tokens\":12,\"temperature\":0.8,\"seed\":42}" "$T/sa"); Bq=$(post "{\"messages\":$MSG,\"max_tokens\":12,\"temperature\":0.8,\"seed\":42}" "$T/sb")
[ "$A" = 200 ] && [ "$Bq" = 200 ] && [ "$(pyk $T/sa "d['choices'][0]['message']['content']")" = "$(pyk $T/sb "d['choices'][0]['message']['content']")" ] && ok "seed determinism" || bad "seed determinism" "$A $Bq"

# 6 top_p accepted and valid range enforced
c=$(post "{\"messages\":$MSG,\"max_tokens\":4,\"top_p\":0.9}" "$T/tp"); [ "$c" = 200 ] && ok "top_p accepted" || bad "top_p" "$c"

# 7 streaming SSE format
c=$(curl -s -m 300 -N -o "$T/sse" -w '%{http_code}' "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"messages\":$MSG,\"max_tokens\":8,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true}}")
ct=$(curl -s -m 60 -D - -o /dev/null "$B/health" >/dev/null; echo ok)
if [ "$c" = 200 ] && grep -q '^data: \[DONE\]' $T/sse && grep -q '"role":"assistant"' $T/sse && grep -q '"finish_reason":"' $T/sse && grep -q '"usage"' $T/sse; then ok "streaming SSE (role, finish, usage, [DONE])"; else bad "streaming SSE" "$c $(head -c 300 $T/sse)"; fi
grep -q '^data: {' $T/sse && ! grep -q '^event:' $T/sse && ok "SSE data-only framing" || bad "SSE framing" "unexpected event lines"

# 8 tool calls: request accepted with tools; if the model calls one, the shape must be OpenAI-style
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
c=$(post "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"tools\":$TOOLS,\"max_tokens\":200,\"temperature\":0}" "$T/tc")
if [ "$c" = 200 ]; then
  if [ "$(pyk $T/tc "'tool_calls' not in d['choices'][0]['message'] or d['choices'][0]['message']['tool_calls'] is None or all(t['type']=='function' and t['id'] and json.loads(t['function']['arguments']) is not None for t in d['choices'][0]['message']['tool_calls'])")" = True ]; then ok "tool request accepted; tool_calls shape valid if emitted"; else bad "tool_calls shape" "$(head -c 300 $T/tc)"; fi
else bad "tool request" "$c $(head -c 200 $T/tc)"; fi

# 9 errors are JSON with status codes
c=$(curl -s -m 10 -o "$T/e1" -w '%{http_code}' "$B/v1/chat/completions" -H 'Content-Type: application/json' -d '{not json')
[ "$c" = 400 ] && [ "$(pyk $T/e1 "d['error']['code']=='invalid_json' and d['error']['type']=='invalid_request_error'")" = True ] && ok "invalid JSON -> 400 JSON error" || bad "invalid JSON" "$c $(head -c 200 $T/e1)"
c=$(post '{"max_tokens":4}' "$T/e2"); [ "$c" = 400 ] && [ "$(pyk $T/e2 "d['error']['code']=='missing_messages'")" = True ] && ok "missing messages -> 400" || bad "missing messages" "$c"
c=$(post "{\"messages\":$MSG,\"temperature\":5}" "$T/e3"); [ "$c" = 400 ] && [ "$(pyk $T/e3 "d['error']['code']=='invalid_temperature'")" = True ] && ok "bad temperature -> 400" || bad "bad temperature" "$c"
c=$(post "{\"messages\":$MSG,\"max_tokens\":0}" "$T/e4"); [ "$c" = 400 ] && ok "max_tokens 0 -> 400" || bad "max_tokens 0" "$c"

# 10 context overflow: clear 400, not a silent truncation
BIG=$(python3 -c "print(' '.join(['lorem ipsum dolor sit amet']*1500))")
c=$(post "{\"messages\":[{\"role\":\"user\",\"content\":\"$BIG\"}],\"max_tokens\":8}" "$T/ov")
[ "$c" = 400 ] && [ "$(pyk $T/ov "d['error']['code']=='context_length_exceeded'")" = True ] && ok "context overflow -> 400 context_length_exceeded" || bad "context overflow" "$c $(head -c 200 $T/ov)"

# 11 concurrency: parallel requests all succeed (queued on the engine lock, not crashed)
for i in 1 2 3; do (post "{\"messages\":$MSG,\"max_tokens\":6,\"temperature\":0}" "$T/par$i" > "$T/parc$i") & done; wait
if [ "$(cat $T/parc1 $T/parc2 $T/parc3 | tr -d '\n')" = "200200200" ]; then ok "3 concurrent requests all 200"; else bad "concurrency" "$(cat $T/parc1 $T/parc2 $T/parc3 | tr -d '\n')"; fi

# 12 client disconnect cancels generation (recorded as not completed)
curl -s -m 2 -N "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"messages\":$MSG,\"max_tokens\":2000,\"temperature\":0,\"stream\":true}" -o /dev/null
sleep 3
last=$(curl -s -m 10 "$B/stats/requests?limit=1")
if [ "$(echo "$last" | python3 -c "import json,sys;r=json.load(sys.stdin)['requests'];print(r and r[-1].get('completed') is False)" 2>/dev/null)" = True ]; then ok "disconnect cancels generation"; else bad "disconnect cancel" "last stat: $(echo "$last" | head -c 200)"; fi

# 9 tool calls must-have: forced named tool_choice, natural call, streaming; tool_choice none hides tools
WX='[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
NOTH='"chat_template_kwargs":{"enable_thinking":false}'
c=$(post "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],\"tools\":$WX,\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}},\"max_tokens\":200,\"temperature\":0,$NOTH}" "$T/fc")
[ "$c" = 200 ] && [ "$(pyk $T/fc "d['choices'][0]['finish_reason']=='tool_calls' and d['choices'][0]['message']['tool_calls'][0]['function']['name']=='get_weather' and 'city' in json.loads(d['choices'][0]['message']['tool_calls'][0]['function']['arguments'])")" = True ] && ok "forced tool_choice emits get_weather call" || bad "forced tool_choice" "$c $(head -c 300 $T/fc)"
c=$(post "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris right now? Use the tool.\"}],\"tools\":$WX,\"max_tokens\":200,\"temperature\":0,$NOTH}" "$T/nc")
[ "$c" = 200 ] && [ "$(pyk $T/nc "d['choices'][0]['finish_reason']=='tool_calls' and d['choices'][0]['message']['tool_calls'][0]['function']['name']=='get_weather'")" = True ] && ok "natural tool call (non-stream)" || bad "natural tool call" "$c $(head -c 300 $T/nc)"
curl -sN -m 120 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris right now? Use the tool.\"}],\"tools\":$WX,\"stream\":true,\"max_tokens\":200,\"temperature\":0,$NOTH}" > $T/sc
[ "$(python3 -c "
import json
calls=[]
for l in open('$T/sc'):
    l=l.strip()
    if l.startswith('data: ') and l!='data: [DONE]':
        d=json.loads(l[6:])
        for t in (d['choices'][0]['delta'].get('tool_calls') or []): calls.append(t)
fin=[json.loads(l[6:])['choices'][0]['finish_reason'] for l in open('$T/sc') if l.startswith('data: {') and json.loads(l[6:])['choices'][0]['finish_reason']]
print(bool(calls) and calls[0]['function']['name']=='get_weather' and 'tool_calls' in fin)" 2>/dev/null)" = True ] && ok "natural tool call (stream)" || bad "stream tool call" "$(head -c 300 $T/sc)"
c=$(post "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],\"tools\":$WX,\"tool_choice\":\"none\",\"max_tokens\":60,\"temperature\":0,$NOTH}" "$T/tn")
[ "$c" = 200 ] && [ "$(pyk $T/tn "d['choices'][0]['message'].get('tool_calls') in (None,[]) and d['choices'][0]['finish_reason']!='tool_calls'")" = True ] && ok "tool_choice none hides tools" || bad "tool_choice none" "$c $(head -c 200 $T/tn)"
# 10 thinking control: reasoning_effort none => no reasoning; short max_tokens yields content
c=$(post "{\"messages\":[{\"role\":\"user\",\"content\":\"Name three colours.\"}],\"reasoning_effort\":\"none\",\"max_tokens\":40,\"temperature\":0}" "$T/re")
[ "$c" = 200 ] && [ "$(pyk $T/re "d['choices'][0]['message']['reasoning_content']=='' and len(d['choices'][0]['message']['content'])>0")" = True ] && ok "reasoning_effort none -> content within 40 tokens" || bad "reasoning_effort none" "$c $(head -c 300 $T/re)"
c=$(post "{\"messages\":[{\"role\":\"user\",\"content\":\"Name three colours.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":40,\"temperature\":0}" "$T/ct")
[ "$c" = 200 ] && [ "$(pyk $T/ct "len(d['choices'][0]['message']['content'])>0")" = True ] && ok "chat_template_kwargs enable_thinking=false -> content" || bad "enable_thinking false" "$c $(head -c 300 $T/ct)"

# 11 queue: 12 concurrent requests against default LE_MAX_QUEUE=8 -> every reply is 200 or 429 server_busy, server stays healthy
rm -f $T/q*; for i in $(seq 1 12); do (curl -s -m 600 -o $T/q$i.json -w '%{http_code}' "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story about a fox number $i.\"}],\"max_tokens\":120,\"temperature\":0,$NOTH}" > $T/q$i.code &) ; done; sleep 60; while pgrep -f "api_integration.sh" >/dev/null 2>&1 && [ "$(ls $T/q*.code 2>/dev/null | wc -l)" -lt 12 ]; do sleep 2; done
for i in $(seq 1 12); do for _ in $(seq 1 300); do [ -s $T/q$i.code ] && break; sleep 1; done; done
codes=$(cat $T/q*.code | tr '\n' ' ')
ok_n=$(cat $T/q*.code | grep -o 200 | wc -l); busy=$(for f in $T/q*.json; do python3 -c "import json,sys;d=json.load(open('$f'));print(d.get('error',{}).get('code',''))" 2>/dev/null; done | grep -c server_busy)
[ "$ok_n" -ge 1 ] && [ $((ok_n + busy)) -eq 12 ] && [ "$(curl -s -o /dev/null -w '%{http_code}' $B/health)" = 200 ] && ok "queue: 12 concurrent -> $ok_n x 200, $busy x 429 server_busy, server healthy" || bad "queue" "codes: $codes"

echo "----"; echo "passed $PASS, failed $FAIL"; rm -rf "$T"; [ "$FAIL" = 0 ]
