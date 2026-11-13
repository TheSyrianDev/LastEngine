#!/usr/bin/env bash
# Non-GPU API error-path tests against the built-in stub backend (no model, no GPU).
# Usage: tests/api_errors_stub.sh [port]   (builds nothing; needs target/release/lastengine)
set -u
P=${1:-8397}; B=http://127.0.0.1:$P
BIN=${BIN:-target/release/lastengine}
PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "FAIL $1: $2"; }
"$BIN" serve --stub --addr 127.0.0.1:$P >/tmp/le_stub_$P.log 2>&1 & SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 50); do curl -sf $B/health >/dev/null && break; sleep 0.2; done
code() { curl -s -o /tmp/le_stub_body -w '%{http_code}' -H 'content-type: application/json' -d "$2" "$B$1"; }
expect() { # name path body want_code want_error_code
  c=$(code "$2" "$3"); e=$(python3 -c "import json;print(json.load(open('/tmp/le_stub_body')).get('error',{}).get('code',''))" 2>/dev/null)
  if [ "$c" = "$4" ] && [ "$e" = "$5" ]; then ok "$1"; else bad "$1" "http $c code '$e' (want $4 '$5')"; fi
}
expect "invalid JSON body"        /v1/chat/completions '{not json' 400 invalid_json
expect "missing messages"         /v1/chat/completions '{"model":"x"}' 400 missing_messages
expect "empty messages"           /v1/chat/completions '{"messages":[]}' 400 missing_messages
expect "temperature out of range" /v1/chat/completions '{"messages":[{"role":"user","content":"hi"}],"temperature":3}' 400 invalid_temperature
expect "top_p out of range"       /v1/chat/completions '{"messages":[{"role":"user","content":"hi"}],"top_p":1.5}' 400 invalid_top_p
expect "negative max_tokens"      /v1/chat/completions '{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}' 400 invalid_max_tokens
c=$(curl -s -o /tmp/le_stub_body -w '%{http_code}' $B/v1/models)
if [ "$c" = 200 ] && python3 -c "import json;d=json.load(open('/tmp/le_stub_body'))['data'][0];assert d['id'] and 'meta' in d" 2>/dev/null; then ok "models list shape"; else bad "models list shape" "http $c"; fi
c=$(curl -s -o /dev/null -w '%{http_code}' $B/health); [ "$c" = 200 ] && ok "health 200" || bad "health" "http $c"
c=$(code /v1/nope '{}'); [ "$c" = 404 ] && ok "unknown route 404" || bad "unknown route" "http $c"
echo "----"; echo "passed $PASS, failed $FAIL"; [ $FAIL = 0 ]
