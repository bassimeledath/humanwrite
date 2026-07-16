#!/bin/bash
# Defense-in-depth PreToolUse guard: block Bash writes into protected paths and
# obvious credential reads. NOT a security boundary -- the real boundary is that
# test data and provider creds do not exist in this environment. See README.
# Only exit code 2 blocks a tool call; exit 1 would be treated as non-blocking.
input=$(cat)
# Extract the command without depending on jq (falls back to python3).
if command -v jq >/dev/null 2>&1; then
  cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
else
  cmd=$(printf '%s' "$input" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null)
fi

blk() { echo "Blocked: $1" >&2; exit 2; }

echo "$cmd" | grep -Eq '(>|>>|tee|cp|mv|rm|sed -i|truncate|dd)[^|]*\b(\./)?(harness|sources|sealed_evaluator)/' \
  && blk "writes to harness/, sources/, or sealed_evaluator/ (Tier 1/2 immutability)."
echo "$cmd" | grep -Eq '(>|>>|tee|cp|mv|rm|sed -i)[^|]*\b(\./)?(ledger/ledger\.py|infra/gpu)\b' \
  && blk "the ledger core and gpu wrapper are immutable to the agent."
echo "$cmd" | grep -Eq '\b(cat|less|head|tail|grep|strings)\b[^|]*\.(env|pem|key)\b' \
  && blk "credential files are not readable by the agent."
echo "$cmd" | grep -Eq '\benv\b|printenv|MODAL_|HF_TOKEN|GPTZERO|OPENROUTER|ANTHROPIC_API_KEY' \
  && blk "environment/credential inspection is not allowed."
exit 0
