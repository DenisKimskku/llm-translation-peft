#!/bin/bash
# Final round: RT-filter all tuple files, build synthetic rows, add a balanced human subset, train 5M and 50M (scoring each epoch).
# Usage: scripts/final_round.sh <name> [human_rows=40000] [cfg5m=configs/lora_5m.yaml] [cfg50m=configs/lora_50m.yaml]
set -euo pipefail
NAME=$1; HUMAN=${2:-40000}; CFG5=${3:-configs/lora_5m.yaml}; CFG50=${4:-configs/lora_50m.yaml}
export HF_HOME=~/hf HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/minseok/llm-advanced-class/project1
RT=()
for f in data/gen/train_tuples_w1.jsonl data/gen/train_tuples_v2a.jsonl data/gen/train_tuples_v2ko.jsonl data/gen/train_tuples_v2b.jsonl; do
  [ -f "$f" ] || continue
  out="${f%.jsonl}.rt.jsonl"
  ~/venvs/vllm/bin/python scripts/roundtrip_filter.py --in "$f" --out "$out" 2>&1 | grep -E "tuples|back-translated|rt mean|Error" || true
  RT+=("$out")
done
~/venvs/csmt/bin/python src/build_dataset.py --tuples "${RT[@]}" --out "data/train/syn_$NAME.jsonl" --single-ratio 0.3 --both-targets --identity 0.05 --min-rt 40 --dedup-against data/dev/dev.jsonl data/dev/dev2.jsonl | tee "logs/build_$NAME.txt"
~/venvs/csmt/bin/python - "$NAME" "$HUMAN" <<'PYEOF'
import json, random, collections, sys
name, n_h = sys.argv[1], int(sys.argv[2]); random.seed(0)
syn = [json.loads(l) for l in open(f"data/train/syn_{name}.jsonl")]
human = [json.loads(l) for l in open("data/train/human_pool.jsonl")]
by = collections.defaultdict(list)
for h in human: by[(h["langs"][0], h["target"])].append(h)
share = {("ja","en"):0.23,("ko","en"):0.2,("en","ko"):0.2,("ja","ko"):0.17,("es","ko"):0.13,("es","en"):0.07}
sub = []
for k, v in by.items():
    random.shuffle(v); sub += v[:int(n_h * share.get(k, 0))]
rows = syn + sub; random.shuffle(rows)
with open(f"data/train/train_{name}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(syn)} synthetic + {len(sub)} human = {len(rows)} rows -> data/train/train_{name}.jsonl")
print(collections.Counter((r["type"], r["target"]) for r in rows))
PYEOF
EVAL_EPOCHS=1 scripts/run_experiment.sh "${NAME}_5m" "$CFG5" "data/train/train_$NAME.jsonl"
EVAL_EPOCHS=1 scripts/run_experiment.sh "${NAME}_50m" "$CFG50" "data/train/train_$NAME.jsonl"
