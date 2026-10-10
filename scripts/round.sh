#!/bin/bash
# One data/training round: round-trip-filter all tuple files, build a training set, train 5M and 50M, evaluate on dev.
# Usage: scripts/round.sh <round_name> [extra build_dataset args...]
set -euo pipefail
NAME=$1; shift
export HF_HOME=~/hf HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/minseok/llm-advanced-class/project1
RT=()
for f in data/gen/train_tuples_w1.jsonl data/gen/train_tuples_v2a.jsonl data/gen/train_tuples_v2ko.jsonl data/gen/train_tuples_v2b.jsonl; do
  [ -f "$f" ] || continue
  out="${f%.jsonl}.rt.jsonl"
  ~/venvs/vllm/bin/python scripts/roundtrip_filter.py --in "$f" --out "$out" 2>&1 | grep -E "tuples|back-translated|rt mean|Error" || true
  RT+=("$out")
done
~/venvs/csmt/bin/python src/build_dataset.py --tuples "${RT[@]}" --out "data/train/train_$NAME.jsonl" --single-ratio 0.3 --both-targets --identity 0.05 --min-rt 40 --dedup-against data/dev/dev.jsonl data/dev/dev2.jsonl "$@" | tee "logs/build_$NAME.txt"
scripts/run_experiment.sh "${NAME}_5m" configs/lora_5m.yaml "data/train/train_$NAME.jsonl"
scripts/run_experiment.sh "${NAME}_50m" configs/lora_50m.yaml "data/train/train_$NAME.jsonl"
