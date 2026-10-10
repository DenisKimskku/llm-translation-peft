#!/bin/bash
# Usage: scripts/run_experiment.sh <name> <config.yaml> <train.jsonl> [extra train args...]
# Env: DEV (default data/dev/dev.jsonl), DEV2 (optional second dev set), EVAL_EPOCHS=1 to also score each epoch checkpoint.
set -euo pipefail
NAME=$1; CFG=$2; TRAIN=$3; shift 3
export HF_HOME=~/hf HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
PY=~/venvs/csmt/bin/python
DEV=${DEV:-data/dev/dev.jsonl}; DEV2=${DEV2:-data/dev/dev2.jsonl}
OUT=outputs/$NAME; mkdir -p "$OUT"
echo "=== [$NAME] train: cfg=$CFG data=$TRAIN extra=$* ===" | tee "$OUT/log.txt"
$PY src/train.py --config "$CFG" --train "$TRAIN" --out "$OUT" "$@" 2>&1 | grep -vE "Warning|warn|deprecated|Loading checkpoint|it/s\]|^\s*$" | tee -a "$OUT/log.txt"
evaluate () {  # $1 adapter dir, $2 tag
  $PY src/infer.py --input "$DEV" --out "$OUT/dev_preds$2.jsonl" --adapter "$1" --batch-size 48 2>&1 | grep -E "inference|Error" | tee -a "$OUT/log.txt"
  $PY src/eval_chrf.py --pred "$OUT/dev_preds$2.jsonl" --ref "$DEV" | tee "$OUT/score$2.json"
  if [ -s "$DEV2" ]; then
    $PY src/infer.py --input "$DEV2" --out "$OUT/dev2_preds$2.jsonl" --adapter "$1" --batch-size 48 2>&1 | grep -E "inference|Error" | tee -a "$OUT/log.txt"
    $PY src/eval_chrf.py --pred "$OUT/dev2_preds$2.jsonl" --ref "$DEV2" > "$OUT/score_dev2$2.json"; echo "dev2 score: $(grep '"score"' $OUT/score_dev2$2.json)"
  fi
}
if [ "${EVAL_EPOCHS:-0}" = "1" ]; then
  for c in "$OUT"/checkpoint-*; do [ -d "$c" ] && evaluate "$c" "_$(basename $c)"; done
fi
evaluate "$OUT/adapter" ""
$PY src/analyze.py --pred "$OUT/dev_preds.jsonl" --ref "$DEV" --worst 8 > "$OUT/analysis.txt"
head -12 "$OUT/analysis.txt"
rm -rf "$OUT"/checkpoint-*
echo "=== [$NAME] done ===" | tee -a "$OUT/log.txt"
