#!/bin/bash
cd /home/minseok/llm-advanced-class/project1
export HF_HOME=~/hf HF_HUB_OFFLINE=1
PY=~/venvs/csmt/bin/python; D=data/dev/dev.jsonl
for a in submit_5m submit_50m; do
  $PY src/infer.py --input $D --out outputs/$a/dev_preds.jsonl --adapter outputs/$a/adapter --batch-size 48 2>&1 | grep -E "Error|Traceback"
  echo "dev $a: $($PY src/eval_chrf.py --pred outputs/$a/dev_preds.jsonl --ref $D | tr -d '\n ')"
done
echo DEVDONE
