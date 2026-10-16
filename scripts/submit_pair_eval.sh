#!/bin/bash
cd /home/minseok/llm-advanced-class/project1
export HF_HOME=~/hf HF_HUB_OFFLINE=1
PY=~/venvs/csmt/bin/python; T=data/dev/official_test.jsonl; D2=data/dev/dev2.jsonl
for a in submit_5m submit_50m; do
  $PY src/infer.py --input $T --out outputs/test/pred_$a.jsonl --adapter outputs/$a/adapter --batch-size 48 2>&1 | grep -E "Error|Traceback"
  echo "test preds: $a ($(wc -l < outputs/test/pred_$a.jsonl) rows)"
  $PY src/infer.py --input $D2 --out outputs/$a/dev2_preds.jsonl --adapter outputs/$a/adapter --batch-size 48 2>&1 | grep -E "Error|Traceback"
  echo "dev2 $a: $($PY src/eval_chrf.py --pred outputs/$a/dev2_preds.jsonl --ref $D2 | tr -d '\n ')"
done
echo PAIRDONE
