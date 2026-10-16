#!/bin/bash
cd /home/minseok/llm-advanced-class/project1
export HF_HOME=~/hf HF_HUB_OFFLINE=1
PY=~/venvs/csmt/bin/python; V=data/dev/official_validation.jsonl
run() { name=$1; src=$2; s=$3
  $PY scripts/scale_adapter.py $src outputs/ut_$name $s
  $PY src/infer.py --input $V --out outputs/official/val_ut_$name.jsonl --adapter outputs/ut_$name --batch-size 48 2>&1 | grep -E "Error|Traceback"
  echo "ut_$name: $($PY src/eval_chrf.py --pred outputs/official/val_ut_$name.jsonl --ref $V | tr -d '\n ')"; }
run 50m_s085 outputs/final3_50m_mlp/adapter 0.85
run 50m_s070 outputs/final3_50m_mlp/adapter 0.70
run 50m_s055 outputs/final3_50m_mlp/adapter 0.55
echo UT50DONE
