#!/bin/bash
# Finish round 3: average epoch checkpoints, score on official validation, write test predictions.
cd /home/minseok/llm-advanced-class/project1
export HF_HOME=~/hf HF_HUB_OFFLINE=1
PY=~/venvs/csmt/bin/python
V=data/dev/official_validation.jsonl
T=data/dev/official_test.jsonl
mkdir -p outputs/test outputs/official
for run in final3_5m final3_50m_mlp final3_5m_mlp; do
  C=outputs/${run}_ckpts
  if [ "$(ls -d $C/checkpoint-* 2>/dev/null | wc -l)" -ge 2 ] && [ ! -f outputs/${run}_avg/adapter_model.safetensors ]; then
    $PY scripts/avg_adapters.py outputs/${run}_avg $(ls -d $C/checkpoint-*) 2>&1 | tail -2
  fi
  if [ -f outputs/${run}_avg/adapter_model.safetensors ] && [ ! -f outputs/official/val_${run}_avg.jsonl ]; then
    $PY src/infer.py --input $V --out outputs/official/val_${run}_avg.jsonl --adapter outputs/${run}_avg --batch-size 48 2>&1 | grep -E "Error|Traceback"
  fi
  [ -f outputs/official/val_${run}_avg.jsonl ] && echo "${run}_avg: $($PY src/eval_chrf.py --pred outputs/official/val_${run}_avg.jsonl --ref $V | tr -d '\n ')"
done
for a in final3_5m final3_50m_mlp final3_5m_mlp final3_5m_avg final3_50m_mlp_avg final3_5m_mlp_avg; do
  AD=outputs/$a/adapter; [ -d "$AD" ] || AD=outputs/$a
  [ -f "$AD/adapter_model.safetensors" ] || continue
  [ -f outputs/test/pred_$a.jsonl ] && { echo "test preds exist: $a"; continue; }
  $PY src/infer.py --input $T --out outputs/test/pred_$a.jsonl --adapter $AD --batch-size 48 2>&1 | grep -E "Error|Traceback"
  echo "test preds: $a ($(wc -l < outputs/test/pred_$a.jsonl) rows)"
done
echo ROUND3DONE
