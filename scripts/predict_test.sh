#!/bin/bash
# Produce predictions for the official test set with both systems.
# Usage: scripts/predict_test.sh <test.jsonl> outputs/final_5m/adapter outputs/final_50m/adapter <student_id>
set -euo pipefail
TEST=$1; A5=$2; A50=$3; SID=${4:-2026xxxxxx}
export HF_HOME=~/hf HF_HUB_OFFLINE=1
PY=~/venvs/csmt/bin/python
mkdir -p outputs/test
$PY src/infer.py --input "$TEST" --out outputs/test/pred_5m.jsonl  --adapter "$A5"  --batch-size 48
$PY src/infer.py --input "$TEST" --out outputs/test/pred_50m.jsonl --adapter "$A50" --batch-size 48
$PY - "$TEST" "$SID" <<'PYEOF'
import json, sys
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
p5 = [json.loads(l) for l in open("outputs/test/pred_5m.jsonl", encoding="utf-8")]
p50 = [json.loads(l) for l in open("outputs/test/pred_50m.jsonl", encoding="utf-8")]
assert len(test) == len(p5) == len(p50), "prediction count mismatch"
out = f"outputs/test/{sys.argv[2]}-project1-predictions.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for t, a, b in zip(test, p5, p50):
        row = dict(t); row["prediction_5m"] = a["prediction"]; row["prediction_50m"] = b["prediction"]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
empty = sum(not a["prediction"].strip() for a in p5) + sum(not b["prediction"].strip() for b in p50)
print(f"wrote {out} ({len(test)} rows; empty predictions: {empty})")
PYEOF
