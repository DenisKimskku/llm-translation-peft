#!/bin/bash
# Build <SID>-project1-submit.zip = { <SID>-project1-src.zip (code only: src/scripts/configs/prompts/README, own dev set, adapter configs
# without weights; NO training data), <SID>-project1-report.pdf, <SID>-project1-predictions.jsonl (+ per-system files) }
set -euo pipefail
SID=${1:-2026711223}
cd /home/minseok/llm-advanced-class/project1
W=/tmp/sub_$SID; rm -rf $W && mkdir -p $W/src_pkg
cp -r src scripts configs prompts README.md $W/src_pkg/
mkdir -p $W/src_pkg/data/dev && cp data/dev/dev.jsonl data/dev/dev2.jsonl data/dev/seeds_dev.jsonl $W/src_pkg/data/dev/ 2>/dev/null || true
for a in final_5m final_50m_mlp; do mkdir -p $W/src_pkg/outputs/$a/adapter; cp outputs/$a/adapter/adapter_config.json outputs/$a/adapter/train_meta.json $W/src_pkg/outputs/$a/adapter/ 2>/dev/null || true; cp outputs/$a/score.json $W/src_pkg/outputs/$a/ 2>/dev/null || true; done
cp outputs/official/val_*.jsonl $W/src_pkg/outputs/ 2>/dev/null || true   # validation predictions (small)
find $W/src_pkg -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
(cd $W/src_pkg && zip -qr ../${SID}-project1-src.zip .)
cp report/report.pdf $W/${SID}-project1-report.pdf
cp submission/${SID}-project1-predictions.jsonl submission/${SID}-project1-predictions-5m.jsonl submission/${SID}-project1-predictions-50m.jsonl $W/
(cd $W && zip -q ${SID}-project1-submit.zip ${SID}-project1-src.zip ${SID}-project1-report.pdf ${SID}-project1-predictions.jsonl ${SID}-project1-predictions-5m.jsonl ${SID}-project1-predictions-50m.jsonl)
mkdir -p submission && cp $W/${SID}-project1-submit.zip submission/
ls -la submission/${SID}-project1-submit.zip && unzip -l submission/${SID}-project1-submit.zip && echo "--- src zip contents (top level) ---" && unzip -l $W/${SID}-project1-src.zip | awk '{print $4}' | cut -d/ -f1-2 | sort | uniq -c | sort -rn | head -12
