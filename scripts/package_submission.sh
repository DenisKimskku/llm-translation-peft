#!/bin/bash
# Build <SID>-project1-submit.zip = { <SID>-project1-src.zip (code, configs, prompts, scripts, README, small data), report PDF, predictions jsonl }
set -euo pipefail
SID=${1:-2026xxxxxx}; PRED=${2:-outputs/test/${SID}-project1-predictions.jsonl}
cd /home/minseok/llm-advanced-class/project1
rm -rf /tmp/sub_$SID && mkdir -p /tmp/sub_$SID/src_pkg
cp -r src scripts configs prompts README.md /tmp/sub_$SID/src_pkg/
mkdir -p /tmp/sub_$SID/src_pkg/data && cp -r data/dev /tmp/sub_$SID/src_pkg/data/ 2>/dev/null || true
[ -f data/train/train_final.jsonl ] && mkdir -p /tmp/sub_$SID/src_pkg/data/train && cp data/train/train_final.jsonl /tmp/sub_$SID/src_pkg/data/train/
for a in outputs/final5m/adapter outputs/final50m/adapter; do [ -d "$a" ] && mkdir -p /tmp/sub_$SID/src_pkg/$a && cp $a/adapter_config.json $a/train_meta.json /tmp/sub_$SID/src_pkg/$a/; done   # configs only, no weights
(cd /tmp/sub_$SID/src_pkg && zip -qr ../${SID}-project1-src.zip . -x "*.pyc" "__pycache__/*")
cp report/report.pdf /tmp/sub_$SID/${SID}-project1-report.pdf
cp "$PRED" /tmp/sub_$SID/${SID}-project1-predictions.jsonl
(cd /tmp/sub_$SID && zip -q ${SID}-project1-submit.zip ${SID}-project1-src.zip ${SID}-project1-report.pdf ${SID}-project1-predictions.jsonl)
mkdir -p submission && cp /tmp/sub_$SID/${SID}-project1-submit.zip submission/ && ls -la submission/ && unzip -l submission/${SID}-project1-submit.zip
