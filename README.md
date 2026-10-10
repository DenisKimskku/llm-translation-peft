# Code-Switching Translation with PEFT (Qwen3-1.7B, 5M / 50M budgets)

Midterm project for *Advanced Topics in Large Language Models*. A frozen `Qwen/Qwen3-1.7B` backbone is extended with
LoRA adapters (≤5M and ≤50M trainable parameters) to translate single-language or code-switched text
(English / Spanish / Korean / Japanese mixed in one sentence) into English or Korean.

## Layout

```
src/common.py          prompt template + Qwen3 non-thinking chat formatting
src/build_dataset.py   turns generated 4-way tuples (+ code-switched variants) into training rows
src/train.py           LoRA/DoRA SFT with completion-only loss (configs/*.yaml)
src/infer.py           greedy, non-thinking batched inference (HF transformers)
src/eval_chrf.py       chrF++ (sacrebleu CHRF word_order=2) per subset + weighted score
src/analyze.py         diagnostics: copy baseline, off-target rate, per-language-combo scores
scripts/gen_claude.py  synthetic data generation driver (headless `claude -p`, JSON output)
scripts/gen_vllm.py    same driver for a local model through vLLM (fallback)
scripts/roundtrip_filter.py  English-pivot round-trip filter for code-switched variants (Qwen3-14B-AWQ)
prompts/               generation prompts (seed situations, v1/v2/v2ko tuple prompts, dev prompt)
configs/               lora_5m.yaml (r=4, all linear) and lora_50m.yaml (r=44, all linear)
data/                  generated tuples, training sets, dev set (no model weights)
```

## Setup

```bash
python3 -m venv ~/venvs/csmt && ~/venvs/csmt/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
~/venvs/csmt/bin/pip install "transformers>=4.57,<5" "peft>=0.17" accelerate datasets sacrebleu sentencepiece pyyaml lingua-language-detector
export HF_HOME=~/hf   # Qwen/Qwen3-1.7B is downloaded on first use
```

## Train

```bash
# 5M system (4,358,144 trainable params)
~/venvs/csmt/bin/python src/train.py --config configs/lora_5m.yaml --train data/train/train_final.jsonl --out outputs/final5m
# 50M system (47,939,584 trainable params)
~/venvs/csmt/bin/python src/train.py --config configs/lora_50m.yaml --train data/train/train_final.jsonl --out outputs/final50m
```
`train.py` prints the exact trainable-parameter count and refuses to run over budget.

## Predict and evaluate

```bash
~/venvs/csmt/bin/python src/infer.py --input test.jsonl --out predictions.jsonl --adapter outputs/final50m/adapter
~/venvs/csmt/bin/python src/eval_chrf.py --pred predictions.jsonl --ref validation.jsonl
```
Input rows need `text` (or `input`) and `target` (`en` | `ko`); `id` is preserved. Decoding is greedy with
`enable_thinking=False`, `max_new_tokens=256`, stopping on `<|im_end|>`.

## Data generation (how the training data was built)

See the report; in short: `scripts/gen_claude.py` + `prompts/train_tuples_v2.txt` produce, per scenario, 8 sentences
with faithful en/ko/es/ja versions and two code-switched variants built from explicit "matrix / embedded / pattern"
specifications; `scripts/roundtrip_filter.py` back-translates every variant to English with a local model and drops
variants whose chrF++ against the English sentence is below 40; `src/build_dataset.py` assembles the rows.
