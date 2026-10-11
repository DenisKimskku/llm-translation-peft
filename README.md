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

## Results (own dev set, 1,535 items; corpus chrF++, score = 0.3·single + 0.7·mixed)

| system | trainable params | single→en | single→ko | mixed→en | mixed→ko | score |
|---|---|---|---|---|---|---|
| Qwen3-1.7B baseline (no PEFT) | 0 | 66.08 | 31.85 | 72.74 | 39.50 | 53.98 |
| 5M system (LoRA r=4, all linear) | 4,358,144 | 78.79 | 58.19 | 83.79 | 67.31 | **73.43** |
| 50M system (LoRA r=44, all linear) | 47,939,584 | 79.98 | 60.51 | 83.99 | 68.76 | **74.54** |

Final adapters: `outputs/final_5m/adapter`, `outputs/final_50m/adapter` (not in git; reproducible with the commands below on
`data/train/train_final.jsonl`, which is built by `scripts/final_round.sh`).

## Setup

```bash
python3 -m venv ~/venvs/csmt && ~/venvs/csmt/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
~/venvs/csmt/bin/pip install "transformers>=4.57,<5" "peft>=0.17" accelerate datasets sacrebleu sentencepiece pyyaml lingua-language-detector
export HF_HOME=~/hf   # Qwen/Qwen3-1.7B is downloaded on first use
```

## Train

```bash
# 5M system (4,358,144 trainable params)
~/venvs/csmt/bin/python src/train.py --config configs/lora_5m.yaml --train data/train/train_final.jsonl --out outputs/final_5m
# 50M system (47,939,584 trainable params)
~/venvs/csmt/bin/python src/train.py --config configs/lora_50m.yaml --train data/train/train_final.jsonl --out outputs/final_50m
```
`train.py` prints the exact trainable-parameter count and refuses to run over budget.

## Predict and evaluate

```bash
~/venvs/csmt/bin/python src/infer.py --input test.jsonl --out predictions.jsonl --adapter outputs/final_50m/adapter
~/venvs/csmt/bin/python src/eval_chrf.py --pred predictions.jsonl --ref validation.jsonl
```
Input rows need `text` (or `input`) and `target` (`en` | `ko`); `id` is preserved. Decoding is greedy with
`enable_thinking=False`, `max_new_tokens=256`, stopping on `<|im_end|>`.

## Reproducing the full pipeline

```bash
scripts/gen_claude.py  --prompt prompts/situations.txt      --seeds data/gen/seeds_situations.jsonl --out data/gen/situations_raw.jsonl
scripts/make_train_seeds_v2.py 1 1 data/gen/seeds_train_v2a.jsonl   # per-sentence matrix/embed/pattern specs
scripts/gen_claude.py  --prompt prompts/train_tuples_v2.txt --seeds data/gen/seeds_train_v2a.jsonl --out data/gen/train_tuples_v2a.jsonl --workers 8
scripts/gen_claude.py  --prompt prompts/train_tuples_v2ko.txt --seeds data/gen/seeds_train_v2ko.jsonl --out data/gen/train_tuples_v2ko.jsonl --workers 8
scripts/build_human_pool.py                                  # optional human pairs (AI Hub / BSD / Tanaka / TED2020 downloads)
scripts/final_round.sh final 40000                           # round-trip filter → build → train 5M + 50M → score
```

## Data generation (how the training data was built)

See the report; in short: `scripts/gen_claude.py` + `prompts/train_tuples_v2.txt` produce, per scenario, 8 sentences
with faithful en/ko/es/ja versions and two code-switched variants built from explicit "matrix / embedded / pattern"
specifications; `scripts/roundtrip_filter.py` back-translates every variant to English with a local model and drops
variants whose chrF++ against the English sentence is below 40; `src/build_dataset.py` assembles the rows.
