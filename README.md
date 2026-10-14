# Code-Switching Translation with PEFT (Qwen3-1.7B, 5M / 50M budgets)

Midterm project for *Advanced Topics in Large Language Models*. A frozen `Qwen/Qwen3-1.7B` backbone is extended with
LoRA adapters (≤5M and ≤50M trainable parameters) to translate single-language or code-switched text
(English / Spanish / Korean / Japanese mixed in one sentence) into English or Korean.

## Layout

```
src/common.py          prompt template, Qwen3 non-thinking chat formatting, Korean punctuation post-processing
src/build_dataset.py   turns generated 4-way tuples (+ code-switched variants) into training rows
src/train.py           LoRA/DoRA SFT with completion-only loss (configs/*.yaml; rank_pattern for MLP-heavy allocation)
src/infer.py           greedy, non-thinking batched inference (HF transformers)
src/eval_chrf.py       chrF++ (sacrebleu CHRF word_order=2) per subset + weighted score
src/analyze.py         diagnostics: copy baseline, off-target rate, length ratio, per-language-combo scores
scripts/gen_claude.py  synthetic data generation driver (headless `claude -p`, JSON output, resumable, parallel)
scripts/gen_vllm.py    same driver for a local model through vLLM (fallback)
scripts/roundtrip_filter.py  English-pivot round-trip filter for code-switched variants (Qwen3-14B-AWQ, vLLM)
scripts/build_final3.py      assembles the final training set (all waves + natural Korean references + human pairs)
scripts/avg_adapters.py      averages LoRA checkpoints (tried; no gain)
scripts/make_submission.py   writes the prediction files in the validation format (`answer` key)
prompts/               generation prompts (situations, tuple prompts v1/v2/v2ko/v3, Korean natural-style rewrite, dev prompt)
configs/               lora_5m.yaml (r=4, all linear) and lora_50m_mlpheavy.yaml (attn r=20 / MLP r=60); other variants tried
data/dev/              own dev sets (the generated training data and the human-pair pool are not included in the submission)
```

## Final systems

| system | config | trainable params |
|---|---|---|
| 5M  | LoRA r=4, alpha 8, all 7 linear projections × 28 layers | 4,358,144 |
| 50M | LoRA attention r=20 (alpha 40), MLP gate/up/down r=60 (alpha 120), all 28 layers | 49,315,840 |

Both trained for 2 epochs on `data/train/train_final3.jsonl` (244,852 rows = 204,852 synthetic rows from 36,334 generated
tuples + 40,000 human-translated pairs), completion-only loss, cosine schedule, effective batch 16, bf16, one RTX 3080.
Adapters: `outputs/final3_5m/adapter`, `outputs/final3_50m_mlp/adapter` (weights not in git / submission; configs and
`train_meta.json` are).

## Results (official validation set, 500 items; corpus chrF++, score = 0.3·single + 0.7·mixed)

| system | single→en | single→ko | mixed→en | mixed→ko | score |
|---|---|---|---|---|---|
| Qwen3-1.7B baseline (no PEFT) | 59.15 | 28.04 | 69.17 | 42.78 | 52.26 |
| 5M system | 63.68 | 48.46 | 74.46 | 69.78 | **67.30** |
| 50M system | 62.73 | 50.27 | 75.04 | 72.12 | **68.46** |

Own dev set (1,535 items): baseline 53.98, round-1 5M 73.43, round-1 50M 74.54. Second dev set (Qwen3-8B, 1,267 items):
baseline 68.82, final 5M 71.76, final 50M 72.59. See the report for the full breakdown and controlled comparisons.

## Setup

```bash
python3 -m venv ~/venvs/csmt && ~/venvs/csmt/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
~/venvs/csmt/bin/pip install "transformers>=4.57,<5" "peft>=0.17" accelerate datasets sacrebleu sentencepiece pyyaml lingua-language-detector
export HF_HOME=~/hf   # Qwen/Qwen3-1.7B is downloaded on first use
```

## Train

```bash
# 5M system (4,358,144 trainable params)
~/venvs/csmt/bin/python src/train.py --config configs/lora_5m.yaml --train data/train/train_final3.jsonl --out outputs/final3_5m
# 50M system (49,315,840 trainable params)
~/venvs/csmt/bin/python src/train.py --config configs/lora_50m_mlpheavy.yaml --train data/train/train_final3.jsonl --out outputs/final3_50m_mlp
```
`train.py` prints the exact trainable-parameter count and refuses to run over budget. Each run takes ~5.9 h on an RTX 3080.

## Predict and evaluate

```bash
~/venvs/csmt/bin/python src/infer.py --input test.jsonl --out predictions.jsonl --adapter outputs/final3_50m_mlp/adapter
~/venvs/csmt/bin/python src/eval_chrf.py --pred predictions.jsonl --ref validation.jsonl
# submission files: one per budget, <SID>-project1-predictions-5M.jsonl and -50M.jsonl (test row + "answer")
~/venvs/csmt/bin/python scripts/make_submission.py --test test.jsonl --pred5 pred_5m.jsonl --pred50 pred_50m.jsonl --sid <SID> --out submission
```
Input rows need `text` (or `input`) and `target` (or `target_language`: `en` | `ko`); `id` is preserved. Decoding is greedy with
`enable_thinking=False`, `max_new_tokens=256`, stopping on `<|im_end|>`. Korean outputs are NFC-normalised and commas are
removed (`postprocess()` in `src/common.py`); `--raw` disables this.

## Reproducing the data pipeline

```bash
scripts/gen_claude.py  --prompt prompts/situations.txt       --seeds data/gen/seeds_situations.jsonl --out data/gen/situations_raw.jsonl
scripts/make_train_seeds_v2.py 1 1 data/gen/seeds_train_v2a.jsonl        # per-sentence matrix/embed/pattern specs
scripts/gen_claude.py  --prompt prompts/train_tuples_v2.txt   --seeds data/gen/seeds_train_v2a.jsonl  --out data/gen/train_tuples_v2a.jsonl  --workers 8
scripts/gen_claude.py  --prompt prompts/train_tuples_v2ko.txt --seeds data/gen/seeds_train_v2ko.jsonl --out data/gen/train_tuples_v2ko.jsonl --workers 8
scripts/make_train_seeds_v3.py ... && scripts/gen_claude.py --prompt prompts/train_tuples_v3.txt --seeds data/gen/seeds_train_v3.jsonl --out data/gen/train_tuples_v3.jsonl --workers 8
~/venvs/vllm/bin/python scripts/roundtrip_filter.py data/gen/train_tuples_v3.jsonl     # English-pivot round trip (Qwen3-14B-AWQ)
scripts/gen_claude.py  --prompt prompts/ko_natural.txt --seeds data/gen/seeds_konat.jsonl --out data/gen/konat_en1st.jsonl --workers 8   # natural Korean references
scripts/build_human_pool.py                                                # optional human pairs (AI Hub / BSD / Tanaka / TED2020 downloads)
~/venvs/csmt/bin/python scripts/build_final3.py                            # round-trip-filtered tuples → rows (+ natural ko refs, + 40k human)
```

In short: per scenario, the generator writes 8 sentences with faithful en/ko/es/ja versions and two code-switched variants
built from explicit "matrix / embedded / pattern" specifications; every variant is script/LID-checked and back-translated to
English with a local model (variants with chrF++ < 40 against the English sentence are dropped); Korean references are
rewritten into natural, comma-free Korean; `src/build_dataset.py` assembles the rows and removes anything overlapping the
dev / validation / test sets.
