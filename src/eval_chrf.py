"""chrF++ evaluation (sacrebleu CHRF word_order=2) with the project's aggregation.

Score(B) = 0.3 * mean(single_en, single_ko) + 0.7 * mean(mixed_en, mixed_ko)

Usage: python src/eval_chrf.py --pred preds.jsonl --ref dev.jsonl [--pred-field prediction]
Rows are matched by 'id' when present, else by order. Each ref row needs: input/text, target (en|ko),
reference, and type (single|mixed). If 'type' is missing it is inferred from scripts in the input.
"""
import argparse, json, re, unicodedata
from collections import defaultdict
from sacrebleu.metrics import CHRF
from common import read_jsonl, norm_target

HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
KANA_KANJI = re.compile(r"[぀-ヿ一-鿿]")
LATIN = re.compile(r"[A-Za-zÀ-ÿ]")

def infer_type(text: str) -> str:
    scripts = sum(bool(p.search(text)) for p in (HANGUL, KANA_KANJI, LATIN))
    return "mixed" if scripts >= 2 else "single"

def get(row, *names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred-field", default="prediction")
    ap.add_argument("--sentence-level", action="store_true", help="also report mean of sentence-level chrF++")
    args = ap.parse_args()
    preds = read_jsonl(args.pred); refs = read_jsonl(args.ref)
    if all("id" in r for r in refs) and all("id" in p for p in preds):
        pm = {p["id"]: p for p in preds}
        pairs = [(pm[r["id"]], r) for r in refs]
    else:
        assert len(preds) == len(refs), f"{len(preds)} preds vs {len(refs)} refs"
        pairs = list(zip(preds, refs))
    buckets = defaultdict(lambda: ([], []))
    for p, r in pairs:
        src = get(r, "input", "text", "source", "src")
        typ = get(r, "type", "input_type", "kind") or infer_type(src)
        typ = "mixed" if str(typ).lower().startswith(("mix", "code", "cs")) else "single"
        tgt = norm_target(get(r, "target", "target_lang", "target_language", "tgt"))
        hyp = unicodedata.normalize("NFC", str(get(p, args.pred_field, "output", "translation", default="")).strip())
        ref = unicodedata.normalize("NFC", str(get(r, "reference", "answer", "ref", "translation", "output")).strip())
        buckets[(typ, tgt)][0].append(hyp); buckets[(typ, tgt)][1].append(ref)
    chrf = CHRF(word_order=2)
    res = {}
    for (typ, tgt), (hyps, rs) in sorted(buckets.items()):
        res[f"{typ}_{tgt}"] = chrf.corpus_score(hyps, [rs]).score
        if args.sentence_level:
            res[f"{typ}_{tgt}_sent"] = sum(chrf.sentence_score(h, [r]).score for h, r in zip(hyps, rs)) / len(hyps)
        res[f"n_{typ}_{tgt}"] = len(hyps)
    def mean(keys):
        vals = [res[k] for k in keys if k in res]
        return sum(vals) / len(vals) if vals else float("nan")
    single = mean(["single_en", "single_ko"]); mixed = mean(["mixed_en", "mixed_ko"])
    res["single"] = single; res["mixed"] = mixed
    res["score"] = 0.3 * single + 0.7 * mixed
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))

if __name__ == "__main__":
    main()
