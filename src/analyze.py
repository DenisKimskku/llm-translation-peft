"""Diagnostics for a prediction file: per-bucket chrF++, copy baseline, off-target rate, truncation, length ratio,
breakdown by language combination. python src/analyze.py --pred preds.jsonl --ref dev.jsonl
"""
import argparse, json, re, statistics, unicodedata, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, norm_target
from sacrebleu.metrics import CHRF

HANGUL = re.compile(r"[가-힣]"); CJK = re.compile(r"[぀-ヿ一-鿿]"); LATIN = re.compile(r"[A-Za-zÀ-ÿ]{2,}")

def off_target(pred, tgt):
    if not pred.strip(): return True
    if tgt == "ko": return (not HANGUL.search(pred)) or bool(CJK.search(pred))
    return bool(HANGUL.search(pred)) or bool(CJK.search(pred))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256); ap.add_argument("--worst", type=int, default=0)
    args = ap.parse_args()
    preds = {p["id"]: p for p in read_jsonl(args.pred)}; refs = read_jsonl(args.ref)
    chrf = CHRF(word_order=2); n = lambda s: unicodedata.normalize("NFC", s).strip()
    buckets = defaultdict(list)
    rows = []
    for r in refs:
        p = preds[r["id"]]["prediction"]; tgt = norm_target(r["target"])
        s = chrf.sentence_score(n(p), [n(r["reference"])]).score
        c = chrf.sentence_score(n(r["text"]), [n(r["reference"])]).score
        rows.append((s, c, r, p))
        buckets[(r["type"], tgt)].append((s, c, r, p))
    print(f"{'bucket':12s} {'n':>5s} {'chrF++':>7s} {'copy':>6s} {'offtgt%':>8s} {'len_ratio':>9s} {'empty':>5s}")
    for key in sorted(buckets):
        b = buckets[key]; hy = [n(p) for _,_,_,p in b]; rf = [n(r["reference"]) for _,_,r,_ in b]; src = [n(r["text"]) for _,_,r,_ in b]
        corpus = chrf.corpus_score(hy, [rf]).score; copy = chrf.corpus_score(src, [rf]).score
        off = 100 * sum(off_target(p, key[1]) for _,_,_,p in b) / len(b)
        lr = statistics.mean(len(h) / max(1, len(r)) for h, r in zip(hy, rf)); emp = sum(not h for h in hy)
        print(f"{key[0]+'_'+key[1]:12s} {len(b):5d} {corpus:7.2f} {copy:6.2f} {off:8.1f} {lr:9.2f} {emp:5d}")
    combos = defaultdict(list)
    for s, c, r, p in rows:
        if r["type"] == "mixed": combos[("+".join(r.get("langs", [])), norm_target(r["target"]))].append(s)
    print("\nmixed by language combo (mean sentence chrF++):")
    for k in sorted(combos, key=lambda k: -len(combos[k])):
        print(f"  {k[0]:12s} ->{k[1]}  n={len(combos[k]):4d}  {statistics.mean(combos[k]):6.2f}")
    if args.worst:
        rows.sort(key=lambda x: x[0])
        print("\nworst cases:")
        for s, c, r, p in rows[:args.worst]:
            print(f"{s:5.1f} [{r['type']}->{r['target']}] {r['text']}\n   REF: {r['reference']}\n   HYP: {p}")

if __name__ == "__main__":
    main()
