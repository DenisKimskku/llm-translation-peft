"""Turn generated 4-way tuples (+ mixed variants) into training rows.

python src/build_dataset.py --tuples data/gen/train_tuples_w1.jsonl [more...] --out data/train/train_v1.jsonl \
    [--single-ratio 0.35] [--both-targets] [--splice 0.5] [--dedup-against data/dev/dev.jsonl]
Each tuple: en, ko, es, ja, mixed1, mixed2 (+ *_langs). Output rows: text, target, reference, type, langs, src.
"""
import argparse, json, random, re, sys, os, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, write_jsonl

HANGUL = re.compile(r"[가-힣]"); KANA = re.compile(r"[぀-ヿ]"); CJK = re.compile(r"[一-鿿]"); LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
def scripts(s):
    return {"ko": bool(HANGUL.search(s)), "ja": bool(KANA.search(s) or CJK.search(s)), "latin": bool(LATIN.search(s))}
def clean(s):
    s = unicodedata.normalize("NFC", str(s)).strip()
    return re.sub(r"\s+", " ", s)

_DET = None
def lid(text):
    global _DET
    if _DET is None:
        from lingua import Language, LanguageDetectorBuilder
        _DET = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH).build()
    l = _DET.detect_language_of(text)
    return None if l is None else l.name.lower()

def ok_tuple(t):
    for k in ("en", "ko", "es", "ja"):
        if not isinstance(t.get(k), str) or len(t[k].strip()) < 3: return False
    if not HANGUL.search(t["ko"]) or (KANA.search(t["ko"]) or CJK.search(t["ko"])): return False  # Korean must be Hangul-only
    if not (KANA.search(t["ja"]) or CJK.search(t["ja"])) or HANGUL.search(t["ja"]): return False
    if HANGUL.search(t["en"]) or KANA.search(t["en"]) or CJK.search(t["en"]): return False
    if HANGUL.search(t["es"]) or KANA.search(t["es"]) or CJK.search(t["es"]): return False
    if lid(t["en"]) != "english" or lid(t["es"]) != "spanish": return False   # catches swapped en/es fields
    return True

def mixed_ok(txt, langs):
    sc = scripts(txt); n = sum(sc.values())
    if n >= 2: return True
    return set(langs) >= {"en", "es"} and n == 1 and sc["latin"]   # en+es mix cannot be verified by script

SPLIT = re.compile(r"(?<=[,，、;:])\s*|\s+(?=(?:and|but|because|so|if|when|while|although|before|after|y|pero|porque|si|cuando|aunque|que)\b)", re.I)

def splice(t, rng):
    """Clause-level splice: take clause boundaries from the English sentence, render alternating clauses from other languages
    where the parallel sentence has the same number of clauses (comma-based). Returns (text, langs) or None."""
    langs = rng.sample(["en", "ko", "es", "ja"], 2)
    a, b = [t[l] for l in langs]
    pa = [p for p in re.split(r"(?<=[,，、])\s*", a) if p.strip()]
    pb = [p for p in re.split(r"(?<=[,，、])\s*", b) if p.strip()]
    if len(pa) < 2 or len(pa) != len(pb): return None
    out = []
    for i in range(len(pa)):
        out.append(pa[i] if i % 2 == 0 else pb[i])
    txt = "".join(p if p.endswith(("、", "，")) else p + " " for p in out).strip()
    return txt, sorted(langs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--single-ratio", type=float, default=0.35, help="fraction of rows that are single-language")
    ap.add_argument("--both-targets", action="store_true", help="use each mixed variant for both en and ko targets")
    ap.add_argument("--splice", type=float, default=0.0, help="prob. of adding one clause-spliced mixed row per tuple")
    ap.add_argument("--identity", type=float, default=0.03, help="prob. of adding an en->en / ko->ko identity row per tuple")
    ap.add_argument("--dedup-against", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rt", type=float, default=0.0, help="drop mixed variants whose English round-trip chrF++ is below this (if scored)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tuples = []
    for p in args.tuples: tuples += read_jsonl(p)
    seen = set(); clean_t = []; bad = 0
    for t in tuples:
        if not ok_tuple(t): bad += 1; continue
        for k in ("en", "ko", "es", "ja", "mixed1", "mixed2"):
            if isinstance(t.get(k), str): t[k] = clean(t[k])
        key = t["en"].lower()
        if key in seen: continue
        seen.add(key); clean_t.append(t)
    block = set()
    for p in args.dedup_against:
        for r in read_jsonl(p):
            block.add(clean(r.get("reference", "")).lower()); block.add(clean(r.get("text", "")).lower())
    clean_t = [t for t in clean_t if t["en"].lower() not in block and t["ko"].lower() not in block]
    print(f"{len(tuples)} tuples -> {len(clean_t)} clean unique ({bad} malformed)")
    mixed_rows, single_rows = [], []
    for i, t in enumerate(clean_t):
        variants = []
        for mk in ("mixed1", "mixed2"):
            if isinstance(t.get(mk), str) and len(t[mk]) > 3:
                langs = sorted(set(t.get(mk + "_langs") or []))
                if args.min_rt and (mk + "_rt") in t and t[mk + "_rt"] < args.min_rt: continue
                if mixed_ok(t[mk], langs): variants.append((t[mk], langs, mk))
        if args.splice and rng.random() < args.splice:
            s = splice(t, rng)
            if s and mixed_ok(*s): variants.append((s[0], s[1], "splice"))
        for j, (txt, langs, src) in enumerate(variants):
            targets = ["en", "ko"] if args.both_targets or src == "splice" else [["en", "ko"][(i + j) % 2]]
            for tgt in targets:
                if txt == t[tgt]: continue
                mixed_rows.append({"text": txt, "target": tgt, "reference": t[tgt], "type": "mixed", "langs": langs, "src": src})
        for src_l in ("es", "ja", "ko"):
            single_rows.append({"text": t[src_l], "target": "en", "reference": t["en"], "type": "single", "langs": [src_l], "src": "single"})
        for src_l in ("en", "es", "ja"):
            single_rows.append({"text": t[src_l], "target": "ko", "reference": t["ko"], "type": "single", "langs": [src_l], "src": "single"})
        if rng.random() < args.identity:
            l = rng.choice(["en", "ko"])
            single_rows.append({"text": t[l], "target": l, "reference": t[l], "type": "single", "langs": [l], "src": "identity"})
    rng.shuffle(single_rows)
    n_single = int(len(mixed_rows) * args.single_ratio / (1 - args.single_ratio))
    rows = mixed_rows + single_rows[:n_single]
    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    from collections import Counter
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(" by type/target:", Counter((r["type"], r["target"]) for r in rows))
    print(" by src:", Counter(r["src"] for r in rows))
    print(" mixed langs:", Counter("+".join(r["langs"]) for r in rows if r["type"] == "mixed").most_common(12))

if __name__ == "__main__":
    main()
