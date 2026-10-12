"""Convert infer.py predictions into the instructor's submission format (validation format + "answer").

python scripts/make_submission.py --test data/dev/official_test.jsonl --pred5 outputs/test/pred_5m.jsonl --pred50 outputs/test/pred_50m.jsonl --sid 2026711223 --out submission/
Writes <sid>-project1-predictions-5m.jsonl, <sid>-project1-predictions-50m.jsonl (each: test row + "answer"), and
<sid>-project1-predictions.jsonl (test row + "answer" (=50M system) + "answer_5m" + "answer_50m").
"""
import argparse, json, os, unicodedata, sys
sys.path.insert(0, 'src'); from common import postprocess
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True); ap.add_argument("--pred5", required=True); ap.add_argument("--pred50", required=True)
ap.add_argument("--sid", required=True); ap.add_argument("--out", default="submission")
a = ap.parse_args()
test, p5, p50 = load(a.test), load(a.pred5), load(a.pred50)
assert len(test) == len(p5) == len(p50) == 500 or len(test) == len(p5) == len(p50), "count mismatch"
m5 = {p["id"]: p["prediction"] for p in p5}; m50 = {p["id"]: p["prediction"] for p in p50}
def clean(s, tgt):
    s = postprocess(s, tgt)
    return s if s else "."   # never submit an empty answer
os.makedirs(a.out, exist_ok=True)
def write(name, fn):
    path = os.path.join(a.out, name)
    with open(path, "w", encoding="utf-8") as f:
        for t in test:
            row = {"id": t["id"], "input": t["input"], "target_language": t["target_language"], "input_type": t["input_type"]}
            row.update(fn(t)); f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", path)
write(f"{a.sid}-project1-predictions-5m.jsonl", lambda t: {"answer": clean(m5[t["id"]], t["target_language"])})
write(f"{a.sid}-project1-predictions-50m.jsonl", lambda t: {"answer": clean(m50[t["id"]], t["target_language"])})
write(f"{a.sid}-project1-predictions.jsonl", lambda t: {"answer": clean(m50[t["id"]], t["target_language"]), "answer_5m": clean(m5[t["id"]], t["target_language"]), "answer_50m": clean(m50[t["id"]], t["target_language"])})
empty = sum(1 for t in test if not m5[t["id"]].strip() or not m50[t["id"]].strip())
print("rows:", len(test), "| empty predictions replaced:", empty, "| ids unique:", len({t["id"] for t in test}) == len(test))
