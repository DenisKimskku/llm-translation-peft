"""Build dev2 (different generator family) from data/dev/dev2_tuples_raw.jsonl with the same recipe as dev."""
import json, re, collections
from lingua import Language, LanguageDetectorBuilder
rows=[json.loads(l) for l in open("data/dev/dev2_tuples_raw.jsonl")]
HANGUL=re.compile(r"[가-힣]"); JA=re.compile(r"[぀-ヿ一-鿿]"); LATIN=re.compile(r"[A-Za-zÀ-ÿ]")
def nscripts(s): return sum(bool(p.search(s)) for p in (HANGUL, JA, LATIN))
det = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH).build()
out=[]; n=0; bad=0; seen=set()
for i, r in enumerate(rows):
    if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("en","ko","es","ja","mixed1","mixed2")): bad+=1; continue
    if not HANGUL.search(r["ko"]) or JA.search(r["ko"]) or not JA.search(r["ja"]) or HANGUL.search(r["ja"]) \
       or det.detect_language_of(r["en"])!=Language.ENGLISH or det.detect_language_of(r["es"])!=Language.SPANISH: bad+=1; continue
    if r["en"].lower() in seen: continue
    seen.add(r["en"].lower())
    m_en, m_ko = (("mixed1","mixed1_langs"),("mixed2","mixed2_langs")) if i % 2 == 0 else (("mixed2","mixed2_langs"),("mixed1","mixed1_langs"))
    for (mk, lk), tgt in ((m_en,"en"),(m_ko,"ko")):
        txt=r[mk].strip(); langs=sorted(set(r.get(lk) or []))
        if nscripts(txt) < 2 and not (set(langs) >= {"en","es"}): continue
        if txt == r[tgt].strip(): continue
        out.append({"id": f"dev2_{n:05d}", "text": txt, "target": tgt, "reference": r[tgt].strip(), "type": "mixed", "langs": langs}); n+=1
    src_en = ["es","ja","ko"][i % 3]; src_ko = ["en","es","ja"][i % 3]
    out.append({"id": f"dev2_{n:05d}", "text": r[src_en].strip(), "target": "en", "reference": r["en"].strip(), "type": "single", "langs": [src_en]}); n+=1
    out.append({"id": f"dev2_{n:05d}", "text": r[src_ko].strip(), "target": "ko", "reference": r["ko"].strip(), "type": "single", "langs": [src_ko]}); n+=1
with open("data/dev/dev2.jsonl","w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
print("dev2 rows:", len(out), "bad tuples:", bad, collections.Counter((o["type"],o["target"]) for o in out))
