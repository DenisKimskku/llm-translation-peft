import json, re, sys
rows=[json.loads(l) for l in open("data/dev/dev_tuples_raw.jsonl")]
HANGUL=re.compile(r"[가-힣]"); JA=re.compile(r"[぀-ヿ一-鿿]"); LATIN=re.compile(r"[A-Za-zÀ-ÿ]")
def nscripts(s): return sum(bool(p.search(s)) for p in (HANGUL, JA, LATIN))
out=[]; n=0
for i, r in enumerate(rows):
    m_en, m_ko = (("mixed1","mixed1_langs"),("mixed2","mixed2_langs")) if i % 2 == 0 else (("mixed2","mixed2_langs"),("mixed1","mixed1_langs"))
    for (mk, lk), tgt in ((m_en,"en"),(m_ko,"ko")):
        txt=r[mk].strip(); langs=sorted(set(r.get(lk) or []))
        if nscripts(txt) < 2 and not (set(langs) >= {"en","es"}): continue
        out.append({"id": f"dev{n:05d}", "text": txt, "target": tgt, "reference": r[tgt].strip(), "type": "mixed", "langs": langs, "domain": r["_seed"]["domain"]}); n+=1
    src_en = ["es","ja","ko"][i % 3]; src_ko = ["en","es","ja"][i % 3]
    out.append({"id": f"dev{n:05d}", "text": r[src_en].strip(), "target": "en", "reference": r["en"].strip(), "type": "single", "langs": [src_en], "domain": r["_seed"]["domain"]}); n+=1
    out.append({"id": f"dev{n:05d}", "text": r[src_ko].strip(), "target": "ko", "reference": r["ko"].strip(), "type": "single", "langs": [src_ko], "domain": r["_seed"]["domain"]}); n+=1
with open("data/dev/dev.jsonl","w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
import collections
print(len(out), "dev examples", collections.Counter((o["type"],o["target"]) for o in out))
