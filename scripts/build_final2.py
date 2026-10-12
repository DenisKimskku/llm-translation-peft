"""Build train_final2: like train_final but Korean references in the official style:
 - natural rewrites (data/gen/konat_en1st.jsonl) replace the literal Korean reference for rows whose INPUT contains no Hangul
   (single en/es/ja -> ko, and mixed inputs without Korean spans); rows with Korean spans in the input keep the literal reference
   (the spans are copied verbatim, as in the official data);
 - natural Korean is also used as the SOURCE of ko -> en single rows;
 - commas are removed from every Korean reference (official references do not use them).
"""
import json, re, random, sys, collections, unicodedata
sys.path.insert(0, "src")
random.seed(0)
H = re.compile(r"[가-힣]")
def nocomma(s): return re.sub(r"\s+", " ", re.sub(r"\s*[,，、]\s*", " ", s)).strip()
nat = {}
for l in open("data/gen/konat_en1st.jsonl"):
    r = json.loads(l); outs = r.get("outputs"); orig = r["_seed"]["orig"]
    if isinstance(outs, list) and len(outs) == len(orig):
        for o, s in zip(outs, orig):
            if isinstance(o, str) and H.search(o) and 0.5 < len(o) / max(1, len(s)) < 1.6: nat[s] = unicodedata.normalize("NFC", o).strip()
print("natural rewrites:", len(nat))
syn = [json.loads(l) for l in open("data/train/syn_final.jsonl")]
human = [json.loads(l) for l in open("data/train/train_final.jsonl")]
human = [r for r in human if r["src"].startswith("human")]
out = []; stats = collections.Counter()
for r in syn:
    r = dict(r)
    if r["target"] == "ko":
        if not H.search(r["text"]) and r["reference"] in nat:
            r["reference"] = nat[r["reference"]]; stats["ko_ref_natural"] += 1
        r["reference"] = nocomma(r["reference"]); stats["ko_ref"] += 1
    elif r["type"] == "single" and r["langs"] == ["ko"] and r["text"] in nat:
        r["text"] = nat[r["text"]]; stats["ko_src_natural"] += 1
    out.append(r)
for r in human:
    r = dict(r)
    if r["target"] == "ko": r["reference"] = nocomma(r["reference"])
    out.append(r)
random.shuffle(out)
with open("data/train/train_final2.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "rows ->", "data/train/train_final2.jsonl", dict(stats))
