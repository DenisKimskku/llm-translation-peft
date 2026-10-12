import json, random, sys
seed = int(sys.argv[1]); out = sys.argv[2]
random.seed(seed)
sits = [json.loads(l) for l in open("data/gen/situations_raw.jsonl")]
LANGS = ["en", "es", "ko", "ja"]; NAMES = {"en": "English", "es": "Spanish", "ko": "Korean", "ja": "Japanese"}
matrix_w = {"ko": 0.3, "en": 0.3, "ja": 0.2, "es": 0.2}
pattern_w = {"noun": 0.5, "clause": 0.2, "phrase": 0.2, "verb": 0.1}
n_embed_w = {1: 0.45, 2: 0.35, 3: 0.2}
def wchoice(d):
    r = random.random(); acc = 0
    for k, v in d.items():
        acc += v
        if r <= acc: return k
    return k
def spec():
    m = wchoice(matrix_w); others = [l for l in LANGS if l != m]
    emb = random.sample(others, wchoice(n_embed_w)); p = wchoice(pattern_w)
    if p == "verb" and m in ("en", "es"): p = "noun"
    if p == "verb" and not any(e in ("en", "es") for e in emb): emb = [random.choice(["en", "es"])] + emb[:1]
    return f"matrix={m}; embed={','.join(emb)}; pattern={p}"
rows = []
for i, s in enumerate(sits):
    src = "es" if i % 2 == 0 else "ja"
    s1 = [spec() for _ in range(8)]; s2 = []
    for a in s1:
        b = spec()
        while b == a: b = spec()
        s2.append(b)
    rows.append({"seed_id": f"v3{src}_{i:04d}", "domain": s["_seed"]["domain"], "situation": s["situation"],
                 "roles": " and ".join(s["roles"]) if isinstance(s.get("roles"), list) else str(s.get("roles")),
                 "source_code": src, "source_name": NAMES[src],
                 "specs1": " | ".join(f"{j+1}:{c}" for j, c in enumerate(s1)), "specs2": " | ".join(f"{j+1}:{c}" for j, c in enumerate(s2))})
random.shuffle(rows)
open(out, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
import collections
print(len(rows), "seeds ->", out, collections.Counter(r["source_code"] for r in rows))
