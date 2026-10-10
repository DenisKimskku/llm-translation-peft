import json, random, sys
seed = int(sys.argv[1]); n_per_sit = int(sys.argv[2]); out = sys.argv[3]
random.seed(seed)
sits = [json.loads(l) for l in open("data/gen/situations_raw.jsonl")]
LANGS = ["en", "es", "ko", "ja"]
matrix_w = {"ko": 0.35, "en": 0.25, "ja": 0.2, "es": 0.2}
pattern_w = {"noun": 0.5, "clause": 0.22, "phrase": 0.18, "verb": 0.10}
def wchoice(d): 
    r = random.random(); acc = 0
    for k, v in d.items():
        acc += v
        if r <= acc: return k
    return k
def spec():
    m = wchoice(matrix_w); others = [l for l in LANGS if l != m]
    emb = random.sample(others, 2 if random.random() < 0.3 else 1)
    p = wchoice(pattern_w)
    if p == "verb" and m in ("en", "es"): p = "noun"
    if p == "verb": emb = [e for e in emb if e in ("en", "es")] or ["en"]
    return f"matrix={m}; embed={','.join(emb)}; pattern={p}"
registers = ["polite spoken", "formal written (email or notice)", "neutral spoken", "casual but polite chat message"]
rows = []
for i, s in enumerate(sits):
    for k in range(n_per_sit):
        s1 = [spec() for _ in range(8)]; s2 = []
        for a in s1:
            b = spec()
            while b == a: b = spec()
            s2.append(b)
        rows.append({"seed_id": f"v2s{seed}_{i:04d}_{k}", "domain": s["_seed"]["domain"], "situation": s["situation"],
                     "roles": " and ".join(s["roles"]) if isinstance(s.get("roles"), list) else str(s.get("roles")),
                     "register": registers[(i + k + seed) % len(registers)], "two_sent": random.randint(1, 8),
                     "specs1": " | ".join(f"{j+1}:{c}" for j, c in enumerate(s1)), "specs2": " | ".join(f"{j+1}:{c}" for j, c in enumerate(s2))})
random.shuffle(rows)
open(out, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(len(rows), "seeds ->", out)
