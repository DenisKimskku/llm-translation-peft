import json, random, sys
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
sits = [json.loads(l) for l in open("data/gen/situations_raw.jsonl")]
pairs = ["en+ko", "en+ja", "en+es", "es+ko", "es+ja", "ko+ja"]
triples = ["en+es+ko", "en+ko+ja", "es+ko+ja", "en+es+ja"]
def combo():
    return random.choice(pairs) if random.random() < 0.7 else random.choice(triples)
registers = ["polite spoken", "formal written (email or notice)", "neutral spoken", "casual but polite chat message"]
rows = []
n_per_sit = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for i, s in enumerate(sits):
    for k in range(n_per_sit):
        c1 = [combo() for _ in range(8)]
        c2 = []
        for c in c1:
            alt = combo()
            while alt == c: alt = combo()
            c2.append(alt)
        rows.append({"seed_id": f"tr{i:04d}_{k}", "domain": s["_seed"]["domain"], "situation": s["situation"],
                     "roles": " and ".join(s.get("roles", ["two people"])) if isinstance(s.get("roles"), list) else str(s.get("roles")),
                     "register": registers[(i + k) % len(registers)], "two_sent": random.randint(1, 8),
                     "combos1": "; ".join(f"{j+1}:{c}" for j, c in enumerate(c1)), "combos2": "; ".join(f"{j+1}:{c}" for j, c in enumerate(c2))})
random.shuffle(rows)
out = sys.argv[3] if len(sys.argv) > 3 else "data/gen/seeds_train.jsonl"
open(out, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(len(rows), "train seeds ->", out)
