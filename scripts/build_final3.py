"""train_final3 = all waves (w1, v2a, v2ko, v3) through build_dataset (RT-filtered) + natural-Korean substitution (as build_final2)
+ comma-free Korean references + 40k human rows (same subset as train_final)."""
import json, re, random, sys, subprocess, collections, unicodedata
random.seed(0)
H = re.compile(r"[\uac00-\ud7a3]")
def nocomma(s): return re.sub(r"\s+", " ", re.sub(r"\s*[,\uff0c\u3001]\s*", " ", s)).strip()
rt = [f"data/gen/train_tuples_{w}.rt.jsonl" for w in ("w1", "v2a", "v2ko", "v3")]
subprocess.run([sys.executable, "src/build_dataset.py", "--tuples", *rt, "--out", "data/train/syn_final3.jsonl", "--single-ratio", "0.3",
                "--both-targets", "--identity", "0.05", "--min-rt", "40", "--dedup-against", "data/dev/dev.jsonl", "data/dev/dev2.jsonl",
                "data/dev/official_validation.jsonl", "data/dev/official_test.jsonl"], check=True)
nat = {}
for l in open("data/gen/konat_en1st.jsonl"):
    r = json.loads(l); outs = r.get("outputs"); orig = r["_seed"]["orig"]
    if isinstance(outs, list) and len(outs) == len(orig):
        for o, s in zip(outs, orig):
            if isinstance(o, str) and H.search(o) and 0.5 < len(o) / max(1, len(s)) < 1.6: nat[s] = unicodedata.normalize("NFC", o).strip()
syn = [json.loads(l) for l in open("data/train/syn_final3.jsonl")]
human = [json.loads(l) for l in open("data/train/train_final.jsonl")]; human = [r for r in human if r["src"].startswith("human")]
out = []; st = collections.Counter()
for r in syn:
    r = dict(r)
    if r["target"] == "ko":
        if not H.search(r["text"]) and r["reference"] in nat: r["reference"] = nat[r["reference"]]; st["ko_ref_natural"] += 1
        r["reference"] = nocomma(r["reference"])
    elif r["type"] == "single" and r["langs"] == ["ko"] and r["text"] in nat: r["text"] = nat[r["text"]]; st["ko_src_natural"] += 1
    out.append(r)
for r in human:
    r = dict(r)
    if r["target"] == "ko": r["reference"] = nocomma(r["reference"])
    out.append(r)
random.shuffle(out)
with open("data/train/train_final3.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(syn), "synthetic +", len(human), "human =", len(out), "rows -> data/train/train_final3.jsonl", dict(st))
print(collections.Counter((r["type"], r["target"]) for r in out))
