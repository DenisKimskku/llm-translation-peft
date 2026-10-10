"""Build an optional pool of human-translated single-language rows (for an ablation only).
Sources: AI Hub ko-en (daily/general/specialized/broadcast), AI Hub ko-ja (non-patent), BSD ja-en, Tanaka ja-en, TED2020 (es-en, es-ko, ja-ko).
Output rows: text, target, reference, type=single, langs, src=human:<corpus>
"""
import os, glob, json, random, re, zipfile, unicodedata
import pandas as pd
random.seed(0)
H = os.path.expanduser("~/.cache/huggingface/hub")
HANGUL = re.compile(r"[가-힣]"); CJK = re.compile(r"[぀-ヿ一-鿿]"); LAT = re.compile(r"[A-Za-zÀ-ÿ]")
def clean(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()
def ok_len(a, b, lo=20, hi=200): return lo <= len(a) <= hi and lo <= len(b) <= hi and 0.3 < len(a) / max(1, len(b)) < 3.5
def ok_ko(s): return bool(HANGUL.search(s)) and not CJK.search(s)
def ok_ja(s): return bool(CJK.search(s)) and not HANGUL.search(s)
def ok_lat(s): return bool(LAT.search(s)) and not HANGUL.search(s) and not CJK.search(s)
rows = []
def add(text, ref, tgt, src_lang, src):
    rows.append({"text": text, "target": tgt, "reference": ref, "type": "single", "langs": [src_lang], "src": f"human:{src}"})

# AI Hub ko-en
f = glob.glob(H + "/datasets--traintogpb--aihub-koen-translation-integrated-large-10m/snapshots/*/train.csv")[0]
want = {71265: 12000, 126: 6000, 111: 6000, 71382: 3000}
df = pd.read_csv(f, nrows=2500000)
for src, n in want.items():
    sub = df[df.source == src].sample(frac=1, random_state=1)
    k = 0
    for _, r in sub.iterrows():
        ko, en = clean(r["ko"]), clean(r["en"])
        if not (ok_ko(ko) and ok_lat(en) and ok_len(ko, en, 15, 160)): continue
        if re.search(r"\d{4}\.\d|\[|\]|\(\d+\)", en): continue   # patent-like / citations
        if k % 2 == 0: add(ko, en, "en", "ko", f"aihub{src}")
        else: add(en, ko, "ko", "en", f"aihub{src}")
        k += 1
        if k >= n: break
    print("aihub ko-en", src, k)
# AI Hub ko-ja (skip patent-heavy sources by inspecting source ids)
f = glob.glob(H + "/datasets--traintogpb--aihub-koja-translation-integrated-large-4.3m/snapshots/*/train.csv")[0]
dj = pd.read_csv(f, nrows=1500000); print("ko-ja sources:", dj["source"].value_counts().head(8).to_dict())
k = 0
for _, r in dj.sample(frac=1, random_state=2).iterrows():
    ko, ja = clean(r["ko"]), clean(r["ja"])
    if not (ok_ko(ko) and ok_ja(ja) and ok_len(ko, ja, 15, 140)): continue
    if re.search(r"특허|청구항|실시예|상기 ", ko): continue
    add(ja, ko, "ko", "ja", "aihub_koja"); k += 1
    if k >= 8000: break
print("aihub ko-ja", k)
# BSD ja-en (business dialogues)
k = 0
for f in glob.glob(H + "/datasets--ryo0634--bsd_ja_en/snapshots/*/**/train-*.parquet", recursive=True):
    b = pd.read_parquet(f)
    for _, r in b.iterrows():
        ja, en = clean(r["ja_sentence"]), clean(r["en_sentence"])
        if ok_ja(ja) and ok_lat(en) and ok_len(ja, en, 10, 160): add(ja, en, "en", "ja", "bsd"); k += 1
print("bsd", k)
# Tanaka ja-en
k = 0
for f in glob.glob(H + "/datasets--hpprc--tanaka-corpus/snapshots/*/**/train-*.parquet", recursive=True):
    t = pd.read_parquet(f); cols = t.columns.tolist(); print("tanaka cols", cols)
    jc = [c for c in cols if "ja" in c.lower()][0]; ec = [c for c in cols if "en" in c.lower()][0]
    for _, r in t.sample(frac=1, random_state=3).iterrows():
        ja, en = clean(r[jc]), clean(r[ec])
        if ok_ja(ja) and ok_lat(en) and ok_len(ja, en, 15, 120): add(ja, en, "en", "ja", "tanaka"); k += 1
        if k >= 4000: break
print("tanaka", k)
# TED2020 pairs
def ted(pair, n, a, b, tgt, src_lang):
    z = zipfile.ZipFile(f"data/raw/TED2020.{pair}.zip"); names = z.namelist()
    fa = [x for x in names if x.endswith("." + a)][0]; fb = [x for x in names if x.endswith("." + b)][0]
    A = z.read(fa).decode("utf-8", "ignore").splitlines(); B = z.read(fb).decode("utf-8", "ignore").splitlines()
    idx = list(range(len(A))); random.shuffle(idx); k = 0
    for i in idx:
        s, t = clean(A[i]), clean(B[i])
        chk = {"ko": ok_ko, "ja": ok_ja, "en": ok_lat, "es": ok_lat}
        if not (chk[a](s) and chk[b](t) and ok_len(s, t, 25, 160)): continue
        if s.startswith("(") or "♪" in s or "..." in s: continue
        add(s, t, tgt, src_lang, f"ted_{pair}"); k += 1
        if k >= n: break
    print("ted", pair, k)
ted("es-ko", 6000, "es", "ko", "ko", "es")
ted("en-es", 4000, "es", "en", "en", "es")
ted("ja-ko", 3000, "ja", "ko", "ko", "ja")
ted("en-ko", 3000, "ko", "en", "en", "ko")
random.shuffle(rows)
os.makedirs("data/train", exist_ok=True)
with open("data/train/human_pool.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(rows), "rows", Counter(r["src"] for r in rows), Counter((r["langs"][0], r["target"]) for r in rows))
