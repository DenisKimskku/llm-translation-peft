"""Download optional human parallel corpora (used only in an ablation)."""
import os, urllib.request, zipfile
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf"))
R = "data/raw"
def opus(corpus, ver, pair):
    url = f"https://object.pouta.csc.fi/OPUS-{corpus}/{ver}/moses/{pair}.txt.zip"; dst = f"{R}/{corpus}.{pair}.zip"
    if not os.path.exists(dst):
        print("GET", url, flush=True); urllib.request.urlretrieve(url, dst)
for pair in ["en-ko", "en-ja", "en-es", "ja-ko", "es-ko", "es-ja"]:
    opus("TED2020", "v1", pair)
for rid in ["traintogpb/aihub-koen-translation-integrated-large-10m", "traintogpb/aihub-koja-translation-integrated-large-4.3m",
            "ryo0634/bsd_ja_en", "hpprc/tanaka-corpus", "google/wmt24pp"]:
    print("HF", rid, flush=True)
    p = snapshot_download(rid, repo_type="dataset", max_workers=8); print("  ->", p, flush=True)
print("done")
