"""Probe which reference style the official validation uses: translate N validation inputs with Claude in a literal vs a natural
style and score each against the references (model/prompt selection on validation inputs only; references never used as input)."""
import json, random, subprocess, sys, re
from sacrebleu.metrics import CHRF
random.seed(0)
V=[json.loads(l) for l in open("data/dev/official_validation.jsonl")]
sample=[]
for it in ("single","mixed"):
    for tl in ("en","ko"):
        sample += random.sample([d for d in V if d["input_type"]==it and d["target_language"]==tl], 10)
STYLES={
 "literal": "Translate literally and faithfully, preserving the sentence structure and every detail; do not paraphrase.",
 "natural": "Translate the way a skilled professional translator writes for native readers: natural, idiomatic, concise, with the most common phrasing a native speaker would use; keep all information but feel free to restructure the sentence.",
}
def call(prompt):
    r=subprocess.run(["claude","-p",prompt,"--model","sonnet","--output-format","text"],capture_output=True,text=True,timeout=600,stdin=subprocess.DEVNULL)
    m=re.search(r"\[.*\]", r.stdout, re.S); return json.loads(m.group(0))
chrf=CHRF(word_order=2); res={}
for sname, sinstr in STYLES.items():
    outs={}
    for i in range(0,len(sample),10):
        chunk=sample[i:i+10]
        items="\n".join(f'{j+1}. (target {d["target_language"]}) {d["input"]}' for j,d in enumerate(chunk))
        prompt=(f"Translate each numbered text into its target language (en = English, ko = Korean, polite formal register). The texts may mix "
                f"English, Spanish, Korean and Japanese; translate everything into the target language. {sinstr}\n"
                f"Output ONLY a JSON array of {len(chunk)} strings in order.\n\n{items}")
        arr=call(prompt)
        for d,o in zip(chunk,arr): outs[d["id"]]=o
    res[sname]=outs
for sname in STYLES:
    for it in ("single","mixed"):
        for tl in ("en","ko"):
            g=[d for d in sample if d["input_type"]==it and d["target_language"]==tl]
            sc=chrf.corpus_score([res[sname][d["id"]] for d in g],[[d["answer"] for d in g]]).score
            print(f"{sname:8s} {it}->{tl}: {sc:5.1f}")
json.dump({"sample":[d["id"] for d in sample],"res":res},open("outputs/official/style_probe.json","w"),ensure_ascii=False)
