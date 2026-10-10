"""Tuple-level round-trip filter (English pivot): back-translate each mixed variant into English with a local model
(Qwen3-14B-AWQ via vLLM) and score chrF++ against the tuple's own "en" field. Adds mixed1_rt / mixed2_rt (+ rt texts).

python scripts/roundtrip_filter.py --in data/gen/train_tuples_w1.jsonl --out data/gen/train_tuples_w1.rt.jsonl
"""
import argparse, json, sys, os, time
sys.path.insert(0, "src")
from common import read_jsonl, write_jsonl
from sacrebleu.metrics import CHRF

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ"); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = read_jsonl(args.inp)
    if args.limit: rows = rows[:args.limit]
    done = {}
    if os.path.exists(args.out):
        for r in read_jsonl(args.out): done[r.get("en", "")] = r
    todo = [r for r in rows if isinstance(r.get("en"), str) and r["en"] not in done]
    print(f"{len(rows)} tuples, {len(done)} already scored, {len(todo)} to do")
    from vllm import LLM, SamplingParams
    big = "14B" in args.model
    llm = LLM(model=args.model, quantization="awq_marlin", dtype="half", max_model_len=1024,
              gpu_memory_utilization=0.98 if big else 0.9, enforce_eager=big, max_num_seqs=48 if big else 128, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    def prompt(text):
        msg = ("Translate the following text into English. The text mixes several languages; translate ALL of it into English, "
               "literally and completely, keeping every detail, number and name. Output only the translation.\n\nText: " + text)
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    jobs = []
    for r in todo:
        for k in ("mixed1", "mixed2"):
            if isinstance(r.get(k), str) and len(r[k]) > 3: jobs.append((r, k))
    t0 = time.time()
    outs = llm.generate([prompt(r[k]) for r, k in jobs], SamplingParams(temperature=0.0, max_tokens=200, stop_token_ids=[151645, 151643]))
    print(f"back-translated {len(jobs)} variants in {time.time()-t0:.0f}s")
    chrf = CHRF(word_order=2)
    for (r, k), o in zip(jobs, outs):
        bt = o.outputs[0].text.strip()
        r[k + "_rt_text"] = bt; r[k + "_rt"] = round(chrf.sentence_score(bt, [r["en"]]).score, 2)
    write_jsonl(args.out, list(done.values()) + todo)
    import statistics
    sc = [r[k] for r in todo for k in ("mixed1_rt", "mixed2_rt") if k in r]
    print("rt mean", round(statistics.mean(sc), 2), "median", statistics.median(sc), "<40:", sum(s < 40 for s in sc), "<50:", sum(s < 50 for s in sc), "of", len(sc))

if __name__ == "__main__":
    main()
