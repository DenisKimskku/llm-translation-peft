"""Fallback generator: same prompt/seed files as gen_claude.py but using a local model through vLLM.
python scripts/gen_vllm.py --prompt prompts/train_tuples_v2.txt --seeds seeds.jsonl --out out.jsonl [--model Qwen/Qwen3-14B-AWQ] [--limit N]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from gen_claude import extract_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=96)
    args = ap.parse_args()
    template = open(args.prompt, encoding="utf-8").read()
    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    if args.limit: seeds = seeds[:args.limit]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            try: done.add(json.loads(l)["_seed"]["seed_id"])
            except Exception: pass
    todo = [s for s in seeds if s["seed_id"] not in done]
    print(f"{len(seeds)} seeds, {len(done)} done, {len(todo)} to do", flush=True)
    from vllm import LLM, SamplingParams
    big = "14B" in args.model
    llm = LLM(model=args.model, quantization="awq_marlin", dtype="half", max_model_len=4096,
              gpu_memory_utilization=0.98 if big else 0.9, enforce_eager=big, max_num_seqs=16 if big else 64, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=3000, stop_token_ids=[151645, 151643])
    n_ok = n_fail = n_rows = 0; t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), args.batch):
            chunk = todo[s:s + args.batch]
            prompts = [tok.apply_chat_template([{"role": "user", "content": template.format(**seed)}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) for seed in chunk]
            outs = llm.generate(prompts, sp)
            for seed, o in zip(chunk, outs):
                try:
                    rows = extract_json(o.outputs[0].text)
                    assert isinstance(rows, list) and rows
                except Exception as e:
                    n_fail += 1; continue
                for r in rows:
                    if isinstance(r, dict):
                        r["_seed"] = seed; f.write(json.dumps(r, ensure_ascii=False) + "\n"); n_rows += 1
                n_ok += 1
            f.flush()
            el = time.time() - t0
            print(f"ok={n_ok} fail={n_fail} rows={n_rows} elapsed={el/60:.1f}m rate={n_rows/el*60:.0f} rows/min", flush=True)

if __name__ == "__main__":
    main()
