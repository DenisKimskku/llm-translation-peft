"""Greedy, non-thinking inference with (optional) PEFT adapter.

python src/infer.py --input dev.jsonl --out preds.jsonl [--adapter outputs/run/adapter] [--engine hf|vllm] [--template-file prompts/t.txt]
Input rows need: text (or input), target (en|ko); id is preserved if present.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BACKBONE, DEFAULT_TEMPLATE, build_prompt, read_jsonl, write_jsonl

def get(row, *names):
    for n in names:
        if n in row and row[n] is not None: return row[n]
    raise KeyError(names)

def run_hf(prompts, adapter, batch_size, max_new_tokens, merge):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BACKBONE); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge: model = model.merge_and_unload()
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outs = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                               eos_token_id=[151645, 151643], pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
        g = g[:, enc["input_ids"].shape[1]:]
        for i, row in zip(idx, g):
            outs[i] = tok.decode(row, skip_special_tokens=True).strip()
        print(f"  {min(s+batch_size, len(order))}/{len(order)}", end="\r", flush=True)
    return outs

def run_vllm(prompts, adapter, max_new_tokens, max_lora_rank):
    from vllm import LLM, SamplingParams
    kw = dict(model=BACKBONE, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.85, enable_prefix_caching=True)
    if adapter: kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=[151645, 151643])
    if adapter:
        from vllm.lora.request import LoRARequest
        outs = llm.generate(prompts, sp, lora_request=LoRARequest("adapter", 1, adapter))
    else:
        outs = llm.generate(prompts, sp)
    return [o.outputs[0].text.strip() for o in outs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None); ap.add_argument("--engine", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--template-file", default=None); ap.add_argument("--system", default=None)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--merge", action="store_true"); ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    template = open(args.template_file, encoding="utf-8").read().strip("\n") if args.template_file else DEFAULT_TEMPLATE
    if args.adapter and os.path.exists(os.path.join(args.adapter, "train_meta.json")) and not args.template_file:
        meta = json.load(open(os.path.join(args.adapter, "train_meta.json")))
        template = meta["config"].get("template", DEFAULT_TEMPLATE); args.system = args.system or meta["config"].get("system")
    rows = read_jsonl(args.input)
    if args.limit: rows = rows[:args.limit]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    prompts = [build_prompt(tok, get(r, "text", "input", "source", "src"), get(r, "target", "target_lang", "target_language", "tgt"), template, args.system) for r in rows]
    t0 = time.time()
    if args.engine == "hf":
        preds = run_hf(prompts, args.adapter, args.batch_size, args.max_new_tokens, args.merge)
    else:
        preds = run_vllm(prompts, args.adapter, args.max_new_tokens, args.max_lora_rank)
    print(f"\ninference: {len(rows)} rows in {time.time()-t0:.1f}s")
    out = []
    for r, p in zip(rows, preds):
        o = {"id": r.get("id")} if "id" in r else {}
        o.update({"text": get(r, "text", "input", "source", "src"), "target": get(r, "target", "target_lang", "target_language", "tgt"), "prediction": p})
        out.append(o)
    write_jsonl(args.out, out)

if __name__ == "__main__":
    main()
