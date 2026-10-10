"""Generic parallel driver for data generation with headless `claude -p`.

python scripts/gen_claude.py --prompt prompts/x.txt --seeds seeds.jsonl --out out.jsonl --workers 6 --model sonnet
The prompt file may contain {placeholders} filled from each seed row. The model must answer with a JSON array;
each element is written as one jsonl row with the seed fields merged in under "_seed". Resumable.
"""
import argparse, json, os, re, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = text.find("["); end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found")
    return json.loads(text[start:end + 1])

def call_claude(prompt: str, model: str, timeout: int = 600) -> str:
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "text"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"claude exit {r.returncode}: {r.stderr[-500:]}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True); ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="sonnet"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
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
    lock = threading.Lock(); n_ok = n_fail = n_rows = 0
    def work(seed):
        prompt = template.format(**seed)
        last = None
        for attempt in range(args.retries + 1):
            try:
                txt = call_claude(prompt, args.model)
                rows = extract_json(txt)
                if not isinstance(rows, list) or not rows: raise ValueError("empty")
                return seed, rows, None
            except Exception as e:
                last = e; time.sleep(3 * (attempt + 1))
        return seed, None, last
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex, open(args.out, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, s) for s in todo]
        for fut in as_completed(futs):
            seed, rows, err = fut.result()
            with lock:
                if rows is None:
                    n_fail += 1; print(f"FAIL seed {seed['seed_id']}: {err}", flush=True); continue
                for r in rows:
                    if isinstance(r, dict):
                        r["_seed"] = seed; f.write(json.dumps(r, ensure_ascii=False) + "\n"); n_rows += 1
                f.flush(); n_ok += 1
                if n_ok % 5 == 0:
                    el = time.time() - t0
                    print(f"ok={n_ok} fail={n_fail} rows={n_rows} elapsed={el/60:.1f}m rate={n_rows/el*60:.0f} rows/min", flush=True)
    print(f"done: ok={n_ok} fail={n_fail} rows={n_rows} in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
