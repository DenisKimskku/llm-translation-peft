"""LoRA/DoRA SFT of frozen Qwen3-1.7B for code-switched translation.

python src/train.py --config configs/lora_5m.yaml --train data/train/train.jsonl --out outputs/run5m [--dev data/dev/dev.jsonl]
Train rows: {"text": ..., "target": "en"|"ko", "reference": ...}. Loss is computed on the completion only.
"""
import argparse, json, math, os, random, sys, time
import yaml, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BACKBONE, DEFAULT_TEMPLATE, build_prompt, read_jsonl

IM_END = "<|im_end|>"

class SFTDataset(Dataset):
    def __init__(self, rows, tok, template, max_len, system=None):
        self.items = []; dropped = 0
        for r in rows:
            p = build_prompt(tok, r["text"], r["target"], template, system)
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            c_ids = tok(str(r["reference"]).strip() + IM_END, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_len:
                dropped += 1; continue
            self.items.append((p_ids, c_ids))
        print(f"dataset: {len(self.items)} examples ({dropped} dropped > {max_len} tokens)")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, c = self.items[i]
        return {"input_ids": p + c, "labels": [-100] * len(p) + c}

class Collator:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), L), -100, dtype=torch.long)
        att = torch.zeros((len(feats), L), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"]); lab[i, :n] = torch.tensor(f["labels"]); att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-adapter", default=None, help="continue training from an adapter dir")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    template = cfg.get("template", DEFAULT_TEMPLATE); system = cfg.get("system", None)
    rows = read_jsonl(args.train)
    if args.max_train: random.shuffle(rows); rows = rows[:args.max_train]
    train_ds = SFTDataset(rows, tok, template, cfg.get("max_len", 384), system)
    dev_ds = SFTDataset(read_jsonl(args.dev), tok, template, cfg.get("max_len", 384), system) if args.dev else None

    model = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.bfloat16, attn_implementation=cfg.get("attn", "sdpa"))
    model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)   # frozen backbone
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        pc = cfg["peft"]
        lcfg = LoraConfig(task_type="CAUSAL_LM", r=pc["r"], lora_alpha=pc["alpha"], lora_dropout=pc.get("dropout", 0.0),
                          target_modules=pc["target_modules"], use_dora=pc.get("dora", False), use_rslora=pc.get("rslora", False),
                          rank_pattern=pc.get("rank_pattern", {}), alpha_pattern=pc.get("alpha_pattern", {}),
                          layers_to_transform=pc.get("layers_to_transform", None), init_lora_weights=pc.get("init", True),
                          bias="none")
        model = get_peft_model(model, lcfg)
    n_tr = count_trainable(model); budget = cfg.get("budget", 5_000_000)
    print(f"trainable params: {n_tr:,} (budget {budget:,})")
    assert n_tr <= budget, "over budget!"
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    tr = cfg["train"]
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=tr.get("batch_size", 8), gradient_accumulation_steps=tr.get("grad_accum", 2),
        per_device_eval_batch_size=16, num_train_epochs=tr.get("epochs", 2), learning_rate=float(tr.get("lr", 2e-4)),
        lr_scheduler_type=tr.get("scheduler", "cosine"), warmup_ratio=tr.get("warmup_ratio", 0.03), weight_decay=tr.get("weight_decay", 0.0),
        bf16=True, logging_steps=tr.get("logging_steps", 25), save_strategy="epoch", save_total_limit=3,
        eval_strategy="epoch" if dev_ds else "no", group_by_length=True, dataloader_num_workers=2,
        report_to=[], seed=args.seed, max_grad_norm=tr.get("max_grad_norm", 1.0), optim=tr.get("optim", "adamw_torch_fused"),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=Collator(tok.pad_token_id))
    # LoRA+ style: higher LR for B matrices (optional)
    if tr.get("loraplus_ratio"):
        ratio = float(tr["loraplus_ratio"]); base_lr = float(tr.get("lr", 2e-4))
        a_params = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n)]
        b_params = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_B" in n)]
        o_params = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" not in n and "lora_B" not in n)]
        groups = [{"params": a_params, "lr": base_lr}, {"params": b_params, "lr": base_lr * ratio}]
        if o_params: groups.append({"params": o_params, "lr": base_lr})
        trainer.optimizer = torch.optim.AdamW(groups, lr=base_lr, weight_decay=tr.get("weight_decay", 0.0), fused=True)
    t0 = time.time()
    trainer.train()
    print(f"train time: {(time.time()-t0)/60:.1f} min")
    model.save_pretrained(os.path.join(args.out, "adapter"))
    json.dump({"trainable_params": n_tr, "config": cfg, "train_file": args.train, "n_train": len(train_ds)},
              open(os.path.join(args.out, "adapter", "train_meta.json"), "w"), indent=1, ensure_ascii=False)
    print("saved adapter to", os.path.join(args.out, "adapter"))

if __name__ == "__main__":
    main()
