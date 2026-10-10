"""GPU sanity check: baseline generation + LoRA memory/throughput probes."""
import time, torch, sys
sys.path.insert(0, "src")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from common import BACKBONE, build_prompt

tok = AutoTokenizer.from_pretrained(BACKBONE); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
print("loaded; VRAM GB:", round(torch.cuda.memory_allocated()/1e9, 2))
examples = [
    ("Aunque el tren llegó tarde, 우리는 회의에 늦지 않았습니다.", "en"),
    ("The meetingが終わったら、send me the revised slides.", "ko"),
    ("El informe의 마지막 section을 내일까지 수정해주세요.", "en"),
    ("일본 주식을 매수하고 싶은데, 東京市場での取引手数料と決済日は 어떻게 되나요?", "en"),
    ("환자분은 어제부터 chest pain이 있었고, 오늘 morning에는 dizziness도 호소했습니다.", "en"),
    ("El informe trimestral debe entregarse antes del viernes.", "ko"),
    ("来週の火曜日に新しいオフィスへ引っ越します。", "en"),
    ("We need to finalize the budget before the board meeting next week.", "ko"),
]
prompts = [build_prompt(tok, t, l) for t, l in examples]
enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
t0 = time.time()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None,
                         eos_token_id=[151645, 151643], pad_token_id=tok.pad_token_id)
dt = time.time() - t0
gen = out[:, enc["input_ids"].shape[1]:]
ntok = (gen != tok.pad_token_id).sum().item()
for (t, l), g in zip(examples, gen):
    print(f"[{l}] {t}\n  -> {tok.decode(g, skip_special_tokens=True).strip()}")
print(f"baseline batch of {len(examples)}: {dt:.1f}s, {ntok/dt:.1f} new tok/s; peak VRAM GB:", round(torch.cuda.max_memory_allocated()/1e9, 2))

# ---- LoRA probes ----
ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
def count(cfg_kwargs):
    base = AutoModelForCausalLM.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    m = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **cfg_kwargs))
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return m, tr
for name, kw in [("r4-all", dict(r=4, lora_alpha=8, target_modules=ALL)),
                 ("r4-all-dora", dict(r=4, lora_alpha=8, target_modules=ALL, use_dora=True)),
                 ("r44-all", dict(r=44, lora_alpha=88, target_modules=ALL)),
                 ("r45-all", dict(r=45, lora_alpha=90, target_modules=ALL))]:
    del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, tr = count(kw)
    print(f"{name}: trainable={tr:,}")
    if name in ("r4-all", "r44-all"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads(); model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        B, L = 8, 256
        x = torch.randint(100, 150000, (B, L), device="cuda")
        torch.cuda.synchronize(); t0 = time.time()
        for step in range(4):
            out = model(input_ids=x, labels=x); out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = (time.time() - t0) / 4
        print(f"  train step B={B} L={L}: {dt:.2f}s/step = {B*L/dt:.0f} tok/s; peak VRAM GB:", round(torch.cuda.max_memory_allocated()/1e9, 2))
        del opt
