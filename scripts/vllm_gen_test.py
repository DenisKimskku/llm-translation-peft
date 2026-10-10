import time, json, sys
from vllm import LLM, SamplingParams
MODEL = "Qwen/Qwen3-14B-AWQ"
llm = LLM(model=MODEL, quantization="awq_marlin", dtype="half", max_model_len=1536, gpu_memory_utilization=0.98,
          enforce_eager=True, max_num_seqs=48, enable_prefix_caching=True)
tok = llm.get_tokenizer()
sents = [
 "Please transfer the remaining balance from your savings account to the checking account before the bank closes at five o'clock today.",
 "The doctor ordered a blood test and an X-ray because the patient has had a fever and a cough for three days.",
 "Passengers traveling to Tokyo must show their passports and boarding passes at gate twelve before the flight departs at nine thirty.",
 "We need to finalize the budget before the board meeting next week.",
 "The shipment was delayed at customs because the invoice did not match the packing list.",
 "Could you send me the revised slides after the meeting ends?",
 "Our team will migrate the database to the new server on Saturday night to avoid downtime.",
 "The professor extended the assignment deadline to Friday because the lab server was down.",
] * 8
instr = ("Translate the following English sentence into Korean, Spanish, and Japanese. Translate literally and faithfully; keep all details. "
         "Korean: polite formal register (-습니다/-세요). Japanese: polite register (です/ます). "
         "Answer ONLY with a JSON object with keys \"ko\", \"es\", \"ja\".\n\nEnglish: ")
prompts = [tok.apply_chat_template([{"role":"user","content": instr + s}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for s in sents]
sp = SamplingParams(temperature=0.0, max_tokens=300)
t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"\n=== {len(prompts)} prompts in {dt:.1f}s: {ntok/dt:.0f} gen tok/s, {len(prompts)/dt*60:.0f} prompts/min ===")
for o in outs[:8]:
    print(o.outputs[0].text.strip()[:400]); print("---")
