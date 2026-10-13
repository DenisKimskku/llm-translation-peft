"""Average the LoRA tensors of two (or more) adapter checkpoints from the same run. python scripts/avg_adapters.py OUT CKPT1 CKPT2 [...]"""
import sys, os, shutil, torch
from safetensors.torch import load_file, save_file
out, srcs = sys.argv[1], sys.argv[2:]
tensors = [load_file(os.path.join(s, "adapter_model.safetensors")) for s in srcs]
avg = {k: sum(t[k].float() for t in tensors) / len(tensors) for k in tensors[0]}
avg = {k: v.to(tensors[0][k].dtype) for k, v in avg.items()}
os.makedirs(out, exist_ok=True); shutil.copy(os.path.join(srcs[0], "adapter_config.json"), out)
save_file(avg, os.path.join(out, "adapter_model.safetensors")); print("averaged", len(srcs), "adapters ->", out, len(avg), "tensors")
