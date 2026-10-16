"""Scale a LoRA adapter's update toward the frozen backbone: lora_B *= s (so B.A -> s*B.A). No training.
usage: scale_adapter.py <src_adapter_dir> <dst_dir> <scale>"""
import sys, shutil, os
from safetensors.torch import load_file, save_file
src, dst, s = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.makedirs(dst, exist_ok=True)
t = load_file(os.path.join(src, "adapter_model.safetensors"))
n = 0
for k in t:
    if "lora_B" in k: t[k] = t[k] * s; n += 1
save_file(t, os.path.join(dst, "adapter_model.safetensors"))
shutil.copy(os.path.join(src, "adapter_config.json"), dst)
print(f"scaled {n} lora_B tensors by {s} -> {dst}")
