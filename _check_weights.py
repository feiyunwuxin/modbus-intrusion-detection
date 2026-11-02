import re, numpy as np, torch, sys
from pathlib import Path

HDR = "MCU/Core/Inc/model_v4_se_23dim_ch32_fp32_s42.h"
txt = open(HDR, encoding="utf-8", errors="replace").read()

arrays = {}
for m in re.finditer(r"static const float\s+(\w+)\[(\d+)\]\s*=\s*\{(.*?)\};", txt, re.S):
    name, n, body = m.group(1), int(m.group(2)), m.group(3)
    vals = [float(x[:-1]) for x in re.findall(r"[-+0-9.eE]+f", body)]
    assert len(vals)==n, (name, len(vals), n)
    arrays[name] = np.array(vals, dtype=np.float32)
print("parsed arrays:", len(arrays))
for k,v in arrays.items(): print(f"  {k:40s} {v.shape}")

# ---- .pt -> BN fold ----
ck = torch.load("model_v4_se_23dim_b64_ch32_do01_window16_s42.pt", map_location="cpu", weights_only=False)
sd = ck["state_dict"]
print("\n.pt state_dict keys:")
for k,v in sd.items(): print(f"  {k:50s} {tuple(v.shape)}")
