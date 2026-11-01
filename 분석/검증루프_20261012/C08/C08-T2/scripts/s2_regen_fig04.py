# 샌드박스(cache\regen)에서 fig04만 다시 그리고 figs\ PNG와 픽셀 비교
import sys, os, hashlib
sys.stdout.reconfigure(encoding="utf-8")
R = sys.argv[1]; FIGS = sys.argv[2]
sys.path.insert(0, R); os.chdir(R)
import make_data_figs as m
m.fig04()
import numpy as np
from PIL import Image
a = np.asarray(Image.open(os.path.join(R, "fig04_within_code_divergence.png")).convert("RGB")).astype(int)
b = np.asarray(Image.open(os.path.join(FIGS, "fig04_within_code_divergence.png")).convert("RGB")).astype(int)
print("shape", a.shape, b.shape)
if a.shape == b.shape:
    d = np.abs(a - b).max(axis=2)
    print("diff_pixels", int((d > 0).sum()), "max", int(d.max()))
for p in [os.path.join(R, "fig04_within_code_divergence.png"), os.path.join(FIGS, "fig04_within_code_divergence.png"),
          os.path.join(R, "fig04_within_code_divergence.csv"), os.path.join(FIGS, "fig04_within_code_divergence.csv")]:
    print(hashlib.sha1(open(p, "rb").read()).hexdigest()[:12], p)
