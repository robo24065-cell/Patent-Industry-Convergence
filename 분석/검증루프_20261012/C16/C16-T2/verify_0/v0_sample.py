# -*- coding: utf-8 -*-
# verify_0: pit_map 행의 20%(최소 15) 무작위 추출, seed 20261017
import pandas as pd, numpy as np, os, sys
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_csv(os.path.join(H, "pit_map.csv"), encoding="utf-8-sig")
k = max(15, int(np.ceil(0.2 * len(m))))
rng = np.random.default_rng(20261017)
idx = sorted(rng.choice(len(m), size=k, replace=False))
s = m.iloc[idx][["트랙", "요소ID", "요소", "등급", "원고_공개판정", "원고_줄"]]
s.to_csv(os.path.join(H, "verify_0", "v0_sample.csv"), index=False, encoding="utf-8-sig")
sys.stdout.reconfigure(encoding="utf-8"); print(len(m), k); print(s[["요소ID", "등급", "원고_공개판정"]].to_string())
