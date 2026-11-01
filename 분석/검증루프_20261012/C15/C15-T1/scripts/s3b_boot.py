# -*- coding: utf-8 -*-
"""C15-T1 s3b: 공식 부트스트랩 방식(1순위 출원인코드 군집, 없으면 특허; B=1000; seed 0; 가중 AUC 동점 0.5)으로 민감도 CI.
사용: python -I s3b_boot.py <S0|S2|S3|S4>  → cache/boot_<S>.json"""
import os, sys, json, time
import numpy as np, pandas as pd
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
Z = np.load(os.path.join(W, "cache", "s3a_arrays.npz"), allow_pickle=True)
pm, pb, y = Z["pm"], Z["pb"], Z["y"]
acd, pid = Z["acd"], Z["pid"]
w0 = np.ones(len(y))
if S == "S2":
    k = Z["keep2"]; pm, pb, y, acd, pid, w0 = pm[k], pb[k], y[k], acd[k], pid[k], w0[k]
elif S == "S3":
    y3 = Z["y3"]; k = y3 >= 0; pm, pb, y, acd, pid, w0 = pm[k], pb[k], y3[k], acd[k], pid[k], w0[k]
elif S == "S4":
    w0 = Z["w4"]
sk = lambda a, b: (a - b) / (1 - b)


class WA:
    def __init__(self, yy, s):
        o = np.argsort(s, kind="stable"); ss = s[o]
        self.o = o; self.y = yy[o].astype(float)
        self.g = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]); self.G = int(self.g[-1]) + 1

    def __call__(self, w):
        wo = w[self.o]
        Pg = np.bincount(self.g, wo * self.y, self.G); Ng = np.bincount(self.g, wo * (1 - self.y), self.G)
        below = np.cumsum(Ng) - Ng
        return float((Pg * (below + 0.5 * Ng)).sum() / (Pg.sum() * Ng.sum()))


cl = np.where(acd == "NA", np.char.add("P", pid.astype(str)), acd)
codes, uniq = pd.factorize(cl); K = len(uniq)
a1, a0 = WA(y, pm), WA(y, pb)
point = dict(auc=a1(w0), auc_base=a0(w0)); point["skill"] = sk(point["auc"], point["auc_base"])
rng = np.random.default_rng(0); s = []; aa = []
for b in range(B):
    w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes] * w0
    x1, x0 = a1(w), a0(w); s.append(sk(x1, x0)); aa.append(x1)
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
R = dict(scenario=S, n=int(len(y)), pos=int(y.sum()), clusters=int(K), B=B, seed=0, point=point, skill_ci=q(s), auc_ci=q(aa), secs=round(time.time() - t0, 1))
json.dump(R, open(os.path.join(W, "cache", f"boot_{S}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False))
