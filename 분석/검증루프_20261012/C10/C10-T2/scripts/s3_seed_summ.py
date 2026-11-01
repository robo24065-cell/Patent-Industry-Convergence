# -*- coding: utf-8 -*-
r"""[3] 보조(판정 외, 축소판): P50−FREEIPC50, P50−HSC1_50의 seed 변동. 구성 = 공식(k0,p0) + (k0,p1)(k0,p2)(k1,p0)(k2,p0).
P50 예측은 같은 학습 집합·seed의 [1] 결과를 재사용(같은 행·같은 seed → 같은 모형). 점추정만 → res/s3_seed.csv, s3_seed.json"""
import sys
sys.dont_write_bytecode = True
from c10_common import *

CF = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]
rows = []
for k, p in CF:
    d = {}
    for c in C_ORIG:
        a = json.load(open(os.path.join(RES, "auc", [f for f in os.listdir(os.path.join(RES, "auc")) if f.startswith(f"auc_c{c}_k{k}_P50-FREE50-TXT50_p") and str(p) in f.split("_p")[-1][:-5]][0]), encoding="utf-8"))["res"]
        if (k, p) == (0, 0):
            z = json.load(open(os.path.join(RES, "auc", f"auc_c{c}_k0_FREEIPC50-HSC1_50-P50HSC1_p0.json"), encoding="utf-8"))["res"]
        else:
            z = json.load(open(os.path.join(RES, "auc", f"auc_c{c}_k{k}_FREEIPC50-HSC1_50_p{p}.json"), encoding="utf-8"))["res"]
        d[c] = {"P50": a[f"P50_p{p}"]["auc"], "FREE50": a[f"FREE50_p{p}"]["auc"], "FREEIPC50": z[f"FREEIPC50_p{p}"]["auc"], "HSC1_50": z[f"HSC1_50_p{p}"]["auc"]}
    for b in ("FREEIPC50", "HSC1_50"):
        dd = np.array([d[c]["P50"] - d[c][b] for c in C_ORIG])
        rows.append({"pair": f"P50-{b}", "k_neg": k, "train_pair": p, "mean_d": dd.mean(), "wins": int((dd > 0).sum()), "per_origin": " ".join(f"{x:+.4f}" for x in dd)})
    dd = np.array([d[c]["FREE50"] - d[c]["HSC1_50"] for c in C_ORIG]); rows.append({"pair": "FREE50-HSC1_50", "k_neg": k, "train_pair": p, "mean_d": dd.mean(), "wins": int((dd > 0).sum()), "per_origin": " ".join(f"{x:+.4f}" for x in dd)})
    dd = np.array([d[c]["FREE50"] - d[c]["FREEIPC50"] for c in C_ORIG]); rows.append({"pair": "FREE50-FREEIPC50", "k_neg": k, "train_pair": p, "mean_d": dd.mean(), "wins": int((dd > 0).sum()), "per_origin": " ".join(f"{x:+.4f}" for x in dd)})
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, "s3_seed.csv"), index=False, encoding="utf-8-sig")
S = {pr: {"n": len(g), "mean": g.mean_d.mean(), "sd": g.mean_d.std(ddof=1), "min": g.mean_d.min(), "max": g.mean_d.max(), "wins_min": int(g.wins.min())} for pr, g in D.groupby("pair")}
json.dump(S, open(os.path.join(RES, "s3_seed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(D.round(5).to_string()); print(json.dumps(S, default=float, indent=1))
