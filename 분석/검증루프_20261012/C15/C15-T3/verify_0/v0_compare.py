# -*- coding: utf-8 -*-
import sys, json, glob, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_auc_score
W = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C15/C15-T3"
R7 = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T3_기술셀부상예측/ESSENTIAL_R7/results"
main = pd.concat([pd.read_parquet(f) for f in glob.glob(W + "/res/t3/hyb_pkr_reg__*_preds.parquet")])
dr = pd.read_parquet(R7 + "/pub_drop_ros_preds.parquet"); kr = pd.read_parquet(R7 + "/pub_kr_rebuild_preds.parquet")
out = {"by_origin": {}}; av, am, ad, ak = [], [], [], []
for v in range(2010, 2019):
    a = pd.read_parquet(f"{W}/verify_0/v0_preds_{v}.parquet").sort_values("cell"); b = main[main.t0 == v].sort_values("cell")
    d = dr[dr.t0 == v].sort_values("cell"); k = kr[kr.t0 == v].sort_values("cell")
    same = bool((a.cell.values == b.cell.values).all() and (a.yb.values == b.yb.values).all() and (a.cell.values == d.cell.values).all() and (a.yb.values == d.yb.values).all())
    x, y = roc_auc_score(a.yb, a.p), roc_auc_score(b.yb, b.p); av.append(x); am.append(y); ad.append(roc_auc_score(d.yb, d.p)); ak.append(roc_auc_score(k.yb, k.p))
    out["by_origin"][v] = dict(verify0=round(x, 5), main=round(y, 5), diff=round(x - y, 5), max_dp=float(np.abs(a.p.values - b.p.values).max()), same_cells=same)
mv, mm, md, mk = map(np.mean, (av, am, ad, ak))
out.update(mean_verify0=mv, mean_main=mm, diff=mv - mm, vs_R8_0p77346=mv - 0.77346,
           retention_verify0_raw=(mv - mk) / (md - mk), retention_verify0_auc4=(round(mv, 4) - round(mk, 4)) / (round(md, 4) - round(mk, 4)),
           drop_ros_minus_v0=md - mv, wins_drop_ros=int(sum(np.array(ad) > np.array(av))))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
json.dump(out, open(W + "/verify_0/v0_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
