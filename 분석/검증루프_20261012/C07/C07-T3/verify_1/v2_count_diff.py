# -*- coding: utf-8 -*-
"""C07-T3 verify_1 v2: 건수 기준 지수(exp_disputes) 대 지속 건수의 ρ 차이 군집 부트스트랩(사전 등록 밖, 민감도)."""
import os, json, sqlite3
import numpy as np, pandas as pd
from scipy.stats import spearmanr
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
N8 = os.path.join(BASE, r"분석\모델경쟁_20260927\N8_특허분쟁표적레이더")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
W = os.path.dirname(os.path.abspath(__file__)); C07 = os.path.dirname(W)
FS = pd.read_parquet(os.path.join(N8, "future_scores_r6.parquet"))
K = pd.read_csv(os.path.join(N8, "future_ksic_r6.csv"), encoding="utf-8-sig", dtype={"ksic": str})
FP = pd.read_parquet(os.path.join(N8, "final_preds_r6.parquet"), columns=["ksic", "y"])
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
EV = pd.read_sql("select rgst_no from right where rgstCsDate between '20250101' and '20260630' and rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록')", c); c.close()
y = FS.rgst_no.astype(str).isin(set(EV.rgst_no.astype(str))).values.astype(float)
ks = sorted(K.ksic); Ki = K.set_index("ksic").loc[ks]; exp_d = Ki.exp_disputes.values; nk = Ki.n.values.astype(float)
pc = FP.groupby("ksic").y.sum().reindex(ks).values.astype(float)
# 지속 건수를 현재 모집단 크기로 옮긴 값(지속률 × 현재 n)
pr = (FP.groupby("ksic").y.mean().reindex(ks).values) * nk
S = pd.read_parquet(os.path.join(C07, "cache", "s02b_scores_t2024.parquet"), columns=["rgst_no", "owner_v2"])
assert (S.rgst_no.values == FS.rgst_no.values).all()
o = S.owner_v2.fillna("").astype(str).values; g = np.where(o == "", "R" + FS.rgst_no.astype(str).values, o)
cg, uq = pd.factorize(g); G = len(uq)
kidx = {k: i for i, k in enumerate(ks)}; ck = np.array([kidx[k] for k in FS.ksic.astype(str)])
sp = lambda a, b: float(spearmanr(a, b).correlation)
cnt = np.bincount(ck, weights=y, minlength=len(ks))
out = dict(rho_exp_cnt=sp(exp_d, cnt), rho_perscnt_cnt=sp(pc, cnt), rho_persrate_x_n_cnt=sp(pr, cnt), rho_n_cnt=sp(nk, cnt))
rng = np.random.default_rng(2027); d1 = []; d2 = []
for _ in range(1000):
    w = rng.multinomial(G, np.full(G, 1.0 / G))[cg].astype(float)
    by = np.bincount(ck, weights=w * y, minlength=len(ks))
    a = sp(exp_d, by); d1.append(a - sp(pc, by)); d2.append(a - sp(pr, by))
for nm, v in (("diff_exp_minus_perscnt", d1), ("diff_exp_minus_persrate_x_n", d2)):
    v = np.array(v); v = v[~np.isnan(v)]
    out[nm] = dict(ci=[float(x) for x in np.percentile(v, [2.5, 97.5])], mean=float(v.mean()), frac_gt0=float((v > 0).mean()))
json.dump(out, open(os.path.join(W, "v2_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
