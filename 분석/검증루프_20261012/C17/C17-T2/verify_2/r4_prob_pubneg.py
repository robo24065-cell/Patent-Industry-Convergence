# -*- coding: utf-8 -*-
"""verify_2 r4: 확률 되돌림 Δ(iii)·결합 Δ(iv)에서, 되돌린 R 쌍의 공개 경로 전용 양성(y_pub=1,y_reg=0)을 음성으로 둔 민감도(전체 재순위, s3와 같은 난수열 앞 40시드).
 산출: verify_2/r4_out.json"""
import pathlib, json, time, runpy
import numpy as np, pandas as pd
from scipy.stats import rankdata
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
V = B/'분석/검증루프_20261012'; C = V/'C17/C17-T2'; CC = C/'cache'; OUT = C/'verify_2'
OFF = 0.47267044268107444
Z = np.load(CC/'eval_base.npz', allow_pickle=True)
px = Z['px']; pl = Z['pl']; y = Z['y'].astype(np.int8); base = Z['base']; firm = Z['firm']; kk = Z['k']
Rpx = Z['Rs_px']; Rpl = Z['Rs_pl']; Rb = Z['Rbase']; Ry_all = Z['Ry'].astype(np.int8)
L = pd.read_parquet(V/'C16/C16-T1/cache/labels.parquet'); L = L[(L.src == 'te') & L.prior_any].reset_index(drop=True)
Ryreg = L.y_reg.values.astype(np.int8)
def auc(v, yy):
    r = rankdata(v); n1 = int(yy.sum()); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def skill_full(addR, flip=np.array([], int), ry=None):
    X1 = np.r_[px, Rpx[addR]]; X2 = np.r_[pl, Rpl[addR]]
    yy = np.r_[y, Ry_all[addR] if ry is None else ry].astype(np.int8).copy()
    if len(flip): yy[flip] = 0
    s = 0.9 * pd.Series(X1).rank(pct=True).values + 0.1 * pd.Series(X2).rank(pct=True).values
    am = auc(s, yy); M0 = max(auc(np.r_[base[j], Rb[j][addR]], yy) for j in range(6))
    return (am - M0) / (1 - M0)
Rkey = pd.Series(np.arange(len(Z['Rfirm'])), index=pd.MultiIndex.from_arrays([Z['Rfirm'], Z['Rk']]))
pr = pd.read_parquet(CC/'p_point_R.parquet'); PBR = np.load(CC/'p_boot_R.npy'); PB = np.load(CC/'p_boot_main.npy')
rrow = Rkey.reindex(pd.MultiIndex.from_arrays([pr.firm.astype(str).values, pr.k.values])).values.astype(int)
key = pd.Series(np.arange(len(y)), index=pd.MultiIndex.from_arrays([firm, kk]))
pp = pd.read_parquet(CC/'p_point.parquet')
rows = key.reindex(pd.MultiIndex.from_arrays([pp.firm.astype(str).values, pp.k.values])).values.astype(int)
rr = np.random.RandomState(20261019); A = []; Bv = []; Cn = []
for b in range(40):
    wR = rr.rand(len(pr)) < PBR[b]; wP = rr.rand(len(pp)) < PB[b]
    ad = rrow[wR]
    A.append(skill_full(ad, ry=Ryreg[ad]) - OFF); Bv.append(skill_full(ad, flip=rows[wP], ry=Ryreg[ad]) - OFF)
    Cn.append(skill_full(ad[Ry_all[ad] == 0]) - OFF)
q = lambda a: dict(mean=float(np.mean(a)), p2_5=float(np.percentile(a, 2.5)), p97_5=float(np.percentile(a, 97.5)))
out = dict(iii_pubonly_neg=q(A), iv_pubonly_neg=q(Bv), iii_neg_only=q(Cn), seeds=40, secs=round(time.time() - t0))
json.dump(out, open(OUT/'r4_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
