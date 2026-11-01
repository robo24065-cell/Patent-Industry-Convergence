# -*- coding: utf-8 -*-
"""verify_2 r2: R 되돌림 Δ(iii)·결합 Δ(iv)를 '전체 재순위'(추가 행 포함 집합에서 블렌드 백분위·AUC·기준선 6종 다시 계산)로 독립 재계산.
 s3의 빠른 계산(공식 집합 기준 백분위 고정)과 대조. 같은 난수열(RandomState 20261019)로 앞 60시드 재현.
 변형: Ry의 공개 경로 전용 양성(y_pub=1,y_reg=0)을 음성으로 / 같은 계보(중복 코드·분할 승계) 의심 쌍 제외.
 산출: verify_2/r2_out.json"""
import pathlib, json, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
V = B/'분석/검증루프_20261012'; C = V/'C17/C17-T2'; CC = C/'cache'; OUT = C/'verify_2'
OFF = 0.47267044268107444
Z = np.load(CC/'eval_base.npz', allow_pickle=True)
px = Z['px']; pl = Z['pl']; y = Z['y'].astype(np.int8); base = Z['base']; firm = Z['firm']; kk = Z['k']
Rpx = Z['Rs_px']; Rpl = Z['Rs_pl']; Rb = Z['Rbase']; Ry_all = Z['Ry'].astype(np.int8)


def auc(v, yy):
    r = rankdata(v); n1 = int(yy.sum()); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def skill_full(addR=np.array([], int), flip=np.array([], int), Ry_override=None):
    """addR: R 행 인덱스(eval_base R 배열), flip: 평가 행을 음성으로."""
    X1 = np.r_[px, Rpx[addR]]; X2 = np.r_[pl, Rpl[addR]]
    ry = Ry_all[addR] if Ry_override is None else Ry_override
    yy = np.r_[y, ry].astype(np.int8).copy()
    if len(flip): yy[flip] = 0
    s = 0.9 * pd.Series(X1).rank(pct=True).values + 0.1 * pd.Series(X2).rank(pct=True).values
    am = auc(s, yy); M0 = max(auc(np.r_[base[j], Rb[j][addR]], yy) for j in range(6))
    return (am - M0) / (1 - M0)


out = {}
out['P0_full'] = skill_full(); out['P0_diff'] = out['P0_full'] - OFF
# ---- R 코드 확정 오매칭 ----
Rkey = pd.Series(np.arange(len(Z['Rfirm'])), index=pd.MultiIndex.from_arrays([Z['Rfirm'], Z['Rk']]))
PJ = pd.read_parquet(CC/'pair_judg.parquet'); PJ['firm'] = PJ.firm.astype(str)
W = PJ[(PJ.grp == 'R') & (PJ.pv == 'wrong')]
ri = Rkey.reindex(pd.MultiIndex.from_arrays([W.firm.values, W.k.values])).values; assert not np.isnan(ri).any(); ri = ri.astype(int)
L = pd.read_parquet(V/'C16/C16-T1/cache/labels.parquet'); L = L[(L.src == 'te') & L.prior_any].reset_index(drop=True); L['firm'] = L.firm.astype(str)
assert (L.firm.values == Z['Rfirm']).all() and (L.y_u.values == Ry_all).all()
yreg = L.y_reg.values[ri]; ypub = L.y_pub.values[ri]
out['n'] = int(len(ri)); out['pos'] = int(Ry_all[ri].sum()); out['pos_reg'] = int(yreg.sum()); out['pos_pub_only'] = int(((ypub == 1) & (yreg == 0)).sum())
out['d_iii_full'] = skill_full(ri) - OFF
# 변형 1: 공개 경로 전용 양성을 음성으로(같은 이름 경로의 진입 문서도 같은 다른 실체일 가능성)
out['d_iii_full_pubonly_to_neg'] = skill_full(ri, Ry_override=np.where(yreg == 1, 1, 0)) - OFF
# 변형 2: 같은 계보 의심(중복 코드: 같은 이름+같은 주소 4어절 / 분할 신설 코드 포스코·오씨아이) 제외
pr1 = pd.read_csv(OUT/'r1_pairs.csv', dtype={'firm': str})
lineage = set(map(tuple, pr1[pr1.all_dup.astype(str) == 'True'][['firm', 'k']].values.tolist()))
lineage |= {(f, int(k)) for f, k in zip(W.firm, W.k) if f in ('120220203266', '120230318740')}
keep2 = np.array([(f, int(k)) not in lineage for f, k in zip(W.firm, W.k)])
out['lineage_pairs'] = int((~keep2).sum()); out['lineage_pos'] = int(Ry_all[ri[~keep2]].sum())
out['d_iii_full_no_lineage'] = skill_full(ri[keep2]) - OFF
out['d_iii_full_no_lineage_pubonly_neg'] = skill_full(ri[keep2], Ry_override=np.where(yreg[keep2] == 1, 1, 0)) - OFF
# 변형 3: 양성 18쌍 제외, 음성 103쌍만 되돌림
out['d_iii_full_neg_only'] = skill_full(ri[Ry_all[ri] == 0]) - OFF
print({k: v for k, v in out.items()}, round(time.time() - t0), flush=True)
# ---- 확률 되돌림·결합: s3와 같은 난수열 앞 60시드 전체 재순위 ----
pr = pd.read_parquet(CC/'p_point_R.parquet'); PBR = np.load(CC/'p_boot_R.npy'); PB = np.load(CC/'p_boot_main.npy')
rrow = Rkey.reindex(pd.MultiIndex.from_arrays([pr.firm.astype(str).values, pr.k.values])).values.astype(int)
key = pd.Series(np.arange(len(y)), index=pd.MultiIndex.from_arrays([firm, kk]))
pp = pd.read_parquet(CC/'p_point.parquet')
rows = key.reindex(pd.MultiIndex.from_arrays([pp.firm.astype(str).values, pp.k.values])).values.astype(int)
assert (y[rows] == 1).all()
rr = np.random.RandomState(20261019); D3 = []; D4 = []; NR = []; NP = []
for b in range(60):
    wR = rr.rand(len(pr)) < PBR[b]; wP = rr.rand(len(pp)) < PB[b]
    D3.append(skill_full(rrow[wR]) - OFF); D4.append(skill_full(rrow[wR], flip=rows[wP]) - OFF); NR.append(int(wR.sum())); NP.append(int(Ry_all[rrow[wR]].sum()))
q = lambda a: dict(mean=float(np.mean(a)), p2_5=float(np.percentile(a, 2.5)), p97_5=float(np.percentile(a, 97.5)))
s3 = json.load(open(CC/'s3_out.json', encoding='utf-8'))
out['full_60'] = dict(iii=q(D3), iv=q(D4), n_readd=q(NR), pos_readd=q(NP))
out['s3_200'] = dict(iii=s3['R_secondary']['iii'], iv=s3['R_secondary']['iv'], d_iii_code=s3['R_lower_code_only']['d_iii'])
out['readd_pos_share'] = float(np.sum(NP) / np.sum(NR))
out['pR_expected'] = float(pr.p.sum()); out['pR_pos_expected'] = float((pr.p.values * Ry_all[rrow]).sum())
# 확률 되돌림 쌍 중 공개 경로 전용 양성의 기대 수
yreg_all = L.y_reg.values; ypub_all = L.y_pub.values
out['pR_pos_pubonly_expected'] = float((pr.p.values * ((ypub_all[rrow] == 1) & (yreg_all[rrow] == 0))).sum())
out['secs'] = round(time.time() - t0)
json.dump(out, open(OUT/'r2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
