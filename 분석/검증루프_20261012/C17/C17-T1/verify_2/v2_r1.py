# -*- coding: utf-8 -*-
"""verify_2: C17-T1 T1-05(R1) 발견 반박 시도. 독립 구현(score17/c17common 미사용, sklearn roc_auc_score).
 읽기 전용: 트랙 캐시·C16 캐시·C17-T1 cache. 쓰기: verify_2/ 만."""
import sys, json, pathlib, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'; TC = T1/'cache'
C16C = B/'분석/검증루프_20261012/C16/C16-T1/cache'; CC = B/'분석/검증루프_20261012/C17/C17-T1/cache'
OUT = pathlib.Path(__file__).resolve().parent
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
res = {}

# ---- 공식 프레임(독립 조립)
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
fz = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'px', 'pl'])
assert (fz.firm.values == te.firm.values).all() and (fz.k.values == te.k.values).all()
bo = pd.read_parquet(FC/'baseline_scores.parquet')
L = pd.read_parquet(C16C/'labels.parquet'); Lt = L[L.src == 'te'].reset_index(drop=True)
assert (Lt.firm.values == te.firm.values).all()
O = te.copy(); O['px'] = fz.px.values; O['pl'] = fz.pl.values; O['y'] = Lt.y_u.values.astype(np.int8)
for c in BK: O[c] = bo[c].values
O = O[~O.prior_any.values].reset_index(drop=True)


def blend(px, pl):
    return 0.9 * pd.Series(px).rank(pct=True).values + 0.1 * pd.Series(pl).rank(pct=True).values


def metr(D, tag):
    y = D.y.values
    s = blend(D.px.values, D.pl.values)
    a = roc_auc_score(y, s); ba = {c: roc_auc_score(y, D[c].values) for c in BK}; m0 = max(ba.values())
    o = np.argsort(-s, kind='stable')
    r = dict(n=int(len(D)), pos=int(y.sum()), firms=int(D.firm.nunique()), auc=a, M0=m0, M0_arg=max(ba, key=ba.get), skill=(a - m0) / (1 - m0), top1000=int(y[o[:1000]].sum()))
    print(tag, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r


res['OFF'] = metr(O, 'OFF')

# ---- RC 레이블 독립 재계산(원천: rc_ins·rc_app, 직접 groupby)
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.unique()); ki = {k: i for i, k in enumerate(K)}
rci = pd.read_parquet(CC/'rc_ins.parquet'); rci = rci[rci.ay >= 2000].copy(); rci['kk'] = rci.ksic.map(ki)
rca = pd.read_parquet(CC/'rc_app.parquet'); rca = rca[rca.ay >= 2000].copy(); rca['kk'] = rca.ksic.astype(str).map(ki); rca = rca.dropna(subset=['kk']); rca['kk'] = rca.kk.astype(int)
# 법인 집합: 범위안 등록(ay 2000–2019) 고유 rgst_no ≥3 (RC는 범위밖 행 없음)
cnt = rci[rci.ay <= 2019].drop_duplicates(['applicant_cd', 'rgst_no']).groupby('applicant_cd').size()
F = sorted(cnt[cnt >= 3].index); res['rc_firms_indep'] = len(F)
held = rci[(rci.ay <= 2019) & rci.applicant_cd.isin(F)].groupby(['applicant_cd', 'kk']).size().index
grid = pd.MultiIndex.from_product([F, range(50)], names=['firm', 'k'])
risk = grid[~grid.isin(held)]
pd_ = rca[(rca.ay <= 2019)].groupby(['applicant_cd', 'kk']).size().index
yreg = rci[(rci.ay >= 2020) & (rci.ay <= 2023)].groupby(['applicant_cd', 'kk']).size().index
pdv = rca.pub_date.fillna('').astype(str); rdv = rca.rgst_date.fillna('').astype(str)
vis = ((pdv.str.len() == 8) & (pdv <= '20250822')) | ((rdv.str.len() == 8) & (rdv <= '20250822'))
ypub = rca[vis & (rca.ay >= 2020) & (rca.ay <= 2023)].groupby(['applicant_cd', 'kk']).size().index
R = pd.DataFrame({'firm': risk.get_level_values(0), 'k': risk.get_level_values(1)})
R['prior_any'] = risk.isin(pd_); R['y_reg'] = risk.isin(yreg); R['y_pub'] = risk.isin(ypub); R['y'] = (R.y_reg | R.y_pub).astype(np.int8)
res['rc_lab_indep'] = dict(pairs=int(len(R)), prior_any=int(R.prior_any.sum()), scored=int((~R.prior_any).sum()), pos=int(R.y[~R.prior_any].sum()),
                           pos_reg=int(R.y_reg[~R.prior_any].sum()), pos_pub=int(R.y_pub[~R.prior_any].sum()))
Lc = pd.read_parquet(CC/'lab_rc.parquet')
mm = R.merge(Lc, on=['firm', 'k'], how='outer', indicator=True, suffixes=('', '_c'))
res['rc_lab_vs_cache'] = dict(only_indep=int((mm._merge == 'left_only').sum()), only_cache=int((mm._merge == 'right_only').sum()),
                              y_diff=int((mm[mm._merge == 'both'].y != mm[mm._merge == 'both'].y_u).sum()), prior_diff=int((mm[mm._merge == 'both'].prior_any != mm[mm._merge == 'both'].prior_any_c).sum()))
print('rc labels', res['rc_lab_indep'], res['rc_lab_vs_cache'], flush=True)
Rk = R[~R.prior_any].reset_index(drop=True)

# ---- rcF 재구축 점수
S = pd.read_parquet(CC/'build'/'rcF'/'scores.parquet')
S = S[~S.prior_any.values]
res['rcF_firms_rc_in_scores'] = int(S.firm.isin(set(F)).values.sum() and S[S.firm.isin(set(F))].firm.nunique())
Sx = S.set_index(['firm', 'k'])
cols = ['px', 'pl'] + BK
# 공식 행 재계산 점수
A = O[['firm', 'k', 'y']].join(Sx[cols], on=['firm', 'k']); assert A[cols].notna().all().all(), 'official rows missing in rcF'
Rr = Rk[['firm', 'k', 'y']].join(Sx[cols], on=['firm', 'k']); res['rc_rows_missing_in_rcF'] = int(Rr.px.isna().sum())
assert Rr[cols].notna().all().all()
# rcF에 있는데 공식·RC 어느 쪽에도 없는 채점 행
kS = pd.MultiIndex.from_frame(S[['firm', 'k']]); kU = pd.MultiIndex.from_frame(pd.concat([O[['firm', 'k']], Rk[['firm', 'k']]]))
res['rcF_scored_rows_unassigned'] = int((~kS.isin(kU)).sum())
DR = pd.concat([A, Rr], ignore_index=True); res['R1re'] = metr(DR, 'R1re')
DF = pd.concat([O[['firm', 'k', 'y'] + cols], Rr], ignore_index=True); res['R1fx'] = metr(DF, 'R1fx')
# 분해: AUC는 재계산, M0만 고정판 / 반대
res['decomp'] = dict(skill_auc_re_M0_fx=(res['R1re']['auc'] - res['R1fx']['M0']) / (1 - res['R1fx']['M0']), skill_auc_fx_M0_re=(res['R1fx']['auc'] - res['R1re']['M0']) / (1 - res['R1re']['M0']))
# 공식 행만(재계산 점수) — RC 행 추가 효과와 공식 행 재채점 효과 분리
res['OFF_rescored_only'] = metr(A, 'OFF_rows_rcF_scores')
# 경계
nrc = len(Rr); isrc = np.r_[np.zeros(len(A), bool), np.ones(nrc, bool)]
for tag, good in (('worst', False), ('best', True)):
    D = DR.copy(); hi = max(D.px.max(), D.pl.max()) + 1; lo = min(D.px.min(), D.pl.min()) - 1
    for c in ('px', 'pl'):
        v = D[c].values.astype(float).copy(); v[isrc & (D.y.values == 1)] = hi if good else lo; v[isrc & (D.y.values == 0)] = lo if good else hi; D[c] = v
    res['R1re_' + tag] = metr(D, 'R1re_' + tag)
# 무작위 대입 여러 시드
rs = []
for sd in range(5):
    D = DR.copy(); rng = np.random.default_rng(1000 + sd); ix = rng.integers(0, len(A), nrc)
    for c in ('px', 'pl'):
        v = D[c].values.astype(float).copy(); v[isrc] = A[c].values[ix]; D[c] = v
    rs.append(metr(D, f'rand{sd}')['skill'])
res['R1re_rand_skills'] = rs
# 추가 반박: 경계 시 기준선도 반대로(전체 범위)
for tag, good in (('worst_bothM0', False), ('best_bothM0', True)):
    D = DR.copy(); hi = 1e6; lo = -1e6
    for c in ['px', 'pl']:
        v = D[c].values.astype(float).copy(); v[isrc & (D.y.values == 1)] = hi if good else lo; v[isrc & (D.y.values == 0)] = lo if good else hi; D[c] = v
    for c in BK:  # 기준선은 모형과 반대 방향
        v = D[c].values.astype(float).copy(); v[isrc & (D.y.values == 1)] = lo if good else hi; v[isrc & (D.y.values == 0)] = hi if good else lo; D[c] = v
    res['R1re_' + tag] = metr(D, 'R1re_' + tag)
OFFs = res['OFF']['skill']
res['deltas'] = {k: res[k]['skill'] - OFFs for k in ['R1re', 'R1fx', 'R1re_worst', 'R1re_best', 'R1re_worst_bothM0', 'R1re_best_bothM0', 'OFF_rescored_only']}
res['deltas']['rand_mean'] = float(np.mean(rs) - OFFs)
# RC 행 백분위
s = blend(DR.px.values, DR.pl.values); r = pd.Series(s).rank(pct=True).values
res['rc_pct_mean'] = float(r[isrc].mean())
# ---- 군집 부트스트랩 Δ(R1re − OFF), 독립 시드, B=200, 법인 단위 짝
U = np.array(sorted(set(DR.firm))); ui = pd.Series(np.arange(len(U)), index=U)
fr_R = ui.reindex(DR.firm.values).values; fr_O = ui.reindex(O.firm.values).values
sR = blend(DR.px.values, DR.pl.values); sO = blend(O.px.values, O.pl.values)
rng = np.random.default_rng(20261017); dl = []
for b in range(200):
    w = np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U)).astype(float)
    wR = w[fr_R]; wO = w[fr_O]
    def sk(D, s_, ww):
        a = roc_auc_score(D.y.values, s_, sample_weight=ww); m = max(roc_auc_score(D.y.values, D[c].values, sample_weight=ww) for c in BK); return (a - m) / (1 - m)
    dl.append(sk(DR, sR, wR) - sk(O, sO, wO))
    if time.time() - t0 > 480: break
dl = np.array(dl); res['boot_delta'] = dict(B=int(len(dl)), lo=float(np.percentile(dl, 2.5)), med=float(np.median(dl)), hi=float(np.percentile(dl, 97.5)), share_le_m0087=float(np.mean(dl <= -0.0087)))
res['secs'] = round(time.time() - t0)
json.dump(res, open(OUT/'v2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
