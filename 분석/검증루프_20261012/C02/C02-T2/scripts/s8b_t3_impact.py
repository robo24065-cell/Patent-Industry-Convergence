# C02-T2 s8b: CN 2021(·2022·2024) IPC 결측이 T3 적격 셀·CN 피처·최종 양성 셀에 주는 영향 (진단, 성적 불변·재학습/재채점 없음)
# 원판 = 제공 그대로(결측 행은 셀 미배정), 가상 A = 결측 고유 출원을 (ksic, 공개연도) 안 비결측 CN 메인그룹 분포 비례 배분(PREREG H3),
# 가상 B(민감도) = 문서종류(발명/실용) 층별 배분. 최종 레이블(KR 2022–24)은 '양성 셀 수 집계'에만 사용.
import os, sys, json, hashlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
W = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2"); C = os.path.join(W, 'cache')
T3 = os.path.join(ROOT, r"분석\모델경쟁_20260927\T3_기술셀부상예측\R3")
Y0, Y1 = 2000, 2024; T = Y1 - Y0 + 1; CCS = ['KR', 'US', 'CN', 'JP', 'EP']
p = pd.read_parquet(os.path.join(C, 's8_panel.parquet')); p['cell'] = p.ksic + '|' + p.mg
cells = np.array(sorted(p.cell.unique())); ci = pd.Series(np.arange(len(cells)), index=cells)
n = np.zeros((len(cells), 5, T), dtype=np.float64)
n[ci[p.cell].values, p.cc.map({c: i for i, c in enumerate(CCS)}).values, p.year.values - Y0] = p.n.values
ksic = np.array([c.split('|')[0] for c in cells])
# ---- 가상 계열
m = pd.read_parquet(os.path.join(C, 's8_cn_missing_ky.parquet'))
mA = m.groupby(['ksic', 'year'])[['n_null', 'n_valid']].sum()
facA = (1 + mA.n_null / mA.n_valid.replace(0, np.nan)).fillna(1.0)
cnA = n[:, 2, :].copy(); cnB = np.zeros_like(cnA)
F = np.ones((len(cells), T))
for (k, y), f in facA.items():
    if y < Y0 or y > Y1 or f == 1: continue
    F[ksic == k, y - Y0] = f
cnA = n[:, 2, :] * F
pdt = pd.read_parquet(os.path.join(C, 's8_cn_panel_dt.parquet')); pdt['cell'] = pdt.ksic + '|' + pdt.mg
mB = m.set_index(['ksic', 'year', 'dt'])
for dt in ['inv', 'um']:
    s = pdt[pdt.dt == dt]; arr = np.zeros((len(cells), T)); arr[ci[s.cell].values, s.year.values - Y0] = s.n.values
    fb = np.ones((len(cells), T))
    mm = m[m.dt == dt]
    for _, r in mm.iterrows():
        if r.n_null == 0 or r.year < Y0: continue
        if r.n_valid > 0: fb[ksic == r.ksic, r.year - Y0] = 1 + r.n_null / r.n_valid
    cnB += arr * fb
assert np.allclose(cnB.sum(), cnA.sum(), rtol=0.02), (cnB.sum(), cnA.sum())
def ws(x, t0, a, b): return x[..., t0 + a - Y0:t0 + b - Y0 + 1].sum(-1)
def g(x, t0, lag=0, w=3):
    if t0 - lag - 2 * w + 1 < Y0: return np.full(x.shape[0], np.nan)
    return np.log((ws(x, t0 - lag, -w + 1, 0) + 1) / (ws(x, t0 - lag - w, -w + 1, 0) + 1))
def slope(x, t0, w=5):
    a = max(Y0, t0 - w + 1); yrs = np.arange(a, t0 + 1); Z = np.log1p(x[:, a - Y0:t0 - Y0 + 1]); tt = yrs - yrs.mean()
    return (Z * tt).sum(1) / (tt ** 2).sum()
def feats(cn, t0):
    nn = n.copy(); nn[:, 2, :] = cn
    kr = nn[:, 0, :]; tot = nn.sum(1); fx = nn[:, 1:, :].sum(1); x = nn[:, 2, :]
    d = dict(CN_mom=g(x, t0), CN_lvl=np.log1p(ws(x, t0, -2, 0)), CN_g1=np.log((x[:, t0 - Y0] + 1) / (x[:, t0 - 1 - Y0] + 1)),
             CN_slope5=slope(x, t0), fx_mom=g(fx, t0), tot_mom=g(tot, t0), tot_lvl=np.log1p(ws(tot, t0, -2, 0)),
             kr_share=np.log((ws(kr, t0, -2, 0) + 1) / (ws(tot, t0, -2, 0) + 1)), tot_W=ws(tot, t0, -2, 0), kr_W=ws(kr, t0, -2, 0),
             CN_W=ws(x, t0, -2, 0))
    if t0 + 3 <= Y1:
        fut = ws(kr, t0, 1, 3); d['kr_fut'] = fut; d['y'] = np.log((fut + 1) / (d['kr_W'] + 1))
    return pd.DataFrame(d, index=cells)
def hkey(cs): return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in cs], dtype=np.int64)
def label(df):
    k = max(1, int(round(0.1 * len(df)))); o = np.lexsort((hkey(df.index.values), -(df.kr_fut - df.kr_W).values, -df.y.values))
    yb = np.zeros(len(df), dtype=int); yb[o[:k]] = 1; return pd.Series(yb, index=df.index)
FE = ['CN_mom', 'CN_lvl', 'CN_g1', 'CN_slope5', 'fx_mom', 'tot_mom', 'tot_lvl', 'kr_share']
res = {'reproduction': {}, 'by_origin': {}}
# ---- 재현 확인: 동결 캐시와 비교
for t0, fn in [(2021, os.path.join(T3, 'cache_pub_final', 'feat_2021.parquet')), (2015, os.path.join(T3, 'cache_pub', 'feat_2015.parquet')), (2018, os.path.join(T3, 'cache_pub_final', 'feat_2018.parquet'))]:
    ref = pd.read_parquet(fn).set_index('cell'); o = feats(n[:, 2, :], t0).reindex(ref.index)
    rep = {'n_ref': int(len(ref)), 'n_missing_in_mine': int(o.tot_W.isna().sum())}
    for c in ['tot_W', 'kr_W', 'CN_lvl', 'CN_mom', 'fx_mom', 'kr_share'] + (['kr_fut'] if t0 == 2021 else []):
        rep[c + '_maxabs'] = float(np.nanmax(np.abs(o[c].values - ref[c].values)))
    rep['n_elig_ref'] = int((ref.tot_W >= 50).sum()); rep['n_elig_mine'] = int((o.tot_W >= 50).sum())
    res['reproduction'][t0] = rep
# ---- 원점별 영향
for t0 in list(range(2010, 2019)) + [2021, 2024]:
    O = feats(n[:, 2, :], t0); A = feats(cnA, t0); B = feats(cnB, t0)
    eO = O.tot_W >= 50; eA = A.tot_W >= 50; eB = B.tot_W >= 50
    r = dict(n_elig=int(eO.sum()), n_elig_A=int(eA.sum()), n_elig_B=int(eB.sum()),
             elig_change_A=float(eA.sum() / eO.sum() - 1), elig_change_B=float(eB.sum() / eO.sum() - 1),
             enter_A=int((eA & ~eO).sum()), exit_A=int((eO & ~eA).sum()))
    sub = eO.values
    # CN 결측 행이 피처 창(t0-5..t0)에 들어가는 비율: 적격 셀 CN 창 합 대비 가상 추가분
    r['cn_window_missing_share_A'] = float(1 - O.CN_W[sub].sum() / A.CN_W[sub].sum())
    r['tot_window_missing_share_A'] = float(1 - O.tot_W[sub].sum() / A.tot_W[sub].sum())
    for f in FE:
        a, b_ = O[f].values[sub], A[f].values[sub]; ok = ~(np.isnan(a) | np.isnan(b_))
        r['rho_A_' + f] = float(spearmanr(a[ok], b_[ok]).statistic) if ok.sum() > 2 and np.std(b_[ok]) > 0 else None
        bb = B[f].values[sub]; ok2 = ~(np.isnan(a) | np.isnan(bb))
        r['rho_B_' + f] = float(spearmanr(a[ok2], bb[ok2]).statistic) if ok2.sum() > 2 else None
        r['mean_shift_A_' + f] = float(np.nanmean(b_ - a))
    r['n_cells_cnW_lt90pct_A'] = int((O.CN_W[sub] < 0.9 * A.CN_W[sub]).sum())
    if 'y' in O:
        yO = label(O[eO]); yA = label(A[eA].assign(y=O.y[eA], kr_fut=O.kr_fut[eA], kr_W=O.kr_W[eA]))
        yB = label(B[eB].assign(y=O.y[eB], kr_fut=O.kr_fut[eB], kr_W=O.kr_W[eB]))
        posO = set(yO[yO == 1].index); posA = set(yA[yA == 1].index); posB = set(yB[yB == 1].index)
        r.update(n_pos=len(posO), n_pos_A=len(posA), pos_lost_A=len(posO - posA), pos_gained_A=len(posA - posO),
                 pos_lost_B=len(posO - posB), pos_gained_B=len(posB - posO),
                 pos_change_share_A=(len(posO - posA) + len(posA - posO)) / max(len(posO), 1),
                 pos_cnW_lt90pct_A=int(sum(1 for c in posO if O.CN_W[c] < 0.9 * A.CN_W[c])),
                 pos_with_cn_A=int(sum(1 for c in posO if A.CN_W[c] > 0)),
                 pos_exit_A=int(len(posO - set(A[eA].index))),
                 pos_enter_new_cells_A=int(len([c for c in posA if not eO[c]])))
    res['by_origin'][t0] = r
fin = res['by_origin'][2021]
fin['verdict_manuscript_caveat'] = '원고 단서 필요' if (abs(fin['elig_change_A']) > 0.05 or fin['pos_change_share_A'] > 0.05) else '단서 불요(5% 이내)'
cnr = {f: fin['rho_A_' + f] for f in FE if f.startswith('CN')}
fin['cn_feature_grade'] = {f: ('영향 작음' if v >= 0.95 else '주의' if v >= 0.90 else '영향 큼') for f, v in cnr.items() if v is not None}
dev = [res['by_origin'][t] for t in range(2010, 2019)]
res['dev_vs_final_rho_gap'] = {f: float(np.mean([d['rho_A_' + f] for d in dev if d['rho_A_' + f] is not None]) - fin['rho_A_' + f]) for f in FE}
json.dump(res, open(os.path.join(W, 't3_impact.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res['reproduction'], ensure_ascii=False, indent=0))
tab = pd.DataFrame(res['by_origin']).T
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 80)
print(tab[['n_elig', 'n_elig_A', 'n_elig_B', 'elig_change_A', 'cn_window_missing_share_A', 'rho_A_CN_mom', 'rho_A_CN_lvl', 'rho_A_CN_g1', 'rho_A_CN_slope5', 'rho_A_fx_mom', 'rho_A_tot_mom', 'rho_A_tot_lvl', 'rho_A_kr_share', 'rho_B_CN_mom', 'rho_B_CN_g1', 'n_cells_cnW_lt90pct_A']].to_string())
print({k: v for k, v in fin.items() if not k.startswith(('rho', 'mean_shift'))})
print(res['dev_vs_final_rho_gap'])
