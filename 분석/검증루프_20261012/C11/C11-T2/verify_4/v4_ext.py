# -*- coding: utf-8 -*-
# verify_4 V2·V3: 셀 외부 정합(prov−txt 25행 + BH) · 그룹 안 34 KSIC를 독립 코드로 재계산.
# 사용: python v4_ext.py <arm> <seed_main> <seed_within> <tag>   (R9 seed: 7, 11)
import sys, json, time
import numpy as np, pandas as pd
B0 = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
T3 = B0 + "/분석/모델경쟁_20260927/T3_기술셀부상예측"; WD = B0 + "/분석/검증루프_20261012/C11/C11-T2"; OUT = WD + "/verify_4"
arm, SM, SW, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
t_0 = time.time(); YRS = list(range(2010, 2019)); NB = 1000

km = pd.read_csv(T3 + '/ESSENTIAL_R7/conc_ksic_map.csv', dtype=str).set_index('ksic')['code']
K = sorted(km.index)
grp_size = km.map(km.value_counts())
MULTI = sorted(km.index[grp_size > 1])

# ---- 외부 성장 ----
ko = pd.read_csv(B0 + "/공모전데이터/작업데이터/kosis_광업제조업_50개산업.csv", dtype={"세세분류코드": str})
ko = ko[(ko['기준'] == '10차') & (ko['마스킹'] == 0)]
lev = {}
for c, nm in (("T04", "출하액"), ("T02", "종사자수"), ("T01", "사업체수")):
    z = ko[ko['항목코드'] == c].groupby(['세세분류코드', '연도'])['값'].sum().unstack()
    lev[nm] = z.reindex(K)
mp = pd.read_csv(B0 + "/분석/fusion/ksic_hs6_map.csv", dtype=str)[['k5', 'hs6']].drop_duplicates()
nk = mp.groupby('hs6').k5.nunique(); mp['w'] = 1.0 / mp.hs6.map(nk)
ex = pd.read_csv(B0 + "/분석/모델경쟁_20260927/_shared/ext/trade_kr_export_hs6_uv_2000_2021.csv", dtype={"hs6": str}, usecols=["year", "hs6", "export_usd"])
im = pd.read_csv(B0 + "/분석/fusion/hs6_import_yearly.csv", dtype={"hs6": str}); im = im[im.partner == 'WLD']
for nm, d, col in (("수출액", ex, "export_usd"), ("수입액", im, "value")):
    z = d[d.year <= 2021].merge(mp, on='hs6'); z['v'] = z[col] * z['w']
    lev[nm] = z.groupby(['k5', 'year'])['v'].sum().unstack().reindex(K)


def grow(L, t0):
    fut = [y for y in (t0 + 1, t0 + 2, t0 + 3) if y in L.columns]; pst = [y for y in (t0 - 2, t0 - 1, t0) if y in L.columns]
    return np.log(L[fut].mean(axis=1)) - np.log(L[pst].mean(axis=1))


GR = {nm: pd.concat({t0: grow(L, t0) for t0 in YRS}, axis=1) for nm, L in lev.items()}

# ---- 신호 ----
def cells(df):
    d = df[df.t0.isin(YRS)].copy()
    d['pr'] = d.groupby('t0')['p'].rank(pct=True)
    real = d.pivot_table(index='ksic', columns='t0', values='yb', aggfunc='mean')
    pred = d.pivot_table(index='ksic', columns='t0', values='pr', aggfunc='mean')
    return real.reindex(index=K, columns=YRS), pred.reindex(index=K, columns=YRS)


txtp = {'stored': T3 + '/TXT_R9/results/txt50_task_preds.parquet'}.get(arm, f'{WD}/res/runs/txt50_task__{arm}_preds.parquet')
txtpan = {'stored': T3 + '/TXT_R9/cache_txt_pub/panel_ccy.parquet'}.get(arm, f'{WD}/cache/txt_{arm}/panel_ccy.parquet')
T50 = pd.read_parquet(txtp, columns=['ksic', 't0', 'cell', 'yb', 'p'])
# V1에서 독립 재구성한 라벨로 yb를 덮어쓴다(같아야 함)
lab = pd.read_parquet(f'{OUT}/v4_labels_{arm}.parquet', columns=['cell', 't0', 'yb']) if arm in ('stored', 'reg', 'pure') else None
if lab is not None:
    T50 = T50.drop(columns='yb').merge(lab, on=['cell', 't0'], how='left'); assert T50.yb.notna().all()
P50 = pd.read_parquet(T3 + '/ESSENTIAL_R7/results/pub_full_preds.parquet', columns=['ksic', 't0', 'yb', 'p'])
mgn = pd.read_parquet(T3 + '/NONCIRC_R8/results/mg_ctx-none_preds.parquet', columns=['mg', 't0', 'yb', 'p'])
mgn = mgn[mgn.t0.isin(YRS)].copy(); mgn['pr'] = mgn.groupby('t0')['p'].rank(pct=True)


def bridges(fn):
    pan = pd.read_parquet(fn, columns=['ksic', 'mg', 'cc', 'year', 'n'])
    br, bp, mo = {}, {}, {}
    for t0 in YRS:
        win = pan[pan.year.between(t0 - 2, t0)]
        w = win.groupby(['ksic', 'mg'])['n'].sum().rename('n').reset_index()
        w['s'] = w['n'] / w.groupby('ksic')['n'].transform('sum')
        m = w.merge(mgn[mgn.t0 == t0], on='mg', how='inner')
        g = m.assign(a=m.s * m.yb, b=m.s * m.pr).groupby('ksic')[['s', 'a', 'b']].sum()
        br[t0] = g.a / g.s; bp[t0] = g.b / g.s
        k0 = win[win.cc == 'KR'].groupby('ksic')['n'].sum().reindex(win.ksic.unique()).fillna(0)
        k1 = pan[pan.year.between(t0 - 5, t0 - 3) & (pan.cc == 'KR')].groupby('ksic')['n'].sum().reindex(k0.index).fillna(0)
        mo[t0] = np.log((k0 + 1) / (k1 + 1))
    f = lambda D: pd.DataFrame(D).reindex(index=K, columns=YRS)
    return f(br), f(bp), f(mo)


S = {}
S['prov', 'cell_real'], S['prov', 'cell_pred'] = cells(P50)
S['txt', 'cell_real'], S['txt', 'cell_pred'] = cells(T50)
S['prov', 'bridge_real'], S['prov', 'bridge_pred'], S['prov', 'pat_mom'] = bridges(T3 + '/R3/cache_pub/panel_ccy.parquet')
S['txt', 'bridge_real'], S['txt', 'bridge_pred'], S['txt', 'pat_mom'] = bridges(txtpan)
print('signals', round(time.time() - t_0), flush=True)


def avg_rank(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def spear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8: return np.nan
    ra, rb = avg_rank(a[ok]), avg_rank(b[ok])
    if ra.std() == 0 or rb.std() == 0: return np.nan
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def rhos(Sa, Gv):
    return np.array([spear(Sa[:, j], Gv[:, j]) for j in range(Sa.shape[1])])


SIG = ('cell_real', 'cell_pred', 'bridge_real', 'bridge_pred', 'pat_mom')
rng = np.random.default_rng(SM); rows = []
for nm in ('출하액', '종사자수', '사업체수', '수출액', '수입액'):
    g = GR[nm]; val = [k for k in K if g.loc[k].notna().any()]; Gv = g.loc[val, YRS].to_numpy(float)
    for sg in SIG:
        A = S['prov', sg].loc[val, YRS].to_numpy(float); Bm = S['txt', sg].loc[val, YRS].to_numpy(float)
        ra, rb = rhos(A, Gv), rhos(Bm, Gv); d = ra - rb
        need = (sg == 'cell_real') or tag.startswith('main')
        bs = np.empty(NB)
        for b in range(NB):
            ii = rng.integers(0, len(val), len(val))
            bs[b] = np.nanmean(rhos(A[ii], Gv[ii]) - rhos(Bm[ii], Gv[ii])) if need else np.nan
        lo, hi = (np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)) if need else (np.nan, np.nan)
        pb = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))) if need else np.nan
        w = int(np.nansum(d > 0)); n = int(np.isfinite(d).sum())
        rows.append(dict(ext=nm, sig=sg, n_ind=len(val), rho_prov=np.nanmean(ra), rho_txt=np.nanmean(rb), diff=np.nanmean(d), lo=lo, hi=hi, wins=w, n=n, p=pb))
    print(nm, round(time.time() - t_0), flush=True)
O = pd.DataFrame(rows)
if O.p.notna().all():
    p = O.p.to_numpy(); m = len(p); o = np.argsort(p, kind='mergesort'); adj = p[o] * m / np.arange(1, m + 1)
    q = np.empty(m); q[o] = np.minimum(1, np.minimum.accumulate(adj[::-1])[::-1]); O['q'] = q
O.to_csv(f'{OUT}/v4_ext_{arm}_{tag}.csv', index=False, encoding='utf-8-sig')
print(O[O.sig == 'cell_real'].to_string())

# ---- 그룹 안 34 KSIC (사업체수 cell_real만 + 같은 rng 소비 순서) ----
codes = sorted(set(km[MULTI])); idx_by = {c: [i for i, k in enumerate(MULTI) if km[k] == c] for c in codes}
cm = km[MULTI].to_numpy()


def demean(sig, nm):
    g = GR[nm].loc[MULTI, YRS]; s = sig.loc[MULTI, YRS]; ok = g.notna() & s.notna()
    g = g.where(ok); s = s.where(ok)
    return (g - g.groupby(cm).transform('mean')).to_numpy(float), (s - s.groupby(cm).transform('mean')).to_numpy(float)


def pears(G_, S_):
    r = np.full(G_.shape[1], np.nan)
    for j in range(G_.shape[1]):
        a, b = S_[:, j], G_[:, j]; ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 8 and a[ok].std() > 0 and b[ok].std() > 0:
            aa = a[ok] - a[ok].mean(); bb = b[ok] - b[ok].mean(); r[j] = (aa * bb).sum() / np.sqrt((aa * aa).sum() * (bb * bb).sum())
    return r


rngw = np.random.default_rng(SW); wrows = []
for nm in ('출하액', '종사자수', '사업체수', '수출액', '수입액'):
    for sg in ('cell_real', 'cell_pred', 'bridge_real'):
        gp, sp = demean(S['prov', sg], nm); gt, st = demean(S['txt', sg], nm)
        rp, rt = pears(gp, sp), pears(gt, st); d = rp - rt; bs = np.empty(NB)
        need = (sg == 'cell_real')
        for b in range(NB):
            pick = rngw.choice(codes, len(codes)); keep = np.concatenate([idx_by[c] for c in pick])
            bs[b] = np.nanmean(pears(gp[keep], sp[keep]) - pears(gt[keep], st[keep])) if need else np.nan
        if need:
            wrows.append(dict(ext=nm, sig=sg, r_prov=np.nanmean(rp), r_txt=np.nanmean(rt), diff=np.nanmean(d), lo=np.nanpercentile(bs, 2.5),
                              hi=np.nanpercentile(bs, 97.5), wins=int(np.nansum(d > 0)), n=int(np.isfinite(d).sum())))
W = pd.DataFrame(wrows); W.to_csv(f'{OUT}/v4_within_{arm}_{tag}.csv', index=False, encoding='utf-8-sig')
print(W.to_string(), round(time.time() - t_0))
