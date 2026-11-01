# -*- coding: utf-8 -*-
"""반박 검증 rf1: S1-full/S1-fixed·L-a, S2, OFF skill을 원천 캐시에서 독자 재구성(작업 스크립트 import 없음).
 모집단: T1 cache(ins_firm, oos, app_pub, ins_pat) + kr_registered 공개일자. 점수: FROZEN o2019, baseline_scores, s1F/ctrlF scores(저장 파일).
 레이블: FROZEN o2019 y_u(공식). 출력: verify_0/rf1_out.json"""
import sys, json, time, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CA = T1/'cache'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'
W = B/'분석/검증루프_20261012/C16/C16-T1'; OUT = W/'verify_0'
V = 20191231


def d8(x):
    s = pd.Series(x).fillna('').astype(str).str.replace('-', '', regex=False).str.strip().str[:8]
    v = pd.to_numeric(s, errors='coerce'); v[(s.str.len() != 8)] = np.nan
    return v.fillna(99999999).astype(np.int64).values


def plus18(x):
    d = pd.to_datetime(pd.Series(x).astype(str).str[:8], format='%Y%m%d', errors='coerce') + pd.DateOffset(months=18)
    return pd.to_numeric(d.dt.strftime('%Y%m%d'), errors='coerce').fillna(99999999).astype(np.int64).values


K = sorted(pd.read_parquet(CA/'ins_pat.parquet', columns=['ksic']).ksic.unique()); ki = {k: i for i, k in enumerate(K)}
ins = pd.read_parquet(CA/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ksic', 'ay', 'appl_date', 'rgst_date'])
ins = ins[(ins.ay >= 2000) & (ins.ay <= 2019)].copy()
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=['등록번호', '공개일자', '등록일자'])
pub = dict(zip(kr['등록번호'].astype(str), kr['공개일자'].astype(str)))
ins['pub'] = d8(ins.rgst_no.astype(str).map(pub).values); ins['rg'] = d8(ins.rgst_date.values)
ins['v'] = np.minimum(ins.rg.values, ins.pub.values); ins['kk'] = ins.ksic.map(ki)
diag = dict(ins_rows=len(ins), ins_pub_missing=int((ins.pub == 99999999).sum()), ins_pub_after_rgst=int((ins.pub > ins.rg).sum()),
            ins_rgst_after_V=int((ins.rg > V).sum()), ins_vis_after_V=int((ins.v > V).sum()))
oos = pd.read_parquet(CA/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'appl_date', 'rgst_date', 'self'])
oos = oos[oos.self & (oos.ay >= 2000) & (oos.ay <= 2019)].copy()
oos['v'] = np.minimum(d8(oos.rgst_date.values), plus18(oos.appl_date.values))
app = pd.read_parquet(CA/'app_pub.parquet', columns=['applicant_cd', 'ay', 'pub_date', 'rgst_date', 'ksic'])
app = app[(app.ay >= 2000) & (app.ay <= 2019)].copy(); app['kk'] = app.ksic.astype(str).map(ki); app = app.dropna(subset=['kk']); app['kk'] = app.kk.astype(int)
app['v'] = np.minimum(d8(app.pub_date.values), d8(app.rgst_date.values))


def nfirm(vis):
    i = ins if not vis else ins[ins.v <= V]; o = oos if not vis else oos[oos.v <= V]
    c = pd.concat([i[['applicant_cd', 'rgst_no']].drop_duplicates().applicant_cd, o[['applicant_cd', 'rgst_no']].drop_duplicates().applicant_cd]).value_counts()
    return set(c[c >= 3].index)


Foff = nfirm(False); Fvis = nfirm(True)
io = ins[ins.applicant_cd.isin(Foff)]
held_off = set(zip(io.applicant_cd, io.kk)); iv = io[io.v <= V]; held_vis = set(zip(iv.applicant_cd, iv.kk))
prior_vis = set(zip(app.applicant_cd[app.v <= V], app.kk[app.v <= V]))
C = sorted((f, k) for (f, k) in held_off - held_vis if f in Fvis and (f, k) not in prior_vis)
vis_any = set(iv.applicant_cd)

o = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'newc', 'prior_any', 'px', 'pl'])
L15 = pd.read_parquet(B/'분석/검증루프_20261012/C15/C15-T2/cache/t1/final_labels.parquet', columns=['firm', 'k', 'y_u']); assert (L15.firm.values == o.firm.values).all() and (L15.k.values == o.k.values).all(); o['y_u'] = L15.y_u.values
pt = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k']); assert (pt.firm.values == o.firm.values).all() and (pt.k.values == o.k.values).all()
bo = pd.read_parquet(FC/'baseline_scores.parquet'); BK = list(bo.columns)
for c in BK: o[c] = bo[c].values
cF = pd.read_parquet(W/'cache/build/ctrlF/scores.parquet', columns=['firm', 'k', 'sbb']); assert (cF.firm.values == o.firm.values).all() and (cF.k.values == o.k.values).all()
o['sbb'] = cF.sbb.values
S = pd.read_parquet(W/'cache/build/s1F/scores.parquet', columns=['firm', 'k', 'px', 'pl', 'sbb', 'is_extra'] + BK)
Sidx = pd.Series(np.arange(len(S)), index=pd.MultiIndex.from_arrays([S.firm.values, S.k.values]))
okey = pd.MultiIndex.from_arrays([o.firm.values, o.k.values])
pv = okey.isin(pd.MultiIndex.from_tuples(list(prior_vis)))
fin = o.firm.isin(Fvis).values
diag.update(Foff=len(Foff), te_firms=int(o.firm.nunique()), Foff_eq_te=set(o.firm.unique()) == Foff, Fvis=len(Fvis), C=len(C), C_firms=len(set(f for f, _ in C)),
            A=int((~o.prior_any.values & fin).sum()), Bn=int((o.prior_any.values & fin & ~pv).sum()), D=int((~o.prior_any.values & ~fin).sum()),
            prior_vis_pairs=len(prior_vis))
# 비-C 법인 행에서 s1F 점수가 공식과 같은가
po = Sidx.reindex(okey).values.astype(np.int64)
Cf = set(f for f, _ in C)
nonC = ~o.firm.isin(set(S.firm[S.is_extra.values.astype(bool)])).values; diag['n_extra']=int(S.is_extra.sum())
diag['s1F_vs_off_maxdiff_nonCfirms'] = {c: float(np.abs(S[c].values[po][nonC] - o[c].values[nonC]).max()) for c in ['px', 'pl', 'sbb'] + BK}
diag['s1F_vs_off_maxdiff_Cfirms_base'] = {c: float(np.abs(S[c].values[po][~nonC] - o[c].values[~nonC]).max()) for c in ['sbb'] + BK}
diag['s1F_vs_off_maxdiff_Cfirms_px'] = float(np.abs(S.px.values[po][~nonC] - o.px.values[~nonC]).max())


def rpct(x): return pd.Series(x).rank(pct=True).values


def build(var):
    cols = ['firm', 'newc', 'y', 'px', 'pl', 'sbb'] + BK
    if var == 'OFF':
        m = ~o.prior_any.values; D = o[m].copy(); D['y'] = D.y_u.values; return D[cols].assign(isC=False)
    if var == 'S2':
        m = ~o.prior_any.values & fin; D = o[m].copy(); D['y'] = D.y_u.values; return D[cols].assign(isC=False)
    m = (~o.prior_any.values & fin) | (o.prior_any.values & fin & ~pv)
    D = o[m].copy(); D['y'] = np.where(D.prior_any.values, 0, D.y_u.values)
    if var == 'S1full':
        pp = po[m]
        for c in ['px', 'pl', 'sbb'] + BK: D[c] = S[c].values[pp]
    ci = Sidx.reindex(pd.MultiIndex.from_tuples(C)).values; assert not np.isnan(ci.astype(float)).any()
    Cd = S.iloc[ci.astype(np.int64)].copy(); Cd['y'] = 0; Cd['newc'] = False
    D = pd.concat([D[cols].assign(isC=False), Cd[cols].assign(isC=True)], ignore_index=True)
    D['newc'] = ~D.firm.isin(vis_any).values
    return D


def wauc_factory(s, y):
    u, inv = np.unique(np.asarray(s, np.float64), return_inverse=True); y = y.astype(np.float64); G = len(u)
    def f(w):
        p = np.bincount(inv, w * y, G); n = np.bincount(inv, w * (1 - y), G)
        return float((p * (np.cumsum(n) - 0.5 * n)).sum() / (p.sum() * n.sum()))
    return f


res = {}; boots = {}
for var in ['OFF', 'S2', 'S1fixed', 'S1full']:
    D = build(var); y = D.y.values.astype(int); s = 0.9 * rpct(D.px.values) + 0.1 * rpct(D.pl.values)
    a = roc_auc_score(y, s); ba = {c: roc_auc_score(y, D[c].values) for c in BK}; M0 = max(ba.values()); asb = roc_auc_score(y, D.sbb.values)
    order = np.argsort(-s, kind='stable')
    r = dict(n=len(D), firms=int(D.firm.nunique()), pos=int(y.sum()), auc=a, M0=M0, M0_arg=max(ba, key=ba.get), skill=(a - M0) / (1 - M0),
             skill_sbb=(a - asb) / (1 - asb), top1000=int(y[order[:1000]].sum()), top1000_C=int(D.isC.values[order[:1000]].sum()))
    # 법인 군집 부트스트랩: 공식 법인 목록(정렬) 기준으로 같은 가중치 → 짝지은 Δ 가능; 그리고 작업과 같은 방식(변형 법인 목록)도
    fl_off = np.array(sorted(o.firm.unique())); fidx = pd.Series(np.arange(len(fl_off)), index=fl_off).reindex(D.firm.values).values.astype(np.int64)
    fl_var = np.array(sorted(D.firm.unique())); vidx = pd.Series(np.arange(len(fl_var)), index=fl_var).reindex(D.firm.values).values.astype(np.int64)
    fs = [wauc_factory(s, y)] + [wauc_factory(D[c].values, y) for c in BK] + [wauc_factory(D.sbb.values, y)]
    A1 = np.zeros((200, 8)); A2 = np.zeros((200, 8))
    for ci_ in range(20):
        rng1 = np.random.RandomState(3000 + ci_); rng2 = np.random.RandomState(9000 + ci_)
        for j in range(10):
            w1 = np.bincount(rng1.randint(0, len(fl_var), len(fl_var)), minlength=len(fl_var)).astype(float)[vidx]
            w2 = np.bincount(rng2.randint(0, len(fl_off), len(fl_off)), minlength=len(fl_off)).astype(float)[fidx]
            A1[ci_ * 10 + j] = [f(w1) for f in fs] if var in ('OFF', 'S1full') else 0; A2[ci_ * 10 + j] = [f(w2) for f in fs]
    sk = lambda A: (A[:, 0] - A[:, 1:7].max(1)) / (1 - A[:, 1:7].max(1))
    sks = lambda A: (A[:, 0] - A[:, 7]) / (1 - A[:, 7])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    r['ci_same_seed_as_work'] = q(sk(A1)); r['ci_sbb_same_seed'] = q(sks(A1)); r['ci_offfirm_seed9000'] = q(sk(A2))
    boots[var] = sk(A2); res[var] = r
    print(var, json.dumps(r, ensure_ascii=False), round(time.time() - t0), flush=True)
for var in ['S2', 'S1fixed', 'S1full']:
    d = boots[var] - boots['OFF']
    res[var]['delta'] = res[var]['skill'] - res['OFF']['skill']
    res[var]['paired_delta_ci_offfirm'] = [float(np.percentile(d, 2.5)), float(np.median(d)), float(np.percentile(d, 97.5))]
    res[var]['paired_share_le_-0.0087'] = float((d <= -0.0087).mean()); res[var]['paired_share_lt0'] = float((d < 0).mean())
res['diag'] = diag; res['secs'] = round(time.time() - t0)
json.dump(res, open(OUT/'rf1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(diag, ensure_ascii=False, default=str))
