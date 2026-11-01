# C04-T3 s2: 재현 관문 → 층별 성적(i) → 출원인 집중·단일 출원인 부상·대체 레이블 재채점(ii) → 개발 원점 → 대기업 몫 → KSIC 목록(iii) → 편향(D1, D3)
# 재학습 없음. 동결 예측 p, p_base 만 사용.
import os, sys, json, hashlib, html, re, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
t_start = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C04\C04-T3"); C = os.path.join(W, 'cache')
R3 = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3")
SEED = 20261012; NB = 2000
OFFICIAL = json.load(open(os.path.join(R3, 'final_result_r3.json'), encoding='utf-8'))
C01 = json.load(open(os.path.join(ROOT, r"검증루프_20261012\C01\C01-T2\verify_0\indep_results.json"), encoding='utf-8'))['T3']
CI_W = OFFICIAL['ci95']['skill'][1] - OFFICIAL['ci95']['skill'][0]

# ---------- 패널 ----------
pan = pd.read_parquet(os.path.join(C, 'panel.parquet')); pan['cell'] = pan.ksic + '|' + pan.mg
cells = np.array(sorted(pan.cell.unique())); ci = np.searchsorted(cells, pan.cell.values); Y0 = 2000
KR = np.zeros((len(cells), 25)); TOT = np.zeros((len(cells), 25))
np.add.at(TOT, (ci, pan.year.values - Y0), pan.n.values)
k_ = pan.cc.values == 'KR'; np.add.at(KR, (ci[k_], pan.year.values[k_] - Y0), pan.n.values[k_])
S = lambda A, a, b: A[:, a - Y0:b - Y0 + 1].sum(1)
hk = lambda cs: np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in cs], dtype=np.int64)

def topk_label(y, tie, cellv, k):
    o = np.lexsort((hk(cellv), -tie, -y)); yb = np.zeros(len(y), int); yb[o[:k]] = 1; return yb

def origin(t0):
    d = pd.DataFrame(dict(cell=cells, tot_W=S(TOT, t0 - 2, t0), kr_W=S(KR, t0 - 2, t0),
                          kr_mom=np.log((S(KR, t0 - 2, t0) + 1) / (S(KR, t0 - 5, t0 - 3) + 1)), kr_fut=S(KR, t0 + 1, t0 + 3)))
    d = d[d.tot_W >= 50].reset_index(drop=True); d['t0'] = t0
    d['y'] = np.log((d.kr_fut + 1) / (d.kr_W + 1)); k = int(round(0.10 * len(d)))
    d['yb'] = topk_label(d.y.values, (d.kr_fut - d.kr_W).values, d.cell.values, k)
    return d

def topk_hits(y, s, k):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if k <= 0: return dict(k=0, min=0, max=0, exp=0.0, quick=0)
    v = np.sort(s)[::-1][k - 1]; ab = s > v; tie = s == v
    need = k - ab.sum(); ha = int(y[ab].sum()); nt = int(tie.sum()); tp = int(y[tie].sum())
    return dict(k=int(k), min=int(ha + max(0, need - (nt - tp))), max=int(ha + min(need, tp)), exp=float(ha + need * tp / nt),
                quick=int(y[np.argsort(-s)[:k]].sum()))

def auc(y, p):
    y = np.asarray(y); return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
def skill(a, b): return (a - b) / (1 - b)

# ---------- 재현 관문 ----------
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
te = origin(2021)
m = P.merge(te, on='cell', how='outer', indicator=True, suffixes=('', '_r'))
gate = dict(merge={k: int(v) for k, v in m._merge.value_counts().items()})
m = m[m._merge == 'both'].drop(columns='_merge').reset_index(drop=True)
gate.update(yb_mismatch=int((m.yb != m.yb_r).sum()), y_maxdiff=float(np.abs(m.y - m.y_r).max()), krW_maxdiff=float(np.abs(m.kr_W - m.kr_W_r).max()),
            n=int(len(m)), pos=int(m.yb_r.sum()))
y = m.yb_r.values; p = m.p.values; pb = m.p_base.values.astype(float)
a_m, a_b = auc(y, p), auc(y, pb)
gate.update(auc=a_m, auc_base=a_b, skill=skill(a_m, a_b), top10=topk_hits(y, p, 504), top100=topk_hits(y, p, 100), top10_base=topk_hits(y, pb, 504),
            top100_base=topk_hits(y, pb, 100))
gate['match_official'] = bool(abs(a_m - OFFICIAL['summary']['auc']) < 1e-12 and abs(a_b - OFFICIAL['summary']['auc_baseline']) < 1e-9
                              and abs(gate['skill'] - OFFICIAL['summary']['mean_skill']) < 1e-9)
gate['match_C01'] = bool(abs(a_m - C01['p']['auc']) < 1e-12 and gate['top10']['quick'] == C01['p']['top10']['quick'] == 183
                         and gate['top100']['quick'] == C01['p']['top100']['quick'] and gate['top10_base']['min'] == C01['p_base']['top10']['min']
                         and gate['top10_base']['max'] == C01['p_base']['top10']['max'])
gate['pass'] = bool(gate['match_official'] and gate['match_C01'] and gate['yb_mismatch'] == 0 and gate['y_maxdiff'] < 1e-6 and gate['merge'].get('both') == 5041)
print('GATE', json.dumps(gate, ensure_ascii=False))
assert gate['pass']
m['y'] = m.y_r; m['yb'] = m.yb_r; m['kr_W'] = m.kr_W_r
m = m[['cell', 'ksic', 'mg', 't0', 'kr_W', 'kr_fut', 'kr_mom', 'tot_W', 'y', 'yb', 'p', 'p_base']]

# ---------- 출원인 키 ----------
docs = pd.read_parquet(os.path.join(C, 'kr_docs.parquet')); docs['cell'] = docs.ksic + '|' + docs.mg
names = pd.Series(docs.app_raw.fillna('').unique())
SUF2 = ['인코포레이티드', '인코퍼레이티드', '인크', '엘엘씨', '엘엘피', '리미티드', '코포레이션', '컴퍼니', '컴파니', '가부시키가이샤', '가부시끼가이샤', '가부시키카이샤',
        '게엠베하', '게젤샤프트', '아게', '에스에이', '에스아', '비브이', '엔브이', '씨오', '엘티디', '피엘씨', '에스피에이', '에스알엘', '유한회사', '유한책임회사', '주식회사']
def n0(s): return (s or '').strip()
def n1(s):
    s = html.unescape(s or '').lower()
    for w in ['주식회사', '(주)', '㈜', '(株)', '（주）']: s = s.replace(w, '')
    return re.sub(r"[\s\.,·\-\(\)\[\]'\"“”‘’&/]+", '', s)
def n2(s):
    s = n1(s)
    for w in SUF2: s = s.replace(w, '')
    return s if s else n1(s)
mp = {nm: (n0(nm), n1(nm), n2(nm)) for nm in names}
docs['k0'] = docs.app_raw.fillna('').map(lambda x: mp[x][0]); docs['k1'] = docs.app_raw.fillna('').map(lambda x: mp[x][1]); docs['k2'] = docs.app_raw.fillna('').map(lambda x: mp[x][2])
keyinfo = dict(n_raw=int(docs.k0.nunique()), n_k1=int(docs.k1.nunique()), n_k2=int(docs.k2.nunique()), n_missing=int((docs.k0 == '').sum()))
# 변이 영향 문서: 같은 k1 키에 원문 표기가 2개 이상인 키의 문서
var_k1 = docs.groupby('k1').k0.nunique(); docs['var1'] = docs.k1.map(var_k1).gt(1)
var_k2 = docs.groupby('k2').k1.nunique(); docs['var2'] = docs.k2.map(var_k2).gt(1)
docs['html'] = docs.app_raw.fillna('').str.contains('&#|&quot;|&amp;|&lt;|&gt;|&laquo;|&raquo;', regex=True)

# ---------- 대기업(상위 1%) ----------
big = {}
for kk in ['k1', 'k2']:
    vol = docs[(docs.year >= 2012) & (docs.year <= 2021) & (docs[kk] != '')].groupby(kk).size().sort_values(ascending=False)
    ntop = int(np.ceil(0.01 * len(vol))); big[kk] = set(vol.index[:ntop])
    keyinfo[f'top1pct_{kk}'] = dict(n_applicants=int(len(vol)), n_top=ntop, top_doc_share=float(vol.iloc[:ntop].sum() / vol.sum()), min_docs_top=int(vol.iloc[ntop - 1]))
docs['big1'] = docs.k1.isin(big['k1'])

def appl_stats(d, t0, key):
    """셀별 출원인 집중과 Δa. d: 평가 셀 DataFrame(cell, kr_W, kr_fut)."""
    sub = docs[docs.cell.isin(set(d.cell)) & (docs.year >= t0 - 2) & (docs.year <= t0 + 3)].copy()
    sub['w'] = np.where(sub.year <= t0, 'past', 'fut')
    g = sub.groupby(['cell', key, 'w']).size().unstack('w', fill_value=0)
    for c in ['past', 'fut']:
        if c not in g: g[c] = 0
    g = g.reset_index(); g['delta'] = g.fut - g.past
    g = g.sort_values(['cell', 'delta', 'fut', key], ascending=[True, False, False, True])
    top = g.groupby('cell').head(1).set_index('cell')
    fsum = g.groupby('cell').fut.sum()
    gf = g[g.fut > 0].copy(); gf['sh'] = gf.fut / gf.cell.map(fsum)
    hhi = gf.groupby('cell').sh.apply(lambda s: float((s ** 2).sum())); top1 = gf.groupby('cell').sh.max(); napp = gf.groupby('cell').size()
    out = pd.DataFrame(index=d.cell.values)
    out['top_app'] = top[key].reindex(out.index); out['top_delta'] = top.delta.reindex(out.index).fillna(0)
    out['top_past'] = top.past.reindex(out.index).fillna(0); out['top_fut'] = top.fut.reindex(out.index).fillna(0)
    out['fut_top1_share'] = top1.reindex(out.index); out['fut_hhi'] = hhi.reindex(out.index); out['fut_n_app'] = napp.reindex(out.index).fillna(0)
    # 상위 1% 몫(결과창 문서, 증가분)
    if key == 'k1':
        bb = sub[sub.big1].groupby(['cell', 'w']).size().unstack('w', fill_value=0)
        for c in ['past', 'fut']:
            if c not in bb: bb[c] = 0
        out['big_fut'] = bb.fut.reindex(out.index).fillna(0); out['big_past'] = bb.past.reindex(out.index).fillna(0)
        out['top_is_big'] = out.top_app.isin(big['k1'])
    out = out.reset_index().rename(columns={'index': 'cell'})
    # 패널 일치 확인: 문서표 합 = kr_W, kr_fut
    chk = sub.groupby(['cell', 'w']).size().unstack('w', fill_value=0).reindex(d.cell.values).fillna(0)
    for c in ['past', 'fut']:
        if c not in chk: chk[c] = 0
    out['_chk'] = (np.abs(chk.past.values - d.kr_W.values) + np.abs(chk.fut.values - d.kr_fut.values))
    return out

def sab_flags(d, a):
    D = (d.kr_fut - d.kr_W).values
    return (d.yb.values == 1) & (D > 0) & (a.top_delta.values >= 0.5 * D)

def alt_C(d, a):
    kw = d.kr_W.values - a.top_past.values; kf = d.kr_fut.values - a.top_fut.values
    yc = np.log((kf + 1) / (kw + 1)); k = int(round(0.10 * len(d)))
    return topk_label(yc, kf - kw, d.cell.values, k)

def score_set(y, p, pb, mask=None):
    if mask is not None: y, p, pb = y[mask], p[mask], pb[mask]
    am, ab = auc(y, p), auc(y, pb); return dict(n=int(len(y)), pos=int(y.sum()), auc=am, auc_base=ab, skill=skill(am, ab))

# ---------- 최종 원점: 출원인 ----------
A = {kk: appl_stats(m, 2021, kk) for kk in ['k0', 'k1', 'k2']}
for kk in A: assert A[kk]._chk.max() == 0, kk
a1 = A['k1']
m['D'] = m.kr_fut - m.kr_W
for kk in A: m[f'sab_{kk}'] = sab_flags(m, A[kk])
m['top_app_k1'] = a1.top_app.values; m['top_delta'] = a1.top_delta.values; m['fut_top1_share'] = a1.fut_top1_share.values
m['fut_hhi'] = a1.fut_hhi.values; m['fut_n_app'] = a1.fut_n_app.values; m['big_fut'] = a1.big_fut.values; m['big_past'] = a1.big_past.values
m['top_is_big'] = a1.top_is_big.values; m['top_past'] = a1.top_past.values; m['top_fut'] = a1.top_fut.values
m['yC'] = alt_C(m, a1)
m['stratum'] = np.select([m.kr_W == 0, m.kr_W == 1], ['S0', 'S1'], 'S2')
m['stratum5'] = np.select([m.kr_W == 0, m.kr_W == 1, m.kr_W <= 4], ['0', '1', '2-4'], '>=5')

# ---------- 부트스트랩 준비 ----------
rng = np.random.default_rng(SEED); mgv = m.mg.values; um = np.unique(mgv)
grp = pd.Series(np.arange(len(m))).groupby(mgv).apply(lambda s: s.values).to_dict()
BOOT = [np.concatenate([grp[u] for u in rng.choice(um, len(um))]) for _ in range(NB)]
def boot_ci(fn):
    vals = []
    for ii in BOOT:
        v = fn(ii)
        if v is not None and np.all(np.isfinite(v)): vals.append(v)
    vals = np.array(vals); return np.percentile(vals, [2.5, 97.5], axis=0), len(vals)

yv, pv, pbv = m.yb.values, m.p.values, m.p_base.values.astype(float)
top504 = np.zeros(len(m), bool); top504[np.argsort(-pv)[:504]] = True
rows = []
for lab, msk in [('ALL', np.ones(len(m), bool)), ('S0', (m.kr_W == 0).values), ('S1', (m.kr_W == 1).values), ('S2', (m.kr_W >= 2).values),
                 ('S0+S1', (m.kr_W <= 1).values), ('2-4', ((m.kr_W >= 2) & (m.kr_W <= 4)).values), ('>=5', (m.kr_W >= 5).values)]:
    yy, pp, bb = yv[msk], pv[msk], pbv[msk]; ns = int(msk.sum()); kk = max(1, int(round(0.1 * ns)))
    am, ab = auc(yy, pp), auc(yy, bb)
    def fn(ii, msk=msk):
        ii = ii[msk[ii]]; y2 = yv[ii]
        if y2.sum() == 0 or y2.sum() == len(y2): return None
        x, x0 = roc_auc_score(y2, pv[ii]), roc_auc_score(y2, pbv[ii]); return np.array([x, x0, x - x0, (x - x0) / (1 - x0)])
    (lo, hi), nb = boot_ci(fn)
    tm, tb = topk_hits(yy, pp, kk), topk_hits(yy, bb, kk)
    Dpos = (m.D.values[msk])[yy == 1]
    rows.append(dict(stratum=lab, n=ns, pos=int(yy.sum()), pos_rate=float(yy.mean()), auc=am, auc_lo=lo[0], auc_hi=hi[0], auc_base=ab, auc_base_lo=lo[1], auc_base_hi=hi[1],
                     auc_diff=am - ab, auc_diff_lo=lo[2], auc_diff_hi=hi[2], skill=skill(am, ab), skill_lo=lo[3], skill_hi=hi[3], n_boot=nb,
                     in_global_top504=int((top504 & msk).sum()), hits_in_global_top504=int((top504 & msk & (yv == 1)).sum()),
                     k_within=kk, p10_within_model=tm['quick'] / kk, p10_within_base_min=tb['min'] / kk, p10_within_base_max=tb['max'] / kk, p10_within_base_exp=tb['exp'] / kk,
                     pos_D_median=float(np.median(Dpos)) if len(Dpos) else np.nan, pos_D_le3=int((Dpos <= 3).sum()), pos_kr_fut_median=float(np.median(m.kr_fut.values[msk][yy == 1])) if len(Dpos) else np.nan,
                     sab_k1=int(m.sab_k1.values[msk].sum()), sab_rate_k1=float(m.sab_k1.values[msk].sum() / max(1, yy.sum()))))
strata = pd.DataFrame(rows); strata.to_csv(os.path.join(W, 'strata_scores.csv'), index=False, encoding='utf-8-sig')
print(strata[['stratum', 'n', 'pos', 'pos_rate', 'auc', 'auc_base', 'auc_diff', 'auc_diff_lo', 'auc_diff_hi', 'skill', 'skill_lo', 'skill_hi', 'in_global_top504', 'hits_in_global_top504', 'p10_within_model', 'p10_within_base_min', 'p10_within_base_max', 'pos_D_le3', 'sab_rate_k1']].round(4).to_string())

# ---------- 대체 레이블 재채점 ----------
base_sc = score_set(yv, pv, pbv)
def alt_rows(d, tag, t0, Ak=None):
    yv_, pv_, pbv_ = d.yb.values, d.p.values, d.p_base.values.astype(float)
    b = score_set(yv_, pv_, pbv_); out = []
    for kk in ['k0', 'k1', 'k2']:
        s = d[f'sab_{kk}'].values
        rA = score_set(yv_, pv_, pbv_, ~s); yB = yv_.copy(); yB[s] = 0; rB = score_set(yB, pv_, pbv_)
        for nm, r in [('A_exclude_SAB', rA), ('B_SAB_as_neg', rB)]:
            out.append(dict(origin=t0, set=tag, key=kk, alt=nm, n=r['n'], pos=r['pos'], sab=int(s.sum()), sab_rate=float(s.sum() / yv_.sum()),
                            auc=r['auc'], auc_base=r['auc_base'], skill=r['skill'], skill_ref=b['skill'], d_skill=r['skill'] - b['skill']))
    if 'yC' in d:
        rC = score_set(d.yC.values, pv_, pbv_)
        out.append(dict(origin=t0, set=tag, key='k1', alt='C_drop_top_applicant', n=rC['n'], pos=rC['pos'], sab=int(d.sab_k1.sum()), sab_rate=float(d.sab_k1.sum() / yv_.sum()),
                        auc=rC['auc'], auc_base=rC['auc_base'], skill=rC['skill'], skill_ref=b['skill'], d_skill=rC['skill'] - b['skill'],
                        overlap_with_orig=int(((d.yC.values == 1) & (yv_ == 1)).sum())))
    return out
alt = alt_rows(m, 'final', 2021)
# Δskill CI (A, k1)  — 같은 부트스트랩 표본
s1 = m.sab_k1.values
def fnA(ii):
    y2, p2, b2, s2 = yv[ii], pv[ii], pbv[ii], s1[ii]
    if y2.sum() == 0: return None
    x, x0 = roc_auc_score(y2, p2), roc_auc_score(y2, b2); k0 = (x - x0) / (1 - x0)
    keep = ~s2; y3 = y2[keep]
    if y3.sum() == 0: return None
    xa, xa0 = roc_auc_score(y3, p2[keep]), roc_auc_score(y3, b2[keep]); yB = y2.copy(); yB[s2] = 0
    xb, xb0 = roc_auc_score(yB, p2), roc_auc_score(yB, b2)
    return np.array([(xa - xa0) / (1 - xa0) - k0, (xb - xb0) / (1 - xb0) - k0, (xa - xa0) / (1 - xa0)])
(loA, hiA), nbA = boot_ci(fnA)
for r in alt:
    if r['key'] == 'k1' and r['alt'] == 'A_exclude_SAB': r.update(d_skill_lo=loA[0], d_skill_hi=hiA[0], skill_lo=loA[2], skill_hi=hiA[2])
    if r['key'] == 'k1' and r['alt'] == 'B_SAB_as_neg': r.update(d_skill_lo=loA[1], d_skill_hi=hiA[1])

# ---------- 개발 원점 ----------
DP = pd.read_parquet(os.path.join(R3, 'preds', 'dev_preds_r3.parquet'))
dev_gate = {}; dev_sab = []
for t0, g in DP.groupby('t0'):
    o = origin(int(t0)); assert t0 + 3 <= 2021
    mm = g.merge(o[['cell', 'kr_fut', 'kr_mom', 'yb', 'y']], on='cell', suffixes=('', '_r'))
    dev_gate[int(t0)] = dict(n=int(len(g)), merged=int(len(mm)), yb_mismatch=int((mm.yb != mm.yb_r).sum()), y_maxdiff=float(np.abs(mm.y - mm.y_r).max()))
    mm = mm.reset_index(drop=True)
    Ad = {kk: appl_stats(mm, int(t0), kk) for kk in ['k0', 'k1', 'k2']}
    for kk in Ad:
        assert Ad[kk]._chk.max() == 0
        mm[f'sab_{kk}'] = sab_flags(mm, Ad[kk])
    mm['yC'] = alt_C(mm, Ad['k1']); mm['D'] = mm.kr_fut - mm.kr_W
    alt += alt_rows(mm, 'dev', int(t0))
    pos = mm.yb == 1; a1d = Ad['k1']
    dev_sab.append(dict(origin=int(t0), n=int(len(mm)), pos=int(pos.sum()), sab_k1=int(mm.sab_k1.sum()), sab_rate_k1=float(mm.sab_k1.sum() / pos.sum()),
                        sab_rate_k0=float(mm.sab_k0.sum() / pos.sum()), sab_rate_k2=float(mm.sab_k2.sum() / pos.sum()),
                        sab_rate_D5=float((mm.sab_k1 & (mm.D >= 5)).sum() / max(1, (pos & (mm.D >= 5)).sum())),
                        pos_krW_le1=float((pos & (mm.kr_W <= 1)).sum() / pos.sum()), pos_D_le3=float((pos & (mm.D <= 3)).sum() / pos.sum()),
                        big_share_of_pos_increase=float(((a1d.big_fut - a1d.big_past).values[pos.values]).sum() / mm.D.values[pos.values].sum()),
                        sab_top_is_big=float((mm.sab_k1.values & a1d.top_is_big.values).sum() / max(1, mm.sab_k1.sum()))))
print('DEV gate', dev_gate)
altdf = pd.DataFrame(alt); altdf.to_csv(os.path.join(W, 'rescore_alt_labels.csv'), index=False, encoding='utf-8-sig')
print(altdf[altdf.set == 'final'].round(4).to_string())
devdf = pd.DataFrame(dev_sab)

# ---------- 최종 원점 요약 ----------
pos = m.yb.values == 1; D = m.D.values
fin = dict(n=int(len(m)), pos=int(pos.sum()),
           sab_rate={kk: float(m[f'sab_{kk}'].sum() / pos.sum()) for kk in ['k0', 'k1', 'k2']},
           sab_n={kk: int(m[f'sab_{kk}'].sum()) for kk in ['k0', 'k1', 'k2']},
           sab_rate_D5=float((m.sab_k1 & (m.D >= 5)).sum() / max(1, (pos & (D >= 5)).sum())), n_pos_D5=int((pos & (D >= 5)).sum()),
           sab_rate_by_stratum={s: float(m.sab_k1[(m.stratum == s) & pos].sum() / max(1, ((m.stratum == s) & pos).sum())) for s in ['S0', 'S1', 'S2']},
           pos_by_stratum={s: int(((m.stratum == s) & pos).sum()) for s in ['S0', 'S1', 'S2']},
           pos_D_dist={'1': int((pos & (D == 1)).sum()), '2': int((pos & (D == 2)).sum()), '3': int((pos & (D == 3)).sum()), '4-9': int((pos & (D >= 4) & (D <= 9)).sum()), '>=10': int((pos & (D >= 10)).sum())},
           pos_D_le3_share=float((pos & (D <= 3)).sum() / pos.sum()), pos_D_min=float(D[pos].min()),
           pos_kr_fut_dist={'<=2': int((pos & (m.kr_fut <= 2)).sum()), '3-5': int((pos & (m.kr_fut >= 3) & (m.kr_fut <= 5)).sum()), '>=6': int((pos & (m.kr_fut >= 6)).sum())},
           label_threshold_y=float(m.y[pos].min()),
           top504_hits=int((top504 & pos).sum()), top504_hits_S01=int((top504 & pos & (m.kr_W <= 1).values).sum()),
           top504_hits_sab=int((top504 & m.sab_k1.values).sum()), top504_hits_D_le3=int((top504 & pos & (D <= 3)).sum()),
           big_share_of_pos_increase=float((m.big_fut - m.big_past).values[pos].sum() / D[pos].sum()),
           big_share_fut_docs_all=float(m.big_fut.sum() / m.kr_fut.sum()), big_share_fut_docs_pos=float(m.big_fut[pos].sum() / m.kr_fut[pos].sum()),
           big_share_fut_docs_by_stratum={s: float(m.big_fut[m.stratum == s].sum() / max(1, m.kr_fut[m.stratum == s].sum())) for s in ['S0', 'S1', 'S2']},
           sab_top_is_big=float((m.sab_k1 & m.top_is_big).sum() / m.sab_k1.sum()),
           fut_top1_share_pos_median=float(m.fut_top1_share[pos].median()), fut_top1_share_neg_median=float(m.fut_top1_share[~pos & (m.kr_fut > 0)].median()),
           fut_hhi_pos_median=float(m.fut_hhi[pos].median()), fut_n_app_pos_median=float(m.fut_n_app[pos].median()),
           top_sab_applicants=m[m.sab_k1].groupby('top_app_k1').size().sort_values(ascending=False).head(10).to_dict())
# 단일 출원인 부상을 낸 출원인의 집중: 상위 10 출원인이 SAB 중 차지하는 몫
fin['sab_top10_applicants_share'] = float(m[m.sab_k1].groupby('top_app_k1').size().sort_values(ascending=False).head(10).sum() / m.sab_k1.sum())
fin['dev_sab_range_k1'] = [float(devdf.sab_rate_k1.min()), float(devdf.sab_rate_k1.max())]
fin['final_vs_dev'] = '최종창 특이' if not (devdf.sab_rate_k1.min() <= fin['sab_rate']['k1'] <= devdf.sab_rate_k1.max()) else '구조적(원점 공통)'

# ---------- applicant_concentration.csv (셀 단위) ----------
ac = m[['cell', 'ksic', 'mg', 'kr_W', 'kr_fut', 'D', 'y', 'yb', 'yC', 'p', 'p_base', 'stratum', 'top_app_k1', 'top_past', 'top_fut', 'top_delta', 'fut_top1_share', 'fut_hhi', 'fut_n_app',
        'big_past', 'big_fut', 'top_is_big', 'sab_k0', 'sab_k1', 'sab_k2']].copy()
ac['top_app_k1'] = np.where(ac.top_is_big, ac.top_app_k1, np.where(ac.top_app_k1.notna(), 'nonbig:' + ac.top_app_k1.fillna('').map(lambda s: hashlib.md5(s.encode()).hexdigest()[:8]), None))
ac.to_csv(os.path.join(W, 'applicant_concentration.csv'), index=False, encoding='utf-8-sig')
devdf.to_csv(os.path.join(W, 'dev_sab_by_origin.csv'), index=False, encoding='utf-8-sig')

# ---------- (iii) KSIC 목록 ----------
LIST = {'26112': '3.2.3', '28902': '3.2.3/4.2', '31312': '3.2.3/4.2', '27211': '3.2.3', '15220': '4.2', '20491': '4.2', '29133': '4.2', '33309': '4.2'}
FF = pd.read_csv(os.path.join(R3, 'future_forecast_r3.csv'), dtype={'ksic': str}); FF['cell'] = FF.ksic + '|' + FF.mg
rec = docs[(docs.year >= 2022) & (docs.year <= 2024)]
k3 = []
for ks, sec in LIST.items():
    pc = set(FF[(FF.ksic == ks) & (FF.pred_breakout_top10 == 1)].cell)
    ra = rec[rec.cell.isin(pc)].groupby('k1').size().sort_values(ascending=False)
    rb = rec[rec.ksic == ks].groupby('k1').size().sort_values(ascending=False)
    mk = m[(m.ksic == ks) & (m.yb == 1)]
    # 실현 부상 셀 증가분 중 상위 출원인 몫: KSIC 안 양성 셀들의 Δa를 출원인별로 합산
    sub = docs[docs.cell.isin(set(mk.cell)) & (docs.year >= 2019) & (docs.year <= 2024)]
    if len(sub):
        dd = sub.assign(w=np.where(sub.year <= 2021, -1, 1)).groupby('k1').w.sum().sort_values(ascending=False)
        c_share = float(dd.iloc[0] / mk.D.sum()) if mk.D.sum() > 0 else np.nan; c_big = bool(dd.index[0] in big['k1'])
    else: c_share, c_big = np.nan, None
    k3.append(dict(ksic=ks, section=sec, n_pred_cells=len(pc), a_docs=int(ra.sum()), a_top1_share=float(ra.iloc[0] / ra.sum()) if len(ra) else np.nan,
                   a_top1_is_big=bool(ra.index[0] in big['k1']) if len(ra) else None, a_n_app=int(len(ra)),
                   b_docs=int(rb.sum()), b_top1_share=float(rb.iloc[0] / rb.sum()) if len(rb) else np.nan, b_top1_is_big=bool(rb.index[0] in big['k1']) if len(rb) else None,
                   c_final_pos_cells=int(len(mk)), c_final_increase=float(mk.D.sum()), c_top1_share_of_increase=c_share, c_top1_is_big=c_big,
                   c_sab_cells=int(mk.sab_k1.sum())))
k3 = pd.DataFrame(k3); k3.to_csv(os.path.join(W, 'ksic_list_top_applicant.csv'), index=False, encoding='utf-8-sig')
print(k3.round(3).to_string())

# ---------- D1·D3 편향 ----------
fd = docs[(docs.year >= 2022) & (docs.year <= 2024)].merge(m[['cell', 'stratum', 'yb', 'sab_k1']], on='cell')
D1 = fd.groupby('stratum').agg(docs=('appl_no', 'size'), missing=('k0', lambda s: int((s == '').sum())), html=('html', 'mean'), var1=('var1', 'mean'), var2=('var2', 'mean'))
D1.loc['POS'] = [int((fd.yb == 1).sum()), int(((fd.yb == 1) & (fd.k0 == '')).sum()), fd.html[fd.yb == 1].mean(), fd.var1[fd.yb == 1].mean(), fd.var2[fd.yb == 1].mean()]
cn = pd.read_parquet(os.path.join(C, 'cn_missing_ky.parquet')); cn21 = cn[cn.year == 2021].set_index('ksic'); cn21['rate'] = cn21.n_null / cn21.n_all
m['cn21_miss'] = m.ksic.map(cn21.rate)
cnw = cn[(cn.year >= 2019) & (cn.year <= 2021)].groupby('ksic')[['n_null', 'n_all']].sum(); m['cnW_miss'] = m.ksic.map(cnw.n_null / cnw.n_all)
D3 = m.groupby('stratum')[['cn21_miss', 'cnW_miss']].mean()
D3.loc['S0+S1'] = m[m.kr_W <= 1][['cn21_miss', 'cnW_miss']].mean()
D3.loc['POS_SAB'] = m[m.sab_k1][['cn21_miss', 'cnW_miss']].mean(); D3.loc['POS_nonSAB'] = m[(m.yb == 1) & ~m.sab_k1][['cn21_miss', 'cnW_miss']].mean()
D3.loc['TOP504'] = m[top504][['cn21_miss', 'cnW_miss']].mean()
print(D1.round(4)); print(D3.round(4))

out = dict(gate=gate, dev_gate=dev_gate, keyinfo=keyinfo, final=fin, dev=devdf.to_dict('records'), ci_width=CI_W, thr_dskill=0.5 * CI_W,
           D1=D1.reset_index().to_dict('records'), D3=D3.reset_index().to_dict('records'), sec=round(time.time() - t_start, 1))
json.dump(out, open(os.path.join(W, 's2_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({k: v for k, v in fin.items()}, ensure_ascii=False, indent=1, default=str))
print(devdf.round(3).to_string()); print('keyinfo', keyinfo); print('sec', out['sec'])
