# -*- coding: utf-8 -*-
"""verify_0 — 독립 검증(c08lib·t4lib·r2lib 미사용). sqlite(mode=ro) 직접 SQL + 파이썬 날짜 계산.
1) 레이블 원천 재생성 → s04 labels.parquet 대조  2) AUC·skill(sklearn) 대조  3) 다른 구현 부트스트랩(B=300, 군집 인덱스 재표집, seed 7)
4) 위약: 레이블 순열 1,000회, KSIC×due_win 층 안 순열 200회  5) 누수 반박: 표본 특허의 L 시점 피처를 SQL로 재계산(pd<=L) vs 동결, 같은 피처를 2026-08 전체로 계산하면 달라지는지(검정력)
6) 결손 민감도(사후): 등록번호 공란 월 코호트를 기대치만큼 가중
출력: verify_0/v0_out.json"""
import os, sys, json, sqlite3, datetime as dt, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
FUT = os.path.join(BASE, r"분석\모델경쟁_20260927\T4_특허포기예측\R2\runs_future\FUT")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
O = {}
# ---- pid: patent rowid 순, rgstDate 유효 & rgst_no '10'
P = pd.read_sql("select rowid rid, rgst_no, rgstDate from patent order by rowid", con)
P = P[P.rgstDate.str.match(r'^\d{8}$', na=False) & P.rgst_no.str.startswith('10')].reset_index(drop=True)
P['pid'] = np.arange(len(P))
fz = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet'))
rs = P.iloc[fz.pid.values]
def addy(s, k):
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    if m == 2 and d == 29: d = 28
    return dt.date(y + int(k), m, d)
# ---- 불납 소멸등록 (SQL)
lap = pd.read_sql("""select rgst_no, rgstCsDate cd from right where rgstCsName = '소멸등록' and rgstCsReason in ('등록료불납','연차료불납')
       and rgstCsDate <= '20260815' and rgstCsDate glob '[12]*'""", con)
lap = lap[lap.rgst_no.isin(set(rs.rgst_no))].reset_index(drop=True)
_pq = pd.read_sql("select rgst_no, payDate, lastAnnl from pay where payDate glob '[12]*'", con)
_pq = _pq[_pq.rgst_no.isin(set(lap.rgst_no))]
_pq['l'] = pd.to_numeric(_pq.lastAnnl, errors='coerce')
_m = lap.reset_index().merge(_pq, on='rgst_no')
_m = _m[_m.payDate < _m.cd]
lap['lmax'] = lap.index.map(_m.groupby('index').l.max())
lap = lap[lap.rgst_no.isin(set(rs.rgst_no)) & lap.lmax.notna()].copy()
rdmap = dict(zip(P.rgst_no, P.rgstDate))
lap['E'] = [addy(rdmap[r], k) for r, k in zip(lap.rgst_no, lap.lmax)]
win = lap[(lap.E > dt.date(2025, 7, 1)) & (lap.E <= dt.date(2025, 10, 31))].copy()
rec = pd.read_sql("select rgst_no, max(rgstCsDate) rcd from right where rgstCsName = '회복등록' and rgstCsDate <= '20260815' group by rgst_no", con)
win = win.merge(rec, on='rgst_no', how='left')
win['restored'] = win.rcd.notna() & (win.rcd >= win.cd)
pos = set(win[~win.restored].rgst_no); posS1 = set(win.rgst_no)
y = rs.rgst_no.isin(pos).astype(int).values; yS1 = rs.rgst_no.isin(posS1).astype(int).values
lb = pd.read_parquet(os.path.join(W, 'cache', 'labels.parquet'))
O['labels'] = dict(pos_indep=int(y.sum()), pos_s04=int(lb.y.sum()), mismatch=int((y != lb.y.values).sum()),
                   pos_S1_indep=int(yS1.sum()), mismatch_S1=int((yS1 != lb.y_S1.values).sum()), restored=int(win.restored.sum()))
# ---- 점수
p, pb, ps = fz.p.values, fz.p_base.values, fz.p_strong.values
A = lambda yy, s: float(roc_auc_score(yy, s)); sk = lambda a, b: (a - b) / (1 - b)
due = fz.due_win.values == 1
O['scores'] = dict(auc=A(y, p), auc_base=A(y, pb), skill=sk(A(y, p), A(y, pb)), skill_strong=sk(A(y, p), A(y, ps)),
                   skill_due=sk(A(y[due], p[due]), A(y[due], pb[due])), skill_due_strong=sk(A(y[due], p[due]), A(y[due], ps[due])))
res = json.load(open(os.path.join(W, 'prospective_result.json'), encoding='utf-8'))
O['scores']['diff_vs_s04'] = dict(auc=abs(O['scores']['auc'] - res['main']['auc_model']), skill=abs(O['scores']['skill'] - res['main']['skill']),
                                  skill_due=abs(O['scores']['skill_due'] - res['due_win']['skill']))
# ---- 다른 구현 부트스트랩: 군집 인덱스 재표집(행 복제)
cl = np.where(fz.acd.values == 'NA', 'P' + fz.pid.astype(str).values, fz.acd.values)
codes, uniq = pd.factorize(cl); K = len(uniq)
order = np.argsort(codes, kind='stable'); starts = np.searchsorted(codes[order], np.arange(K)); ends = np.r_[starts[1:], len(codes)]
rng = np.random.default_rng(7); bsk, bskd = [], []
for it in range(300):
    pick = rng.integers(0, K, K)
    idx = np.concatenate([order[starts[c]:ends[c]] for c in pick])
    yy = y[idx]; bsk.append(sk(A(yy, p[idx]), A(yy, pb[idx])))
    dm = due[idx]; bskd.append(sk(A(yy[dm], p[idx][dm]), A(yy[dm], pb[idx][dm])))
O['bootstrap_alt'] = dict(B=300, seed=7, skill_ci=[float(np.percentile(bsk, 2.5)), float(np.percentile(bsk, 97.5))],
                          skill_due_ci=[float(np.percentile(bskd, 2.5)), float(np.percentile(bskd, 97.5))])
# ---- 위약
from scipy.stats import rankdata
rp, rb = rankdata(p), rankdata(pb); n1 = int(y.sum()); n0 = len(y) - n1
fa = lambda yy, r: (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rng = np.random.default_rng(11); pl = []
for it in range(1000):
    yy = rng.permutation(y); pl.append(sk(fa(yy, rp), fa(yy, rb)))
pl = np.array(pl)
strata = pd.factorize(fz.ksic.astype(str).values + '_' + fz.due_win.astype(str).values)[0]
grp = pd.Series(np.arange(len(y))).groupby(strata).apply(np.array).tolist()
ps_ = []
for it in range(200):
    yy = y.copy()
    for g in grp: yy[g] = rng.permutation(y[g])
    ps_.append((sk(fa(yy, rp), fa(yy, rb)), fa(yy, rp), fa(yy, rb)))
ps_ = np.array(ps_)
O['placebo'] = dict(full_perm=dict(B=1000, mean=float(pl.mean()), p97_5=float(np.percentile(pl, 97.5)), max=float(pl.max()), n_ge_obs=int((pl >= O['scores']['skill']).sum())),
                    strat_ksic_due=dict(B=200, skill_mean=float(ps_[:, 0].mean()), skill_p97_5=float(np.percentile(ps_[:, 0], 97.5)),
                                        auc_model_mean=float(ps_[:, 1].mean()), auc_base_mean=float(ps_[:, 2].mean()),
                                        n_ge_obs=int((ps_[:, 0] >= O['scores']['skill']).sum())))
# ---- 누수 반박: 표본 3,000건 L 시점 paid_year 재계산(SQL) vs 동결, 그리고 2026-08 전체 기준
ev = pd.read_parquet(os.path.join(FUT, 'ev.parquet'), columns=['pid', 'paid_year', 'n_renew'])
smp = np.random.default_rng(5).choice(len(fz), 3000, replace=False)
rn = rs.rgst_no.values[smp]
q = pd.read_sql("select rgst_no, payDate, cast(lastAnnl as integer) l, seq from pay where payDate glob '[12]*'", con)
q = q[q.rgst_no.isin(set(rn))]
le = q[q.payDate <= '20250701'].groupby('rgst_no').l.max(); full = q.groupby('rgst_no').l.max()
pyL = pd.Series(rn).map(le).fillna(0).values; pyF = pd.Series(rn).map(full).fillna(0).values
evp = ev.paid_year.values[smp]
O['leak_refute'] = dict(n=3000, paid_year_le_L_eq_frozen=float((pyL == evp).mean()), paid_year_full_eq_frozen=float((pyF == evp).mean()),
                        note='동결 피처가 L 필터판과 일치하고 전체판과는 다르면, 피처가 L 이후 납부를 쓰지 않았다는 근거')
# 결과(납부 연장 여부)와 동결 점수의 관계가 창 안 기한 특허에서만 의미 있는지(창 밖 특허는 사건 0)
O['leak_refute']['pos_outside_due'] = int(y[~due].sum())
# ---- 결손 민감도(사후, 비사전등록): 공란 월 코호트 가중
rdm = pd.to_datetime(rs.rgstDate.values, format='%Y%m%d')
ym = rdm.strftime('%Y-%m')
ref = pd.Series(ym[(rdm >= '2022-11-01') & (rdm <= '2023-10-31')]).value_counts().mean()   # 결손 전 12개월 평균(위험집합 기준)
cnt = pd.Series(ym).value_counts()
gap = [m for m in cnt.index if ('2023-12' <= m <= '2024-08') or ('2025-02' <= m <= '2025-06')]
wt = np.ones(len(y))
for m in gap:
    if cnt[m] < ref: wt[ym == m] = ref / cnt[m]
O['gap_sensitivity'] = dict(ref_monthly=float(ref), gap_months=gap, added_weight=float(wt.sum() - len(y)),
                            auc=float(roc_auc_score(y, p, sample_weight=wt)), auc_base=float(roc_auc_score(y, pb, sample_weight=wt)))
O['gap_sensitivity']['skill'] = sk(O['gap_sensitivity']['auc'], O['gap_sensitivity']['auc_base'])
O['secs'] = round(time.time() - t0, 1)
json.dump(O, open(os.path.join(HERE, 'v0_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))
