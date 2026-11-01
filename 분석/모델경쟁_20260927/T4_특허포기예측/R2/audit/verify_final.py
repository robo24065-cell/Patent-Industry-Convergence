# -*- coding: utf-8 -*-
"""최종시험 결과 독립 재계산 (r2lib·t4lib 를 import 하지 않는다).
1) 평가 레이블을 registry_full.sqlite(읽기 전용) 원자료에서 직접 다시 만든다(납부완료일 기준, label_np 와 같은 정의를 독립 구현).
2) 구성요소 예측(npy)에서 앙상블을 spec 가중치로 다시 합성하고, AUC 를 순위합(Mann-Whitney)으로 직접 계산.
3) 1순위 출원인 군집 부트스트랩을 다른 구현(인덱스 재표집·sklearn AUC)으로 B=300 재계산해 CI 를 대조.
결과: R2/audit/verify_final.json
"""
import os, sys, json, glob, sqlite3
import numpy as np, pandas as pd
from scipy.stats import rankdata

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T4 = os.path.dirname(R2)
OD = os.path.join(R2, 'runs_final', 'F22')
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
L = pd.Timestamp('2022-01-01'); END = pd.Timestamp('2025-01-01'); DATA_END = pd.Timestamp('2025-08-22'); GRACE = pd.Timedelta(days=213)


def addy(ts, k):
    """기념일 + k년 (2/29 -> 2/28)"""
    y = ts.dt.year + k; m = ts.dt.month; d = ts.dt.day.where(~((ts.dt.month == 2) & (ts.dt.day == 29)), 28)
    ok = ts.notna() & pd.notna(y)
    out = pd.Series(pd.NaT, index=ts.index, dtype='datetime64[ns]')
    out[ok] = pd.to_datetime(pd.DataFrame({'year': y[ok].astype(int), 'month': m[ok], 'day': d[ok]}), errors='coerce')
    return out


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- 평가 행
ev = pd.read_parquet(os.path.join(OD, 'ev.parquet'), columns=['pid', 'L', 'ksic', 'acd', 'y'])
pt = pd.read_parquet(os.path.join(T4, 'cache', 'patent.parquet'), columns=['rgst_no', 'rgstDate'])
pt = pt[pd.to_datetime(pt.rgstDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce').notna() & (pt.rgst_no.str[:2] == '10')].reset_index(drop=True)
ev['rgst_no'] = pt.rgst_no.values[ev.pid.values]
print('eval rows', len(ev), 'L', ev.L.unique(), flush=True)

# ---------------------------------------------------------------- 원자료에서 레이블 재구성
con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
con.execute('create temp table q(r text primary key)')
con.executemany('insert into q values (?)', [(r,) for r in ev.rgst_no.unique()])
P = pd.read_sql_query('select p.rgst_no, p.rgstDate, p.cndrtExptnDate, p.applDate from patent p join q on p.rgst_no=q.r', con)
pay = pd.read_sql_query('select a.rgst_no, a.lastAnnl, a.payDate from pay a join q on a.rgst_no=q.r', con)
rt = pd.read_sql_query("select b.rgst_no, b.rgstCsName, b.rgstCsDate from right b join q on b.rgst_no=q.r where b.rgstCsName in ('권리의 말소등록','말소등록','직권말소등록')", con)
con.close()
P = P.drop_duplicates('rgst_no').set_index('rgst_no')
P['rd'] = pd.to_datetime(P.rgstDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
P['exp'] = pd.to_datetime(P.cndrtExptnDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
ad = pd.to_datetime(P.applDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
P['exp'] = P.exp.fillna(addy(ad, 20))
pay['l'] = pd.to_numeric(pay.lastAnnl, errors='coerce'); pay['pd'] = pd.to_datetime(pay.payDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
pay = pay[pay.l.notna() & pay.pd.notna() & (pay.pd <= DATA_END)]
P['lmax'] = pay.groupby('rgst_no').l.max()
P['pth'] = pd.NaT
ok = P.lmax.notna()
P.loc[ok, 'pth'] = addy(P.rd[ok], P.lmax[ok].astype(int))
P['pth'] = pd.to_datetime(P.pth)
rt['cd'] = pd.to_datetime(rt.rgstCsDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
mz = set(rt[(rt.cd > L) & (rt.cd <= min(END, DATA_END))].rgst_no)
Q = P.reindex(ev.rgst_no.values)
p, ex = Q.pth.values, Q.exp.values
m_z = np.isin(ev.rgst_no.values, list(mz))
zero = (p > np.datetime64(END)) | ((ex <= np.datetime64(END)) & (p >= ex)) | m_z
one = (p > np.datetime64(L)) & (p <= np.datetime64(END)) & (p < ex) & (p + np.timedelta64(213, 'D') <= np.datetime64(DATA_END)) & ~m_z
y_ind = np.full(len(ev), np.nan); y_ind[zero] = 0; y_ind[one & ~zero] = 1
both = ~np.isnan(y_ind) & ev.y.notna().values
agree = float((y_ind[both] == ev.y.values[both]).mean())
print('label agreement', agree, 'n_both', int(both.sum()), 'ind NaN', int(np.isnan(y_ind).sum()), 'pipe NaN', int(ev.y.isna().sum()), flush=True)

# ---------------------------------------------------------------- 앙상블 재합성
spec = json.load(open(os.path.join(R2, 'final_spec_r2.json'), encoding='utf-8'))['spec']
W = spec['ensemble']['weights']
pf = {}
for comp in list(W) + ['base']:
    fs = [f for f in glob.glob(os.path.join(OD, 'pred', f'{comp}_*.npy')) if '_b' not in os.path.basename(f)[len(comp) + 1:]]
    assert len(fs) == 1, (comp, fs)
    pf[comp] = np.load(fs[0])
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
z = sum(W[c] * lg(pf[c]) for c in W) / sum(W.values())
pens = 1 / (1 + np.exp(-z))
m = ev.y.notna().values
y = ev.y.values[m].astype(int)
res = dict(n=int(m.sum()), base_rate=float(y.mean()), label_agreement_independent=agree, n_label_compared=int(both.sum()),
           n_label_independent_nan=int(np.isnan(y_ind).sum()))
res['auc_model'] = auc(y, pens[m]); res['auc_base'] = auc(y, pf['base'][m])
res['skill'] = (res['auc_model'] - res['auc_base']) / (1 - res['auc_base'])
res['components'] = {c: auc(y, pf[c][m]) for c in W}
# 독립 레이블로 채점(두 레이블 모두 확정된 행)
yi = y_ind[both].astype(int)
res['auc_model_indlabel'] = auc(yi, pens[both]); res['auc_base_indlabel'] = auc(yi, pf['base'][both])
res['skill_indlabel'] = (res['auc_model_indlabel'] - res['auc_base_indlabel']) / (1 - res['auc_base_indlabel'])
# 상위 10%
k = int(0.1 * m.sum())
res['precision_top10'] = dict(model=float(y[np.argsort(-pens[m])[:k]].mean()), baseline=float(y[np.argsort(-pf['base'][m])[:k]].mean()))
# 군집 부트스트랩(다른 구현)
from sklearn.metrics import roc_auc_score
acd = ev.acd.values[m]; pid = ev.pid.values[m]
cl = np.where(acd == 'NA', 'P' + pid.astype(str), acd)
g = pd.Series(np.arange(len(cl))).groupby(cl).indices
idx = list(g.values()); K = len(idx)
rng = np.random.default_rng(12345)
pm, pb = pens[m], pf['base'][m]
sk, am = [], []
for b in range(300):
    s = rng.integers(0, K, K)
    ii = np.concatenate([idx[i] for i in s])
    a1, a0 = roc_auc_score(y[ii], pm[ii]), roc_auc_score(y[ii], pb[ii])
    sk.append((a1 - a0) / (1 - a0)); am.append(a1)
res['boot_B'] = 300; res['n_clusters'] = K
res['skill_ci95'] = [float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))]
res['auc_ci95'] = [float(np.percentile(am, 2.5)), float(np.percentile(am, 97.5))]
# 파이프라인 결과와 대조
fr = json.load(open(os.path.join(R2, 'final_result.json'), encoding='utf-8'))
res['pipeline'] = dict(auc_model=fr['model']['auc'], auc_base=fr['baseline']['auc'], skill=fr['skill_auc'], n=fr['n'], ci_skill=fr['bootstrap_95ci']['skill'])
res['match'] = dict(auc_model=abs(res['auc_model'] - fr['model']['auc']) < 1e-6, auc_base=abs(res['auc_base'] - fr['baseline']['auc']) < 1e-6,
                    skill=abs(res['skill'] - fr['skill_auc']) < 1e-6, n=res['n'] == fr['n'])
json.dump(res, open(os.path.join(R2, 'audit', 'verify_final.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
