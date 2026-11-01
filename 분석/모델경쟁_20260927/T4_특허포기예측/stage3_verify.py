# -*- coding: utf-8 -*-
"""3단계 독립 대조: final_run.py 홀드아웃 예측(pred_*.parquet)에 대해
 (1) 결과변수를 registry_full.sqlite(읽기전용)에서 별도 코드로 다시 판정해 일치율 확인
 (2) AUC·PR-AUC·Brier를 sklearn으로 재계산
 (3) 고정 기준선을 frame_base.parquet 로 별도 재적합해 AUC 대조
 (4) 출원인 군집 부트스트랩 95% CI를 다른 구현(sample_weight 방식 sklearn AUC)으로 재계산
 (5) 비전문가용 정확도(상위 10%/20% 적중률, 포착률)
모형은 건드리지 않는다. 출력: final_models/run_holdout/stage3_verify.json
"""
import os, sys, json, sqlite3, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'final_models', 'run_holdout')
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
Ls = ['2019-01-01', '2019-07-01', '2020-01-01', '2020-07-01', '2021-01-01', '2021-07-01']
os.environ['T4_STAGE3_HOLDOUT'] = '1'
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

d = pd.concat([pd.read_parquet(os.path.join(OUT, f'pred_{L.replace("-", "")}.parquet')) for L in Ls], ignore_index=True)
import t4lib as T
from run import labels
pt = T.load()['pt']
d['rgst_no'] = pt.rgst_no.values[d.pid.values]
d['y'] = np.concatenate([labels(L, d[d.L == L].pid.values, None) for L in Ls])

# ---------------- (1) 독립 결과 판정 (sqlite 직접)
con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
ids = pd.DataFrame({'rgst_no': d.rgst_no.unique()})
pat = pd.read_sql('select rgst_no, rgstDate from patent', con)
pat = pat[pat.rgst_no.isin(ids.rgst_no)]
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName='소멸등록'", con)
rt = rt[rt.rgst_no.isin(ids.rgst_no) & rt.rgstCsReason.isin(['등록료불납', '연차료불납'])]
pay = pd.read_sql('select rgst_no, lastAnnl, payDate from pay', con)
con.close()
pay = pay[pay.rgst_no.isin(rt.rgst_no)]
pay['pdt'] = pd.to_datetime(pay.payDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
pay['la'] = pd.to_numeric(pay.lastAnnl, errors='coerce')
rt['cdt'] = pd.to_datetime(rt.rgstCsDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
m = rt.merge(pay[['rgst_no', 'pdt', 'la']], on='rgst_no')
m = m[m.pdt < m.cdt].groupby(['rgst_no', 'cdt']).la.max().reset_index()
m = m.merge(pat, on='rgst_no')
rd = pd.to_datetime(m.rgstDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
m['E'] = pd.to_datetime(T.add_years(rd.values.astype('datetime64[D]'), m.la.values.astype(int)))
print('labels built', len(m), m.dtypes.to_dict(), flush=True)
y2 = np.zeros(len(d))
for L in Ls:
    Ld = pd.Timestamp(L); end = Ld + pd.DateOffset(years=3)
    hit = set(m[(m.E > Ld) & (m.E <= end)].rgst_no.astype(str))
    idx = np.where(d.L.values == L)[0]
    y2[idx] = pd.Series(d.rgst_no.values[idx]).astype(str).isin(hit).values.astype(float)
    print(L, len(hit), y2[idx].mean(), flush=True)
agree = float((y2 == d.y.values).mean())
res = dict(label_agreement=agree, n_disagree=int((y2 != d.y.values).sum()), rate_pipeline=float(d.y.mean()), rate_independent=float(y2.mean()))

print('agree', res, flush=True)
# ---------------- (2) 지표 재계산
def mt(y, p):
    return dict(auc=float(roc_auc_score(y, p)), prauc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)))
res['model'] = mt(d.y, d.p); res['baseline_pipeline'] = mt(d.y, d.p_base)
res['model_on_independent_label'] = mt(y2, d.p)
res['baseline_on_independent_label'] = mt(y2, d.p_base)

# ---------------- (3) 기준선 별도 재적합
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
fb = pd.read_parquet(os.path.join(OUT, 'frame_base.parquet'))
res['baseline_train'] = dict(rows=len(fb), Ls=[str(fb.L.min()), str(fb.L.max()), int(fb.L.nunique())], pos_rate=float(fb.y.mean()))


def bx(x, med):
    X = pd.DataFrame({'age': x.age.astype(float), 'lclaims': np.log1p(pd.to_numeric(x.claims, errors='coerce').fillna(med)),
                      'rgst_year': x.rgst_year.astype(float), 'atype': x.atype.astype(str), 'ksic': x.ksic.astype(str)})
    return X


med = float(pd.to_numeric(fb.claims, errors='coerce').median())
ct = ColumnTransformer([('n', StandardScaler(), ['age', 'lclaims', 'rgst_year']), ('c', OneHotEncoder(handle_unknown='ignore'), ['atype', 'ksic'])])
bm = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=3000)).fit(bx(fb, med), fb.y.values)
feats = pd.concat([T.features(L)[['pid', 'age', 'claims', 'rgst_year', 'atype', 'ksic']].assign(L=L) for L in Ls], ignore_index=True)
dd = d[['pid', 'L']].merge(feats, on=['pid', 'L'], how='left', validate='one_to_one')
pb2 = bm.predict_proba(bx(dd, med))[:, 1]
res['baseline_refit_independent'] = mt(d.y, pb2)
res['baseline_refit_corr_with_pipeline'] = float(np.corrcoef(pb2, d.p_base)[0, 1])

print('baseline refit', res['baseline_refit_independent'], flush=True)
# ---------------- (4) 군집 부트스트랩(다른 구현)
cl = np.where(d.acd.values == 'NA', 'P' + d.pid.astype(str).values, d.acd.values)
codes, uniq = pd.factorize(cl); K = len(uniq)
rng = np.random.default_rng(12345)
R = {k: [] for k in ['auc', 'auc_base', 'diff', 'skill']}
import time as _t; _t0 = _t.time()
NB = int(os.environ.get('NB', '200'))
for b in range(NB):
    w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
    a = roc_auc_score(d.y, d.p, sample_weight=w); ab = roc_auc_score(d.y, d.p_base, sample_weight=w)
    R['auc'].append(a); R['auc_base'].append(ab); R['diff'].append(a - ab); R['skill'].append((a - ab) / (1 - ab))
res['boot_indep_B'] = NB
res['boot_indep'] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in R.items()}
res['n_clusters'] = int(K); res['n_rows'] = int(len(d)); res['n_patents'] = int(d.pid.nunique()); res['n_events'] = int(d.y.sum())

# ---------------- (5) 비전문가용 정확도
pa = {}
for q in [0.1, 0.2]:
    for nm, col in [('model', 'p'), ('baseline', 'p_base')]:
        acc, cap = [], []
        for L in Ls:
            s = d[d.L == L]
            thr = s[col].quantile(1 - q)
            top = s[s[col] >= thr]
            acc.append(top.y.mean()); cap.append(top.y.sum() / s.y.sum())
        pa[f'{nm}_top{int(q*100)}_precision'] = round(float(np.mean(acc)), 4)
        pa[f'{nm}_top{int(q*100)}_capture'] = round(float(np.mean(cap)), 4)
pa['base_rate'] = round(float(d.y.mean()), 4)
# 쌍 비교 정확도(AUC 해석): 포기될 특허와 유지될 특허를 하나씩 뽑았을 때 포기 쪽 위험을 더 높게 매기는 확률 = AUC
res['plain'] = pa
# 보정: 기준일별 평균 예측 vs 실현
res['calibration_perL'] = {L: dict(mean_pred=round(float(d.p[d.L == L].mean()), 4), realized=round(float(d.y[d.L == L].mean()), 4)) for L in Ls}
json.dump(res, open(os.path.join(OUT, 'stage3_verify.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
