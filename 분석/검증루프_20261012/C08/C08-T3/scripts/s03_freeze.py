# -*- coding: utf-8 -*-
"""채점 전 점수·비교 지표 동결 (결과 창 사건을 읽지 않는다).
- 동결 미래 점수 pred_20250701.parquet(p, p_base, 구성요소) 그대로
- p_haz_win: 동결 위험모형 결정행 중 납부기한 d <= 창 끝(2025-10-31)만으로 1-Π(1-h) (부 지표용 창 전용 점수, 저장 npy만 사용)
- p_strong: VERIFY_R5 강화 기준선 정의(13변수 로지스틱, frame_cls 60만 표본 seed 0)를 FUT 학습 프레임(레이블 컷 2025-08-22)에 적합 → L=2025-07 예측
- due_win: L 시점 납부완료일이 (L, 창 끝] → 창 안 납부기한 보유(구조적으로 사건 가능)
- big10/big2: 위험집합 특허 수 기준 상위 10개·2위 출원인코드
- 지속 지수: KSIC별 직전 1년 같은 창 실현율 (위험집합 2024-07-01, E∈(2024-07-01, 2024-10-31], 소멸등록 cd<=2025-08-22, 회복 cd<=2025-08-22 제외)
출력: frozen_scores.parquet, persist_index.csv, cache/s03_freeze.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import c08lib as C
t0 = time.time()
WEND = np.datetime64('2025-10-31'); L = C.L0; CFRZ = np.datetime64('2025-08-22')
D = C.load(); pt, pay, rt = D['pt'], D['pay'], D['rt']
pr = pd.read_parquet(os.path.join(C.FUT, 'pred_20250701.parquet'))
out = pr[['pid', 'ksic', 'atype', 'acd', 'p', 'p_base', 'p_cls', 'p_haz', 'p_logit', 'p_hgb']].copy()
R = {}
# ---- 창 전용 위험모형 점수
sig = lambda z: 1 / (1 + np.exp(-z))
zh = np.mean([np.load(os.path.join(C.FUT, 'pred', f'haz_0568d4297b_b{k}.npy')) for k in range(3)], 0)
h = sig(zh)
hrow = pd.read_parquet(os.path.join(C.FUT, 'ev_haz.parquet'), columns=['row', 'k', 'pid'])
dd = C.plus_years(pt.rd.values[hrow.pid.values], hrow.k.values.astype(int) - 1)
w = dd <= WEND
lsum = pd.Series(np.log1p(-np.clip(h[w], 1e-6, 1 - 1e-6))).groupby(hrow.row.values[w]).sum()
phw = np.zeros(len(out)); phw[lsum.index.values] = 1 - np.exp(lsum.values)
out['p_haz_win'] = phw
# 전체(3년) 재합성 = p_haz 확인
lall = pd.Series(np.log1p(-np.clip(h, 1e-6, 1 - 1e-6))).groupby(hrow.row.values).sum()
ph3 = np.zeros(len(out)); ph3[lall.index.values] = 1 - np.exp(lall.values)
R['p_haz_recompose_maxdiff'] = float(np.abs(ph3 - out.p_haz.values).max())
R['haz_rows_in_window'] = int(w.sum()); R['patents_with_window_haz_row'] = int(len(lsum))
# ---- 창 안 납부기한 보유
pthL, lvL = C.paid_thru(pt, pay, L)
pp = pthL[out.pid.values]
out['due_win'] = ((pp > L) & (pp <= WEND)).astype('int8')
R['due_win_n'] = int(out.due_win.sum())
R['due_win_vs_hazrow_agree'] = float(((phw > 0).astype(int) == out.due_win.values).mean())
# ---- 강화 기준선
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
NUM = ['age', 'claims', 'rgst_year', 'next_annl', 'paid_ahead', 'n_due_win', 'tier_in_win', 'a_lapse_rate', 'a_lapse_rate2', 'n_late', 'yrs_since_pay']
fr = pd.read_parquet(os.path.join(C.FUT, 'frame_cls.parquet'), columns=['y', 'L'] + NUM + ['atype', 'ksic'])
R['frame_cls'] = dict(rows=len(fr), y_nan=int(fr.y.isna().sum()), L_max=str(fr.L.max()))
fr = fr[fr.y.notna()]
fr = fr.sample(n=min(600_000, len(fr)), random_state=0)
ct = ColumnTransformer([('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), NUM),
                        ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=20), ['atype', 'ksic'])])
def prep(d):
    X = d[NUM + ['atype', 'ksic']].copy(); X['claims'] = np.log1p(X.claims.astype(float))
    for c in ['atype', 'ksic']: X[c] = X[c].astype(str)
    return X
m = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000)).fit(prep(fr), fr.y.values)
ev = pd.read_parquet(os.path.join(C.FUT, 'ev.parquet'), columns=['pid'] + NUM + ['atype', 'ksic'])
assert (ev.pid.values == out.pid.values).all()
out['p_strong'] = m.predict_proba(prep(ev))[:, 1]
del fr, ev
# ---- 대형 출원인 (위험집합 특허 수, 'NA' 제외)
vc = out[out.acd != 'NA'].acd.value_counts()
top10 = vc.index[:10].tolist()
out['big10'] = out.acd.isin(top10).astype('int8'); out['big2'] = (out.acd == vc.index[1]).astype('int8')
R['top10'] = [dict(rank=i + 1, acd=a, n=int(vc[a]), atype=str(out.atype[out.acd == a].iloc[0]),
                   top_ksic=str(out.ksic[out.acd == a].value_counts().index[0])) for i, a in enumerate(top10)]
# ---- 지속 지수(직전 1년 같은 창, 동결 시점 2025-08-22 정보)
Lp = np.datetime64('2024-07-01'); Wp = np.datetime64('2024-10-31')
rp, pthp, _ = C.risk_set(D, Lp)
lap = C.lapse_events(D)
rec = rt[(rt.name == '회복등록') & (rt.cd <= CFRZ)]
recmax = rec.groupby('pid').cd.max()
e = lap[(lap.E > Lp) & (lap.E <= Wp) & (lap.cd <= CFRZ)].copy()
e['rec'] = e.pid.map(recmax)
e = e[~(e.rec.notna() & (e.rec >= e.cd))]
yp = np.isin(rp, e.pid.unique()).astype(int)
duep = ((pthp[rp] > Lp) & (pthp[rp] <= Wp)).astype(int)
pk = pd.DataFrame({'ksic': pt.ksic.values[rp], 'y': yp, 'due': duep})
pi = pk.groupby('ksic').agg(n_prev=('y', 'size'), persist_rate=('y', 'mean'), n_due_prev=('due', 'sum'), ev_prev=('y', 'sum')).reset_index()
pi['persist_rate_due'] = pi.ev_prev / pi.n_due_prev.replace(0, np.nan)
pi.to_csv(os.path.join(C.W, 'persist_index.csv'), index=False, encoding='utf-8-sig')
R['persist'] = dict(risk_set_prev=len(rp), events_prev=int(yp.sum()), rate_prev=float(yp.mean()), n_ksic=len(pi),
                    note='직전 1년 같은 창(2024-07-01, 2024-10-31], cd<=2025-08-22 — 동결 시점에 알 수 있던 정보')
# ---- 연례 지수 값(future_forecast_r2.csv)과 동결 점수 KSIC 평균 일치
ff = pd.read_csv(os.path.join(C.R2, 'future_forecast_r2.csv'), dtype={'ksic': str})
f7 = ff[ff.L == '2025-07-01'].set_index('ksic')
g = out.groupby(out.ksic.astype(str)).p.mean()
R['forecast_csv_match'] = dict(n_ksic=int(len(g)), maxdiff=float(np.abs(g - f7.pred_risk.reindex(g.index)).max()),
                               all_maxdiff=float(abs(out.p.mean() - f7.loc['ALL', 'pred_risk'])),
                               realized_rate_nonnull_L_ge_2022_07=int(ff[ff.L >= '2022-07-01'].realized_rate.notna().sum()),
                               realized_rate_nonnull_L_le_2022_01=int(ff[ff.L <= '2022-01-01'].realized_rate.notna().sum()))
fp = os.path.join(C.W, 'frozen_scores.parquet')
out.to_parquet(fp, index=False)
R['frozen_scores'] = dict(path=fp, rows=len(out), sha256=C.sha(fp), cols=list(out.columns),
                          means={c: float(out[c].mean()) for c in ['p', 'p_base', 'p_strong', 'p_haz_win', 'due_win', 'big10']})
R['persist_index_sha256'] = C.sha(os.path.join(C.W, 'persist_index.csv'))
R['secs'] = round(time.time() - t0, 1); R['peak_gb'] = C.peak_gb()
C.jdump(R, os.path.join(C.CACHE, 's03_freeze.json'))
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
