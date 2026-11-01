"""T3 3라운드 최종시험 독립 재계산 (D단계, final_run_r3.py --mode final 실행 뒤에만).
r3lib/r3core·feat 캐시를 전혀 쓰지 않고 base_rows.parquet(공개연도 ≤2024)에서 직접 다시 만든다.
  (1) 셀×국가×공개연도 출원 수 패널(2000–2024, 셀 내 출원번호 중복 제거)
  (2) t0=2021 적격 셀(5개국 2019–21 ≥50)·레이블(KR 2022–24 대 2019–21 log 성장)·상위 10%(동률: 절대증가 → md5)
  (3) 저장된 예측(preds/FINAL_R3_t0_2021.parquet)으로 AUC(순위합 직접 계산 + sklearn)·PR-AUC·P@10·skill
  (4) 기준선: 이 패널에서 t0 2005–2018 적격 셀의 kr_mom·레이블을 새로 만들어 로지스틱(C=1e6) 재적합 → t0=2021 예측과 저장된 p_base 대조
  (5) IPC 메인그룹 군집 부트스트랩(시드 11, 1,000회)
y 허용오차 1e-6(저장 예측의 y 는 float32 패널에서 계산됨).
출력: verify_final_r3d.json
"""
import os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
import numpy as np, pandas as pd, duckdb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
CA = os.path.join(HERE, 'cache_pub_final').replace(os.sep, '/')
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
pan = con.sql(f"""select ksic||'|'||mg cell, cc, year, count(distinct appl_no) n from (select ksic, mgn(ipc_raw) mg, cc, year, appl_no
   from read_parquet('{CA}/base_rows.parquet')) where mg is not null group by all""").df()
cells = np.array(sorted(pan.cell.unique())); ci = np.searchsorted(cells, pan.cell.values)
Y0 = 2000; KR = np.zeros((len(cells), 25)); TOT = np.zeros((len(cells), 25))
np.add.at(TOT, (ci, pan.year.values - Y0), pan.n.values)
kr = pan.cc.values == 'KR'; np.add.at(KR, (ci[kr], pan.year.values[kr] - Y0), pan.n.values[kr])
S = lambda A, a, b: A[:, a - Y0:b - Y0 + 1].sum(1)
hk = lambda cs: np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in cs], dtype=np.int64)

def origin(t0, lab=True):
    d = pd.DataFrame(dict(cell=cells, tot_W=S(TOT, t0 - 2, t0), kr_W=S(KR, t0 - 2, t0),
                          kr_mom=np.log((S(KR, t0 - 2, t0) + 1) / (S(KR, t0 - 5, t0 - 3) + 1))))
    d = d[d.tot_W >= 50].reset_index(drop=True); d['t0'] = t0
    if lab:
        assert t0 + 3 <= 2024
        d['kr_fut'] = S(KR, t0 + 1, t0 + 3)[np.searchsorted(cells, d.cell.values)]
        d['y'] = np.log((d.kr_fut + 1) / (d.kr_W + 1)); k = int(round(0.10 * len(d)))
        o = np.lexsort((hk(d.cell.values), -(d.kr_fut - d.kr_W).values, -d.y.values)); d['yb'] = 0; d.loc[o[:k], 'yb'] = 1
    return d

te = origin(2021)
P = pd.read_parquet(os.path.join(HERE, 'preds', 'FINAL_R3_t0_2021.parquet'))
m = P.merge(te, on='cell', how='outer', indicator=True, suffixes=('', '_ind'))
out = dict(cells_saved=int(len(P)), cells_indep=int(len(te)), merge={k: int(v) for k, v in m._merge.value_counts().items()})
m = m[m._merge == 'both']
out['label_check'] = dict(y_maxdiff=float(np.abs(m.y - m.y_ind).max()), yb_mismatch=int((m.yb != m.yb_ind).sum()),
                          kr_W_maxdiff=float(np.abs(m.kr_W - m.kr_W_ind).max()), n_evt_indep=int(m.yb_ind.sum()))
def auc_rank(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
y = m.yb_ind.values
a, a0 = auc_rank(y, m.p.values), auc_rank(y, m.p_base.values)
k = int(round(0.10 * len(m)))
out['metrics_indep'] = dict(auc=a, auc_sklearn=float(roc_auc_score(y, m.p)), auc_base=a0, skill=(a - a0) / (1 - a0),
                            prauc=float(average_precision_score(y, m.p)), prauc_base=float(average_precision_score(y, m.p_base)),
                            p_at10=float(y[np.argsort(-m.p.values)[:k]].mean()), p_at10_base=float(y[np.argsort(-m.p_base.values)[:k]].mean()))
# 기준선 재적합(독립 패널)
tr = pd.concat([origin(t) for t in range(2005, 2019)], ignore_index=True)
assert tr.t0.max() + 3 < 2022
lr = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values)
pb = lr.predict_proba(m[['kr_mom']].values)[:, 1]
out['baseline_refit'] = dict(n_train=int(len(tr)), coef=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]),
                             p_base_maxdiff=float(np.abs(pb - m.p_base.values).max()), auc_refit=auc_rank(y, pb))
# 군집 부트스트랩(시드 11)
rng = np.random.default_rng(11); mg = m.mg.values; um = np.unique(mg)
grp = pd.Series(np.arange(len(m))).groupby(mg).apply(lambda s: s.values).to_dict()
bs = []
for _ in range(1000):
    ii = np.concatenate([grp[u] for u in rng.choice(um, len(um))]); yy = y[ii]
    if yy.min() == yy.max(): continue
    x, x0 = roc_auc_score(yy, m.p.values[ii]), roc_auc_score(yy, m.p_base.values[ii]); bs.append(((x - x0) / (1 - x0), x, x0))
bs = np.array(bs)
out['boot_seed11'] = dict(skill=[float(v) for v in np.percentile(bs[:, 0], [2.5, 97.5])], auc=[float(v) for v in np.percentile(bs[:, 1], [2.5, 97.5])],
                          n_clusters=int(len(um)))
F = json.load(open(os.path.join(HERE, 'final_result_r3.json'), encoding='utf-8'))
out['vs_final_result'] = dict(skill_final=F['by_origin']['2021']['skill'], skill_indep=out['metrics_indep']['skill'],
                              diff=abs(F['by_origin']['2021']['skill'] - out['metrics_indep']['skill']))
out['all_match'] = bool(out['merge'].get('left_only', 0) == 0 and out['merge'].get('right_only', 0) == 0 and out['label_check']['yb_mismatch'] == 0
                        and out['label_check']['y_maxdiff'] < 1e-6 and out['vs_final_result']['diff'] < 1e-9 and out['baseline_refit']['p_base_maxdiff'] < 1e-6)
json.dump(out, open(os.path.join(HERE, 'verify_final_r3d.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
