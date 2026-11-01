"""T3 3라운드 최종시험 독립 재계산 (final_run_r3.py --mode final 실행 뒤에만).
패널·피처 코드(r3lib/r3core)를 거치지 않고 base_rows.parquet(공개연도 ≤2024)에서 직접:
  (1) 적격 셀(5개국 공개연도 2019–21 합 ≥50), (2) KR 공개연도 2022–24 대 2019–21 성장 레이블과 상위 10%(동률: 절대증가 → md5),
  (3) 저장된 예측(preds/FINAL_R3_t0_2021.parquet)으로 AUC·PR-AUC·P@10·skill 재계산, (4) 기준선(kr_mom 로지스틱)을 feat 파일로 다시 적합,
  (5) IPC 메인그룹 군집 부트스트랩을 다른 시드(7)로 재계산.
출력: verify_final_r3.json
"""
import os, sys, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
CA = os.path.join(HERE, 'cache_pub_final').replace(os.sep, '/')
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
src = open(os.path.join(T3, 'build_panel.py'), encoding='utf-8').read()
exec(src[src.index('con.execute(r"""create macro'):src.index('con.execute(rf"""create view b')])
con.execute(f"""create temp table a as select distinct ksic, mgn(ipc_raw) mg, cc, year, appl_no
  from read_parquet('{CA}/base_rows.parquet') where mgn(ipc_raw) is not null and year between 2019 and 2024""")
c = con.sql("""select ksic||'|'||mg cell,
   count(*) filter (where year between 2019 and 2021) tot_W,
   count(*) filter (where cc='KR' and year between 2019 and 2021) kr_W,
   count(*) filter (where cc='KR' and year between 2022 and 2024) kr_fut
  from a group by 1""").df()
c = c[c.tot_W >= 50].reset_index(drop=True)
c['y'] = np.log((c.kr_fut + 1) / (c.kr_W + 1))
k = int(round(0.10 * len(c)))
key = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in c.cell], dtype=np.int64)
o = np.lexsort((key, -(c.kr_fut - c.kr_W).values, -c.y.values)); c['yb_ind'] = 0; c.loc[o[:k], 'yb_ind'] = 1
P = pd.read_parquet(os.path.join(HERE, 'preds', 'FINAL_R3_t0_2021.parquet'))
m = P.merge(c, on='cell', how='outer', indicator=True, suffixes=('', '_ind'))
res = dict(n_elig_indep=len(c), n_pred=len(P), merge=m._merge.value_counts().astype(int).to_dict())
m = m[m._merge == 'both']
res['label_y_maxdiff'] = float(np.abs(m.y - m.y_ind).max()); res['yb_mismatch'] = int((m.yb != m.yb_ind).sum()); res['n_evt_indep'] = int(c.yb_ind.sum())
yb = m.yb_ind.values
def met(p):
    kk = int(round(0.1 * len(p))); top = np.argsort(-p)[:kk]
    return dict(auc=float(roc_auc_score(yb, p)), prauc=float(average_precision_score(yb, p)), p_at10=float(yb[top].mean()))
res['model'] = met(m.p.values); res['baseline_saved'] = met(m.p_base.values)
# 기준선 재적합: 학습 t0 2005..2018 적격(≥50) feat 파일의 kr_mom, 레이블은 개발 캐시와 같은 규칙
tr = []
for t0 in range(2005, 2019):
    f = pd.read_parquet(os.path.join(HERE, 'cache_pub_final', f'feat_{t0}.parquet')); f = f[f.tot_W >= 50].reset_index(drop=True)
    kk = int(round(0.1 * len(f))); hk = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in f.cell], dtype=np.int64)
    oo = np.lexsort((hk, -(f.kr_fut - f.kr_W).values, -f.y.values)); f['yb'] = 0; f.loc[oo[:kk], 'yb'] = 1; tr.append(f[['kr_mom', 'yb']])
tr = pd.concat(tr); te = pd.read_parquet(os.path.join(HERE, 'cache_pub_final', 'feat_2021.parquet')).set_index('cell').loc[m.cell.values]
lb = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values)
res['baseline_refit'] = met(lb.predict_proba(te[['kr_mom']].values)[:, 1]); res['n_train_baseline'] = len(tr)
res['skill'] = (res['model']['auc'] - res['baseline_refit']['auc']) / (1 - res['baseline_refit']['auc'])
# 군집 부트스트랩(시드 7)
rng = np.random.default_rng(7); mg = m.cell.str.split('|').str[1].values; ums = np.unique(mg)
gi = pd.Series(np.arange(len(mg))).groupby(mg).apply(lambda s: s.values).to_dict(); pm, pb = m.p.values, m.p_base.values; bs = []
for _ in range(1000):
    ii = np.concatenate([gi[u] for u in rng.choice(ums, len(ums))]); a = roc_auc_score(yb[ii], pm[ii]); a0 = roc_auc_score(yb[ii], pb[ii]); bs.append((a, (a - a0) / (1 - a0)))
bs = np.array(bs); res['ci95_seed7'] = dict(auc=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), skill=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), n_clusters=int(len(ums)))
# KSIC별 부상률(해석용) 상위
m['ksic'] = m.cell.str[:5]
sub = m[m.kr_W >= 5]; ybs = sub.yb_ind.values
res['kr_ge5_subset'] = dict(n=int(len(sub)), n_evt=int(ybs.sum()), auc_model=float(roc_auc_score(ybs, sub.p)), auc_base=float(roc_auc_score(ybs, sub.p_base)))
json.dump(res, open(os.path.join(HERE, 'verify_final_r3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
