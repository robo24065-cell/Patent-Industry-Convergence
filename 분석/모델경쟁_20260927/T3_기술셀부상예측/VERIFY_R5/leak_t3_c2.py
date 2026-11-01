# (c2) R2 국내우선권 레이블 민감도의 개발 원점 대조 + 2023 해외우선권 절단 보정 레이블. 읽기 전용, 산출 leak_t3_c2.json
import os, sys, json, hashlib
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
BR = os.path.join(T3, 'R2', 'cache_2023', 'base_rows.parquet').replace(os.sep, '/')
con.execute(f"create temp table kr as select ksic||'|'||mgn(ipc_raw) cell, year, appl_no, max(prio_for) pf from read_parquet('{BR}') where cc='KR' and mgn(ipc_raw) is not null group by 1,2,3")
def hkey(c): return np.array([int(hashlib.md5(x.encode()).hexdigest()[:8], 16) for x in c], dtype=np.int64)
def lab(g, fut, past):
    y = np.log((fut + 1) / (past + 1)); k = int(round(0.1 * len(g))); o = np.lexsort((hkey(g.cell.values), -(fut - past), -y)); yb = np.zeros(len(g), int); yb[o[:k]] = 1; return yb
sk = lambda a, b: (a - b) / (1 - b)
out = {}
P = pd.concat([pd.read_parquet(os.path.join(T3, 'R2', 'preds', 'dev_preds.parquet')), pd.read_parquet(os.path.join(T3, 'R2', 'preds', 'FINAL_t0_2019_2020.parquet'))])
for t, g in P.groupby('t0'):
    c = con.sql(f"""select cell, sum(case when year between {t-2} and {t} then 1 else 0 end) past, sum(case when year between {t+1} and {t+3} then 1 else 0 end) fut,
                   sum(case when year between {t-2} and {t} and pf=0 then 1 else 0 end) past_d, sum(case when year between {t+1} and {t+3} and pf=0 then 1 else 0 end) fut_d,
                   sum(case when year=2023 and pf=1 then 1 else 0 end) pf23 from kr where year between {t-2} and {t+3} group by 1""").df()
    g = g.merge(c, on='cell', how='left').fillna(0)
    yb0 = lab(g, g.fut.values, g.past.values); assert (yb0 == g.yb.values).all(), t
    ybd = lab(g, g.fut_d.values, g.past_d.values)
    r = dict(skill_official=sk(roc_auc_score(g.yb, g.p), roc_auc_score(g.yb, g.p_base)), skill_dom=sk(roc_auc_score(ybd, g.p), roc_auc_score(ybd, g.p_base)))
    if t + 3 >= 2023:   # 2023 해외우선권 출원 절단 보정(2019–22 평균 비중 대비 결손을 셀 비례로 복원)
        ybi = lab(g, g.fut.values + 0.13 * g.pf23.values, g.past.values)
        r['skill_pf2023_imputed'] = sk(roc_auc_score(ybi, g.p), roc_auc_score(ybi, g.p_base)); r['label_overlap_imputed'] = float((ybi & g.yb).sum() / g.yb.sum())
    out[str(t)] = r
dev = [out[str(t)] for t in range(2010, 2017)]
out['dev_mean'] = dict(skill_official=float(np.mean([r['skill_official'] for r in dev])), skill_dom=float(np.mean([r['skill_dom'] for r in dev])))
out['final_mean'] = dict(skill_official=float(np.mean([out[t]['skill_official'] for t in ('2019', '2020')])), skill_dom=float(np.mean([out[t]['skill_dom'] for t in ('2019', '2020')])))
json.dump(out, open(os.path.join(HERE, 'leak_t3_c2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=0))
