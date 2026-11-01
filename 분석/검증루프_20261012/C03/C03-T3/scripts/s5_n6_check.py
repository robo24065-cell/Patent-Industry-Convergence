import sys, json; sys.path.insert(0, sys.argv[1]); sys.stdout.reconfigure(encoding='utf-8')
from _paths import *
import pandas as pd, numpy as np
R=pd.read_parquet(EXT+r"\ntis_project_raw.parquet",columns=["year","corp_rgst_no","perform_agent_cd","perform_grp"])
R=R[R.year.astype(int).between(2019,2021)]
R["bz"]=R.corp_rgst_no.astype("string").str.replace(r"[^0-9]","",regex=True)
P=pd.read_parquet(MC+r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet")
pop=set(P.bz.astype(str))
FIRMW={"중소기업","중견기업","대기업"}
codes=R[R.bz.isin(pop)].groupby("bz").perform_agent_cd.agg(lambda s:set(s.dropna().astype(str)))
fw=R[R.bz.isin(pop)&R.perform_grp.isin(FIRMW)].groupby("bz").perform_agent_cd.agg(lambda s:set(s.dropna().astype(str)))
only06=int(sum(1 for s in fw if s=={"06"}))
# 기업 코드(04,05,08) 과제가 있는데 모집단 밖인 사업자번호 중 08만 있는 수(다른 조건 미적용, 상한)
firm_corr=R[R.perform_agent_cd.isin(["04","05","08"])].groupby("bz").perform_agent_cd.agg(lambda s:set(s.dropna().astype(str)))
only08_out=int(sum(1 for b,s in firm_corr.items() if s=={"08"} and b not in pop))
res=dict(n_pop=len(pop),pos=int(P.y.sum()),pop_firms_whose_firm_projects_are_only_code06=only06,
         outside_pop_bz_with_only_code08_projects_2019_21_upper_bound=only08_out)
json.dump(res,open(OUT+r"\_s5_n6.json","w"),indent=1); print(res)
