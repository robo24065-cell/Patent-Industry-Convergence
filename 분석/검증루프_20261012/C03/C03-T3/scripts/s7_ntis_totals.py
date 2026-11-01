import sys, json; sys.path.insert(0, sys.argv[1]); sys.stdout.reconfigure(encoding='utf-8')
from _paths import *
import pandas as pd, numpy as np
R=pd.read_parquet(EXT+r"\ntis_project_raw.parquet",columns=["pjt_no","year","secret_flag","gov_funds","total_funds","ministry"])
R["year"]=R.year.astype(int)
g=R.groupby("year").agg(n=("pjt_no","size"),n_uniq=("pjt_no","nunique"),n_secret=("secret_flag","sum"))
g["gov_nonsecret_eok"]=R[~R.secret_flag].groupby("year").gov_funds.sum()/1e8
g["tot_secret_eok"]=R[R.secret_flag].groupby("year").total_funds.sum()/1e8
A=pd.read_csv(LOOP+r"\C02\C02-T1\audit_tables.csv",encoding="utf-8-sig",low_memory=False)
A=A[A.table=="a_totals_official_vs_collected"].set_index(A[A.table=="a_totals_official_vs_collected"].year.astype(int))
g=g.join(A[["official_n","official_exec","official_defense"]])
g["ratio_n"]=g.n/g.official_n; g["ratio_gov"]=g.gov_nonsecret_eok/g.official_exec; g["ratio_gov_plus_secret"]=(g.gov_nonsecret_eok+g.tot_secret_eok)/g.official_exec
g["ratio_gov_vs_off_minus_def"]=g.gov_nonsecret_eok/(g.official_exec-g.official_defense); g["secret_share_n"]=g.n_secret/g.n
dm=R[~R.secret_flag & R.ministry.astype(str).str.contains("방위사업")].shape[0]
out=dict(table=g.loc[2007:2024].round(4).reset_index().to_dict("records"),nonsecret_방위사업청_rows=int(dm),
         total_rows=int(len(R)),secret_rows=int(R.secret_flag.sum()))
json.dump(out,open(OUT+r"\_s7_ntis_totals.json","w",encoding="utf8"),ensure_ascii=False,indent=1,default=float)
print(g.loc[2007:2024][["n","n_secret","official_n","ratio_n","ratio_gov","ratio_gov_plus_secret","ratio_gov_vs_off_minus_def","secret_share_n"]].round(4).to_string()); print(dm, len(R))
