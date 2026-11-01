import os, sqlite3, numpy as np, pandas as pd, json
d=os.path.dirname(os.path.abspath(__file__))
REG=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c=sqlite3.connect(f"file:{REG}?mode=ro",uri=True)
e=pd.read_sql("select substr(rgstCsDate,1,4) y, rgstCsName n, count(*) k from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록') group by 1,2",c)
pv=e.pivot(index="y",columns="n",values="k").fillna(0).astype(int)
print(pv.loc["2010":].to_string())
r2=sqlite3.connect(r"file:C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry.sqlite?mode=ro",uri=True)
print(pd.read_sql("select min(fetched_at) a, max(fetched_at) b from registry",r2))
print(pd.read_sql("select max(rgstCsDate) from right where rgstCsDate < '2027'",c))
L=pd.read_csv(os.path.join(d,"labels_rebuilt.csv"),comment="#",dtype={"rgst_no":str})
S=L[L.n_sco_win>0].copy(); S["yr"]=S.first_sco//10000; S["det"]=S.sco_dir.isin(["active","passive","mixed"])
print(S.groupby("yr").det.agg(["mean","size"]))
print(S.groupby("det")[["inv_after_sco_365","inv_before_sco_365","indiv","prior_disp_pre2023","withdrawn_after"]].mean())
