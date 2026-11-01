# E-C02T2-01/02 독립 재현: T3 최종 캐시(base_rows) 직접 집계 → CN IPC 결측 비례배분 시 적격 셀·부상 셀 변화. 재학습·재채점 없음.
import sys, json, time; sys.path.insert(0, sys.argv[1]); sys.stdout.reconfigure(encoding='utf-8')
from _paths import *
import duckdb, numpy as np, pandas as pd
from scipy.stats import spearmanr
t=time.time()
B=(MC+r"\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet").replace("\\","/")
con=duckdb.connect(); con.execute("set memory_limit='2GB'; set threads=2")
g=con.sql(f"""select ksic, split_part(ipc_raw,'/',1) mg, cc, cast(year as int) as yr, count(*) n from read_parquet('{B}')
 where year between 2017 and 2024 and ipc_raw is not null and ipc_raw<>'' group by all""").df()
m=con.sql(f"""select ksic, cast(year as int) as yr, count(*) filter (where ipc_raw is null or ipc_raw='') n_null, count(*) filter (where not(ipc_raw is null or ipc_raw='')) n_valid
 from read_parquet('{B}') where cc='CN' and year between 2017 and 2024 group by all""").df()
yrs=con.sql(f"""select cast(year as int) as yr, count(*) n, count(*) filter (where ipc_raw is null or ipc_raw='') n_null from read_parquet('{B}') where cc='CN' and year between 2017 and 2024 group by 1 order by 1""").df()
g=g.rename(columns={"yr":"year"}); m=m.rename(columns={"yr":"year"}); yrs=yrs.rename(columns={"yr":"year"})
g["cell"]=g.ksic+"|"+g.mg
pv=g.pivot_table(index=["cell","ksic"],columns=["cc","year"],values="n",aggfunc="sum",fill_value=0)
def col(cc,y): return pv[(cc,y)] if (cc,y) in pv.columns else 0
fac=(1+m.n_null/m.n_valid.replace(0,np.nan)).fillna(1.0); fac.index=pd.MultiIndex.from_frame(m[["ksic","year"]])
ks=pv.index.get_level_values("ksic")
def cnA(y): return col("CN",y)*np.array([fac.get((k,y),1.0) for k in ks])
def run(t0):
    W=[t0-2,t0-1,t0]
    tot=sum(col(c,y) for c in ["KR","US","JP","CN","EP"] for y in W)
    totA=tot-sum(col("CN",y) for y in W)+sum(cnA(y) for y in W)
    kr=sum(col("KR",y) for y in W)
    r=dict(t0=t0,n_elig=int((tot>=50).sum()),n_elig_A=int((totA>=50).sum()))
    r["enter_A"]=int(((totA>=50)&(tot<50)).sum()); r["exit_A"]=int(((totA<50)&(tot>=50)).sum())
    r["pct_more_vs_official"]=round(100*(r["n_elig_A"]/r["n_elig"]-1),2); r["pct_fewer_vs_imputed"]=round(100*(1-r["n_elig"]/r["n_elig_A"]),2)
    e=tot>=50
    g1=np.log((col("CN",t0)+1)/(col("CN",t0-1)+1)); g1A=np.log((cnA(t0)+1)/(cnA(t0-1)+1))
    r["rho_CN_g1_official_cells"]=float(spearmanr(np.asarray(g1)[e.values],np.asarray(g1A)[e.values]).correlation)
    if t0+3<=2024:
        fut=sum(col("KR",y) for y in [t0+1,t0+2,t0+3])
        def lab(mask):
            idx=np.where(mask.values)[0]; y=np.log((fut.values[idx]+1)/(kr.values[idx]+1)); d=(fut.values[idx]-kr.values[idx])
            k=int(round(0.1*len(idx))); o=np.lexsort((-d,-y))[:k]; yb=np.zeros(len(pv),bool); yb[idx[o]]=True; return yb
        y0=lab(e); yA=lab(totA>=50)
        r["n_pos"]=int(y0.sum()); r["n_pos_A"]=int(yA.sum()); r["pos_in_entered_cells_A"]=int((yA&~e.values).sum()); r["pos_lost_A_among_official"]=int((y0&~yA).sum())
    return r
out=dict(cn_by_year=yrs.to_dict("records"),by_origin={t0:run(t0) for t0 in (2018,2021,2024)},elapsed=round(time.time()-t,1))
json.dump(out,open(OUT+r"\t3_cn_ipc_recheck.json","w"),indent=1,default=int); print(json.dumps(out,indent=1,default=int))
