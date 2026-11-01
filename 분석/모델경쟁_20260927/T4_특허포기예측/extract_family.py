import duckdb, pandas as pd, numpy as np, re, time
M=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta\*.parquet"
duckdb.sql("SET threads=2; SET memory_limit='3GB'")
t=time.time()
d=duckdb.sql(f"select appl_no, any_value(m.family) AS fam, count(distinct ksic) AS n_ksic_meta, any_value(ipc_all) AS ipc_all from read_parquet('{M}') m where m.cc='KR' group by appl_no").df()
print(len(d), time.time()-t)
fam=d.fam.fillna('')
def parse(s):
    if not s: return (0,0,0,0,0,0,0)
    toks=set(s.split('|'))
    cc=set(x[:2] for x in toks)
    return (len(toks), len(cc-{'WO','KR'}), int('US' in cc), int('EP' in cc), int('JP' in cc), int('CN' in cc), int('WO' in cc))
arr=np.array([parse(s) for s in fam])
out=pd.DataFrame(arr,columns=['fam_n','fam_cc','fam_us','fam_ep','fam_jp','fam_cn','fam_wo'])
out['applNo']=d.appl_no.values; out['n_ksic_meta']=d.n_ksic_meta.values
out['n_ipc_all']=d.ipc_all.fillna('').str.count(r'\|')+1
out.to_parquet('cache/kr_family.parquet',index=False)
print(out.describe().T[['mean','max']])
