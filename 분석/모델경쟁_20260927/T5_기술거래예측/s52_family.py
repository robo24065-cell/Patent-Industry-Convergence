# [v31 피처] 제공 5개국 융합데이터(_shared/meta) KR 행의 family 필드 → KR 등록특허별 해외 패밀리(US·EP·JP·CN·WO) 구성 → cache/fam.parquet
# 시점: 해외 패밀리는 우선일로부터 30개월 안에 진입하므로, 적용 시 출원일이 L-2.5년 이전인 특허만 값을 쓰고 나머지는 결측(v2lib EXTRA_FN 'fam').
import sys, duckdb, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
M = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet"
con = duckdb.connect(); con.execute("SET threads=2; SET memory_limit='3GB'")
df = con.execute(f"select appl_no, any_value(family) as family from read_parquet('{M}') where cc='KR' group by appl_no").df()
fam = df.family.fillna('')
out = pd.DataFrame({'applNo': df.appl_no.astype(str)})
for c in ['US', 'EP', 'JP', 'CN', 'WO']:
    out[f'fam_{c}'] = fam.str.split('|').apply(lambda t: int(any(x.startswith(c) for x in t)))
out['n_fam_cc'] = out[['fam_US', 'fam_EP', 'fam_JP', 'fam_CN']].sum(axis=1)
out['n_fam_docs'] = fam.str.split('|').apply(lambda t: len(set(x for x in t if x and not x.startswith('KR'))))
pat = pd.read_parquet('cache/pat.parquet', columns=['rgst_no', 'applNo']); pat['applNo'] = pat.applNo.astype(str)
r = pat.merge(out, on='applNo', how='left').drop(columns=['applNo'])
print('matched', r.fam_US.notna().mean().round(3), r[['fam_US', 'fam_EP', 'fam_JP', 'fam_CN', 'fam_WO']].mean().round(3).to_dict())
r.to_parquet('cache/fam.parquet', index=False)
