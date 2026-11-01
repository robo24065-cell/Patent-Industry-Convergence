import pandas as pd, duckdb
from r4_common import *
af = pd.read_parquet(f"{R3I}/cache/app_firm.parquet"); print(af.dtypes); print(af.head(3).T); print(len(af))
print(af.applicantCd.astype(str).str[0].value_counts())
ow = pd.read_parquet(f"{CACHE}/reg_owner_lt2023.parquet")
print("null d share", ow.d.isna().mean(), (ow.d=='').mean()); print(ow[ow.seq==0].d.isna().mean())
print(ow.groupby('rgst_no').d.apply(lambda s: s.isna().any()).mean())
pt = pd.read_parquet(f"{CACHE}/reg_patent.parquet"); print(pt.head(3).T)
lk = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=['appl_no','rgst_no','Year','RegistType','ProjectID','ProjectYear','perform_grp']); print(lk.head(3).T); print(lk.rgst_no.notna().mean())
