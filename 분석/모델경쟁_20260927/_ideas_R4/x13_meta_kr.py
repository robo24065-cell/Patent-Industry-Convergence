# 제공 meta KR: 출원번호→KSIC·메인IPC 메인그룹·출원연도·공개/등록일(알려진 날짜) — 셀 수요(M2)용
import duckdb, pandas as pd
from r4_common import *
M = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
d = duckdb.sql(f"""select appl_no, ksic, appl_year ay, main_ipc, pub_date, rgst_date, applicant from read_parquet('{M}') where cc='KR'""").df()
print(len(d), d.head(3).T)
d["mg"] = d.main_ipc.fillna("").str.replace(" ", "").str.split("/").str[0]
pd_ = pd.to_numeric(d.pub_date.astype(str).str[:8], errors="coerce"); rd = pd.to_numeric(d.rgst_date.astype(str).str[:8], errors="coerce")
d["known"] = pd.concat([pd_, rd], axis=1).min(axis=1)
d[["appl_no", "ksic", "ay", "mg", "known", "applicant"]].to_parquet(f"{CACHE}/meta_kr.parquet", index=False)
print(d.mg.head(), d.known.isna().mean())
