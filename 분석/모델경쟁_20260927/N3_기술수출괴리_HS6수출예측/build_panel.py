# -*- coding: utf-8 -*-
"""5개국 특허 집계 패널 (KSIC × 국가 × 출원연도 × 공개(최초공표)연도 × 우선권국가).
최초공표연도 dy = min(공개일, 등록일)의 연도 (KR 조기등록·CN 실용신안은 공개일이 없고 등록공보로 공표).
🔒 dy ≤ 2023 만 집계 (G3 최종시험 목표연도 2024 봉인). 2024 이후 건수는 계산하지 않는다.
출력: cache/pat_panel.parquet
"""
import os, duckdb
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "..", "_shared", "meta", "*.parquet").replace("\\", "/")
os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
con = duckdb.connect()
con.execute("set memory_limit='3GB'; set threads=2")
q = f"""
with b as (
  select ksic, cc, cast(appl_year as int) ay,
    least(case when pub_date is not null and length(pub_date)>=4 then cast(substr(pub_date,1,4) as int) end,
          case when rgst_date is not null and length(rgst_date)>=4 then cast(substr(rgst_date,1,4) as int) end) dy,
    case when prio_cc is null or prio_cc='' then cc else split_part(prio_cc,'|',1) end p1
  from read_parquet('{META}')
)
select ksic, cc, ay, dy,
  case when p1 in ('KR','US','JP','CN') then p1
       when p1 in ('EP','DE','FR','GB','IT','NL','CH','SE','AT','BE','DK','FI','ES','WO') then 'EU' else 'OT' end p1,
  count(*)::int n
from b where dy is not null and dy <= 2023 and ay <= 2023
group by all
"""
df = con.sql(q).df()
df.to_parquet(os.path.join(HERE, "cache", "pat_panel.parquet"), index=False)
print(len(df), df.dy.min(), df.dy.max(), df.n.sum())
