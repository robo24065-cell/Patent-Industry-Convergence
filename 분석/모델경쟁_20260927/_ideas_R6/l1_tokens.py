# -*- coding: utf-8 -*-
"""패밀리 토큰 파싱 → (cc, appl_no)별 '관청별 첫 공개연도' 표.
family 필드는 공개·등록번호 토큰(현재 스냅샷)이다. 출원번호가 아니다(KR 'KR10YYYY#######'은 KR 공개번호).
연도 식별: US 공개 US+YYYY+6~7자리, JP 공개 JP+YYYY+6자리, WO WO+YYYY+6자리, KR 공개 KR(10|20)+YYYY+7자리 또는 KR+YYYY+7자리,
EP 공개번호(연도 없음)는 EP 행 자신의 공개일로 번호→연도 보정표를 만든다. CN 공개번호는 연도를 알 수 없어 쓰지 않는다.
🔒 연도 ≥ 2023 인 토큰은 파싱 단계에서 버린다(결과·피처 모두 2022년 말까지만)."""
import os, time, duckdb, numpy as np, pandas as pd
from r6_common import CACHE

t = time.time()
C = CACHE.replace("\\", "/")
con = duckdb.connect(); con.sql("set threads=2; set memory_limit='3GB'; set preserve_insertion_order=false; set enable_progress_bar=false")
V = "least(coalesce(nullif(trim(pubd),''),'99999999'), coalesce(nullif(trim(rgd),''),'99999999'))"

# 1) EP 번호→공개연도 보정표: EP 행 중 EP 번호 토큰이 1개뿐인 행의 공개일
con.sql(f"""create table epcal as
 select min(num) num, any_value(py) py from (
 select appl_no, cast(regexp_extract(tok,'^EP0?([0-9]{{7}})$',1) as int) num, cast(substr(pubd,1,4) as int) py
 from (select appl_no, pubd, unnest(string_split(fam,'|')) tok from '{C}/p_EP.parquet' where pubd is not null and trim(pubd)<>'')
 where regexp_matches(tok,'^EP0?[0-9]{{7}}$')) group by appl_no having count(distinct num)=1""")
cal = con.sql("""select num//5000 b, median(py) py, count(*) n from (
   select e.* from epcal e) group by 1 having count(*)>=5 order by 1""").df()
cal["py"] = np.maximum.accumulate(cal["py"].values)  # 단조 보정
full = pd.DataFrame({"b": np.arange(0, 2000)}).merge(cal[["b", "py"]], on="b", how="left")
full["py"] = full["py"].ffill().bfill()
full.loc[full.b > cal.b.max(), "py"] = np.nan  # 🔒 2022년 말 이후 번호대는 연도 미상(=2023+)으로 둔다
cal = full
cal.to_csv(os.path.join(CACHE, "ep_num_year_cal.csv"), index=False)
con.register("cal", cal)
print("EP cal bins", len(cal), cal.head(3).values.tolist(), cal.tail(3).values.tolist())

rows = []
for cc in ["KR", "US", "JP", "EP"]:
    con.sql(f"""create or replace table tk as
      select appl_no, tok from (select appl_no, unnest(string_split(fam,'|')) tok from '{C}/p_{cc}.parquet' where fam is not null)""")
    con.sql("""create or replace table ty0 as
      select appl_no, tok,
        case when regexp_matches(tok,'^EP0?[0-9]{7}$') then try_cast(regexp_extract(tok,'^EP0?([0-9]{7})$',1) as int)//5000 end epb,
        case when regexp_matches(tok,'^US20[0-9]{2}[0-9]{6,7}$') then 'US'
             when regexp_matches(tok,'^JP20[0-9]{2}[0-9]{6}$') then 'JP'
             when regexp_matches(tok,'^WO[12][0-9]{3}[0-9]{6}$') then 'WO'
             when regexp_matches(tok,'^KR(10|20)[12][0-9]{3}[0-9]{7}$') or regexp_matches(tok,'^KR[12][0-9]{3}[0-9]{7}$') then 'KR'
             when regexp_matches(tok,'^EP0?[0-9]{7}$') then 'EP'
             when regexp_matches(tok,'^CN[0-9]') then 'CN'
             when regexp_matches(tok,'^[A-Z]{2}') then 'OT' end ofc,
        case when regexp_matches(tok,'^(US|JP|WO)[12][0-9]{3}') then cast(substr(tok,3,4) as int)
             when regexp_matches(tok,'^KR(10|20)[12][0-9]{3}[0-9]{7}$') then cast(substr(tok,5,4) as int)
             when regexp_matches(tok,'^KR[12][0-9]{3}[0-9]{7}$') then cast(substr(tok,3,4) as int)
        end yr0
      from tk""")
    con.sql("""create or replace table ty as
      select ty0.appl_no, ty0.ofc, coalesce(ty0.yr0, cal.py)::int yr from ty0 left join cal on ty0.epb = cal.b""")
    agg = con.sql(f"""select appl_no,
        min(case when ofc='US' and yr<2023 then yr end) y_us,
        min(case when ofc='JP' and yr<2023 then yr end) y_jp,
        min(case when ofc='EP' and yr<2023 then yr end) y_ep,
        min(case when ofc='WO' and yr<2023 then yr end) y_wo,
        min(case when ofc='KR' and yr<2023 then yr end) y_kr,
        max((ofc='CN')::int) has_cn_any, max((ofc='OT')::int) has_ot_any, count(*) ntok
      from ty group by 1""").df()
    # 🔒 연도 ≥ 2023 토큰은 위 min()에서 제외됨. 존재 여부(has_cn_any 등)는 연도를 모르는 스냅샷이라 피처로 쓰지 않는다(기록용).
    for c in ["y_us", "y_jp", "y_ep", "y_wo", "y_kr"]:
        assert agg[c].dropna().max() < 2023 if agg[c].notna().any() else True
    agg.insert(0, "cc", cc)
    agg.to_parquet(os.path.join(CACHE, f"tok_{cc}.parquet"), index=False)
    print(cc, len(agg), {c: int(agg[c].notna().sum()) for c in ["y_us", "y_jp", "y_ep", "y_wo", "y_kr"]}, round(time.time() - t, 1), flush=True)
