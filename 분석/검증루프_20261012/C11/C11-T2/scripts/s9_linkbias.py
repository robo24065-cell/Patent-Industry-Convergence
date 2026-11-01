# -*- coding: utf-8 -*-
r"""C11-T2 s9: 데이터 점검(새 데이터 없음) — T3 풀 TXT 패밀리 상속의 선택 편향·시점 정합.
 (b)(c) 팔별: 해외 복수 코드 문서 중 상속(family) 문서와 표 배정 문서의 국가·KSIC(제공) 분포 TVD, 제공 라벨 일치율.
 (d) 시점: 상속 문서의 연결 KR 문서 공개연도(base_rows year, T3 'pub' 정의)가 해외 문서 공개연도보다 늦은 비율(라벨이 '미래 KR 문서'에서 오는 비율).
쓰기: res/s9_linkbias.csv, res/s9_timing.csv"""
import sys
sys.dont_write_bytecode = True
import duckdb, pandas as pd
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; C10 = BASE + "/분석/검증루프_20261012/C10/C10-T3"
con = duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str); multi = tuple(km.code.value_counts().loc[lambda s: s > 1].index)
con.execute(f"create table yr as select cc, appl_no, min(year) y from read_parquet('{T3}/R3/cache_pub/base_rows.parquet') group by 1,2")
rows, trows = [], []
for arm in ("r8", "reg", "pub"):
    A = f"read_parquet('{W}/cache/txt_assign_{arm}.parquet')"
    d = con.sql(f"""select cc, sum((\"method\"='family')::int) n_fam, count(*) n_multi,
        avg(case when \"method\"='family' then (ksic_txt=ksic_prov)::int end) agree_fam,
        avg(case when \"method\"<>'family' then (ksic_txt=ksic_prov)::int end) agree_tab
        from {A} where cc<>'KR' and code in {multi} group by rollup(cc) order by cc""").df()
    tv = con.sql(f"""with a as (select ksic_prov k, (\"method\"='family') f, count(*) n from {A} where cc<>'KR' and code in {multi} group by all),
        s as (select k, f, n / sum(n) over (partition by f) p from a)
        select 0.5*sum(abs(coalesce(x.p,0)-coalesce(y.p,0))) from (select * from s where f) x full join (select * from s where not f) y using(k)""").fetchone()[0]
    d.insert(0, "arm", arm); d["tvd_ksic_fam_vs_tab_all"] = tv; rows.append(d)
    if arm != "r8":
        L = f"(select cc, appl_no, kr_appl_no from read_parquet('{C10}/cache/links.parquet') where link='{arm.upper()}')"
    else:
        L = f"(select cc, appl_no, kr_appl_no from read_parquet('{MC}/_ideas_R8/cache/fam_link.parquet'))"
    t = con.sql(f"""select o.cc, count(distinct o.appl_no) n_fam_docs, avg((ky.y > fy.y)::int) share_kr_later, avg((ky.y > fy.y + 2)::int) share_kr_later_3y
        from {A} o join {L} l using(cc, appl_no) join yr fy on fy.cc=o.cc and fy.appl_no=o.appl_no join yr ky on ky.cc='KR' and ky.appl_no=l.kr_appl_no
        where o.\"method\"='family' group by rollup(o.cc) order by 1""").df()
    t.insert(0, "arm", arm); trows.append(t)
B = pd.concat(rows); B.to_csv(W + "/res/s9_linkbias.csv", index=False, encoding="utf-8-sig"); print(B.round(4).to_string())
T = pd.concat(trows); T.to_csv(W + "/res/s9_timing.csv", index=False, encoding="utf-8-sig"); print(T.round(4).to_string())
