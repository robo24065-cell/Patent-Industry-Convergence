# verify_1: C-B 반박 재계산. (A) 원 스크립트 정의 그대로 재현 (B) IPC 메인그룹 표기 정규화(공백 유무·선행 0) 후 재계산.
# 잠금 원점(2024) 결과는 만들지 않는다: KR fp<=2024-06-30만 결과용으로 읽는다.
import os, json, time
import duckdb, pandas as pd, numpy as np
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\verify_1")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
out = {}
# 표기 형식 분포(국가별, 공백 없는 형식 비중)
out["fmt"] = con.sql(f"""select cc, count(*) n, sum((main_ipc is null)::int) nnull,
  sum(regexp_matches(main_ipc,'^[A-H][0-9]{{2}}[A-Z] +[0-9]+/')::int) sp,
  sum(regexp_matches(main_ipc,'^[A-H][0-9]{{2}}[A-Z][0-9]+/')::int) nosp,
  sum(regexp_matches(main_ipc,'^[A-H][0-9]{{2}}[A-Z] *0[0-9]+/')::int) lead0
  from (select distinct cc, appl_no, ksic, main_ipc from read_parquet('{META}')) group by cc order by cc""").fetchall()
con.sql(f"""create temp table d as
 select cc, appl_no, ksic,
   least(try_strptime(nullif(pub_date,''),'%Y%m%d'), try_strptime(nullif(rgst_date,''),'%Y%m%d')) fp,
   upper(regexp_replace(split_part(main_ipc,'/',1),'\s+',' ','g')) mg_orig,
   case when regexp_matches(upper(main_ipc),'^[A-H][0-9]{{2}}[A-Z] *[0-9]+')
        then regexp_extract(upper(main_ipc),'^([A-H][0-9]{{2}}[A-Z])',1) || ' ' ||
             cast(cast(regexp_extract(upper(main_ipc),'^[A-H][0-9]{{2}}[A-Z] *([0-9]+)',1) as integer) as varchar)
        else null end mg_norm
 from (select distinct cc, appl_no, ksic, pub_date, rgst_date, main_ipc from read_parquet('{META}'))""")
out["n_rows"] = con.sql("select cc, count(*), sum((mg_orig<>coalesce(mg_norm,'')) ::int), sum((mg_norm is null)::int) from d group by cc order by cc").fetchall()
out["kr_2025h1_appl_year"] = con.sql(f"""select appl_year, count(*) from (select distinct appl_no, ksic, appl_year,
   least(try_strptime(nullif(pub_date,''),'%Y%m%d'), try_strptime(nullif(rgst_date,''),'%Y%m%d')) fp from read_parquet('{META}') where cc='KR')
   where fp between '2025-01-01' and '2025-06-30' group by 1 order by 1""").fetchall()
res_all = {}
for MG in ("mg_orig", "mg_norm"):
    fo = con.sql(f"select ksic, {MG} mg, cc, year(fp) y, count(*) n from d where cc<>'KR' and fp < '2025-01-01' and {MG} is not null group by 1,2,3,4").df()
    kr_dev = con.sql(f"select ksic, {MG} mg, min(fp) kr_first from d where cc='KR' and fp <= '2024-06-30' and {MG} is not null group by 1,2").df()
    kr_g = con.sql(f"select {MG} mg, min(fp) kr_g_first from d where cc='KR' and fp <= '2024-06-30' and {MG} is not null group by 1").df()
    kr_dev["kr_first"] = pd.to_datetime(kr_dev.kr_first); kr_g["kr_g_first"] = pd.to_datetime(kr_g.kr_g_first)
    rows = []
    for t0 in range(2012, 2024):
        f = fo[(fo.y >= t0 - 2) & (fo.y <= t0)]
        fc = f.groupby(["ksic", "mg"]).n.sum().rename("nf").reset_index()
        fcn = f[f.cc == "CN"].groupby(["ksic", "mg"]).n.sum().rename("ncn")
        fc = fc.merge(fcn, on=["ksic", "mg"], how="left").fillna({"ncn": 0})
        c = fc.merge(kr_dev, on=["ksic", "mg"], how="left")
        c = c[~(c.kr_first <= pd.Timestamp(f"{t0}-12-31"))]
        c["y6"] = c.kr_first.between(pd.Timestamp(f"{t0+1}-01-01"), pd.Timestamp(f"{t0+1}-06-30"))
        c = c.merge(kr_g, on="mg", how="left")
        c["label_gap"] = c.kr_g_first <= pd.Timestamp(f"{t0}-12-31")
        for F in (10, 20, 50):
            s = c[c.nf >= F]
            rows.append(dict(t0=t0, F=F, cells=len(s), pos=int(s.y6.sum()),
                             pos_label_gap=int((s.y6 & s.label_gap).sum()),
                             label_gap_share=round(float(s.label_gap.mean()), 4),
                             cn_share=round(float(s.ncn.sum() / max(s.nf.sum(), 1)), 4)))
    R = pd.DataFrame(rows); R["mg"] = MG
    res_all[MG] = R
con.close()
R = pd.concat(res_all.values())
R.to_csv(os.path.join(OUTD, "v1_counts.csv"), index=False, encoding="utf-8-sig")
summ = R.groupby(["mg", "F"]).agg(pos_sum=("pos", "sum"), pos_min=("pos", "min"), pos_max=("pos", "max"),
                                   lg_min=("label_gap_share", "min"), lg_max=("label_gap_share", "max"),
                                   pos_lg=("pos_label_gap", "sum")).reset_index()
out["summary"] = summ.to_dict("records")
json.dump(out, open(os.path.join(OUTD, "v1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str, indent=0)); print(R[R.F == 20].to_string()); print(time.time() - t)
