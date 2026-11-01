# verify_1 v3: CN main_ipc 결측(2021 38%)을 ipc_all 첫 코드로 보충했을 때 C-B 양성(F=20) 민감도. 참고용 12개월 창(t0<=2022)도 센다.
import os, json, time
import duckdb, pandas as pd
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\verify_1")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
out = {}
out["cn_null_ipcall"] = con.sql(f"""select count(*), sum((ipc_all is not null and ipc_all<>'')::int) from (select distinct appl_no, ksic, main_ipc, ipc_all from read_parquet('{META}') where cc='CN' and main_ipc is null)""").fetchall()
out["ipcall_sample"] = con.sql(f"select ipc_all from read_parquet('{META}') where cc='CN' and main_ipc is null and ipc_all is not null limit 3").fetchall()
con.sql(f"""create temp table d as
 select cc, appl_no, ksic, least(try_strptime(nullif(pub_date,''),'%Y%m%d'), try_strptime(nullif(rgst_date,''),'%Y%m%d')) fp,
   upper(regexp_replace(split_part(coalesce(main_ipc, regexp_extract(ipc_all,'([A-H][0-9]{{2}}[A-Z] *[0-9]+/[0-9]+)',1)),'/',1),'\s+',' ','g')) mg
 from (select distinct cc, appl_no, ksic, pub_date, rgst_date, main_ipc, ipc_all from read_parquet('{META}'))""")
out["null_after_fill"] = con.sql("select cc, sum((mg is null or mg='')::int) from d group by cc order by cc").fetchall()
fo = con.sql("select ksic, mg, cc, year(fp) y, count(*) n from d where cc<>'KR' and fp < '2025-01-01' and mg<>'' group by 1,2,3,4").df()
kr = con.sql("select ksic, mg, min(fp) kr_first from d where cc='KR' and fp <= '2024-06-30' and mg<>'' group by 1,2").df()
con.close()
kr["kr_first"] = pd.to_datetime(kr.kr_first)
rows = []
for t0 in range(2012, 2024):
    f = fo[(fo.y >= t0 - 2) & (fo.y <= t0)]
    fc = f.groupby(["ksic", "mg"]).n.sum().rename("nf").reset_index()
    c = fc.merge(kr, on=["ksic", "mg"], how="left"); c = c[~(c.kr_first <= pd.Timestamp(f"{t0}-12-31"))]
    s = c[c.nf >= 20]
    y6 = s.kr_first.between(pd.Timestamp(f"{t0+1}-01-01"), pd.Timestamp(f"{t0+1}-06-30"))
    y12 = s.kr_first.between(pd.Timestamp(f"{t0+1}-01-01"), pd.Timestamp(f"{t0+1}-12-31")) if t0 <= 2022 else None
    rows.append(dict(t0=t0, cells=len(s), pos6=int(y6.sum()), pos12=(int(y12.sum()) if y12 is not None else None)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUTD, "v3_cnfill_F20.csv"), index=False, encoding="utf-8-sig")
out["pos6_sum"] = int(R.pos6.sum()); out["pos6_range"] = [int(R.pos6.min()), int(R.pos6.max())]
out["pos12_sum_t0_2012_2022"] = int(R.pos12.dropna().sum())
json.dump(out, open(os.path.join(OUTD, "v3_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(out); print(R.to_string()); print(time.time() - t)
