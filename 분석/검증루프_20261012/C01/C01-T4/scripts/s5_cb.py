# s5: C-B 「해외 부상 셀의 한국 첫 도착」 개발 타당성. 잠금 원점(2024) 결과는 만들지 않는다:
#     KR 문서는 결과용으로 fp ≤ 2024-06-30만, 잠금 적격 판정용으로 fp ≤ 2024-12-31만 읽는다.
import os, json, time
import duckdb, pandas as pd, numpy as np
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
res = {}
# IPC 형식 점검
res["ipc_samples"] = con.sql(f"select cc, any_value(main_ipc) from read_parquet('{META}') group by cc").fetchall()
con.sql(f"""create temp table d as
 select cc, appl_no, ksic,
   least(try_strptime(nullif(pub_date,''),'%Y%m%d'), try_strptime(nullif(rgst_date,''),'%Y%m%d')) fp,
   upper(regexp_replace(split_part(main_ipc,'/',1),'\\s+',' ','g')) mg
 from (select distinct cc, appl_no, ksic, pub_date, rgst_date, main_ipc from read_parquet('{META}'))""")
res["rows_distinct"] = con.sql("select cc, count(*), sum((fp is null)::int), sum((mg is null or mg='')::int) from d group by cc order by cc").fetchall()
# 해외: 셀×연도×국가 건수
fo = con.sql("""select ksic, mg, cc, year(fp) y, count(*) n from d where cc<>'KR' and fp is not null and fp < '2025-01-01' group by 1,2,3,4""").df()
# KR 첫 공개(결과용: ≤2024-06-30)
kr_dev = con.sql("""select ksic, mg, min(fp) kr_first from d where cc='KR' and fp is not null and fp <= '2024-06-30' group by 1,2""").df()
# KR 존재(잠금 적격용: ≤2024-12-31)
kr_lock = con.sql("""select ksic, mg, 1 kr_le2024 from d where cc='KR' and fp is not null and fp <= '2024-12-31' group by 1,2""").df()
# 메인그룹 단위 KR 첫 공개(라벨 공백 vs 기술 공백)
kr_g = con.sql("""select mg, min(fp) kr_g_first from d where cc='KR' and fp is not null and fp <= '2024-12-31' group by 1""").df()
# P4: 잠금 구간 KR 건수(KSIC별) vs 2024 상반기 — 건수만
p4 = con.sql("""select ksic, sum((fp between '2024-01-01' and '2024-06-30')::int) h1_2024, sum((fp between '2025-01-01' and '2025-06-30')::int) h1_2025
                from d where cc='KR' group by 1""").df()
con.close()
p4.to_csv(os.path.join(OUTD, "s5_p4_kr_ksic_h1.csv"), index=False, encoding="utf-8-sig")
a, b = p4.h1_2024 / p4.h1_2024.sum(), p4.h1_2025 / p4.h1_2025.sum()
res["p4_kr_h1_total"] = [int(p4.h1_2024.sum()), int(p4.h1_2025.sum())]
res["p4_ksic_dissimilarity_h1_2024_vs_2025"] = round(float(0.5 * (a - b).abs().sum()), 4)
res["p4_ksic_max_abs_share_diff"] = round(float((a - b).abs().max()), 4)
kr_dev["kr_first"] = pd.to_datetime(kr_dev.kr_first); kr_g["kr_g_first"] = pd.to_datetime(kr_g.kr_g_first)
rows = []
for t0 in range(2012, 2025):
    f = fo[(fo.y >= t0 - 2) & (fo.y <= t0)]
    fc = f.groupby(["ksic", "mg"]).n.sum().rename("nf").reset_index()
    fcn = f[f.cc == "CN"].groupby(["ksic", "mg"]).n.sum().rename("ncn")
    fc = fc.merge(fcn, on=["ksic", "mg"], how="left").fillna({"ncn": 0})
    if t0 < 2024:
        c = fc.merge(kr_dev, on=["ksic", "mg"], how="left")
        c = c[~(c.kr_first <= pd.Timestamp(f"{t0}-12-31"))]
        c["y"] = c.kr_first.between(pd.Timestamp(f"{t0+1}-01-01"), pd.Timestamp(f"{t0+1}-06-30"))
    else:
        c = fc.merge(kr_lock, on=["ksic", "mg"], how="left")
        c = c[c.kr_le2024.isna()]
        c["y"] = np.nan
    c = c.merge(kr_g, on="mg", how="left")
    c["label_gap"] = c.kr_g_first <= pd.Timestamp(f"{t0}-12-31")    # 메인그룹엔 KR 문서가 이미 있음(다른 KSIC 라벨)
    for F in (10, 20, 50):
        s = c[c.nf >= F]
        r = dict(t0=t0, F=F, lock=(t0 == 2024), cells=len(s), ksic_n=s.ksic.nunique(), mg_n=s.mg.nunique(),
                 cn_share_foreign_docs=round(float(s.ncn.sum() / max(s.nf.sum(), 1)), 4),
                 label_gap_share=round(float(s.label_gap.mean()), 4) if len(s) else np.nan)
        if t0 < 2024:
            r.update(pos=int(s.y.sum()), base_rate=round(float(s.y.mean()), 4),
                     pos_label_gap=int((s.y & s.label_gap).sum()), pos_tech_gap=int((s.y & ~s.label_gap).sum()),
                     rate_label_gap=round(float(s[s.label_gap].y.mean()), 4) if s.label_gap.any() else np.nan,
                     rate_tech_gap=round(float(s[~s.label_gap].y.mean()), 4) if (~s.label_gap).any() else np.nan,
                     cn_only_cells_share=round(float((s.ncn == s.nf).mean()), 4),
                     pos_cn_only=int((s.y & (s.ncn == s.nf)).sum()))
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUTD, "s5_cb_counts.csv"), index=False, encoding="utf-8-sig")
json.dump(res, open(os.path.join(OUTD, "s5_cb.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(R[R.F == 20].to_string()); print(res); print(time.time() - t)
