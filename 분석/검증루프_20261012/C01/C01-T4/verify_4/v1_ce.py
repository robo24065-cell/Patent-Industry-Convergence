# verify_4: C-E 기각 근거 재계산 (원천: firm_nps_firm_panel, wkpl 스냅샷, 등록원부 applicant, pat_base)
import os, sqlite3, json
import pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\ext")
T4 = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4")
OUT = os.path.join(T4, "verify_4")
res = {}
d = pd.read_parquet(os.path.join(EXT, "firm_nps_firm_panel.parquet"), columns=["applicantCd", "snap", "active", "indcd", "n_wkpl"])
d = d[d.active & d.indcd.fillna("").ne("")]
mp = pd.read_csv(os.path.join(EXT, "firm_nps_indcd_ksic50_map.csv"), dtype=str, encoding="utf-8-sig")
i2k = mp.groupby("indcd").ksic.first().to_dict()
d["k50"] = d.indcd.map(i2k)
# 1) 스냅샷별 식별되는 KSIC 수(패널 기업 기준)
res["ksic_identified_by_snap"] = d.dropna(subset=["k50"]).groupby("snap").k50.nunique().to_dict()
# wkpl 원자료 기준도
ws = {}
for f in sorted(os.listdir(EXT)):
    if f.startswith("firm_nps_wkpl_") and f.endswith(".parquet"):
        w = pd.read_parquet(os.path.join(EXT, f), columns=["indcd"])
        ws[f[14:20]] = int(w.indcd.map(i2k).dropna().nunique())
res["ksic_identified_by_wkpl_snap"] = ws
# 2) 특허 KSIC 집합
pb = pd.read_parquet(os.path.join(T4, "cache", "pat_base.parquet"), columns=["rgst_no", "ksic", "rg"])
con = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
app = pd.read_sql("select distinct rgst_no, applicantCd from applicant where applicantCd like '1%'", con)
con.close()
pa = app.merge(pb, on="rgst_no")
pa["ry"] = pa.rg.dt.year
snaps = sorted(d.snap.unique())
rows = []
for gap in (1, 2):
    for i in range(len(snaps) - gap):
        s0, s1 = snaps[i], snaps[i + gap]
        y0 = int(s0[:4])
        x = d[d.snap == s0][["applicantCd", "indcd", "k50"]].merge(d[d.snap == s1][["applicantCd", "indcd", "k50"]], on="applicantCd")
        ks = pa[pa.ry <= y0][["applicantCd", "ksic"]].drop_duplicates()
        patf = set(ks.applicantCd)
        xp = x[x.applicantCd.isin(patf)]
        ch = xp[xp.indcd_x != xp.indcd_y]
        toward = ch.merge(ks, left_on=["applicantCd", "k50_y"], right_on=["applicantCd", "ksic"])
        toward = toward[toward.k50_x.fillna("") != toward.k50_y]
        rows.append(dict(gap=gap, s0=s0, s1=s1, both_active=len(x), pat_firms=len(xp), changed_pat=len(ch),
                         changed_into_k50=int(ch.k50_y.notna().sum()), toward_own_patent_ksic=int(toward.applicantCd.nunique())))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(OUT, "v1_ce_counts.csv"), index=False, encoding="utf-8-sig")
print(r.to_string())
# 3) 2024-12->2025-12 변경의 상위 전이
x = d[d.snap == "2024-12"][["applicantCd", "indcd"]].merge(d[d.snap == "2025-12"][["applicantCd", "indcd"]], on="applicantCd")
c = x[x.indcd_x != x.indcd_y]
res["top_transitions_2024_2025"] = c.groupby(["indcd_x", "indcd_y"]).size().sort_values(ascending=False).head(15).reset_index().values.tolist()
res["new_codes_2025_not_in_2024"] = int(len(set(d[d.snap == "2025-12"].indcd) - set(d[d.snap == "2024-12"].indcd)))
x = d[d.snap == "2022-12"][["applicantCd", "indcd"]].merge(d[d.snap == "2023-12"][["applicantCd", "indcd"]], on="applicantCd")
c = x[x.indcd_x != x.indcd_y]
res["top_transitions_2022_2023"] = c.groupby(["indcd_x", "indcd_y"]).size().sort_values(ascending=False).head(10).reset_index().values.tolist()
json.dump(res, open(os.path.join(OUT, "v1_ce.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str)[:4000])
