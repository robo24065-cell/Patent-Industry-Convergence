# -*- coding: utf-8 -*-
"""C12-T1 s4b: (b) 세대·셀룰러 분포 정정판, (a) 공개 수치(IPlytics 5G 선언 패밀리 점유율) 대조용 기업 점유율.
입력: cache/etsi2(s1b). 출력: cache/s4b_out.json, cache/ext_compare.csv"""
import os, sys, glob, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(W, "cache", "etsi2")
O = {}
ag = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(C, "agg2_*.parquet")))], ignore_index=True)
tot = ag.rows.sum()
O["rows"] = int(tot)
O["gen_rows_share_all"] = {"5G=1": float(ag[ag.gen.str[0] == "1"].rows.sum() / tot), "4G=1": float(ag[ag.gen.str[1] == "1"].rows.sum() / tot),
                           "3G=1": float(ag[ag.gen.str[2] == "1"].rows.sum() / tot), "2G=1": float(ag[ag.gen.str[3] == "1"].rows.sum() / tot),
                           "세대 열 모두 0": float(ag[ag.gen == "0000"].rows.sum() / tot), "세대 열 모두 빈칸": float(ag[ag.gen == "blank"].rows.sum() / tot)}
k = ag[ag.cc == "KR"]; tk = k.rows.sum()
O["gen_rows_share_KR"] = {"5G=1": float(k[k.gen.str[0] == "1"].rows.sum() / tk), "4G=1": float(k[k.gen.str[1] == "1"].rows.sum() / tk),
                          "3G=1": float(k[k.gen.str[2] == "1"].rows.sum() / tk), "2G=1": float(k[k.gen.str[3] == "1"].rows.sum() / tk),
                          "세대 열 모두 0": float(k[k.gen == "0000"].rows.sum() / tk), "세대 열 모두 빈칸": float(k[k.gen == "blank"].rows.sum() / tk)}
O["cellular_rows_share_all"] = float(ag[ag.cell == 1].rows.sum() / tot)
O["cellular_rows_share_KR"] = float(k[k.cell == 1].rows.sum() / tk)
O["cellular_or_gen_rows_share_all"] = float(ag[(ag.cell == 1) | ag.gen.str.contains("1")].rows.sum() / tot)
O["cellular_or_gen_rows_share_KR"] = float(k[(k.cell == 1) | k.gen.str.contains("1")].rows.sum() / tk)
# 반영연도별 5G 비중(KR 행)
ky = k.groupby("refl_y").apply(lambda d: float(d[d.gen.str[0] == "1"].rows.sum() / d.rows.sum()), include_groups=False)
O["KR_5G_row_share_by_refl_year"] = ky.round(4).to_dict()
# ---- 패밀리×기업
fm = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(C, "famgen_*.parquet")))], ignore_index=True)
fm = fm.groupby(["DIPG_PATF_ID", "COMP_LEGAL_NAME"]).agg(refl=("refl", "min"), refl5=("refl5", "min"), refl4=("refl4", "min"), reflc=("reflc", "min"),
                                                       kr=("kr", "max"), ess=("ess", "max"), gblank=("gblank", "min")).reset_index()
fm.to_parquet(os.path.join(W, "cache", "fam_company_gen.parquet"), index=False)
O["families"] = int(fm.DIPG_PATF_ID.nunique()); O["families_5G"] = int(fm[fm.refl5 < "9999"].DIPG_PATF_ID.nunique())
O["families_cellular"] = int(fm[fm.reflc < "9999"].DIPG_PATF_ID.nunique())
# 기업명 정리(같은 그룹의 법인 표기 묶기 — 공개 보고서도 그룹 단위)
GRP = [("Huawei", "HUAWEI"), ("Qualcomm", "QUALCOMM"), ("Samsung", "SAMSUNG"), ("ZTE", "ZTE"), ("LG Electronics", "LG ELECTRONICS"),
       ("Nokia", "NOKIA|ALCATEL"), ("Ericsson", "ERICSSON"), ("CATT/Datang", "DATANG|CATT|CHINA ACADEMY OF TELECOMMUNICATIONS"),
       ("OPPO", "OPPO"), ("vivo", "VIVO MOBILE"), ("Sharp", "SHARP"), ("Apple", "APPLE"), ("NTT Docomo", "DOCOMO"), ("Xiaomi", "XIAOMI|BEIJING XIAOMI"),
       ("InterDigital", "INTERDIGITAL"), ("Intel", "INTEL"), ("MediaTek", "MEDIATEK"), ("Lenovo/Motorola", "LENOVO|MOTOROLA MOBILITY"), ("Sony", "SONY"),
       ("ETRI", "ELECTRONICS AND TELECOMMUNICATIONS RESEARCH"), ("WILUS", "WILUS"), ("KT", "^KT CORP")]
up = fm.COMP_LEGAL_NAME.str.upper()
fm["grp"] = "other:" + fm.COMP_LEGAL_NAME
for g, pat in GRP:
    m = up.str.contains(pat, regex=True) & fm.grp.str.startswith("other:")
    fm.loc[m, "grp"] = g
rows = []
for lab, cut, col in [("IPlytics 2021-02 비교(5G, 반영 ≤2021-01-31)", "2021-01-31", "refl5"), ("IPlytics 2021-11 비교(5G, 반영 ≤2021-10-31)", "2021-10-31", "refl5"),
                      ("현재(5G, 반영 ≤2026-09-28)", "2026-09-28", "refl5"), ("현재(전 세대, 반영 ≤2026-09-28)", "2026-09-28", "refl")]:
    q = fm[fm[col] <= cut]
    nf = q.DIPG_PATF_ID.nunique()
    sh = q.drop_duplicates(["DIPG_PATF_ID", "grp"]).groupby("grp").DIPG_PATF_ID.nunique().sort_values(ascending=False) / nf
    for r, (g, v) in enumerate(sh.head(15).items(), 1):
        rows.append(dict(comparison=lab, rank=r, group=g, share=round(float(v), 4), families_total=int(nf)))
X = pd.DataFrame(rows)
X.to_csv(os.path.join(W, "cache", "ext_compare.csv"), index=False, encoding="utf-8-sig")
# KR 구성원이 있는 5G 패밀리 비율, KR 국적 기업(삼성·LG·ETRI·KT·WILUS 등) 점유
q = fm[fm.refl5 <= "2026-09-28"]
O["share_5G_families_with_KR_member"] = float(q[q.kr == 1].DIPG_PATF_ID.nunique() / q.DIPG_PATF_ID.nunique())
krco = q.grp.isin(["Samsung", "LG Electronics", "ETRI", "WILUS", "KT"]) | q.COMP_LEGAL_NAME.str.upper().str.contains("KOREA|PANTECH|INNOVATIVE TECHNOLOGY LAB|SK TELECOM|HYUNDAI|KAIST|SEOUL NATIONAL", regex=True)
O["share_5G_families_KR_declarant_approx"] = float(q[krco].DIPG_PATF_ID.nunique() / q.DIPG_PATF_ID.nunique())
O["ess_flag_share_families"] = float(fm.groupby("DIPG_PATF_ID").ess.max().mean())
O["secs"] = round(time.time() - t0, 1)
json.dump(O, open(os.path.join(W, "cache", "s4b_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(O, ensure_ascii=False, indent=1))
print(X.to_string())
