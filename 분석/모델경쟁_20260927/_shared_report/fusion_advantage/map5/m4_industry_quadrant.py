# -*- coding: utf-8 -*-
r"""map5 4단계: 한국 특허 포지션 ↔ 같은 KSIC 5자리로만 이어지는 국내 산업 지표 — 정합·괴리 사분면 (예측 주장 아님, 기술통계).
입력(읽기만): m2_panel_appl.csv, m1_origin_appl.parquet, 공모전데이터/작업데이터/kosis_광업제조업_50개산업.csv(10차 2008–2020),
            분석/fusion/ksic_hs6_map.csv, _shared/ext/R2ind_comtrade_exports_5c_hs6_2004_2021.csv + R2N3_comtrade_exports_5c_hs6.csv
사전 고정
- 개발 창: 특허 P2 출원 2015–17, 수출 2015–17, KOSIS 2015–17(10차). 변화: 특허 2008–10→2015–17, KOSIS 2008–09→2016–17, 수출 2008–10→2015–17.
- 특허 포지션 x = 한국 RTA(5개 관청 동일가중). 국제 산업 포지션 y = 한국 수출 RCA(KR·US·JP·CN·DE 5개 보고국 동일가중, 48 KSIC 범위).
  HS6가 여러 KSIC에 걸리면 같은 몫으로 나눈다. DE는 EP 대응(유럽 대표)으로 쓴다.
- 국내 생산 포지션 = KSIC의 50개 산업 내 출하액 비중 / 같은 범위 KR 특허 비중 (log, '생산 대비 기술 집약').
- 사분면 경계 = 1(RTA·RCA). 상관 = Spearman, KSIC 부트스트랩 2000회 95% CI.
- 사후(최종시험 구간 포함): 특허 P4 2020–22, 수출 2021, KOSIS 11차 2022–24 — 같은 사분면 판정의 안정성만 본다.
- 연계표 대조: 공식 IPC-KSIC 연계표가 한 그룹으로 묶는 34개 KSIC(label_validity 그룹 규칙)에서 같은 그룹 안 KSIC들이 서로 다른 사분면에 놓이는지 센다.
"""
import os, json, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(OUT, r"..\..\.."))
BASE = os.path.abspath(os.path.join(MC, r"..\.."))
CCS = ["KR", "US", "JP", "CN", "EP"]
rng = np.random.default_rng(42)

prof = pd.read_csv(os.path.join(OUT, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")
pa = pd.read_csv(os.path.join(OUT, "m2_panel_appl.csv"), dtype={"ksic": str})


def rta(y0, y1):
    n = pa[pa.year.between(y0, y1) & pa.valid].groupby(["ksic", "cc"]).n_inv.sum().unstack()
    s = n / n.sum()
    return (s.div(s.mean(1), axis=0)), n


rta_p2, n_p2 = rta(2015, 2017); rta_p1, n_p1 = rta(2008, 2010); rta_p4, n_p4 = rta(2020, 2022)

# ---- 수출 ----
mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype={"k5": str, "hs6": str})
mp["w"] = 1 / mp.groupby("hs6").k5.transform("nunique")
mp = mp.drop_duplicates(["k5", "hs6"])
ex = pd.concat([pd.read_csv(os.path.join(MC, r"_shared\ext", f), dtype={"hs6": str}) for f in
                ["R2ind_comtrade_exports_5c_hs6_2004_2021.csv", "R2N3_comtrade_exports_5c_hs6.csv"]]).drop_duplicates(["year", "reporter", "hs6"])
ex["hs6"] = ex.hs6.str.zfill(6)
rep = {410: "KR", 842: "US", 392: "JP", 156: "CN", 276: "DE"}
ex["rep"] = ex.reporter.map(rep)
ek = ex.merge(mp[["k5", "hs6", "w"]], on="hs6").assign(v=lambda d: d.export_usd * d.w).groupby(["k5", "rep", "year"]).v.sum().reset_index()
ek.to_csv(os.path.join(OUT, "m4_exports_ksic_5rep.csv"), index=False, encoding="utf-8-sig")


def rca(years):
    v = ek[ek.year.isin(years)].groupby(["k5", "rep"]).v.sum().unstack()
    s = v / v.sum()
    return s.div(s.mean(1), axis=0), v


rca_p2, v_p2 = rca([2015, 2016, 2017]); rca_p1, v_p1 = rca([2008, 2009, 2010]); rca_p4, v_p4 = rca([2021])

# ---- KOSIS ----
ko = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
ko = ko[ko.마스킹 == 0]
k10 = ko[ko.기준 == "10차"]; k11 = ko[ko.기준 == "11차"]


def kv(df, item, years):
    return df[(df.항목코드 == item) & df.연도.isin(years)].groupby("세세분류코드").값.mean()


ship_p2 = kv(k10, "T04", [2015, 2016, 2017]); emp_p2 = kv(k10, "T02", [2015, 2016, 2017])
ship_g = np.log(kv(k10, "T04", [2016, 2017]) / kv(k10, "T04", [2008, 2009]))
emp_g = np.log(kv(k10, "T02", [2016, 2017]) / kv(k10, "T02", [2008, 2009]))
ship_p4 = kv(k11, "T04", [2022, 2023, 2024])

T = pd.DataFrame(index=prof.index)
T["name"] = prof.name
T["pat_rta_P2"] = rta_p2.KR; T["pat_rta_P1"] = rta_p1.KR; T["pat_rta_P4"] = rta_p4.KR
T["pat_kr_n_P2"] = n_p2.KR
T["kr_origin_abroad_rta_P3"] = prof.kr_origin_abroad_rta_P3
T["exp_rca_P2"] = rca_p2.KR.reindex(T.index); T["exp_rca_P1"] = rca_p1.KR.reindex(T.index); T["exp_rca_2021"] = rca_p4.KR.reindex(T.index)
T["exp_kr_usd_P2"] = v_p2.KR.reindex(T.index)
T["exp_lead_rep_P2"] = rca_p2.idxmax(1).reindex(T.index)
T["ship_P2"] = ship_p2.reindex(T.index); T["emp_P2"] = emp_p2.reindex(T.index)
T["ship_share50"] = T.ship_P2 / T.ship_P2.sum()
T["pat_share50"] = T.pat_kr_n_P2 / T.pat_kr_n_P2.sum()
T["log_ship_over_pat"] = np.log(T.ship_share50 / T.pat_share50)
T["ship_g_0817"] = ship_g.reindex(T.index); T["emp_g_0817"] = emp_g.reindex(T.index)
T["pat_g_0817"] = np.log(n_p2.KR / n_p1.KR)
T["pat_rta_g_0817"] = np.log(T.pat_rta_P2 / T.pat_rta_P1)
T["exp_rca_g_0817"] = np.log(T.exp_rca_P2 / T.exp_rca_P1)
T["ship_share50_P4"] = (ship_p4 / ship_p4.sum()).reindex(T.index)


def quad(x, y, lx="기술", ly="수출"):
    return np.where(x.isna() | y.isna(), "판정불가",
           np.where(x >= 1, np.where(y >= 1, f"{lx} 강·{ly} 강(정합)", f"{lx} 강·{ly} 약(괴리)"),
                    np.where(y >= 1, f"{lx} 약·{ly} 강(괴리)", f"{lx} 약·{ly} 약(정합)")))


T["quad_tech_export_P2"] = quad(T.pat_rta_P2, T.exp_rca_P2)
T["quad_tech_export_P4"] = quad(T.pat_rta_P4, T.exp_rca_2021)
T["quad_abroad_export_P2"] = quad(T.kr_origin_abroad_rta_P3, T.exp_rca_P2, "해외출원")
T["quad_domestic"] = np.where(T.log_ship_over_pat.isna(), "판정불가",
                              np.where(T.log_ship_over_pat > 0, "생산 집약>기술 집약", "기술 집약≥생산 집약"))
T["quad_momentum"] = np.where(T.pat_g_0817.isna() | T.ship_g_0817.isna(), "판정불가",
                              np.where(T.pat_g_0817 >= T.pat_g_0817.median(),
                                       np.where(T.ship_g_0817 >= T.ship_g_0817.median(), "특허↑·출하↑", "특허↑·출하↓(괴리)"),
                                       np.where(T.ship_g_0817 >= T.ship_g_0817.median(), "특허↓·출하↑(괴리)", "특허↓·출하↓")))

# 연계표 그룹 (label_validity GROUP_PREFIX 규칙 그대로 복사)
GP = [("C1500", ["15"]), ("C2030", ["202"]), ("C2042", ["2041"]), ("C2043", ["2042"]), ("C2049", ["2049"]), ("C2100", ["21"]), ("C2300", ["23"]),
      ("C2410", ["241"]), ("C2420", ["242"]), ("C2590", ["259"]), ("C2610", ["261"]), ("C2620", ["262"]), ("C2630", ["263"]), ("C2640", ["264"]),
      ("C2650", ["265"]), ("C2710", ["271"]), ("C2720", ["272"]), ("C2730", ["273"]), ("C2810", ["281"]), ("C2850", ["285"]), ("C2890", ["289"]),
      ("C2918", ["2918"]), ("C2919", ["291"]), ("C2920", ["292"]), ("C3000", ["30"]), ("C3120", ["312"]), ("C3130", ["313"]), ("C3300", ["33"])]


def grp(c):
    for g, ps in GP:
        if any(c.startswith(p) for p in ps):
            return g


T["cw_group"] = [grp(k) for k in T.index]
T["cw_group_size"] = T.groupby("cw_group").name.transform("size")
T.to_csv(os.path.join(OUT, "m4_quadrant_table.csv"), encoding="utf-8-sig")


def sp_ci(a, b, B=2000):
    d = pd.concat([a, b], axis=1).dropna(); x, y = d.iloc[:, 0].values, d.iloc[:, 1].values
    r = spearmanr(x, y).statistic
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return dict(rho=round(r, 3), ci=[round(np.nanpercentile(bs, 2.5), 3), round(np.nanpercentile(bs, 97.5), 3)], n=len(x))


res = {
    "rho_patRTA_expRCA_P2": sp_ci(T.pat_rta_P2, T.exp_rca_P2),
    "rho_patRTA_P1_expRCA_P2(7년 시차)": sp_ci(T.pat_rta_P1, T.exp_rca_P2),
    "rho_abroadRTA_expRCA_P2": sp_ci(T.kr_origin_abroad_rta_P3, T.exp_rca_P2),
    "rho_patshare50_shipshare50_P2": sp_ci(T.pat_share50, T.ship_share50),
    "rho_patshare50_empshare50_P2": sp_ci(T.pat_share50, T.emp_P2 / T.emp_P2.sum()),
    "rho_patgrowth_shipgrowth_0817": sp_ci(T.pat_g_0817, T.ship_g_0817),
    "rho_patgrowth_empgrowth_0817": sp_ci(T.pat_g_0817, T.emp_g_0817),
    "rho_patRTAchange_expRCAchange_0817": sp_ci(T.pat_rta_g_0817, T.exp_rca_g_0817),
    "posthoc_rho_patRTA_P4_expRCA_2021": sp_ci(T.pat_rta_P4, T.exp_rca_2021),
    "quad_tech_export_P2_counts": T.quad_tech_export_P2.value_counts().to_dict(),
    "quad_tech_export_P4_counts(사후)": T.quad_tech_export_P4.value_counts().to_dict(),
    "quad_P2_vs_P4_same_share": round((T.quad_tech_export_P2 == T.quad_tech_export_P4)[T.quad_tech_export_P2 != "판정불가"].mean(), 3),
    "quad_abroad_export_counts": T.quad_abroad_export_P2.value_counts().to_dict(),
    "quad_domestic_counts": T.quad_domestic.value_counts().to_dict(),
    "quad_momentum_counts": T.quad_momentum.value_counts().to_dict(),
    "export_ksic_n": int(T.exp_rca_P2.notna().sum()), "kosis_ksic_n": int(T.ship_P2.notna().sum()),
}
# 연계표 대조: 묶이는 그룹 안 사분면 다양성
multi = T[T.cw_group_size > 1]
gq = multi.groupby("cw_group").agg(n=("name", "size"), n_quad=("quad_tech_export_P2", lambda s: s[s != "판정불가"].nunique()),
                                   rta_min=("pat_rta_P2", "min"), rta_max=("pat_rta_P2", "max"),
                                   rca_min=("exp_rca_P2", "min"), rca_max=("exp_rca_P2", "max"))
gq.to_csv(os.path.join(OUT, "m4_crosswalk_group_quadrants.csv"), encoding="utf-8-sig")
res["cw_multi_groups"] = int(len(gq)); res["cw_multi_ksic"] = int(gq.n.sum())
res["cw_groups_with_mixed_quadrants"] = int((gq.n_quad > 1).sum())
res["cw_ksic_in_mixed_groups"] = int(gq[gq.n_quad > 1].n.sum())
# 그룹 내 분산 비중(한국 특허 RTA log, 수출 RCA log)
for col in ["pat_rta_P2", "exp_rca_P2", "pat_g_0817", "ship_g_0817"]:
    v = np.log(T[col]) if col in ("pat_rta_P2", "exp_rca_P2") else T[col]
    d = pd.DataFrame({"v": v, "g": T.cw_group}).dropna()
    within = ((d.v - d.groupby("g").v.transform("mean")) ** 2).sum(); total = ((d.v - d.v.mean()) ** 2).sum()
    res[f"within_group_var_share_{col}"] = round(within / total, 3)
res["lists"] = {q: [f"{k} {T.at[k, 'name']}" for k in T.index[T.quad_tech_export_P2 == q]] for q in T.quad_tech_export_P2.unique()}
json.dump(res, open(os.path.join(OUT, "m4_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print(gq.round(2).to_string())
