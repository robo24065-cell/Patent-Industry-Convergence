# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5\m2_trend_map.py — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)
# -*- coding: utf-8 -*-
r"""map5 2단계: 50 KSIC × 5개국 × 연도 기술 지형도 지표 + WIPO 커버리지 보정.
입력(읽기만): m1_*.parquet, m1b_ksic_field_pub.parquet, _shared/ext/quality_raw/quality_wipo_pi/dc_indicator_patent_4_publication_by_technology.csv,
            공모전데이터/작업데이터/kosis_광업제조업_50개산업.csv(산업명만)
산출: m2_coverage_factors_pub.csv, m2_panel_appl.csv, m2_panel_pub_corrected.csv, m2_ksic_profile.csv, m2_country_summary.csv,
      m2_jp_uniformity.csv, m2_checks.json
규칙(사전 고정)
- 주 계열: 출원연도·발명특허(m1 정의). CN은 2004년부터(2000–03 수록 미미), 5개국 비교 지표는 2004–2022.
- 점유: 국가 내 KSIC 비중 s_ck = n_ck / Σ_k n_ck (50 KSIC 범위 안 상대 비중).
- RTA 주 정의 = 5개국 동일가중: s_ck / mean_c(s_ck). 보조 = Balassa(5개국 합계 기준, CN 물량이 지배).
- 구간: P0 2004–06, P1 2008–10, P2 2015–17(개발 '최근'), P3 2017–19(개발 끝, T3 F=2020 이전), P4 2020–22(사후: 최종시험 구간 포함, JP 커버리지 의심).
- 성장: 구간 평균 건수의 연평균 증가율(CAGR) P1→P3 (주), P2→P4 (사후).
- 한국 대비 격차: P3 RTA 1위 국가(선도국)와 log(RTA_KR / RTA_선도국).
- WIPO 보정(공개연도 보조 계열): 국가×연도×WIPO분야 커버리지 비율 r = 제공/WIPO. 기준 r* = 같은 분야 ±3년 중앙값(자기 연도 제외).
  국가×연도 총계에서 |log(r*/r)| > log(1.15)인 해만 이상치로 보고(완만한 추세는 보정하지 않음), 그 해에 한해 분야별 계수 r*/r(분야 ±3년 중앙값 기준)를 곱한다.
  작은 칸(WIPO<300 또는 제공<30)·미매핑은 국가×연도 총계 계수. 공개 2000–02는 좌측 절단이라 보정하지 않고 계열에서 뺀다(2003~).
"""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
import os, json
import numpy as np, pandas as pd
OUT = _C19_OUT
MC = os.path.abspath(os.path.join(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5", r"..\..\.."))
BASE = os.path.abspath(os.path.join(MC, r"..\.."))
CCS = ["KR", "US", "JP", "CN", "EP"]
P = {"P0": (2004, 2006), "P1": (2008, 2010), "P2": (2015, 2017), "P3": (2017, 2019), "P4": (2020, 2022)}

names = (pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
         .drop_duplicates("세세분류코드").set_index("세세분류코드")["산업명"])

# ---------- 1. 출원연도 패널 ----------
a = pd.read_parquet(os.path.join(OUT, "m1_counts_appl.parquet"))
a["ksic"] = a.ksic.astype(str)
a = a[a.appl_year.between(2000, 2024)]
full = pd.MultiIndex.from_product([sorted(a.ksic.unique()), CCS, range(2000, 2025)], names=["ksic", "cc", "year"])
pa = a.rename(columns={"appl_year": "year"}).set_index(["ksic", "cc", "year"])[["n_all", "n_inv", "n_inv_jp8"]].reindex(full, fill_value=0).reset_index()
pa["valid"] = ~(((pa.cc == "CN") & (pa.year < 2004)) | (pa.year > 2022))
pa["flag"] = np.select([(pa.cc == "JP") & pa.year.between(2019, 2022), pa.year >= 2023, (pa.cc == "CN") & (pa.year < 2004)],
                       ["JP 2019–22 수록 감소 의심", "절단", "CN 수록 미미"], "")
tot = pa.groupby(["cc", "year"]).n_inv.transform("sum")
pa["share"] = pa.n_inv / tot
pa.loc[~pa.valid, "share"] = np.nan
pa["share_ew_mean"] = pa.groupby(["ksic", "year"]).share.transform("mean")
pa["rta_ew"] = pa.share / pa.share_ew_mean
w = pa[pa.valid].groupby(["ksic", "year"]).n_inv.sum().rename("n_k5")
W = pa[pa.valid].groupby("year").n_inv.sum().rename("n_5")
pa = pa.merge(w, on=["ksic", "year"], how="left").merge(W, on="year", how="left")
pa["rta_balassa"] = pa.share / (pa.n_k5 / pa.n_5)
pa.loc[~pa.valid, ["rta_ew", "rta_balassa"]] = np.nan
pa.drop(columns=["share_ew_mean", "n_k5", "n_5"]).to_csv(os.path.join(OUT, "m2_panel_appl.csv"), index=False, encoding="utf-8-sig")


def window(df, col, p):
    y0, y1 = P[p]
    return df[df.year.between(y0, y1)].groupby(["ksic", "cc"])[col].sum()


prof = []
for p in P:
    n = window(pa, "n_inv", p).rename("n").reset_index()
    n["share"] = n.n / n.groupby("cc").n.transform("sum")
    n["rta_ew"] = n.share / n.groupby("ksic").share.transform("mean")
    nk = n.groupby("ksic").n.transform("sum"); nt = n.n.sum()
    n["rta_bal"] = n.share / (nk / nt)
    n["period"] = p
    prof.append(n)
prof = pd.concat(prof)
wide = prof.pivot_table(index="ksic", columns=["period", "cc"], values=["n", "share", "rta_ew", "rta_bal"])
out = pd.DataFrame(index=wide.index)
out["name"] = names.reindex(out.index).values
for p in ["P1", "P3", "P4"]:
    for c in CCS:
        out[f"rta_{c}_{p}"] = wide[("rta_ew", p, c)]
for c in CCS:
    out[f"n_{c}_P3"] = wide[("n", "P3", c)]
    out[f"rtabal_{c}_P3"] = wide[("rta_bal", "P3", c)]
    out[f"share_{c}_P3"] = wide[("share", "P3", c)]
for c in CCS:
    out[f"cagr_{c}_P1P3"] = (wide[("n", "P3", c)] / wide[("n", "P1", c)]) ** (1 / 9) - 1
    out[f"cagr_{c}_P2P4"] = (wide[("n", "P4", c)] / wide[("n", "P2", c)]) ** (1 / 5) - 1
out["cagr_others_P1P3"] = out[[f"cagr_{c}_P1P3" for c in CCS if c != "KR"]].mean(1)
out["kr_rel_growth_P1P3"] = out.cagr_KR_P1P3 - out.cagr_others_P1P3
out["kr_rel_growth_P2P4"] = out.cagr_KR_P2P4 - out[[f"cagr_{c}_P2P4" for c in CCS if c != "KR"]].mean(1)
rt3 = out[[f"rta_{c}_P3" for c in CCS]]
out["leader_P3"] = rt3.idxmax(1).str.slice(4, 6)
out["leader_rta_P3"] = rt3.max(1)
out["kr_gap_log_P3"] = np.log(out.rta_KR_P3 / out.leader_rta_P3)
rt4 = out[[f"rta_{c}_P4" for c in CCS]]
out["leader_P4"] = rt4.idxmax(1).str.slice(4, 6)
out["kr_gap_log_P4"] = np.log(out.rta_KR_P4 / rt4.max(1))
out["kr_rta_change_P1P3"] = np.log(out.rta_KR_P3 / out.rta_KR_P1)
out["kr_rank_P3"] = rt3.rank(axis=1, ascending=False)["rta_KR_P3"]

# 한국 기원(우선권 첫 국가=KR) 해외 출원 위상: US·EP·JP·CN 관청에서 KR 기원 비중의 KSIC 상대값
o = pd.read_parquet(os.path.join(OUT, "m1_origin_appl.parquet"))
o["ksic"] = o.ksic.astype(str)
oo = o[(o.office != "KR") & o.appl_year.between(*P["P3"])]
kr_ab = oo[oo.origin == "KR"].groupby("ksic").n_inv.sum()
all_ab = oo.groupby("ksic").n_inv.sum()
out["kr_origin_abroad_n_P3"] = kr_ab.reindex(out.index).fillna(0)
out["kr_origin_abroad_share_P3"] = out.kr_origin_abroad_n_P3 / all_ab.reindex(out.index)
out["kr_origin_abroad_rta_P3"] = out.kr_origin_abroad_share_P3 / (kr_ab.sum() / all_ab.sum())
oo1 = o[(o.office != "KR") & o.appl_year.between(*P["P1"])]
k1 = oo1[oo1.origin == "KR"].groupby("ksic").n_inv.sum(); a1 = oo1.groupby("ksic").n_inv.sum()
out["kr_origin_abroad_rta_P1"] = (k1 / a1).reindex(out.index) / (k1.sum() / a1.sum())
# 국내 출원 대비 해외 진출 강도: KR 기원 해외 출원 / KR 관청 출원
out["kr_abroad_per_domestic_P3"] = out.kr_origin_abroad_n_P3 / out.n_KR_P3

# ---------- 2. WIPO 커버리지 보정 (공개연도) ----------
wp = pd.read_csv(os.path.join(MC, r"_shared\ext\quality_raw\quality_wipo_pi\dc_indicator_patent_4_publication_by_technology.csv"))
wp = wp[wp.office.isin(CCS) & (wp.tec_id > 0)].groupby(["office", "year", "tec_id"])["count"].sum().reset_index()
wp.columns = ["cc", "year", "wipo_field", "wipo_pub"]
kf = pd.read_parquet(os.path.join(OUT, "m1b_ksic_field_pub.parquet"))
kf["ksic"] = kf.ksic.astype(str)
pf = kf.groupby(["cc", "pub_year", "wipo_field"]).n.sum().reset_index().rename(columns={"pub_year": "year", "n": "prov"})
m = wp.merge(pf, on=["cc", "year", "wipo_field"], how="left").fillna({"prov": 0})
m = m[(m.year >= 2003) & ~((m.cc == "CN") & (m.year < 2010))]  # 2000–02 공개는 출원 2000 시작에 따른 좌측 절단 → 보정 대상 아님(계열 시작 2003)
m["r"] = m.prov / m.wipo_pub
tt = m.groupby(["cc", "year"])[["prov", "wipo_pub"]].sum(); tt["r"] = tt.prov / tt.wipo_pub; tt = tt.reset_index()


def ref_median(df, keys):
    res = []
    for _, g in df.groupby(keys):
        g = g.sort_values("year")
        yr = g.year.values; rr = g.r.values
        ref = [np.median(rr[(np.abs(yr - y) <= 3) & (yr != y)]) if ((np.abs(yr - y) <= 3) & (yr != y)).sum() >= 2 else np.nan for y in yr]
        gg = g.copy(); gg["r_ref"] = ref; res.append(gg)
    return pd.concat(res)


tt = ref_median(tt, ["cc"])
tt["f_tot"] = np.where(np.abs(np.log(tt.r_ref / tt.r)) > np.log(1.15), tt.r_ref / tt.r, 1.0)
m = ref_median(m, ["cc", "wipo_field"])
m = m.merge(tt[["cc", "year", "f_tot"]], on=["cc", "year"], how="left")
small = (m.wipo_pub < 300) | (m.prov < 30) | m.r_ref.isna() | (m.r <= 0)
m["f"] = np.where(m.f_tot == 1, 1.0, np.where(small, m.f_tot, m.r_ref / m.r))  # 국가 총계가 이상치인 해에만 분야별 계수 적용
m["f"] = m.f.clip(0.5, 2.5)
tt.to_csv(os.path.join(OUT, "m2_coverage_factors_pub_total.csv"), index=False, encoding="utf-8-sig")
m.to_csv(os.path.join(OUT, "m2_coverage_factors_pub.csv"), index=False, encoding="utf-8-sig")
kf2 = kf.rename(columns={"pub_year": "year"}).merge(m[["cc", "year", "wipo_field", "f"]], on=["cc", "year", "wipo_field"], how="left")
kf2 = kf2.merge(tt[["cc", "year", "f_tot"]], on=["cc", "year"], how="left")
kf2 = kf2[kf2.year >= 2003]
kf2["f"] = kf2.f.fillna(kf2.f_tot).fillna(1.0)  # 2023–25(WIPO 없음)·미매핑은 총계계수, 없으면 1
kf2["n_corr"] = kf2.n * kf2.f
pc = kf2.groupby(["ksic", "cc", "year"])[["n", "n_corr"]].sum().reset_index()
pc["wipo_corrected"] = pc.year <= 2022
pc.to_csv(os.path.join(OUT, "m2_panel_pub_corrected.csv"), index=False, encoding="utf-8-sig")
# 공개연도 보정 계열의 성장(P1→P3, 공개연도), 보정 전후
for c in CCS:
    for col, tag in [("n", "raw"), ("n_corr", "corr")]:
        s = pc[pc.cc == c].pivot_table(index="ksic", columns="year", values=col)
        if c == "CN":
            out[f"pubcagr_{c}_{tag}_1221"] = (s[[2019, 2020, 2021]].mean(1) / s[[2011, 2012, 2014]].mean(1)) ** (1 / 8.33) - 1
        else:
            out[f"pubcagr_{c}_{tag}_1221"] = (s[[2019, 2020, 2021]].mean(1) / s[[2011, 2012, 2013]].mean(1)) ** (1 / 8) - 1
out.to_csv(os.path.join(OUT, "m2_ksic_profile.csv"), encoding="utf-8-sig")

# ---------- 3. JP 2019–22 감소의 KSIC 균일성 (RTA가 이 감소에 둔감한지) ----------
js = pa[pa.cc == "JP"].pivot_table(index="ksic", columns="year", values="share")
rows = []
for y in range(2005, 2023):
    d = np.log(js[y] / js[y - 1]).replace([np.inf, -np.inf], np.nan)
    rows.append({"year": y, "sd_dlog_share": d.std(), "jp_total": int(pa[(pa.cc == "JP") & (pa.year == y)].n_inv.sum())})
ju = pd.DataFrame(rows)
ju.to_csv(os.path.join(OUT, "m2_jp_uniformity.csv"), index=False, encoding="utf-8-sig")

# ---------- 4. 국가 요약 ----------
cs = []
for c in CCS:
    cs.append({"cc": c, "n_P3": int(out[f"n_{c}_P3"].sum()),
               "n_ksic_rta_gt1_P3": int((out[f"rta_{c}_P3"] > 1).sum()),
               "n_ksic_leader_P3": int((out.leader_P3 == c).sum()),
               "n_ksic_leader_P4": int((out.leader_P4 == c).sum()),
               "median_cagr_P1P3": round(out[f"cagr_{c}_P1P3"].median(), 4),
               "median_cagr_P2P4": round(out[f"cagr_{c}_P2P4"].median(), 4)})
cs = pd.DataFrame(cs)
cs.to_csv(os.path.join(OUT, "m2_country_summary.csv"), index=False, encoding="utf-8-sig")

from scipy.stats import spearmanr
chk = {
    "coverage_flagged_total": tt[tt.f_tot != 1][["cc", "year", "r", "r_ref", "f_tot"]].round(3).values.tolist(),
    "coverage_flagged_cells_n": int((m.f != 1).sum()), "coverage_cells_n": int(len(m)),
    "coverage_flagged_cells_by_ccyear": m[m.f != 1].groupby(["cc", "year"]).size().loc[lambda s: s >= 5].to_dict().__repr__(),
    "rta_ew_vs_balassa_KR_P3_spearman": round(spearmanr(out.rta_KR_P3, out.rtabal_KR_P3).statistic, 3),
    "rta_KR_P3_vs_P4_spearman": round(spearmanr(out.rta_KR_P3, out.rta_KR_P4).statistic, 3),
    "rta_JP_P3_vs_P4_spearman": round(spearmanr(out.rta_JP_P3, out.rta_JP_P4).statistic, 3),
    "kr_rta_gt1_P3": int((out.rta_KR_P3 > 1).sum()),
    "kr_leader_P3": int((out.leader_P3 == "KR").sum()),
    "kr_rta_vs_origin_abroad_rta_spearman": round(spearmanr(out.rta_KR_P3, out.kr_origin_abroad_rta_P3).statistic, 3),
    "pub_growth_KR_raw_vs_corr_spearman": round(spearmanr(out.pubcagr_KR_raw_1221, out.pubcagr_KR_corr_1221, nan_policy="omit").statistic, 3),
    "pub_growth_JP_raw_vs_corr_spearman": round(spearmanr(out.pubcagr_JP_raw_1221, out.pubcagr_JP_corr_1221, nan_policy="omit").statistic, 3),
    "jp_sd_dlog_share_2019_2022": ju[ju.year.between(2019, 2022)].sd_dlog_share.round(4).tolist(),
    "jp_sd_dlog_share_2005_2018_median": round(ju[ju.year.between(2005, 2018)].sd_dlog_share.median(), 4),
    "jp_sd_dlog_share_2005_2018_max": round(ju[ju.year.between(2005, 2018)].sd_dlog_share.max(), 4),
}
json.dump(chk, open(os.path.join(OUT, "m2_checks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(chk, ensure_ascii=False, indent=1))
print(cs.to_string())
