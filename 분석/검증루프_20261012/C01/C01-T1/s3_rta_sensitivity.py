# -*- coding: utf-8 -*-
r"""C01-T1 s3: (c) 영향. PREREG.txt 규칙 그대로.
(0) 원 P3 RTA 재현(fig1_rta_map_P3.csv 대조)  (i) WIPO35 분야 RTA 제공판 vs WIPO판  (ii) KSIC50 RTA 방식 A·B 보정
(iii) CN 제외 4개국  (iv) P4  산출: rta_sensitivity.csv, rta_ksic_P3_variants.csv, rta_wipo35_P3.csv, s3_results.json
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(OUT, "cache")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
M5 = os.path.join(MC, r"_shared_report\fusion_advantage\map5")
CCS = ["KR", "US", "JP", "CN", "EP"]
P = {"P3": (2017, 2019), "P4": (2020, 2022)}
ag = pd.read_parquet(os.path.join(C, "agg_ksic.parquet")); ag = ag[ag.is_inv].copy(); ag["ksic"] = ag.ksic.astype(str)
m = pd.read_parquet(os.path.join(C, "cov_r.parquet"))  # cc, year, f, wipo, prov, r, small

# ---------- 가중치 표 ----------
tot = m.groupby(["cc", "year"])[["prov", "wipo"]].sum(); tot["r_tot"] = tot.prov / tot.wipo; tot = tot.reset_index()
gt = tot[tot.year >= 2010].groupby("year").r_tot.apply(lambda s: np.exp(np.log(s).mean())).rename("g_tot")
tot = tot.merge(gt, on="year", how="left"); tot["q_tot"] = tot.r_tot / tot.g_tot
m = m.merge(tot[["cc", "year", "r_tot", "q_tot"]], on=["cc", "year"])
m["rA"] = np.where(m.small | (m.r <= 0), m.r_tot, m.r)
g = m.groupby(["year", "f"]).rA.apply(lambda s: np.exp(np.log(s.clip(lower=1e-6)).mean())).rename("g")
m = m.merge(g, on=["year", "f"]); m["qB"] = np.where(m.small | (m.r <= 0), m.q_tot, m.r / m.g)
clipA = int(((m.rA < 0.02) | (m.rA > 1.5)).sum()); clipB = int(((m.qB < 0.25) | (m.qB > 4)).sum())
clipB_list = m[((m.qB < 0.25) | (m.qB > 4)) & (m.year.between(2017, 2024))][["cc", "year", "f", "qB"]].round(2).values.tolist()
m["wA"] = 1 / m.rA.clip(0.02, 1.5); m["wB"] = 1 / m.qB.clip(0.25, 4)
W = m[["cc", "year", "f", "wA", "wB"]]
T = tot[["cc", "year", "r_tot", "q_tot"]].copy(); T["wA_tot"] = 1 / T.r_tot.clip(0.02, 1.5); T["wB_tot"] = 1 / T.q_tot.clip(0.25, 4)


def weights(df):
    d = df.copy()
    d["wy"] = d.pub_year.clip(upper=2022)
    miss = d.pub_year.isna() | (d.pub_year < 2003)
    d.loc[miss, "wy"] = (d.loc[miss, "appl_year"] + 1).clip(2003, 2022)
    d["wy"] = d.wy.astype(int)
    d = d.merge(W.rename(columns={"year": "wy", "f": "wipo_field"}), on=["cc", "wy", "wipo_field"], how="left")
    d = d.merge(T[["cc", "year", "wA_tot", "wB_tot"]].rename(columns={"year": "wy"}), on=["cc", "wy"], how="left")
    miss = d.pub_year.isna() | (d.pub_year < 2003)  # merge 후 인덱스가 바뀌므로 다시 계산(1차 실행의 정렬 오류 수정)
    use_tot = miss | (d.wipo_field == 0) | d.wA.isna()
    d.loc[use_tot, "wA"] = d.loc[use_tot, "wA_tot"]; d.loc[use_tot, "wB"] = d.loc[use_tot, "wB_tot"]
    d["nA"] = d.n * d.wA; d["nB"] = d.n * d.wB
    return d, float((d.n * use_tot).sum() / d.n.sum())


def rta(n, ccs):
    """n: Series index (ksic, cc). 동일가중 RTA."""
    x = n.unstack("cc")[ccs].fillna(0)
    s = x / x.sum(0)
    return s.div(s.mean(1), axis=0)


def summarize(R, ref=None):
    lead = R.idxmax(1)
    out = {"leaders": {c: int((lead == c).sum()) for c in R.columns}, "kr_rta_gt1": int((R["KR"] > 1).sum()) if "KR" in R else None,
           "kr_leader_list": sorted(lead[lead == "KR"].index.tolist())}
    return out, lead


res = {"clip_cells_A": clipA, "clip_cells_B": clipB, "clipB_cells_2017_2024": clipB_list}
rows = []; variants = {}
for p, (y0, y1) in P.items():
    d = ag[ag.appl_year.between(y0, y1)]
    d, tot_share = weights(d)
    res[f"{p}_share_n_total_fallback"] = round(tot_share, 4)
    for tag, col in [("raw", "n"), ("A", "nA"), ("B", "nB")]:
        n = d.groupby(["ksic", "cc"])[col].sum()
        for cset, ccs in [("5cc", CCS), ("4cc_noCN", [c for c in CCS if c != "CN"])]:
            R = rta(n, ccs); variants[(p, tag, cset)] = R
# 0) 재현
f1 = pd.read_csv(os.path.join(M5, "fig1_rta_map_P3.csv"), dtype={"ksic": str}).set_index("ksic")
R0 = variants[("P3", "raw", "5cc")].reindex(f1.index)
res["repro_maxabs_vs_fig1"] = float(np.abs(R0[CCS].values - f1[CCS].values).max())
prof = pd.read_csv(os.path.join(M5, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")
R4 = variants[("P4", "raw", "5cc")].reindex(prof.index)
res["repro_maxabs_P4_vs_profile"] = float(np.abs(R4[CCS].values - prof[[f"rta_{c}_P4" for c in CCS]].values).max())
# 비교
for p in P:
    for cset in ["5cc", "4cc_noCN"]:
        Rr = variants[(p, "raw", cset)]; sr, lr = summarize(Rr)
        for tag in ["raw", "A", "B"]:
            Rv = variants[(p, tag, cset)]; sv, lv = summarize(Rv)
            row = dict(period=p, countries=cset, method=tag)
            for c in Rv.columns:
                row[f"lead_{c}"] = sv["leaders"][c]; row[f"dlead_{c}"] = sv["leaders"][c] - sr["leaders"][c]
                row[f"rho_{c}_vs_raw"] = round(spearmanr(Rr[c], Rv[c]).statistic, 4)
            row["kr_rta_gt1"] = sv["kr_rta_gt1"]; row["d_kr_rta_gt1"] = sv["kr_rta_gt1"] - sr["kr_rta_gt1"]
            row["leader_changed_n"] = int((lv != lr).sum())
            row["kr_leaders"] = "|".join(sv["kr_leader_list"])
            row["kr_leaders_kept_of_raw"] = len(set(sv["kr_leader_list"]) & set(sr["kr_leader_list"]))
            rows.append(row)
sens = pd.DataFrame(rows)
# 4개국 RTA의 1위를 5개국 원판과 비교
l5 = variants[("P3", "raw", "5cc")].idxmax(1); l4 = variants[("P3", "raw", "4cc_noCN")].idxmax(1)
res["P3_4cc_vs_5cc_leader_changed_excluding_CNled"] = int(((l5 != l4) & (l5 != "CN")).sum())
res["P3_5cc_CN_led_ksic_new_leader_in_4cc"] = l4[l5 == "CN"].value_counts().to_dict()
sens.to_csv(os.path.join(OUT, "rta_sensitivity.csv"), index=False, encoding="utf-8-sig")
# 산업별 상세(P3, 5개국)
names = prof["name"]
det = pd.DataFrame({"name": names})
for tag in ["raw", "A", "B"]:
    R = variants[("P3", tag, "5cc")].reindex(det.index)
    for c in CCS:
        det[f"rta_{c}_{tag}"] = R[c].round(4)
    det[f"leader_{tag}"] = R[CCS].idxmax(1)
R4c = variants[("P3", "raw", "4cc_noCN")].reindex(det.index); det["leader_raw_4cc"] = R4c.idxmax(1)
det.to_csv(os.path.join(OUT, "rta_ksic_P3_variants.csv"), encoding="utf-8-sig")
# KR 1위 7개 값
for tag in ["raw", "A", "B"]:
    res[f"kr_top_values_{tag}"] = {k: round(float(det.at[k, f"rta_KR_{tag}"]), 3) for k in ["20423", "26212", "31312", "24191", "26121", "25922", "28511"]}

# ---------- (i) WIPO35 분야 RTA (공개 2017–19) ----------
p3 = m[m.year.between(2017, 2019)].groupby(["f", "cc"])[["prov", "wipo"]].sum()
Rp = rta(p3.prov, CCS); Rw = rta(p3.wipo, CCS)
w35 = {c: round(spearmanr(Rp[c], Rw[c]).statistic, 4) for c in CCS}
res["wipo35_rta_spearman_prov_vs_wipo"] = w35
med_others = {c: float(np.median([w35[o] for o in CCS if o != c])) for c in CCS}
res["wipo35_flag_country_lt_median_minus_0.15"] = [c for c in CCS if w35[c] < med_others[c] - 0.15]
res["wipo35_leaders_prov"] = Rp.idxmax(1).value_counts().to_dict(); res["wipo35_leaders_wipo"] = Rw.idxmax(1).value_counts().to_dict()
res["wipo35_kr_rta_gt1_prov_wipo"] = [int((Rp.KR > 1).sum()), int((Rw.KR > 1).sum())]
res["wipo35_leader_agree"] = int((Rp.idxmax(1) == Rw.idxmax(1)).sum())
# 16번(의약) 제외
Rp2 = rta(p3.prov.drop(16, level="f"), CCS); Rw2 = rta(p3.wipo.drop(16, level="f"), CCS)
res["wipo35_rta_spearman_excl_field16"] = {c: round(spearmanr(Rp2[c], Rw2[c]).statistic, 4) for c in CCS}
pd.concat([Rp.add_prefix("prov_"), Rw.add_prefix("wipo_")], axis=1).round(4).to_csv(os.path.join(OUT, "rta_wipo35_P3.csv"), encoding="utf-8-sig")

# ---------- 판정 ----------
s = sens[(sens.period == "P3") & (sens.countries == "5cc")].set_index("method")
verdict = {}
for tag in ["A", "B"]:
    r = s.loc[tag]
    c1 = all(abs(r[f"dlead_{c}"]) <= 2 for c in CCS); c2 = abs(r.d_kr_rta_gt1) <= 3; c3 = r.rho_KR_vs_raw >= 0.90
    verdict[tag] = dict(c1_leaders=bool(c1), c2_kr_gt1=bool(c2), c3_kr_rho=bool(c3), robust=bool(c1 and c2 and c3))
res["verdict"] = verdict
json.dump(res, open(os.path.join(OUT, "s3_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(sens.to_string())
