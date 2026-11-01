# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5\m7_supplement.py — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)
# -*- coding: utf-8 -*-
r"""map5 7단계(보완 점검, 재시도 세션): 입력은 m2 산출물만, 쓰기는 map5 폴더에만.
(a) 한국 상대 성장 재정의: m2의 kr_rel_growth_P1P3는 '해외 4개국 CAGR 평균' 대비라 CN(중앙값 +17.8%/년)이 지배한다.
    → CN 제외 3개국(US·JP·EP) 평균 대비, 그리고 '국가 내 비중 변화'(log share P3/P1) 기준을 보조로 낸다.
(b) 공개연도 보조 계열(WIPO 보정): 2017–19 → 2022–24 공개 CAGR, 보정 전·후. 2020년 이후를 포함하므로 '사후 분석' 표기.
    2023–24는 WIPO 값이 없어 보정 계수 = 해당 국가 총계 계수(없으면 1) → 사실상 미보정.
(c) JP 2021 보정 크기: 공개 2021 JP 총량 보정 전·후, 그리고 KSIC 순위 보존(Spearman).
"""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = _C19_OUT
CCS = ["KR", "US", "JP", "CN", "EP"]
prof = pd.read_csv(os.path.join(OUT, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")
pa = pd.read_csv(os.path.join(OUT, "m2_panel_appl.csv"), dtype={"ksic": str})
pc = pd.read_csv(os.path.join(OUT, "m2_panel_pub_corrected.csv"), dtype={"ksic": str})
res = {}

# (a)
T = pd.DataFrame(index=prof.index); T["name"] = prof.name
T["kr_rel_growth_vs_USJPEP_P1P3"] = prof.cagr_KR_P1P3 - prof[["cagr_US_P1P3", "cagr_JP_P1P3", "cagr_EP_P1P3"]].mean(1)
sh = lambda y0, y1: (lambda n: n / n.sum())(pa[pa.year.between(y0, y1) & pa.valid].groupby(["ksic", "cc"]).n_inv.sum().unstack())
s1, s3 = sh(2008, 2010), sh(2017, 2019)
T["kr_dlog_share_P1P3"] = np.log(s3.KR / s1.KR)
T["kr_dlog_share_minus_foreign_mean"] = T.kr_dlog_share_P1P3 - np.log(s3[["US", "JP", "CN", "EP"]] / s1[["US", "JP", "CN", "EP"]]).mean(1)
res["kr_rel_growth_vs_4_pos_n"] = int((prof.kr_rel_growth_P1P3 > 0).sum())
res["kr_rel_growth_vs_USJPEP_pos_n"] = int((T.kr_rel_growth_vs_USJPEP_P1P3 > 0).sum())
res["kr_rel_growth_vs_USJPEP_median"] = round(T.kr_rel_growth_vs_USJPEP_P1P3.median(), 4)
res["spearman_relgrowth_vs4_vs_USJPEP"] = round(spearmanr(prof.kr_rel_growth_P1P3, T.kr_rel_growth_vs_USJPEP_P1P3).statistic, 3)
res["spearman_relgrowth_vs4_vs_dlogshare_rel"] = round(spearmanr(prof.kr_rel_growth_P1P3, T.kr_dlog_share_minus_foreign_mean).statistic, 3)
res["kr_dlogshare_rel_pos_n"] = int((T.kr_dlog_share_minus_foreign_mean > 0).sum())
nm = lambda idx: [f"{k} {T.at[k, 'name']}" for k in idx]
res["kr_share_gain_rel_top5"] = nm(T.kr_dlog_share_minus_foreign_mean.nlargest(5).index)
res["kr_share_loss_rel_top5"] = nm(T.kr_dlog_share_minus_foreign_mean.nsmallest(5).index)
# 국가 총량 성장(50 KSIC 합) 출원 P1→P3
tot = pa[pa.valid].groupby(["cc", "year"]).n_inv.sum().unstack(0)
res["total_cagr_P1P3"] = {c: round((tot.loc[2017:2019, c].mean() / tot.loc[2008:2010, c].mean()) ** (1 / 9) - 1, 4) for c in CCS}

# (b) 공개연도 보조(사후)
pv = {}
for col in ["n", "n_corr"]:
    for c in CCS:
        s = pc[pc.cc == c].pivot_table(index="ksic", columns="year", values=col)
        pv[(c, col)] = (s[[2022, 2023, 2024]].mean(1) / s[[2017, 2018, 2019]].mean(1)) ** (1 / 5) - 1
for c in CCS:
    T[f"pubcagr_{c}_raw_1724_posthoc"] = pv[(c, "n")]; T[f"pubcagr_{c}_corr_1724_posthoc"] = pv[(c, "n_corr")]
res["posthoc_pub_cagr_1719_2224_median"] = {c: {"raw": round(pv[(c, "n")].median(), 4), "corr": round(pv[(c, "n_corr")].median(), 4)} for c in CCS}
T["kr_pub_rel_vs_USJPEP_posthoc"] = pv[("KR", "n_corr")] - pd.concat([pv[(c, "n_corr")] for c in ["US", "JP", "EP"]], axis=1).mean(1)
# 개발 상대 성장과 사후 공개 상대 성장의 순위 지속성
res["posthoc_persistence_spearman_relgrowth_dev_vs_pub"] = round(spearmanr(T.kr_rel_growth_vs_USJPEP_P1P3, T.kr_pub_rel_vs_USJPEP_posthoc).statistic, 3)

# (c) JP 2021
j = pc[(pc.cc == "JP") & (pc.year.between(2019, 2023))].groupby("year")[["n", "n_corr"]].sum()
res["jp_pub_total_raw_corr_2019_2023"] = {int(y): [int(r.n), int(round(r.n_corr))] for y, r in j.iterrows()}
jj = pc[(pc.cc == "JP") & (pc.year == 2021)]
res["jp_2021_ksic_rank_raw_vs_corr_spearman"] = round(spearmanr(jj.n, jj.n_corr).statistic, 4)
T.to_csv(os.path.join(OUT, "m7_kr_growth_supplement.csv"), encoding="utf-8-sig")
json.dump(res, open(os.path.join(OUT, "m7_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
