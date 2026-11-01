# -*- coding: utf-8 -*-
"""C04-T1 s5: audit_tables.csv(긴 형식) 묶기 — s1·s2·s4 산출물만 읽음."""
import os, json, pandas as pd
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T1"; C = os.path.join(H, "cache")
rows = []
p1 = pd.read_csv(os.path.join(C, "p1_trend.csv"), encoding="utf-8-sig")
for _, r in p1.iterrows():
    for k in ["r3nts_corp_close", "tier1_corp_close", "official_corp_close", "ratio_all", "ratio_tier1", "official_close_rate"]:
        rows.append(dict(table="P1_trend_by_year", key=int(r.y), metric=k, value=r[k]))
for _, r in pd.read_csv(os.path.join(C, "s4_hazard.csv"), encoding="utf-8-sig").iterrows():
    for k in ["n", "pos", "rate", "off_2y_rate", "ratio_n4_over_off"]:
        rows.append(dict(table="P1b_hazard_by_t0", key=int(r.t0), metric=k, value=r[k]))
m = pd.read_csv(os.path.join(C, "nts_close_month_corp.csv"), index_col=0, encoding="utf-8-sig")
for y in m.index:
    for mo in m.columns:
        rows.append(dict(table="P3_corp_close_month", key=f"{y}-{int(mo):02d}", metric="n", value=m.loc[y, mo]))
s1 = json.load(open(os.path.join(C, "s1_profile.json"), encoding="utf-8"))
for y, v in s1["P3_h2h1_corp"].items():
    rows.append(dict(table="P3_h2h1_corp", key=y, metric="h2_over_h1", value=v))
for _, r in pd.read_csv(os.path.join(C, "s2_smd.csv"), encoding="utf-8-sig").iterrows():
    for k in ["n_a", "n_b", "mean_a", "mean_b", "smd"]:
        rows.append(dict(table="P2_smd:" + r["compare"], key=r["var"], metric=k, value=r[k]))
for _, r in pd.read_csv(os.path.join(C, "s2_dist_testpop.csv"), encoding="utf-8-sig", dtype={"level": str}).iterrows():
    for k in ["size", "sum", "mean"]:
        rows.append(dict(table="b_dist_testpop:" + r.dim, key=r.level, metric={"size": "n", "sum": "pos", "mean": "rate"}[k], value=r[k]))
for grp in ["anom", "dup", "b_stt_cd", "kind_counts"]:
    for k, v in s1[grp].items():
        rows.append(dict(table="b_nts_" + grp, key=k, metric="n", value=v))
s2 = json.load(open(os.path.join(C, "s2_population.json"), encoding="utf-8"))
for k in ["pool_n", "pred_n", "pred_pos", "pred_in_pool", "registered_open_not_in_pred", "pred_not_registered_open", "proxy_auc_in_registered",
          "proxy_mean_registered_live", "obs_rate_registered_live", "proxy_mean_nobizno_live", "n_nobizno_live", "n_nobizno_nolive",
          "pos_merge_pm1y", "pos_assign_pm1y", "pos_pay_after_close30_no_owner_change", "pos_n", "pos_opp_n", "pos_opp_pay_after"]:
    rows.append(dict(table="P2P3_population", key=k, metric="value", value=s2[k]))
for k, v in s2["pool_grp_excl_closed_by_t0"].items():
    rows.append(dict(table="P2_pool_t0_2023_open", key=k, metric="n", value=v))
for k, v in s2["test_rate_by_rl_query"].items():
    rows.append(dict(table="P2_test_rate_by_rightlist_query", key=k, metric="n", value=v["size"]))
    rows.append(dict(table="P2_test_rate_by_rightlist_query", key=k, metric="rate", value=v["mean"]))
for k, v in s2["test_hyu"].items():
    rows.append(dict(table="P4_hyu_testpop", key=k, metric="n", value=v))
pd.DataFrame(rows).to_csv(os.path.join(H, "audit_tables.csv"), index=False, encoding="utf-8-sig")
print(len(rows))
