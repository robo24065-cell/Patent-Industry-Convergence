# -*- coding: utf-8 -*-
# C16-T2 verify_0 v1: N4 DART 목록 'B 의심' 발견 반박 시도. 읽기 전용. 재채점 없음.
import os, json, sys, sqlite3
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
N4 = os.path.join(BASE, r"분석\모델경쟁_20260927\N4_특허행태_기업폐업조기경보")
OUT = os.path.join(BASE, r"분석\검증루프_20261012\C16\C16-T2\verify_0")
R = {}
digits = lambda s: s.fillna("").astype(str).str.replace(r"\D", "", regex=True)

ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"))
R["ns_cols"] = list(ns.columns)
ns["bz"] = digits(ns.bizno)
reg = ns[ns.registered.fillna(False).astype(bool)].copy()
R["reg_rows"] = len(reg); R["reg_bz_empty"] = int((reg.bz.str.len() != 10).sum())
# 같은 bz 다중 행의 end_dt 불일치
g = reg.groupby("bz").end_dt.agg(lambda s: s.nunique(dropna=False))
R["reg_bz_multi_end_conflict"] = int((g > 1).sum())
# bz 단위: 폐업일은 최소값(있으면), tier1 = 어느 행이든 n_prov_rgst>0
U = reg[reg.bz.str.len() == 10].groupby("bz").agg(end=("end_dt", "min"), nprov=("n_prov_rgst", "max")).reset_index()
R["U_n"] = len(U)
in_win = lambda e: (e >= "2016-01-01") & (e < "2026-01-01")
alive15 = lambda e: e.isna() | (e >= "2016-01-01")
R["U_closed_any"] = float(U.end.notna().mean())
R["U_closed_1625"] = float(in_win(U.end).mean())
R["U_closed_1625_alive15"] = float(in_win(U.end)[alive15(U.end)].mean())
T1 = U[U.nprov.fillna(0) > 0]
R["T1_n"] = len(T1); R["T1_closed_1625"] = float(in_win(T1.end).mean())
R["T1_closed_1625_alive15"] = float(in_win(T1.end)[alive15(T1.end)].mean())

# DART 재무 패널
dp = pd.read_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"), columns=["corp_code", "fy", "bizr_no", "corp_cls"])
dp["bz"] = digits(dp.bizr_no)
D = dp.groupby("corp_code").agg(bz=("bz", "first"), cls=("corp_cls", "first"), fymax=("fy", "max")).reset_index()
R["D_firms"] = len(D); R["D_bz_nunique"] = int(D.bz.nunique()); R["D_cls"] = D.cls.value_counts().to_dict()
M = D.merge(U, on="bz", how="inner")
R["D_matched"] = len(M)
R["D_closed_any"] = int(M.end.notna().sum())
R["D_closed_lt2026"] = int((M.end < "2026-01-01").sum())
R["D_closed_1625"] = int(in_win(M.end).sum()); R["D_closed_1625_share"] = float(in_win(M.end).mean())
R["D_closed_1625_alive15"] = float(in_win(M.end)[alive15(M.end)].mean())
R["ratio_rule_any"] = float(M.end.notna().mean() / U.end.notna().mean())
R["ratio_1625_univ"] = R["D_closed_1625_share"] / R["U_closed_1625"]
R["ratio_1625_tier1"] = R["D_closed_1625_share"] / R["T1_closed_1625"]
R["ratio_1625_alive15_tier1"] = R["D_closed_1625_alive15"] / R["T1_closed_1625_alive15"]
R["D_closed_by_cls"] = M.assign(c=in_win(M.end)).groupby("cls").c.agg(["size", "sum"]).to_dict(orient="index")

# 상위 목록(DART 대상 12,322사, 기업개황) — 목록 자체가 폐업 법인을 빼는가
dc = sqlite3.connect("file:" + os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite") + "?mode=ro", uri=True)
C = pd.read_sql("select corp_code, bizr_no, corp_cls from company", dc); C["bz"] = digits(C.bizr_no)
C = C[C.bz.str.len() == 10].drop_duplicates("bz")
MC = C.merge(U, on="bz", how="inner")
R["C_n"] = len(C); R["C_matched"] = len(MC)
R["C_closed_1625_by_cls"] = MC.assign(c=in_win(MC.end)).groupby("corp_cls").c.agg(["size", "sum", "mean"]).to_dict(orient="index")
R["C_closed_1625_E_tier1"] = float(in_win(MC[(MC.corp_cls == "E") & (MC.nprov.fillna(0) > 0)].end).mean())
R["C_closed_any_lt2026"] = int((MC.end < "2026-01-01").sum())

# N4 시험 패널
fp = pd.read_parquet(os.path.join(N4, "cache", "final_panel.parquet"), columns=["applicantCd", "bizno", "t0", "dart_has", "log_emp", "nps_present"])
pr = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"), columns=["applicantCd", "bizno", "y"])
R["pred_n"] = len(pr); R["pred_dup"] = int(pr.duplicated(["applicantCd", "bizno"]).sum())
te = fp[fp.t0 == 2023].merge(pr, on=["applicantCd", "bizno"], how="inner")
R["te_n"] = len(te); R["te_pos"] = int(te.y.sum())
R["te_dart_has_n"] = int((te.dart_has == 1).sum())
for k in (0, 1):
    s = te[te.dart_has.fillna(0) == k]; R[f"te_rate_dh{k}"] = [len(s), int(s.y.sum()), float(s.y.mean())]
# 규모 표준화(독립 구현: 경계 정수 고용 기준)
emp = np.expm1(te.log_emp)
b = np.select([te.nps_present != 1, emp < 9.5, emp < 49.5, emp < 299.5], ["NPS없음", "<10", "10-49", "50-299"], "300+")
te["b"] = b; dh = te.dart_has.fillna(0)
w = te[dh == 1].b.value_counts(normalize=True)
r0 = te[dh == 0].groupby("b").y.mean()
R["std_ratio"] = float(te[dh == 1].y.mean() / (w * r0.reindex(w.index)).sum())
# 역방향 표준화(dart_has=0 분포 가중) — 0 사건 층이 많아 불안정성 확인
w0 = te[dh == 0].b.value_counts(normalize=True); r1 = te[dh == 1].groupby("b").y.mean()
R["std_ratio_rev"] = float((w0 * r1.reindex(w0.index).fillna(np.nan)).sum() / te[dh == 0].y.mean())
R["dh1_bins"] = te[dh == 1].groupby("b").y.agg(["size", "sum"]).to_dict(orient="index")
json.dump(R, open(os.path.join(OUT, "v1_refute_q1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
