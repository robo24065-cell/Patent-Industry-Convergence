# -*- coding: utf-8 -*-
"""C16-T2 Q1: N4 DART 재무 대상 목록(현재 DART 회사 목록 기반)의 생존 선택 여부와 영향 표본 비율.
읽기 전용 입력만 사용. 재채점 없음. 출력: C16-T2\cache\q1_out.json
PREREG §4 Q1 판정 규칙을 그대로 적용한다."""
import os, json, sys
import pandas as pd, numpy as np

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
EXT = os.path.join(BASE, r"모델경쟁_20260927\_shared\ext")
N4 = os.path.join(BASE, r"모델경쟁_20260927\N4_특허행태_기업폐업조기경보")
OUT = os.path.join(BASE, r"검증루프_20261012\C16\C16-T2\cache")

dart = pd.read_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"), columns=["corp_code", "fy", "bizr_no", "avail_date", "corp_cls", "est_dt"])
dart["bz"] = dart.bizr_no.fillna("").astype(str).str.replace(r"\D", "", regex=True)
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "end_dt", "registered", "query_date", "src", "n_prov_rgst"])
ns["bz"] = ns.bizno.fillna("").astype(str).str.replace(r"\D", "", regex=True)
reg = ns[ns.registered.fillna(False).astype(bool)].drop_duplicates("bz")
res = {}
# 데이터 점검
res["dart_rows"] = int(len(dart)); res["dart_firms"] = int(dart.corp_code.nunique())
res["dart_bz10_firms"] = int(dart.loc[dart.bz.str.len() == 10, "corp_code"].nunique())
res["dart_fy_range"] = [int(dart.fy.min()), int(dart.fy.max())]
res["dart_dup_corp_fy"] = int(dart.duplicated(["corp_code", "fy"]).sum())
res["nts_rows"] = int(len(ns)); res["nts_registered_bz"] = int(len(reg))
res["nts_query_date"] = sorted(map(str, ns.query_date.dropna().astype(str).unique()))[:3]
# (a) DART 재무 패널 기업 중 국세청 폐업일 보유
df = dart[dart.bz.str.len() == 10].groupby("bz").agg(fy_min=("fy", "min"), fy_max=("fy", "max"), cls=("corp_cls", "first")).reset_index()
m = df.merge(reg[["bz", "end_dt", "n_prov_rgst"]], on="bz", how="left", indicator=True)
res["a_dart_bz_firms"] = int(len(df))
res["a_matched_to_nts_registered"] = int((m._merge == "both").sum())
mm = m[m._merge == "both"].copy()
mm["closed"] = mm.end_dt.notna()
mm["end_y"] = mm.end_dt.dt.year
res["a_closed_any"] = int(mm.closed.sum())
res["a_closed_share"] = float(mm.closed.mean()) if len(mm) else None
res["a_closed_by_year"] = {str(int(k)): int(v) for k, v in mm.loc[mm.closed, "end_y"].value_counts().sort_index().items()}
res["a_closed_lt2026"] = int((mm.end_dt < pd.Timestamp("2026-01-01")).sum())
# 폐업 후 DART 재무가 끊겼는지(폐업 연도 대비 마지막 fy)
cl = mm[mm.closed].copy()
res["a_closed_lastfy_minus_endy_median"] = float((cl.fy_max - cl.end_y).median()) if len(cl) else None
# (b) 국세청 대상 모집단 전체 폐업 비율(같은 registered 기준), 제공범위 등록 보유 tier1만도
res["b_univ_closed_share"] = float(reg.end_dt.notna().mean())
t1 = reg[reg.n_prov_rgst.fillna(0) > 0]
res["b_tier1_n"] = int(len(t1)); res["b_tier1_closed_share"] = float(t1.end_dt.notna().mean())
res["b_univ_closed_2016_2025_share"] = float(((reg.end_dt >= "2016-01-01") & (reg.end_dt < "2026-01-01")).mean())
res["a_closed_2016_2025_share"] = float(((mm.end_dt >= "2016-01-01") & (mm.end_dt < "2026-01-01")).mean()) if len(mm) else None
ratio = (res["a_closed_2016_2025_share"] / res["b_univ_closed_2016_2025_share"]) if res["b_univ_closed_2016_2025_share"] else None
res["ratio_dart_vs_univ_2016_2025"] = ratio
# (c) N4 시험 패널(t0=2023)의 dart_has
fp = pd.read_parquet(os.path.join(N4, "cache", "final_panel.parquet"), columns=["applicantCd", "bizno", "t0", "dart_has", "y"])
pr = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"), columns=["applicantCd", "bizno", "y"])
res["c_final_panel_t0"] = {str(k): int(v) for k, v in fp.t0.value_counts().sort_index().items()}
te = fp[fp.t0 == 2023].drop(columns=["y"]).merge(pr, on=["applicantCd", "bizno"], how="inner")
res["c_test_n"] = int(len(te)); res["c_test_pos"] = int(te.y.sum())
res["c_dart_has_rate"] = float((te.dart_has == 1).mean())
g = te.groupby(te.dart_has.fillna(0)).y.agg(["size", "sum", "mean"])
res["c_by_dart_has"] = {str(k): {"n": int(r["size"]), "pos": int(r["sum"]), "rate": float(r["mean"])} for k, r in g.iterrows()}
# 시험 패널 안에서 DART 패널 기업이 2024–25 폐업한 수(=dart_has 층 양성)
# 판정
if res["a_closed_lt2026"] >= 20 and ratio is not None and ratio >= 0.25:
    res["verdict"] = "현재 목록 생존 선택 근거 약함 → A(준-정합 C 유지)"
else:
    res["verdict"] = "B 의심"
res["affected_share_test"] = res["c_dart_has_rate"]
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "q1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
