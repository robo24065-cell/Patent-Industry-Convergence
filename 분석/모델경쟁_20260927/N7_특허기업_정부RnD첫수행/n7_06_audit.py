# -*- coding: utf-8 -*-
r"""🔒 최종시험 전 감사(레이블·피처 시점·기준선·CI 군집). 최종시험 결과변수(과제연도 ≥2023)는 읽지 않는다.
산출: audit_r5.json"""
import os, json, inspect
import numpy as np, pandas as pd
import n7lib, n7feat
from n7lib import *

A = {}
SPEC = json.load(open(os.path.join(OUT, "final_spec_r5.json"), encoding="utf-8"))
# (a) 레이블 결과 구간 < F
A["a_label"] = dict(
    final_train_t0=[t for t in origins_for("final") if label_allowed(t, "final") and t != FINAL_T0],
    final_train_label_max_year=max(t + 2 for t in origins_for("final") if t != FINAL_T0),
    F=F_YEAR, t2021_in_final=2021 in origins_for("final"),
    ok=max(t + 2 for t in origins_for("final") if t != FINAL_T0) < F_YEAR and 2021 not in origins_for("final"))
# (b) 피처 시점
ky = pd.read_parquet(os.path.join(EXT, "ntis_project_ksic_year.parquet"))
lag = (ky.avail_year - ky.year).value_counts().to_dict()
A["b_ntis_ksic_avail_lag"] = {str(k): int(v) for k, v in lag.items()}
A["b_nps_snaps_final"] = snaps_for("final"); A["b_nps_snaps_future"] = snaps_for("future")
src = inspect.getsource(n7lib) + inspect.getsource(n7feat)
A["b_business_rights_in_frozen_bundles"] = [b for b in SPEC["feature_bundles_needed"] if b == "C07_design_utility"]
A["b_frozen_feature_set_has_rl"] = any(f.startswith("rl_") for f in SPEC["features"])
# 사후 목록 경유 사업자번호(r3nts src=RL, 즉 2026 권리목록에서만 사업자번호를 얻은 출원인) 규모 — dev 패널 기준
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "src", "registered"])
dp = pd.read_parquet(os.path.join(CACHE, "dev_panel.parquet"), columns=["applicantCd", "t0"])
rl_only = set(ns.loc[ns.src == "RL", "applicantCd"])
A["b_rl_bizno_only_rows_dev_panel"] = int(dp.applicantCd.isin(rl_only).sum()); A["b_dev_panel_rows"] = int(len(dp))
# NTIS 성과 연결표 Year·ProjectYear 결측 비율(결측이면 ≤ t0 비교가 False → 피처에서 빠짐: 보수적)
lk = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["Year", "ProjectYear"])
A["b_link_year_na"] = float(pd.to_numeric(lk.Year, errors="coerce").isna().mean())
A["b_link_projyear_na"] = float(pd.to_numeric(lk.ProjectYear, errors="coerce").isna().mean())
# 코드 속 시점 필터 존재 여부(정적 점검)
checks = {"M2 Year<=t0": "(a.Year <= t0) & (a.ProjectYear <= t0)", "M3 avail_year<=t0": "KY[KY.avail_year <= t0]",
          "C06 window (t0-2,t0]": "yv = ((fy > s) & (fy <= s + 2))", "AF known_date<=end": "AF[AF.known_date <= end]",
          "NTS end<=end 제외": "f = f[~(f.end <= end)]", "registry rgstDate<=end": "pt[pt.rgstDate <= end]",
          "pay d<=end": "py[(py.d <= end)", "inventor rgstDate<=end": "iv[(iv.rgstDate <= end)",
          "NPS cur snapshot": 'NPS[f"{t0}-12"]', "dev NTIS assert": 'assert d.year.max() < F_YEAR'}
A["b_static_filters_present"] = {k: (v in src) for k, v in checks.items()}
# (c) M0 고정
A["c_M0"] = dict(spec=SPEC["baseline_M0"], lib=M0, same=SPEC["baseline_M0"] == M0, model="xgb seeds " + str(SPEC["model"]["seeds"]))
# (d) CI 군집
A["d_cluster"] = "bizno(사업자번호) 군집 부트스트랩: final_run_r5.py cluster_boot(..., cl=te.bizno)"
A["d_ok"] = "cl = te.bizno.values" in open(os.path.join(OUT, "final_run_r5.py"), encoding="utf-8").read()
A["verdict"] = "위반 없음" if (A["a_label"]["ok"] and not A["b_frozen_feature_set_has_rl"] and A["c_M0"]["same"] and A["d_ok"]
                              and all(A["b_static_filters_present"].values()) and set(A["b_ntis_ksic_avail_lag"]) == {"1"}) else "점검 필요"
json.dump(A, open(os.path.join(OUT, "audit_r5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(A, ensure_ascii=False, indent=1))
