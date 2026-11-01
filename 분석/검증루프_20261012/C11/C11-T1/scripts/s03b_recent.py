# -*- coding: utf-8 -*-
"""2025–26 발급 코드의 법인번호 체크섬 실패 집중 원인 탐색."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np
ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet"))
J = ca[ca.jur != ""].copy()
J["fail"] = J.jur_ok == False
J["iy"] = J.cd.str[1:5].astype(int)
J["serial"] = J.jur.str[6:12].astype(int)
J["pre6"] = J.jur.str[:6]
O = {}
O["fail_counts_by_period"] = {"<=2024": int(J.fail[J.iy <= 2024].sum()), "2025": int(J.fail[J.iy == 2025].sum()), "2026": int(J.fail[J.iy == 2026].sum())}
O["rows_by_period"] = {"<=2024": int((J.iy <= 2024).sum()), "2025": int((J.iy == 2025).sum()), "2026": int((J.iy == 2026).sum())}
R = J[J.iy >= 2025]
O["recent_fail_by_pre6_top"] = R.groupby("pre6").fail.agg(["mean", "size"]).sort_values("size", ascending=False).head(12).round(3).to_dict("index")
# 같은 등기관서 110111에서 발급연도별 serial 범위
for p in ["110111", "131111", "134511", "284111"]:
    g = J[J.pre6 == p]
    O[f"serial_by_year_{p}"] = g.groupby(g.iy.clip(lower=2018)).serial.agg(["min", "median", "max", "size"]).astype(int).to_dict("index")
    O[f"fail_by_year_{p}"] = g.groupby(g.iy.clip(lower=2018)).fail.mean().round(3).to_dict()
# 최근 코드에서 serial 상위(오래된 순번 소진) 여부: 실패 vs 통과 serial 분포
O["recent_serial_fail_vs_ok"] = {k: R[R.fail == v].serial.describe().round(0).to_dict() for k, v in [("fail", True), ("ok", False)]}
# 13번째 자리를 순번 일부로 보는 가설: 실패 번호의 앞 12자리가 통과 번호와 겹치는가(같은 앞 12자리 다른 끝자리)
p12 = J.jur.str[:12]
dup12 = p12.duplicated(keep=False)
O["recent_fail_share_with_dup12"] = float(dup12[J.fail & (J.iy >= 2025)].mean())
O["older_fail_share_with_dup12"] = float(dup12[J.fail & (J.iy <= 2024)].mean())
# 재발급 코드인가: 최근 코드인데 기존 코드와 같은 법인번호 공유
jc = J.groupby("jur").cd.nunique()
O["recent_fail_share_jur_shared_with_other_code"] = float(J.jur[J.fail & (J.iy >= 2025)].map(jc).gt(1).mean())
# 국세청 상태와 개업연도 대리(사업자번호 체크섬)
O["recent_fail_biz_present"] = float((R.loc[R.fail, "biz"] != "").mean())
O["recent_ok_biz_present"] = float((R.loc[~R.fail, "biz"] != "").mean())
json.dump(O, open(guard_path(os.path.join(CACHE, "s03b_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
