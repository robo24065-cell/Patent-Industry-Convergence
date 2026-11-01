# -*- coding: utf-8 -*-
# C16-T2 Q1b (사후 진단, PREREG 밖): N4 시험 패널에서 dart_has 층 폐업률을 NPS 고용 규모 층 안에서 비교. 재채점 없음.
import os, json, sys
import pandas as pd, numpy as np
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
N4 = os.path.join(BASE, r"모델경쟁_20260927\N4_특허행태_기업폐업조기경보")
OUT = os.path.join(BASE, r"검증루프_20261012\C16\C16-T2\cache")
fp = pd.read_parquet(os.path.join(N4, "cache", "final_panel.parquet"), columns=["applicantCd", "bizno", "t0", "dart_has", "log_emp", "nps_present", "age"])
pr = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"), columns=["applicantCd", "bizno", "y"])
te = fp[fp.t0 == 2023].merge(pr, on=["applicantCd", "bizno"])
te["dh"] = te.dart_has.fillna(0)
te["emp_bin"] = pd.cut(te.log_emp, [-1, np.log1p(9), np.log1p(49), np.log1p(299), 99], labels=["<10", "10-49", "50-299", "300+"]).astype(str)
te.loc[te.nps_present != 1, "emp_bin"] = "NPS없음"
t = te.groupby(["emp_bin", "dh"]).y.agg(["size", "sum", "mean"]).reset_index()
# 규모 층 표준화(dart_has=1 분포로 가중) 폐업률 비
w = te[te.dh == 1].emp_bin.value_counts(normalize=True)
r0 = te[te.dh == 0].groupby("emp_bin").y.mean().reindex(w.index)
std0 = float((w * r0).sum()); r1 = float(te[te.dh == 1].y.mean())
res = dict(table=t.to_dict(orient="records"), dart1_rate=r1, dart0_rate_size_std=std0, ratio_size_std=r1 / std0 if std0 else None, dart1_emp_dist=w.to_dict())
json.dump(res, open(os.path.join(OUT, "q1b_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
sys.stdout.reconfigure(encoding="utf-8"); print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
