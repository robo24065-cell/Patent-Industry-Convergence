# -*- coding: utf-8 -*-
# C16-T2 verify_0 v1b: 시험 패널 dart_has=1 층을 DART corp_cls(현재값)로 나눠 폐업률 비교. 읽기 전용, 재채점 없음.
import os, sys, json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
EXT = os.path.join(B, r"모델경쟁_20260927\_shared\ext"); N4 = os.path.join(B, r"모델경쟁_20260927\N4_특허행태_기업폐업조기경보")
dp = pd.read_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"), columns=["bizr_no", "corp_cls"])
dp["bz"] = dp.bizr_no.fillna("").astype(str).str.replace(r"\D", "", regex=True); cls = dp.drop_duplicates("bz").set_index("bz").corp_cls
fp = pd.read_parquet(os.path.join(N4, "cache", "final_panel.parquet"), columns=["applicantCd", "bizno", "t0", "dart_has"])
pr = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"), columns=["applicantCd", "bizno", "y"])
te = fp[fp.t0 == 2023].merge(pr, on=["applicantCd", "bizno"])
te["bz"] = te.bizno.fillna("").astype(str).str.replace(r"\D", "", regex=True)
te["cls"] = te.bz.map(cls).fillna("-")
t = te.groupby([te.dart_has.fillna(0), "cls"]).y.agg(["size", "sum", "mean"])
print(t.to_string())
json.dump({str(k): v for k, v in t.to_dict(orient="index").items()}, open(os.path.join(B, r"검증루프_20261012\C16\C16-T2\verify_0\v1b_test_by_cls.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
