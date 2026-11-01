# -*- coding: utf-8 -*-
"""C07-T3 s03: t0=2024 모집단 독립 재구성(n8lib 미사용, sqlite mode=ro)과 결과 무관 점검.
- 일치율 / 차집합 원인
- 등록월별 모집단 수(2021-01~2024-12): KR 2023-11 이후 등록 결손 확인
- 미래 생존 조건 여부: t0=2024 모집단 중 2025년 소멸·말소 비율 vs t0=2022 모집단 중 2023년 비율
→ ../cache/s03_out.json"""
import os, sys, json, sqlite3
import numpy as np, pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
pt = pd.read_sql("select rgst_no, ksic, rgstDate, cndrtExptnDate, applDate from patent", c)
ex = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName from right where rgstCsName in ('소멸등록','권리의 말소등록','말소등록')", c)
ex["d"] = pd.to_numeric(ex.rgstCsDate, errors="coerce")
rg = pd.to_numeric(pt.rgstDate, errors="coerce"); xd = pd.to_numeric(pt.cndrtExptnDate, errors="coerce")
def pop(E):
    dead = set(ex[ex.d <= E].rgst_no)
    return pt[(rg <= E) & (xd > E) & ~pt.rgst_no.isin(dead)]
P24 = pop(20241231); P22 = pop(20221231)
S = pd.read_parquet(os.path.join(H, "future_scores_r6.parquet"), columns=["rgst_no", "ksic"])
a, b = set(P24.rgst_no), set(S.rgst_no)
out = dict(rebuilt=len(a), future=len(b), inter=len(a & b), only_rebuilt=len(a - b), only_future=len(b - a), jaccard=len(a & b) / len(a | b))
FP = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no"])
out["t2022_rebuilt"] = len(P22); out["t2022_official_same_set"] = bool(set(P22.rgst_no) == set(FP.rgst_no))
out["rgst_date_null"] = int(rg.isna().sum()); out["expiry_null"] = int(xd.isna().sum())
# 등록월별
m = (rg // 100).astype("Int64")
mm = m[(m >= 202101) & (m <= 202412)].value_counts().sort_index()
out["reg_month_counts_2021_2024"] = {str(k): int(v) for k, v in mm.items()}
y = (rg // 10000).astype("Int64").value_counts().sort_index()
out["reg_year_counts_2015_2026"] = {str(k): int(v) for k, v in y.items() if 2015 <= k <= 2026}
# 미래 생존 조건: 다음 해 소멸·말소 비율
def lapse(P, lo, hi):
    s = set(ex[(ex.d >= lo) & (ex.d <= hi)].rgst_no)
    return float(P.rgst_no.isin(s).mean())
out["lapse_next_year"] = dict(t2022_in_2023=lapse(P22, 20230101, 20231231), t2024_in_2025=lapse(P24, 20250101, 20251231),
                              t2024_in_2026H1=lapse(P24, 20260101, 20260630), t2022_in_2023H1=lapse(P22, 20230101, 20230630))
# KSIC 분포 비교(t0=2022 대 2024)
k22 = P22.ksic.value_counts(normalize=True); k24 = P24.ksic.value_counts(normalize=True)
kk = pd.concat([k22, k24], axis=1).fillna(0)
out["ksic_tvd_2022_vs_2024"] = float(0.5 * np.abs(kk.iloc[:, 0] - kk.iloc[:, 1]).sum())
out["n_ksic"] = int(P24.ksic.nunique())
out["utility_share"] = dict(t2022=float((P22.rgst_no.str[:2] == "20").mean()), t2024=float((P24.rgst_no.str[:2] == "20").mean()))
json.dump(out, open(os.path.join(W, "cache", "s03_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
