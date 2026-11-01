# verify_2 v2: 2025-09 이후 소멸등록 증가 = 위험집합(코호트) 증가인가, 소멸률 증가인가
import os, json
import pandas as pd, numpy as np
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\verify_2")
df = pd.read_parquet(os.path.join(OUT, "v1_base.parquet"))
r = {}
r["rg_year_counts"] = {int(k): int(v) for k, v in df.rg.dt.year.value_counts().sort_index().items()}
r["max_annl_counts"] = {int(k): int(v) for k, v in df.max_annl.value_counts().sort_index().items()}
df["lapsed"] = df.lap.notna() & ~(df.rest.notna() & (df.rest >= df.lap))
rows = []
for k in range(3, 21):
    ann = df.rg + pd.DateOffset(years=k)
    m = ann.dt.to_period("M")
    at = (df.max_annl >= k)
    lp = (df.max_annl == k) & df.lapsed
    t = pd.DataFrame({"m": m, "atr": at, "lp": lp, "k": k})
    t = t[(t.m >= pd.Period("2022-01", "M")) & (t.m <= pd.Period("2025-10", "M"))]
    rows.append(t.groupby(["m", "k"]).agg(atr=("atr", "sum"), lp=("lp", "sum")).reset_index())
R = pd.concat(rows)
R["m"] = R.m.astype(str)
tot = R.groupby("m")[["atr", "lp"]].sum()
tot["haz"] = (tot.lp / tot.atr).round(4)
r["by_month_total"] = {k: [int(v.atr), int(v.lp), float(v.haz)] for k, v in tot.iterrows()}
R["kb"] = np.where(R.k <= 4, "k3-4", np.where(R.k <= 7, "k5-7", "k8+"))
def per(a, b):
    s = R[(R.m >= a) & (R.m <= b)].groupby("kb")[["atr", "lp"]].sum()
    s["haz"] = (s.lp / s.atr).round(4); s["at_pm"] = (s.atr / (len(pd.period_range(a, b, freq="M")))).round(0)
    return {k: [float(v.at_pm), int(v.lp), float(v.haz)] for k, v in s.iterrows()}
r["per_2023_2024"] = per("2023-01", "2024-12")
r["per_2024_01_11"] = per("2024-01", "2024-11")
r["per_2024_12_2025_10"] = per("2024-12", "2025-10")
# 등록 코호트별: 납부완료 2024-12~2025-10 소멸의 등록연도 분포 vs 2023-12~2024-10
def coh(a, b):
    s = df[df.lapsed & (df.paid_thru >= a) & (df.paid_thru <= b)]
    return {int(k): int(v) for k, v in s.rg.dt.year.value_counts().sort_index().items()}
r["lapse_rgyear_2023-12_2024-10"] = coh("2023-12-01", "2024-10-31")
r["lapse_rgyear_2024-12_2025-10"] = coh("2024-12-01", "2025-10-31")
json.dump(r, open(os.path.join(OUT, "v2_cohort.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k in ["rg_year_counts", "per_2023_2024", "per_2024_01_11", "per_2024_12_2025_10", "lapse_rgyear_2023-12_2024-10", "lapse_rgyear_2024-12_2025-10"]:
    print(k, r[k])
for k, v in r["by_month_total"].items():
    if k >= "2023-06": print(k, v)
