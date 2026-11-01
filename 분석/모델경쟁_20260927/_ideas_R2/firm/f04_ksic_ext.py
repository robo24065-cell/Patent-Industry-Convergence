# -*- coding: utf-8 -*-
"""KSIC 5자리 × 연도 외부 산업 신호 (기업 노출도 가중에 사용).
- KOSIS 광업제조업조사(출하·종사자·부가가치): 10차(≤2020)·11차(2020–) 2020년 비율로 연결, 2010 결측은 기하보간.
  t0 시점엔 t0 조사가 미공표(다음 해 말 공표) → 원점 t0 피처는 t0-1 기준 2년 성장률.
- 무역: ksic_hs6_map × hs6_import_yearly(WLD, CN) → KSIC 수입액, t0 기준 2년 성장률(무역통계는 익년 2월 공표).
- 5개국 특허(제공 데이터 meta): 공개연도 기준 KSIC×국가 건수, 모멘텀 log((t0-1..t0)/(t0-3..t0-2)).
출력: cache/ksic_ext.parquet (ksic, t0, 피처)"""
import os
import numpy as np, pandas as pd, duckdb

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
C = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
META = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
T0S = range(2010, 2024)

k = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"))
k = k[k.항목코드.isin(["T02", "T04", "T06"])]
pv = k.pivot_table(index=["세세분류코드", "항목코드", "연도"], columns="기준", values="값").reset_index()
out = []
for (ks, it), g in pv.groupby(["세세분류코드", "항목코드"]):
    g = g.set_index("연도")
    s10 = g["10차"].dropna(); s11 = g["11차"].dropna()
    s = s10.copy()
    if 2020 in s10.index and 2020 in s11.index and s11[2020] > 0:
        r = s10[2020] / s11[2020]
        for y in s11.index:
            if y > 2020:
                s[y] = s11[y] * r
    s = s.sort_index()
    if 2009 in s.index and 2011 in s.index and s[2009] > 0 and s[2011] > 0:
        s[2010] = np.sqrt(s[2009] * s[2011])
    s = s.sort_index()
    for y in s.index:
        out.append((str(ks), it, y, s[y]))
K = pd.DataFrame(out, columns=["ksic", "it", "y", "v"]).pivot_table(index=["ksic", "y"], columns="it", values="v").reset_index()
rows = []
for t0 in T0S:
    a = K[K.y == t0 - 1].set_index("ksic"); b = K[K.y == t0 - 3].set_index("ksic")
    r = pd.DataFrame({"g_ship": np.log(a.T04 / b.T04), "g_emp": np.log(a.T02 / b.T02), "g_va": np.log(a.T06 / b.T06),
                      "va_per_emp": np.log(a.T06 / a.T02)})
    r["t0"] = t0
    rows.append(r.reset_index())
KX = pd.concat(rows, ignore_index=True)

m = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str)
h = pd.read_csv(os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv"), dtype={"hs6": str})
h["hs6"] = h.hs6.str.zfill(6); m["hs6"] = m.hs6.str.zfill(6)
im = h.merge(m[["k5", "hs6"]].drop_duplicates(), on="hs6").groupby(["k5", "partner", "year"]).value.sum().unstack("partner").reset_index()
rows = []
for t0 in T0S:
    a = im[im.year == t0].set_index("k5"); b = im[im.year == t0 - 2].set_index("k5")
    r = pd.DataFrame({"g_imp": np.log((a.WLD + 1) / (b.WLD + 1)), "g_imp_cn": np.log((a.CN + 1) / (b.CN + 1)),
                      "cn_share": a.CN / a.WLD})
    r["t0"] = t0
    rows.append(r.reset_index().rename(columns={"k5": "ksic"}))
IX = pd.concat(rows, ignore_index=True)

q = f"""select ksic::varchar ksic, cc, cast(substr(pub_date,1,4) as int) py, count(distinct appl_no) n
from read_parquet('{META}') where pub_date is not null and length(pub_date)>=4 group by 1,2,3"""
M = duckdb.sql(q).df()
M.to_parquet(os.path.join(C, "meta_ksic_cc_py.parquet"), index=False)
rows = []
for t0 in T0S:
    a = M[M.py.between(t0 - 1, t0)].groupby(["ksic", "cc"]).n.sum().unstack().fillna(0)
    b = M[M.py.between(t0 - 3, t0 - 2)].groupby(["ksic", "cc"]).n.sum().unstack().fillna(0)
    a, b = a.align(b, fill_value=0)
    r = np.log((a + 1) / (b + 1)).add_prefix("pmom_")
    tot_a = a.drop(columns=["KR"]).sum(1); tot_b = b.drop(columns=["KR"]).sum(1)
    r["pmom_foreign"] = np.log((tot_a + 1) / (tot_b + 1))
    r["kr_gap"] = r["pmom_foreign"] - r["pmom_KR"]
    r["t0"] = t0
    rows.append(r.reset_index())
PX = pd.concat(rows, ignore_index=True)
X = PX.merge(KX, on=["ksic", "t0"], how="left").merge(IX, on=["ksic", "t0"], how="left")
X.to_parquet(os.path.join(C, "ksic_ext.parquet"), index=False)
print(X.groupby("t0")[["pmom_KR", "pmom_foreign", "g_ship", "g_imp"]].mean().round(3))
print("ksic", X.ksic.nunique(), "cover g_ship", X.g_ship.notna().mean().round(3), "g_imp", X.g_imp.notna().mean().round(3))
