# -*- coding: utf-8 -*-
"""아이디어 1(고성장 기업)·2(신규 부실화) 기업×원점 패널.
원점 t0: 피처 = t0 회계연도 재무 + t0 말까지의 특허 사건 + t0 시점 공표된 산업 신호(KSIC 노출도 가중).
결과(h=2): 1) g2 = log(매출_{t0+2}/매출_{t0}), y_hg = 원점 내 상위 25%
           2) y_dist = (영업손실 t0+1·t0+2 연속) 또는 (t0+2 자본총계 < 0.5×자본금), t0 비부실 기업만
🔒 최종시험 원점 2023(결과 2024–25, F=2024): fin.parquet 에 2024 이후 값이 없다(수집 안 함). 개발 원점 t0 ≤ 2021.
출력: cache/panel12.parquet"""
import os
import numpy as np, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
C = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
F = pd.read_parquet(os.path.join(C, "fin.parquet"))
assert F.year.max() <= 2023
P = pd.read_parquet(os.path.join(C, "patfeat.parquet"))
S = pd.read_parquet(os.path.join(C, "ksic_share.parquet"))
X = pd.read_parquet(os.path.join(C, "ksic_ext.parquet"))
dart = pd.read_parquet(os.path.join(C, "dart_firms.parquet"))

f = F.set_index(["corp_code", "year"])
def lag(col, k):
    s = F.set_index(["corp_code", "year"])[col]
    s.index = s.index.set_levels(s.index.levels[1] + k, level=1)
    return s

B = F[["corp_code", "year"]].rename(columns={"year": "t0"}).copy()
idx = pd.MultiIndex.from_frame(B.rename(columns={"t0": "year"}))
def get(col, k):  # 값 at t0+k
    s = F.set_index(["corp_code", "year"])[col]
    return s.reindex(pd.MultiIndex.from_arrays([B.corp_code, B.t0 + k])).values
for col in ["rev", "op", "ni", "assets", "liab", "equity", "capital", "cur_assets", "re"]:
    B[col] = get(col, 0)
B["rev_m1"] = get("rev", -1); B["rev_m2"] = get("rev", -2); B["op_m1"] = get("op", -1)
B["rev_p2"] = get("rev", 2); B["op_p1"] = get("op", 1); B["op_p2"] = get("op", 2)
B["eq_p2"] = get("equity", 2); B["cap_p2"] = get("capital", 2)
# 재무 피처
B["log_rev"] = np.log1p(B.rev.clip(lower=0)); B["log_assets"] = np.log1p(B.assets.clip(lower=0))
B["g_rev1"] = np.log(B.rev / B.rev_m1); B["g_rev2"] = np.log(B.rev / B.rev_m2)
B["opm"] = B.op / B.rev; B["roa"] = B.ni / B.assets; B["lev"] = B.liab / B.assets
B["eq_cap"] = B.equity / B.capital; B["cur_ratio"] = B.cur_assets / B.assets; B["re_assets"] = B.re / B.assets
B["op_neg"] = (B.op < 0).astype(float); B["op_neg_m1"] = (B.op_m1 < 0).astype(float)
for c in ["g_rev1", "g_rev2", "opm", "roa", "lev", "eq_cap", "cur_ratio", "re_assets"]:
    B[c] = B[c].replace([np.inf, -np.inf], np.nan).clip(B[c].quantile(0.01), B[c].quantile(0.99))
d = dart.set_index("corp_code")
B["firm_age"] = B.t0 - pd.to_numeric(B.corp_code.map(d.est_dt).str[:4], errors="coerce")
# 결과
ok1 = (B.rev > 1e9) & (B.rev_p2 > 0)
B["g2"] = np.where(ok1, np.log(B.rev_p2 / B.rev), np.nan)
B["y_hg"] = np.nan
for t, g in B[ok1].groupby("t0"):
    B.loc[g.index, "y_hg"] = (g.g2 >= g.g2.quantile(0.75)).astype(float)
dist0 = (B.op < 0) | (B.equity < 0.5 * B.capital)
ok2 = B.op.notna() & B.equity.notna() & ~dist0 & B.op_p1.notna() & B.op_p2.notna() & B.eq_p2.notna()
B["y_dist"] = np.where(ok2, (((B.op_p1 < 0) & (B.op_p2 < 0)) | (B.eq_p2 < 0.5 * B.cap_p2)).astype(float), np.nan)
B = B.drop(columns=["rev_p2", "op_p1", "op_p2", "eq_p2", "cap_p2"])
# 특허 피처 (t0 말까지 등록 특허가 없으면 0)
B = B.merge(P, on=["corp_code", "t0"], how="left")
B["has_pat"] = B.n_all.notna().astype(float)
B[P.columns.drop(["corp_code", "t0"])] = B[P.columns.drop(["corp_code", "t0"])].fillna(0)
# KSIC 노출도 가중 외부 신호
ext_cols = [c for c in X.columns if c not in ("ksic", "t0")]
SX = S.merge(X, on=["ksic", "t0"], how="left")
agg = {}
for c in ext_cols:
    w = SX.share.where(SX[c].notna(), 0)
    SX[c + "_w"] = SX[c].fillna(0) * w
    SX[c + "_ws"] = w
G = SX.groupby(["corp_code", "t0"])
E = pd.DataFrame({("x_" + c): G[c + "_w"].sum() / G[c + "_ws"].sum().replace(0, np.nan) for c in ext_cols}).reset_index()
B = B.merge(E, on=["corp_code", "t0"], how="left")
# 주력 KSIC (생존 스톡 최대 비중)
top = S.sort_values("share", ascending=False).drop_duplicates(["corp_code", "t0"])[["corp_code", "t0", "ksic", "share"]]
B = B.merge(top.rename(columns={"ksic": "top_ksic", "share": "top_share"}), on=["corp_code", "t0"], how="left")
B.to_parquet(os.path.join(C, "panel12.parquet"), index=False)
print(B.groupby("t0").agg(n=("corp_code", "size"), n_hg=("y_hg", "count"), hg=("y_hg", "mean"), n_dist=("y_dist", "count"),
                           dist=("y_dist", "mean"), has_pat=("has_pat", "mean")).round(3))
