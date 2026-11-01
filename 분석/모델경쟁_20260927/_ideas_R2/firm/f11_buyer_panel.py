# -*- coding: utf-8 -*-
"""아이디어 3: 기술 구매기업(특허 양수) 예측 패널. 단위 = (DART 연결 특허보유 법인, 원점 t0).
결과(h=2): y_buy = t0+1..t0+2 에 소규모(≤5건) 양수 거래 1건 이상. y_buy_any = 규모 무관.
🔒 최종시험 원점 2022(결과 2023–24, F=2023). 개발 원점 t0 ≤ 2020. buy_events 는 ≤2022 만 들어 있다.
출력: cache/panel3.parquet"""
import os
import numpy as np, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
C = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
P = pd.read_parquet(os.path.join(C, "patfeat.parquet"))
P = P[P.t0 <= 2020]
E = pd.read_parquet(os.path.join(C, "buy_events.parquet"))
assert E.y.max() <= 2022
D = E.drop_duplicates(["corp_code", "seller", "d"])[["corp_code", "y", "deal_size", "ksic"]]
X = pd.read_parquet(os.path.join(C, "ksic_ext.parquet"))
S = pd.read_parquet(os.path.join(C, "ksic_share.parquet"))
dart = pd.read_parquet(os.path.join(C, "dart_firms.parquet")).set_index("corp_code")

def cnt(lo, hi, small=None):
    d = D[D.y.between(lo, hi)]
    if small is True:
        d = d[d.deal_size <= 5]
    return d.groupby("corp_code").size()

rows = []
for t0, g in P.groupby("t0"):
    g = g.copy()
    g["y_buy"] = (g.corp_code.map(cnt(t0 + 1, t0 + 2, True)).fillna(0) > 0).astype(float)
    g["y_buy_any"] = (g.corp_code.map(cnt(t0 + 1, t0 + 2)).fillna(0) > 0).astype(float)
    g["buy_past3"] = g.corp_code.map(cnt(t0 - 2, t0)).fillna(0)
    g["buy_past_all"] = g.corp_code.map(cnt(1990, t0)).fillna(0)
    g["buy_small_past3"] = g.corp_code.map(cnt(t0 - 2, t0, True)).fillna(0)
    rows.append(g)
B = pd.concat(rows, ignore_index=True)
B["firm_age"] = B.t0 - pd.to_numeric(B.corp_code.map(dart.est_dt).str[:4], errors="coerce")
B["log_alive"] = np.log1p(B.n_alive); B["log_reg3"] = np.log1p(B.n_reg3)
ext_cols = [c for c in X.columns if c not in ("ksic", "t0")]
SX = S.merge(X, on=["ksic", "t0"], how="left")
for c in ext_cols:
    w = SX.share.where(SX[c].notna(), 0)
    SX[c + "_w"] = SX[c].fillna(0) * w; SX[c + "_ws"] = w
G = SX.groupby(["corp_code", "t0"])
EX = pd.DataFrame({("x_" + c): G[c + "_w"].sum() / G[c + "_ws"].sum().replace(0, np.nan) for c in ext_cols}).reset_index()
B = B.merge(EX, on=["corp_code", "t0"], how="left")
B.to_parquet(os.path.join(C, "panel3.parquet"), index=False)
print(B.groupby("t0").agg(n=("corp_code", "size"), buy=("y_buy", "mean"), buy_n=("y_buy", "sum"), any=("y_buy_any", "mean")).round(4))
