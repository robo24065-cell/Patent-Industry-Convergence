# -*- coding: utf-8 -*-
"""
평활 창 민감도 — 4Q 이동합이 타이밍 신호를 소거했는가?
대형 산업(E4 충분)에 한해 1Q/2Q/4Q 창으로 산업내편차 AUC를 비교한다.
표본 크기 문제가 없는 셀에서만 비교하므로 창 축소의 순수 효과를 본다.
"""
import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V2 = os.path.join(BASE, "분석", "v2")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
PANEL_START = 2008
log = lambda m: print(m, flush=True)


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~(np.isnan(y) | np.isnan(s)); y, s = y[ok], s[ok]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


panel = pd.read_parquet(os.path.join(V2, "padi_panel_v2.parquet"))
panel["k3"] = panel.ksic.str[:3]

k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k = k[k.항목명.str.contains("생산지수")].copy()
k["year"] = k.연월.str[:4].astype(int)
k["q"] = ((k.연월.str[4:6].astype(int) - 1) // 3) + 1
k["val"] = pd.to_numeric(k.값, errors="coerce")
kq = k.dropna(subset=["val"]).groupby(["산업코드", "year", "q"]).val.mean().reset_index()
kq["qi"] = (kq.year - PANEL_START) * 4 + (kq.q - 1)
kq["k3"] = kq.산업코드.str[1:]
kq = kq[kq.k3.str.len() == 3].rename(columns={"val": "pidx"})

# 소분류 집계 후 창 길이별 지수 산출
g0 = panel.groupby(["k3", "qi"]).agg(O=("O", "sum"), E=("E_adj", "sum")).reset_index()
out = []
for win in (1, 2, 4):
    rows = []
    for u, gg in g0.groupby("k3"):
        gg = gg.sort_values("qi").copy()
        gg["Ow"] = gg.O.rolling(win).sum()
        gg["Ew"] = gg.E.rolling(win).sum()
        gg["padi"] = gg.Ow / gg.Ew.replace(0, np.nan)
        gg = gg[gg.Ew >= 10]               # 표본 하한 통일
        rows.append(gg)
    g = pd.concat(rows, ignore_index=True)
    m = g.merge(kq[["k3", "qi", "pidx"]], on=["k3", "qi"], how="inner").sort_values(["k3", "qi"])
    m["yoy"] = m.groupby("k3")["pidx"].pct_change(4) * 100
    m = m.dropna(subset=["padi", "yoy"])
    m["onset"] = 0
    for u, gg in m.groupby("k3"):
        gg = gg.sort_values("qi")
        m.loc[gg.index, "onset"] = ((gg.yoy < -5) & (gg.yoy.shift(1) >= 0)).astype(int).values
    m["within"] = m.padi - m.groupby("k3").padi.transform("mean")
    for h in (1, 2, 3):
        y = m.groupby("k3").onset.shift(-h)
        out.append({"win": win, "h": h, "n": len(m), "onset": int(m.onset.sum()),
                    "AUC_수준": auc(y, m.padi), "AUC_산업내": auc(y, m.within)})

d = pd.DataFrame(out)
log("[평활 창 민감도] 소분류 집계 트랙, 산업내편차 AUC가 핵심")
log(d.round(3).to_string(index=False))
log("\n창 길이별 산업내 AUC 평균:")
log(d.groupby("win").AUC_산업내.mean().round(4).to_string())
best = d.groupby("win").AUC_산업내.mean().idxmax()
log(f">>> 최적 창 = {best}Q, 그러나 0.5와의 거리 = {abs(d.groupby('win').AUC_산업내.mean().max()-0.5):.4f}")
d.to_csv(os.path.join(V2, "window_sensitivity.csv"), index=False, encoding="utf-8-sig")
