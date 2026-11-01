# -*- coding: utf-8 -*-
"""
스트레스 테스트 2차 — T1에서 드러난 '침체 정의 민감성'의 정체 규명
발견: 생산 -5% 기준에서는 r=+0.43 유의, 0% 기준에서는 r=-0.05 무신호
가설 H: PADI는 '침체 빈도'가 아니라 '침체 심도(severity)'와 연관된다
       (완만한 마이너스 전환과는 무관, 급락과만 연관 = 취약성 지표의 전형적 성질)
검정: 빈도 대신 심도 지표들과의 상관 — 낙폭, 하방변동성, 꼬리위험
"""
import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V3 = os.path.join(BASE, "분석", "v3")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
OUT = os.path.join(BASE, "분석", "stress")
PSTART = 2008
rng = np.random.default_rng(11)
log = lambda m: print(m, flush=True)

k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k["val"] = pd.to_numeric(k.값, errors="coerce")
k = k.dropna(subset=["val"])
k["mi"] = (k.연월.str[:4].astype(int) - PSTART) * 12 + k.연월.str[4:6].astype(int) - 1
k["k3"] = k.산업코드.str[1:]
k = k[k.k3.str.len() == 3]
piv = k.pivot_table(index=["k3", "mi"], columns="항목명", values="val", aggfunc="mean").reset_index()
piv.columns.name = None
piv = piv.rename(columns={c: ("prod" if "생산" in c else "ship" if "출하" in c else "inv" if "재고" in c else c)
                          for c in piv.columns if c not in ("k3", "mi")})
piv = piv.sort_values(["k3", "mi"])
piv["yoy"] = piv.groupby("k3")["prod"].pct_change(12) * 100

p = pd.read_parquet(os.path.join(V3, "padi_v3_k5.parquet"))
p["k3"] = p.unit.astype(str).str[:3]
m = p.merge(piv[["k3", "mi", "prod", "yoy"]], on=["k3", "mi"], how="inner")
m = m.replace([np.inf, -np.inf], np.nan)

# ── 산업별 침체 '심도' 지표 ─────────────────────────────────────
rows = []
for u, g in m.groupby("unit"):
    g = g.sort_values("mi").dropna(subset=["yoy"])
    if len(g) < 60:
        continue
    y = g.yoy.values
    pr = g["prod"].values
    peak = np.maximum.accumulate(pr)
    dd = (pr / peak - 1) * 100                      # 낙폭(drawdown)
    rows.append({
        "unit": u,
        "padi": g.padi.mean(),
        "sev_mean_neg": y[y < 0].mean() if (y < 0).any() else 0,   # 마이너스 구간 평균 깊이
        "sev_p05": np.percentile(y, 5),                            # 하위 5% 성장률
        "sev_min": y.min(),                                        # 최악 성장률
        "sev_maxdd": dd.min(),                                     # 최대낙폭
        "downside_sd": y[y < 0].std() if (y < 0).sum() > 2 else np.nan,  # 하방변동성
        "freq_neg": (y < 0).mean(),                                # 마이너스 빈도(참고)
        "n": len(g),
    })
a = pd.DataFrame(rows).dropna()
log(f"산업 {len(a)}개")


def boot(x, y, groups=None, n=4000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    g = np.asarray(groups)[ok] if groups is not None else None
    r0 = np.corrcoef(x, y)[0, 1]
    bs = []
    if g is None:
        for _ in range(n):
            i = rng.integers(0, len(x), len(x))
            if np.std(x[i]) > 1e-9 and np.std(y[i]) > 1e-9:
                bs.append(np.corrcoef(x[i], y[i])[0, 1])
    else:
        uq = np.unique(g)
        for _ in range(n):
            pick = rng.choice(uq, len(uq), replace=True)
            idx = np.concatenate([np.where(g == c)[0] for c in pick])
            if len(idx) > 3 and np.std(x[idx]) > 1e-9 and np.std(y[idx]) > 1e-9:
                bs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return r0, np.percentile(bs, [2.5, 97.5])


a["mid"] = a.unit.str[:2]
log("\n[가설 H 검정] PADI vs 침체 '심도' (부호 주의: 심도 지표는 낮을수록 심각 → 음의 상관이 가설 부합)")
tests = [("sev_mean_neg", "마이너스 구간 평균 깊이"), ("sev_p05", "하위5% 성장률"),
         ("sev_min", "최악 성장률"), ("sev_maxdd", "최대낙폭"),
         ("downside_sd", "하방변동성(↑심각)"), ("freq_neg", "마이너스 빈도(참고)")]
res = []
for c, lab in tests:
    r, ci = boot(a.padi, a[c])
    rc, cic = boot(a.padi, a[c], groups=a.mid)
    sig = "유의" if (ci[0] > 0 or ci[1] < 0) else "비유의"
    sigc = "유의" if (cic[0] > 0 or cic[1] < 0) else "비유의"
    rs = a.padi.corr(a[c], method="spearman")
    res.append({"지표": lab, "r": r, "CI_lo": ci[0], "CI_hi": ci[1], "판정": sig,
                "클러스터CI_lo": cic[0], "클러스터CI_hi": cic[1], "클러스터판정": sigc, "Spearman": rs})
    log(f"  {lab:22s} r={r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] {sig:4s} | 클러스터[{cic[0]:+.3f},{cic[1]:+.3f}] {sigc:4s} | ρ={rs:+.3f}")

pd.DataFrame(res).to_csv(os.path.join(OUT, "severity_test.csv"), index=False, encoding="utf-8-sig")

log("\n[해석 보조] PADI 상위/하위 5개 산업의 침체 심도")
a2 = a.sort_values("padi", ascending=False)
cols = ["unit", "padi", "sev_min", "sev_maxdd", "downside_sd", "freq_neg"]
log("  상위(포기율 높음):\n" + a2.head(5)[cols].round(2).to_string(index=False))
log("  하위(포기율 낮음):\n" + a2.tail(5)[cols].round(2).to_string(index=False))
a.to_csv(os.path.join(OUT, "severity_agg.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {OUT}")
