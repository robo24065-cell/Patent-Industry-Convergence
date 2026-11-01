# -*- coding: utf-8 -*-
"""
신호 분해 검정 — PADI의 예측력은 횡단면인가 시계열인가?
  H_cross : "포기율이 구조적으로 높은 산업 = 원래 침체가 잦은 산업" (조기경보 아님)
  H_time  : "한 산업 안에서 포기율이 오르면 그 뒤 침체가 온다" (진짜 조기경보)
AUC(padi 수준) - AUC(산업내 편차) 로 두 성분을 분리한다.
"""
import os
import numpy as np
import pandas as pd

V2 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\v2"
log = lambda m: print(m, flush=True)


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~(np.isnan(y) | np.isnan(s)); y, s = y[ok], s[ok]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


for level in ("k5", "k3"):
    m = pd.read_csv(os.path.join(V2, f"verify_track_{level}.csv"), encoding="utf-8-sig")
    m = m.sort_values(["unit", "qi"])
    log(f"\n{'='*64}\n[트랙 {level}]  단위 {m.unit.nunique()}개")

    # ── 1) 횡단면 성분: 산업 평균 PADI vs 산업 침체빈도 ──────────
    agg = m.groupby("unit").agg(padi_mean=("padi", "mean"),
                                onset_rate=("onset", "mean"),
                                yoy_mean=("yoy", "mean"), n=("qi", "size")).reset_index()
    c1 = agg.padi_mean.corr(agg.onset_rate)
    c2 = agg.padi_mean.corr(agg.yoy_mean)
    log(f"  [횡단면] corr(산업평균 PADI, 침체빈도) = {c1:+.3f}")
    log(f"           corr(산업평균 PADI, 평균 생산성장) = {c2:+.3f}")

    # ── 2) 시계열 성분: 산업내 편차만으로 예측 ──────────────────
    m["padi_within"] = m.padi - m.groupby("unit").padi.transform("mean")
    log("  [분해] h별 AUC — 수준(level) vs 산업내 편차(within)")
    rows = []
    for h in (1, 2, 3, 4):
        y = m.groupby("unit").onset.shift(-h)
        a_lvl = auc(y, m.padi)
        a_wth = auc(y, m.padi_within)
        # 산업 더미만으로 얻는 AUC (신호 없이 산업 정체성만)
        a_id = auc(y, m.unit.map(agg.set_index("unit").onset_rate))
        rows.append({"h": h, "AUC_수준": a_lvl, "AUC_산업내편차": a_wth,
                     "AUC_산업정체성만": a_id})
    d = pd.DataFrame(rows)
    log(d.round(3).to_string(index=False))
    gap = (d.AUC_수준 - d.AUC_산업내편차).mean()
    log(f"  >>> 수준 − 산업내편차 평균 격차 = {gap:+.3f}")
    if abs(d.AUC_산업내편차.mean() - 0.5) < 0.02:
        log("  >>> 판정: 시계열 성분 사실상 없음 → 0.57 AUC는 횡단면(산업 정체성) 효과")
    else:
        log("  >>> 판정: 시계열 성분 존재")

    # ── 3) 산업내 편차의 방향성 (부호 확인) ─────────────────────
    for h in (1, 2, 3):
        y = m.groupby("unit").onset.shift(-h)
        ok = y.notna() & m.padi_within.notna()
        hi = m.padi_within[ok & (y == 1)].mean()
        lo = m.padi_within[ok & (y == 0)].mean()
        log(f"    h=+{h}: 침체 직전 산업내편차 {hi:+.4f} vs 평시 {lo:+.4f} (차 {hi-lo:+.4f})")
