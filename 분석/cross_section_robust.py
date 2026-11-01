# -*- coding: utf-8 -*-
"""
횡단면 발견의 강건성 — "특허를 많이 버리는 산업 = 경기 변동이 큰 산업"이 진짜인가?
교란 후보: 산업 규모, 특허 스톡 연령, 실용신안 비중, 관측기간
"""
import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V2 = os.path.join(BASE, "분석", "v2")
rng = np.random.default_rng(7)
log = lambda m: print(m, flush=True)

for level in ("k5", "k3"):
    m = pd.read_csv(os.path.join(V2, f"verify_track_{level}.csv"), encoding="utf-8-sig")
    agg = m.groupby("unit").agg(
        padi=("padi", "mean"),
        onset_rate=("onset", "mean"),
        yoy_sd=("yoy", "std"),          # 경기 변동성
        yoy_mean=("yoy", "mean"),
        size=("R4", "mean"),            # 위험집합 규모 = 산업 크기
        n=("qi", "size")).reset_index().dropna()
    log(f"\n{'='*60}\n[{level}] 단위 {len(agg)}개")

    def boot_corr(x, y, n=2000):
        r0 = np.corrcoef(x, y)[0, 1]
        bs = [np.corrcoef(x[i], y[i])[0, 1]
              for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
        return r0, np.percentile(bs, [2.5, 97.5])

    for tgt, label in [("onset_rate", "침체빈도"), ("yoy_sd", "생산 변동성"), ("yoy_mean", "평균 성장률")]:
        r, ci = boot_corr(agg.padi.values, agg[tgt].values)
        star = "유의" if (ci[0] > 0 or ci[1] < 0) else "비유의"
        log(f"  corr(PADI, {label}) = {r:+.3f}  95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  {star}")

    # 산업 규모 통제 후 부분상관
    def partial(x, y, z):
        bx = np.polyfit(z, x, 1); by = np.polyfit(z, y, 1)
        rx = x - np.polyval(bx, z); ry = y - np.polyval(by, z)
        return np.corrcoef(rx, ry)[0, 1]

    lz = np.log(agg["size"].values + 1)
    log(f"  [규모 통제] 부분상관(PADI, 침체빈도 | log규모) = {partial(agg.padi.values, agg.onset_rate.values, lz):+.3f}")
    log(f"  [규모 통제] 부분상관(PADI, 변동성   | log규모) = {partial(agg.padi.values, agg.yoy_sd.values, lz):+.3f}")

    top = agg.nlargest(5, "padi")[["unit", "padi", "onset_rate", "yoy_sd"]]
    bot = agg.nsmallest(5, "padi")[["unit", "padi", "onset_rate", "yoy_sd"]]
    log("  포기율 상위5:\n" + top.round(3).to_string(index=False))
    log("  포기율 하위5:\n" + bot.round(3).to_string(index=False))
    agg.to_csv(os.path.join(V2, f"cross_section_{level}.csv"), index=False, encoding="utf-8-sig")
