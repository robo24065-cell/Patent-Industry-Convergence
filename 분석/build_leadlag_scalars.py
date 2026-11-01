# -*- coding: utf-8 -*-
"""
선행성 패널 → 횡단면 스칼라(전반기 전용) 변환

설계원칙 1(시간분할) 준수: 종속변수 dd_late 는 2017년 이후 값이므로
여기서 만드는 모든 스칼라는 **2016년 이전 값만** 사용한다.
predictive_agg.csv 의 39행에 ksic 키로 바로 붙일 수 있게 만든다.

산출: 분석/fusion/ctrl_leadlag_early.csv
"""
import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
F = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

EARLY_END = 2016

Q = pd.read_csv(os.path.join(F, "leadlag_panel.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
tg = pd.read_csv(os.path.join(F, "predictive_agg.csv"), encoding="utf-8-sig",
                 dtype={"ksic": str})
E = Q[Q.year <= EARLY_END].copy()
log(f"전반기 패널: {len(E)}행, {E.year.min()}~{E.year.max()}")

g = E.groupby("ksic")
out = pd.DataFrame({
    # 전반기 평균 대중 수입침투(수입 중 중국비중, %)
    "cnimp_share_e": g.cn_share_imp.mean(),
    # 전반기 대중 수입침투 상승폭 (2016 - 2000 근방, 3년평균 차분으로 잡음완화)
    "cnimp_rise_e": (g.apply(lambda d: d[d.year >= 2014].cn_share_imp.mean()
                             - d[d.year <= 2002].cn_share_imp.mean(),
                             include_groups=False)),
    # 전반기 대중 수입액 로그 (규모)
    "cnimp_lnusd_e": g.cn_imp_usd.mean().pipe(lambda s: np.log(s.clip(lower=1))),
    # 전반기 대중 수입 연평균 증가율
    "cnimp_growth_e": g.g_impv.mean(),
})
out.index.name = "ksic"
out = out.reset_index()

m = tg[["ksic"]].merge(out, on="ksic", how="left")
log(f"39산업 중 매칭 {m.cnimp_share_e.notna().sum()}개 "
    f"(결측 {sorted(m[m.cnimp_share_e.isna()].ksic)})")
for c in out.columns[1:]:
    log(f"  {c:16s} nonnull={m[c].notna().sum():2d}  "
        f"mean={m[c].mean():.3f}  sd={m[c].std():.3f}")

# 해상도 진단: 중분류 내 변동이 남아있는가 (수입침투율이 탈락한 이유를 되풀이하지 않기 위해)
m["mid"] = m.ksic.str[:2]
log("")
log("중분류 내 표준편차 (0이면 클러스터 추론에서 검정력 소실):")
for c in out.columns[1:]:
    w = m.groupby("mid")[c].std()
    w = w[m.groupby("mid").ksic.nunique() > 1]
    log(f"  {c:16s} 복수산업 중분류 {len(w)}개 중 within-sd 평균 {w.mean():.3f}, "
        f"0인 곳 {int((w.fillna(0) < 1e-9).sum())}개")

m.to_csv(os.path.join(F, "ctrl_leadlag_early.csv"), index=False, encoding="utf-8-sig")
log("")
log(f"저장 ctrl_leadlag_early.csv ({len(m)}행)")
