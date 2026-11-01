# -*- coding: utf-8 -*-
"""
모델 확장 — 선형 조합 밖에서 개선 여지를 찾는다
────────────────────────────────────────────────────────────────
탐색이 밝힌 것: 진짜 기준선은 '전반기 낙폭'(지속성)이고 LOCO-R² 이미 +0.482 다.
따라서 모든 확장은 **지속성 위에서** 평가한다.

시험할 확장 (전부 사전 근거가 있는 것만. 무작정 뒤지지 않는다)
  [E1] 이중압박 상호작용  PADI × 중국비중
       근거: 우리 4유형 진단 프레임워크가 이미 "이중압박형"을 별도 유형으로 규정했다.
             그렇다면 두 축의 곱이 합보다 나아야 한다. 프레임워크의 자기검정이다.
  [E2] 비선형          PADI², 순위변환, 윈저화
       근거: O/E 비율은 하한 0, 상한 없는 비대칭 분포다. 극단값이 회귀를 끌 수 있다.
  [E3] 임계효과        PADI > 1 (기대보다 많이 포기) 더미
       근거: O/E=1 은 '기대만큼'이라는 자연스러운 기준점이다. 임의 절단이 아니다.
  [E4] 층화 PADI       저가치층 PADI, 층간 격차
       근거: 지금까지 모든 층화 검정에서 저가치·중소 층에만 신호가 있었다.
  [E5] 상대화          중분류 내 PADI 순위 (산업군 효과 제거)
       근거: 산업구조 공선성 문제에 대한 직접 대응.

평가: LOCO-CV R² (분할연도 7개 평균). 지속성 단독 대비 증분을 본다.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
SPLITS = list(range(2013, 2020))
rng = np.random.default_rng(4649)
log = lambda m: print(m, flush=True)

ALL = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str, "k3": str, "mid": str})
PA = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                 dtype={"ksic": str})[["ksic", "padi_lo_e", "um_share", "hhi", "ipc_entropy"]]
ALL = ALL.merge(PA, on="ksic", how="left")


def loco_r2(X, y, grp):
    uq = np.unique(grp); pred = np.full(len(y), np.nan)
    for c in uq:
        te = grp == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try:
            b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception:
            return np.nan
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pred)
    sse = np.sum((y[ok] - pred[ok]) ** 2); sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    return float(1 - sse / sst) if sst > 0 else np.nan


def prep(d):
    """확장 변수 생성 — 전부 전반기 정보만 사용"""
    d = d.copy()
    d["padi_sq"] = d.padi_e ** 2
    d["padi_rank"] = rankdata(d.padi_e) / len(d)
    lo, hi = d.padi_e.quantile([.05, .95])
    d["padi_win"] = d.padi_e.clip(lo, hi)
    d["padi_over1"] = (d.padi_e > 1.0).astype(float)
    d["padi_x_cn"] = (d.padi_e - d.padi_e.mean()) * (d.cn_share_e - d.cn_share_e.mean())
    d["padi_x_cng"] = (d.padi_e - d.padi_e.mean()) * (d.cn_growth_e - d.cn_growth_e.mean())
    # 중분류 내 상대 PADI (산업군 효과 제거)
    d["padi_wm"] = d.padi_e - d.groupby("mid").padi_e.transform("mean")
    d["cn_wm"] = d.cn_share_e - d.groupby("mid").cn_share_e.transform("mean")
    d["gap_lo"] = d.padi_lo_e - d.padi_e
    return d


SPECS = {
    "① 지속성만 (기준선)":                 ["dd_prod_early"],
    "② +PADI (현 최선 2변수)":             ["dd_prod_early", "padi_e"],
    "③ +PADI +중국비중 (탐색 1위)":         ["dd_prod_early", "padi_e", "cn_share_e"],
    "─ E1 이중압박 상호작용 ─":            None,
    "  +PADI×중국비중":                    ["dd_prod_early", "padi_e", "cn_share_e", "padi_x_cn"],
    "  +PADI×중국증가율":                  ["dd_prod_early", "padi_e", "cn_growth_e", "padi_x_cng"],
    "─ E2 비선형 ─":                       None,
    "  PADI 제곱항 추가":                  ["dd_prod_early", "padi_e", "padi_sq"],
    "  PADI 순위변환":                     ["dd_prod_early", "padi_rank"],
    "  PADI 윈저화(5/95)":                 ["dd_prod_early", "padi_win"],
    "─ E3 임계효과 ─":                     None,
    "  PADI>1 더미":                       ["dd_prod_early", "padi_over1"],
    "  PADI + PADI>1 더미":                ["dd_prod_early", "padi_e", "padi_over1"],
    "─ E4 층화 ─":                         None,
    "  저가치층 PADI":                     ["dd_prod_early", "padi_lo_e"],
    "  전체 + 층간격차":                    ["dd_prod_early", "padi_e", "gap_lo"],
    "─ E5 중분류 내 상대화 ─":              None,
    "  중분류내 PADI":                     ["dd_prod_early", "padi_wm"],
    "  중분류내 PADI + 중국":               ["dd_prod_early", "padi_wm", "cn_wm"],
    "─ 참고: 현행 모형 ─":                  None,
    "  PADI + 중국증가율 (지속성 없음)":     ["padi_e", "cn_growth_e"],
}

log("[LOCO-CV R² — 분할연도 7개 평균 / 최소]")
log(f"  {'사양':34s} {'DV=생산':>18s} {'DV=출하':>18s}")
log(f"  {'':34s} {'평균':>8s}{'최소':>10s} {'평균':>8s}{'최소':>10s}")
rows = []
for name, cols in SPECS.items():
    if cols is None:
        log(f"  {name}")
        continue
    cells = []
    for tag in ("prod", "ship"):
        ycol = f"dd_{tag}_late"
        vals = []
        for s in SPLITS:
            d = prep(ALL[ALL.split == s])
            d = d.dropna(subset=[ycol] + cols)
            if len(d) < 15 or d.mid.nunique() < 6: continue
            X = d[cols].values.astype(float)
            if np.any(np.std(X, axis=0) < 1e-12): continue
            r = loco_r2(X, d[ycol].values.astype(float), d.mid.values)
            if not np.isnan(r): vals.append(r)
        cells.append((np.mean(vals) if vals else np.nan,
                      np.min(vals) if vals else np.nan, len(vals)))
    log(f"  {name:34s} {cells[0][0]:+8.3f}{cells[0][1]:+10.3f} "
        f"{cells[1][0]:+8.3f}{cells[1][1]:+10.3f}")
    rows.append({"spec": name.strip(), "cols": "+".join(cols),
                 "prod_mean": cells[0][0], "prod_min": cells[0][1],
                 "ship_mean": cells[1][0], "ship_min": cells[1][1]})

R = pd.DataFrame(rows)
R.to_csv(os.path.join(FUS, "model_extend_results.csv"), index=False, encoding="utf-8-sig")

log("\n[판정 — 지속성 기준선 대비 증분]")
base = R[R.spec.str.startswith("① ")].iloc[0]
R["d_ship"] = R.ship_mean - base.ship_mean
R["d_prod"] = R.prod_mean - base.prod_mean
top = R.sort_values("ship_mean", ascending=False).head(6)
for r in top.itertuples():
    log(f"  {r.spec:34s} 출하 LOCO {r.ship_mean:+.3f} (기준선 대비 {r.d_ship:+.3f})")
log(f"\n  기준선(지속성만) 출하 LOCO = {base.ship_mean:+.3f}")
log(f"  개선이 있는 사양 수: {(R.d_ship > 0.01).sum()}/{len(R)}")
log(f"\n저장 → {FUS}/model_extend_results.csv")
