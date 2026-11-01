# -*- coding: utf-8 -*-
"""
종속변수 해상도 승격 — KSIC 3자리 → 5자리
────────────────────────────────────────────────────────────────
[문제] 기존 종속변수 dd_late 는 KOSIS 광공업생산지수에서 만들었는데, 이 지수는
       소분류(3자리)까지만 발표된다. 그래서 세세분류 39개의 종속변수 고유값이 23개뿐이고
       k3 내 분산이 정확히 0이었다. 결과적으로
         · n=39 는 n=23 을 임의 가중한 것에 불과했고
         · 산업구조 6지표(5자리 해상도)와 PADI(5자리)를 분리할 자유도가 없었다

[해법] 광업제조업조사는 출하액·부가가치를 **세세분류 5자리**로 연 단위 발표한다.
       488개 산업 × 2008~2024(2010 결측, 16개 연도) 보유.
       이걸 종속변수로 쓰면 39개 산업이 39개 고유값을 갖는다.

[대가와 대응]
  (1) 연 단위 → 최대낙폭이 거칠어진다. 월별 8년(96점) → 연별 8점.
  (2) 출하액은 명목값 → 물가상승이 하락을 가린다.
      대응: 제조업 전체 대비 **상대 출하액**(점유율) 낙폭을 병행 산출한다. 점유율은
           공통 물가효과가 소거되므로 명목/실질 문제에서 자유롭다.
  (3) 정의가 달라졌으므로 기존 종속변수와의 상관을 반드시 확인한다.
      상관이 낮으면 같은 현상을 재는 것이 아니므로 교체 자체를 재검토해야 한다.
"""
import os, glob, re, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SRC  = os.path.join(BASE, "공모전데이터", "외부데이터", "산업분류별")
FUS  = os.path.join(BASE, "분석", "fusion")
SPLIT = 2017
log = lambda m: print(m, flush=True)

# ══════════════════════════════════════════════════════════════
log("[1] 광업제조업조사 적재 (전국, 세세분류 5자리)")
recs = []
for f in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
    d = pd.read_csv(f, encoding="cp949", dtype=str)
    d = d[d["시도별"].astype(str).str.strip() == "전국"]
    ycols = [c for c in d.columns if "년" in c]
    for _, r in d.iterrows():
        code = str(r["[B]산업별"]).strip()
        if len(code) != 6 or not code[1:].isdigit():
            continue
        item = str(r["항목"]).strip()
        for yc in ycols:
            yr = int(re.match(r"(\d{4})", yc).group(1))
            v = str(r[yc]).replace(",", "").strip()
            if v in ("", "-", "X", "x", "nan", "None"):
                continue
            try:
                recs.append((code[1:], yr, item, float(v)))
            except ValueError:
                pass
R = pd.DataFrame(recs, columns=["ksic", "year", "item", "val"]).drop_duplicates(
    ["ksic", "year", "item"], keep="first")
log(f"  레코드 {len(R):,} | 산업 {R.ksic.nunique()} | 연도 {sorted(R.year.unique())}")

W = R.pivot_table(index=["ksic", "year"], columns="item", values="val").reset_index()
W.columns.name = None
ship = [c for c in W.columns if "출하액" in c][0]
va   = [c for c in W.columns if "부가가치" in c][0]
W = W.rename(columns={ship: "shipment", va: "value_added"})
log(f"  패널 {len(W):,}행 | 출하액 결측 {W.shipment.isna().mean()*100:.1f}% "
    f"| 부가가치 결측 {W.value_added.isna().mean()*100:.1f}%")

# 제조업(10~34) 한정 + 제조업 전체 대비 점유율
W = W[W.ksic.str[:2].astype(int).between(10, 34)].copy()
tot = W.groupby("year").shipment.transform("sum")
W["ship_share"] = W.shipment / tot * 100
totv = W.groupby("year").value_added.transform("sum")
W["va_share"] = W.value_added / totv * 100
log(f"  제조업 한정 {len(W):,}행 | 산업 {W.ksic.nunique()}")


# ══════════════════════════════════════════════════════════════
def maxdd(s):
    """최대낙폭(%) — 누적최고 대비 최저 하락률. 값이 음수일수록 깊은 침체."""
    s = np.asarray(s, float)
    s = s[~np.isnan(s)]
    if len(s) < 3 or np.nanmax(s) <= 0:
        return np.nan
    peak = np.maximum.accumulate(s)
    return float(np.min(s / peak - 1.0) * 100)


log(f"\n[2] 기간분할 최대낙폭 산출 (전반기 <{SPLIT}, 후반기 >={SPLIT})")
rows = []
for k, g in W.groupby("ksic"):
    g = g.sort_values("year")
    e, l = g[g.year < SPLIT], g[g.year >= SPLIT]
    if len(e) < 4 or len(l) < 4:
        continue
    rows.append({
        "ksic": k,
        "dd5_late":        maxdd(l.shipment),      "dd5_early":      maxdd(e.shipment),
        "dd5s_late":       maxdd(l.ship_share),    "dd5s_early":     maxdd(e.ship_share),
        "dd5v_late":       maxdd(l.value_added),   "dd5v_early":     maxdd(e.value_added),
        "dd5vs_late":      maxdd(l.va_share),      "dd5vs_early":    maxdd(e.va_share),
        "n_yr_early": len(e), "n_yr_late": len(l),
    })
DV = pd.DataFrame(rows)
log(f"  산출 {len(DV)}산업")
for c in ("dd5_late", "dd5s_late", "dd5v_late", "dd5vs_late"):
    log(f"    {c:12s} 평균 {DV[c].mean():+7.2f}  중앙 {DV[c].median():+7.2f}  "
        f"범위 [{DV[c].min():+7.2f}, {DV[c].max():+7.2f}]  결측 {DV[c].isna().sum()}")

# ══════════════════════════════════════════════════════════════
log("\n[3] ★ 기존 종속변수와의 정합성 — 같은 현상을 재고 있는가")
P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
M = P.merge(DV, on="ksic", how="left")
log(f"  39개 대상 중 매칭 {M.dd5_late.notna().sum()}개")
for c in ("dd5_late", "dd5s_late", "dd5v_late", "dd5vs_late"):
    s = M.dropna(subset=[c, "dd_late"])
    if len(s) > 5:
        log(f"    corr(기존 dd_late, {c:11s}) = {s.dd_late.corr(s[c]):+.3f}  "
            f"(Spearman {s.dd_late.corr(s[c], method='spearman'):+.3f}, n={len(s)})")

log("\n[4] ★ 핵심 — 새 종속변수의 k3 내 분산이 살아났는가")
for c in ("dd_late", "dd5_late", "dd5s_late", "dd5v_late", "dd5vs_late"):
    s = M.dropna(subset=[c])
    nuq = s[c].nunique()
    wv = s.groupby("k3")[c].var(ddof=0).fillna(0).sum()
    multi = s.groupby("k3").size()
    multi = multi[multi > 1]
    log(f"    {c:12s} n={len(s):3d} 고유값={nuq:3d}  k3내 총분산={wv:12.4f}  "
        f"복수산업 k3 {len(multi)}개")

log("\n[5] 전체 제조업으로 확장했을 때의 표본 크기")
log(f"  광업제조업조사 제조업 세세분류: {W.ksic.nunique()}개")
log(f"  전·후반기 4년 이상 확보: {len(DV)}개")
log(f"  → 융합데이터가 50개 세세분류만 담고 있어 현재 PADI 산출 가능은 39개.")
log(f"    특허청 전체 분류결과를 쓰면 종속변수 쪽 상한은 {len(DV)}개까지 열려 있다.")

DV.to_csv(os.path.join(FUS, "dv5_maxdd.csv"), index=False, encoding="utf-8-sig")
W.to_csv(os.path.join(FUS, "mm_panel5.csv"), index=False, encoding="utf-8-sig")
M.to_csv(os.path.join(FUS, "predictive_agg_dv5.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {FUS}  (dv5_maxdd.csv, mm_panel5.csv, predictive_agg_dv5.csv)")
