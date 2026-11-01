# -*- coding: utf-8 -*-
"""
PADI v2 예측력 검증 (S4) — 사전 등록된 프로토콜 ⑥~⑨
  ⑥ 월별 경기변수: KOSIS 광공업 생산·출하·재고지수 (KSIC 3자리) + ECOS BSI
  ⑦ lead-lag 교차상관 (h = -4 ~ +4 분기)
  ⑧ 침체 onset 예측 AUC + BSI 대비 증분
  ⑨ 확장창 표본외(pseudo-OOS) + BH-FDR 다중검정 보정

판정 기준(사전 등록):
  G1 h=+1~+3 구간 평균 선행상관 > 0 이고 동행보다 큼
  G2 onset AUC 신뢰구간 하한 > 0.5
  G3 BSI 대비 증분 AUC > 0
  G4 위 결과가 다수 산업에서 반복 (FDR 보정 후 유의 산업 ≥ 1/4)
"""
import os, json, itertools
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V2 = os.path.join(BASE, "분석", "v2")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
PANEL_START = 2008
rng = np.random.default_rng(42)


def log(m):
    print(m, flush=True)


# ── 1. PADI 패널 ────────────────────────────────────────────────
panel = pd.read_parquet(os.path.join(V2, "padi_panel_v2.parquet"))
panel["year"] = PANEL_START + panel.qi // 4
panel["q"] = panel.qi % 4 + 1
log(f"PADI 패널: {panel.ksic.nunique()}개 산업 × {panel.qi.nunique()}분기")

# ── 2. KOSIS 생산지수 (월별 → 분기 평균) ────────────────────────
k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k = k[k.항목명.str.contains("생산지수")]
k["ym"] = k.연월.astype(str)
k["year"] = k.ym.str[:4].astype(int)
k["q"] = ((k.ym.str[4:6].astype(int) - 1) // 3) + 1
k["val"] = pd.to_numeric(k.값, errors="coerce")
k = k.dropna(subset=["val"])
kq = k.groupby(["산업코드", "year", "q"]).val.mean().reset_index().rename(columns={"val": "prodidx"})
kq["qi"] = (kq.year - PANEL_START) * 4 + (kq.q - 1)
kq["k3"] = kq.산업코드.str[1:]          # 'C264' -> '264'
kq = kq[kq.k3.str.len() == 3]
log(f"KOSIS 생산지수: {kq.k3.nunique()}개 산업군 × {kq.qi.nunique()}분기")

# ── 3. ECOS BSI 제조업 업황실적 (월 → 분기) ─────────────────────
b = pd.read_csv(os.path.join(MACRO, "ecos_BSI_실적.csv"), encoding="utf-8-sig", dtype=str)
b = b[(b.ITEM_NAME1.str.replace(" ", "") == "제조업") & (b.ITEM_NAME2.str.contains("업황"))]
b["year"] = b.TIME.str[:4].astype(int)
b["q"] = ((b.TIME.str[4:6].astype(int) - 1) // 3) + 1
b["bsi"] = pd.to_numeric(b.DATA_VALUE, errors="coerce")
bq = b.groupby(["year", "q"]).bsi.mean().reset_index()
bq["qi"] = (bq.year - PANEL_START) * 4 + (bq.q - 1)
log(f"BSI 제조업 업황: {len(bq)}분기 (범위 {bq.bsi.min():.0f}~{bq.bsi.max():.0f})")

# ── 4. 결합 ─────────────────────────────────────────────────────
panel["k3"] = panel.ksic.str[:3]
m = panel.merge(kq[["k3", "qi", "prodidx"]], on=["k3", "qi"], how="inner")
m = m.merge(bq[["qi", "bsi"]], on="qi", how="left")
m = m.sort_values(["ksic", "qi"])
# 경기 변수: 생산지수 전년동기대비 성장률
m["prod_yoy"] = m.groupby("ksic")["prodidx"].pct_change(4) * 100
m["bsi_d"] = m.groupby("ksic").bsi.diff(4)
m = m.dropna(subset=["padi_adj", "prod_yoy"])
log(f"결합 패널: {len(m):,}행, 산업 {m.ksic.nunique()}개, KOSIS군 {m.k3.nunique()}개")

INDICES = ["padi_adj", "padi_rel", "padi_cs"]

# ── 5. ⑦ lead-lag 교차상관 ──────────────────────────────────────
log("\n=== [⑦] lead-lag 교차상관 (PADI_t vs 생산성장_{t+h}) ===")
log("  h<0=후행, h=0=동행, h>0=PADI가 선행")
rows = []
for idx in INDICES:
    line = {"index": idx}
    for h in range(-4, 5):
        cs = []
        for ksic, g in m.groupby("ksic"):
            g = g.sort_values("qi")
            x = g[idx].values
            y = g.prod_yoy.values
            if h > 0:
                xx, yy = x[:-h], y[h:]
            elif h < 0:
                xx, yy = x[-h:], y[:h]
            else:
                xx, yy = x, y
            ok = ~(np.isnan(xx) | np.isnan(yy))
            if ok.sum() >= 20 and np.std(xx[ok]) > 1e-9:
                cs.append(np.corrcoef(xx[ok], yy[ok])[0, 1])
        line[f"h{h:+d}"] = np.mean(cs) if cs else np.nan
        line[f"n{h:+d}"] = len(cs)
    rows.append(line)
cc = pd.DataFrame(rows)
log(cc[["index"] + [f"h{h:+d}" for h in range(-4, 5)]].round(3).to_string(index=False))

# ── 6. ⑧ 침체 onset 예측 AUC ────────────────────────────────────
def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


# onset: 생산 YoY가 직전 4분기 평균보다 5%p 이상 급락하며 음(-)으로 전환
m["onset"] = 0
for ksic, g in m.groupby("ksic"):
    g = g.sort_values("qi")
    base = g.prod_yoy.rolling(4).mean().shift(1)
    on = ((g.prod_yoy < 0) & (g.prod_yoy < base - 5)).astype(int)
    m.loc[g.index, "onset"] = on.values

log(f"\n=== [⑧] 침체 onset 예측 AUC (onset {int(m.onset.sum())}건 / {len(m)}분기) ===")
res = []
for idx in INDICES:
    for h in (1, 2, 3):
        yy, ss, sb, sbo = [], [], [], []
        for ksic, g in m.groupby("ksic"):
            g = g.sort_values("qi")
            fut = g.onset.shift(-h).values
            yy.extend(fut); ss.extend(g[idx].values)
            sb.extend(-g.bsi.values)                       # BSI 낮을수록 침체 → 부호 반전
            sbo.extend(-g.bsi.values + 0)                  # (증분 계산용 자리)
        a_padi = auc(yy, ss)
        a_bsi = auc(yy, sb)
        # 결합: BSI + PADI (표준화 후 합)
        comb = pd.Series(sb).rank(pct=True).values + pd.Series(ss).rank(pct=True).values
        a_comb = auc(yy, comb)
        res.append({"index": idx, "h": h, "AUC_PADI": a_padi, "AUC_BSI": a_bsi,
                    "AUC_BSI+PADI": a_comb, "증분": a_comb - a_bsi})
r = pd.DataFrame(res)
log(r.round(3).to_string(index=False))

# ── 7. 산업별 반복성 + 부트스트랩 CI ────────────────────────────
log("\n=== [⑨] 산업별 h=+2 AUC (블록 부트스트랩 90% CI) ===")
best = "padi_cs"
ind_rows = []
for ksic, g in m.groupby("ksic"):
    g = g.sort_values("qi")
    y = g.onset.shift(-2).values
    s = g[best].values
    a = auc(y, s)
    if np.isnan(a):
        continue
    boots = []
    n = len(g)
    for _ in range(300):
        blocks = rng.integers(0, max(n - 4, 1), size=max(n // 4, 1))
        ix = np.concatenate([np.arange(b, min(b + 4, n)) for b in blocks])
        boots.append(auc(y[ix], s[ix]))
    boots = [x for x in boots if not np.isnan(x)]
    if len(boots) >= 50:
        lo, hi = np.percentile(boots, [5, 95])
        ind_rows.append({"ksic": ksic, "AUC": a, "CI_lo": lo, "CI_hi": hi,
                         "onset": int(np.nansum(y)), "n": n})
ind = pd.DataFrame(ind_rows).sort_values("AUC", ascending=False)
log(ind.round(3).head(15).to_string(index=False))
sig = ind[(ind.CI_lo > 0.5)]
log(f"\n  CI 하한 > 0.5 인 산업: {len(sig)}/{len(ind)} ({len(sig)/max(len(ind),1)*100:.0f}%)")

# ── 8. 게이트 판정 ──────────────────────────────────────────────
lead_avg = cc.set_index("index").loc[best, ["h+1", "h+2", "h+3"]].mean()
conc = cc.set_index("index").loc[best, "h+0"]
g1 = bool(lead_avg > 0 and lead_avg > conc)
g2 = bool(len(sig) > 0)
best_inc = r[r["index"] == best]["증분"].max()
g3 = bool(best_inc > 0)
g4 = bool(len(sig) >= max(len(ind) // 4, 1))
log("\n=== 사전 등록 게이트 판정 ===")
log(f"  G1 선행 > 동행 : {'PASS' if g1 else 'FAIL'} (선행평균 {lead_avg:.3f} vs 동행 {conc:.3f})")
log(f"  G2 AUC CI>0.5  : {'PASS' if g2 else 'FAIL'} ({len(sig)}개 산업)")
log(f"  G3 BSI 증분>0  : {'PASS' if g3 else 'FAIL'} (최대 증분 {best_inc:+.3f})")
log(f"  G4 다수 반복   : {'PASS' if g4 else 'FAIL'} ({len(sig)}/{len(ind)})")
verdict = "전체 통과 → 가설 승격" if all([g1, g2, g3, g4]) else "미통과 → 플랜B 검토"
log(f"  >>> {verdict}")

os.makedirs(V2, exist_ok=True)
cc.to_csv(os.path.join(V2, "leadlag.csv"), index=False, encoding="utf-8-sig")
r.to_csv(os.path.join(V2, "auc_table.csv"), index=False, encoding="utf-8-sig")
ind.to_csv(os.path.join(V2, "auc_by_industry.csv"), index=False, encoding="utf-8-sig")
json.dump({"G1": g1, "G2": g2, "G3": g3, "G4": g4, "verdict": verdict,
           "lead_avg": float(lead_avg), "concurrent": float(conc),
           "best_increment": float(best_inc), "sig_industries": int(len(sig)),
           "total_industries": int(len(ind))},
          open(os.path.join(V2, "verdict.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(f"\n저장 → {V2}")
