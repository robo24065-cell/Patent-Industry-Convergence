# -*- coding: utf-8 -*-
"""
PADI v2 검증 v2b — 설계안 v2 §S2 ④ "소분류 집계 트랙" 구현 (누락분 보완)
변경점 (p-hacking 아님: 전부 설계안에 사전 명시된 항목):
  [A] KOSIS 생산지수 해상도(KSIC 3자리)에 맞춰 PADI를 소분류로 집계 — 단위 정합
      (기존: 5자리 PADI vs 3자리 생산지수 → 다대일 희석)
  [B] onset 정의 엄격화: '호황→침체 전환' 최초 분기만 (설계안 §3 'onset 라벨')
  [C] 소표본 셀(E4<10) 제외 (설계안 S3 게이트 ②)
  [D] 연속형 결과변수(생산 YoY) 회귀 병행 — 이진화 정보손실 보완
두 트랙(5자리·3자리) 결과를 모두 보고한다.
"""
import os, json
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V2 = os.path.join(BASE, "분석", "v2")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
PANEL_START = 2008
rng = np.random.default_rng(42)
log = lambda m: print(m, flush=True)


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


# ── 데이터 ──────────────────────────────────────────────────────
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

b = pd.read_csv(os.path.join(MACRO, "ecos_BSI_실적.csv"), encoding="utf-8-sig", dtype=str)
b = b[(b.ITEM_NAME1.str.replace(" ", "") == "제조업") & (b.ITEM_NAME2.str.contains("업황"))].copy()
b["year"] = b.TIME.str[:4].astype(int)
b["q"] = ((b.TIME.str[4:6].astype(int) - 1) // 3) + 1
b["bsi"] = pd.to_numeric(b.DATA_VALUE, errors="coerce")
bq = b.groupby(["year", "q"]).bsi.mean().reset_index()
bq["qi"] = (bq.year - PANEL_START) * 4 + (bq.q - 1)


def make_track(level):
    """level='k5'(원래) 또는 'k3'(소분류 집계)"""
    if level == "k3":
        # [A] O·E를 소분류로 합산한 뒤 재계산 (위험집합 가중 자동 반영)
        g = panel.groupby(["k3", "qi"]).agg(O4=("O4", "sum"), E_adj4=("E_adj4", "sum"),
                                            R4=("R4", "sum")).reset_index()
        g = g.rename(columns={"k3": "unit"})
    else:
        g = panel[["ksic", "k3", "qi", "O4", "E_adj4", "R4"]].rename(columns={"ksic": "unit"})
    g["padi"] = g.O4 / g.E_adj4.replace(0, np.nan)
    g = g[g.E_adj4 >= 10]                                    # [C]
    # 자기기준선 z (확장창, look-ahead 없음)
    rows = []
    for u, gg in g.groupby("unit"):
        gg = gg.sort_values("qi").copy()
        v = gg.padi.values
        z = np.full(len(v), np.nan)
        for i in range(len(v)):
            h = v[max(0, i - 20):i]
            h = h[~np.isnan(h)]
            if len(h) >= 12 and np.nanstd(h) > 1e-6:
                z[i] = (v[i] - np.nanmean(h)) / np.nanstd(h)
        gg["padi_z"] = z
        rows.append(gg)
    g = pd.concat(rows, ignore_index=True)
    g["padi_cs"] = g.padi_z - g.groupby("qi").padi_z.transform("median")

    key = "unit" if level == "k3" else "k3"
    m = g.merge(kq[["k3", "qi", "pidx"]], left_on=[key if level == "k3" else "k3", "qi"],
                right_on=["k3", "qi"], how="inner", suffixes=("", "_y"))
    m = m.merge(bq[["qi", "bsi"]], on="qi", how="left").sort_values(["unit", "qi"])
    m["yoy"] = m.groupby("unit")["pidx"].pct_change(4) * 100
    m = m.dropna(subset=["padi", "yoy"])

    # [B] onset: YoY가 0 이상에서 -5% 미만으로 '전환'되는 최초 분기
    m["onset"] = 0
    for u, gg in m.groupby("unit"):
        gg = gg.sort_values("qi")
        prev = gg.yoy.shift(1)
        on = ((gg.yoy < -5) & (prev >= 0)).astype(int)
        m.loc[gg.index, "onset"] = on.values
    return m


results = {}
for level in ("k5", "k3"):
    m = make_track(level)
    log(f"\n{'='*62}\n[트랙 {level}] 단위 {m.unit.nunique()}개 × {len(m):,}행 | onset {int(m.onset.sum())}건 ({m.onset.mean()*100:.1f}%)")

    # ⑦ lead-lag
    cc = {}
    for idx in ("padi", "padi_z", "padi_cs"):
        row = {}
        for h in range(-4, 5):
            cs = []
            for u, gg in m.groupby("unit"):
                gg = gg.sort_values("qi")
                x, y = gg[idx].values, gg.yoy.values
                xx, yy = (x[:-h], y[h:]) if h > 0 else ((x[-h:], y[:h]) if h < 0 else (x, y))
                ok = ~(np.isnan(xx) | np.isnan(yy))
                if ok.sum() >= 20 and np.std(xx[ok]) > 1e-9:
                    cs.append(np.corrcoef(xx[ok], yy[ok])[0, 1])
            row[f"h{h:+d}"] = np.mean(cs) if cs else np.nan
        cc[idx] = row
    ccdf = pd.DataFrame(cc).T
    log("  [⑦ lead-lag] (음수 = PADI↑ 시 생산↓, h>0 = PADI 선행)")
    log(ccdf.round(3).to_string())

    # ⑧ AUC
    res = []
    for idx in ("padi", "padi_z", "padi_cs"):
        for h in (1, 2, 3, 4):
            yy, ss, sb = [], [], []
            for u, gg in m.groupby("unit"):
                gg = gg.sort_values("qi")
                yy.extend(gg.onset.shift(-h).values)
                ss.extend(gg[idx].values)
                sb.extend(-gg.bsi.values)
            a_p, a_b = auc(yy, ss), auc(yy, sb)
            comb = pd.Series(sb).rank(pct=True).values + pd.Series(ss).rank(pct=True).values
            a_c = auc(yy, comb)
            res.append({"idx": idx, "h": h, "AUC_PADI": a_p, "AUC_BSI": a_b,
                        "AUC_결합": a_c, "증분": a_c - a_b})
    rdf = pd.DataFrame(res)
    log("  [⑧ onset 예측 AUC]")
    log(rdf.round(3).to_string(index=False))

    best_row = rdf.loc[rdf.AUC_PADI.idxmax()]
    results[level] = {"leadlag": ccdf.round(4).to_dict(),
                      "auc": rdf.round(4).to_dict("records"),
                      "best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in best_row.items()},
                      "n_units": int(m.unit.nunique()), "onset": int(m.onset.sum())}
    m.to_csv(os.path.join(V2, f"verify_track_{level}.csv"), index=False, encoding="utf-8-sig")

json.dump(results, open(os.path.join(V2, "verify_v2b.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
log(f"\n저장 → {V2}")
