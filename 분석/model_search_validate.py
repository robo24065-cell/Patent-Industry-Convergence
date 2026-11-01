# -*- coding: utf-8 -*-
"""
탐색 결과 검증 — 승자의 저주 보정 + PADI 증분값
────────────────────────────────────────────────────────────────
탐색 결과:
  1위  ship ~ padi_e + cn_share_e + dd_prod_early   LOCO평균 +0.534
  14위 ship ~ dd_prod_early (단독)                   LOCO평균 +0.482
  현행 prod ~ padi_e + cn_growth_e                   LOCO평균 +0.057 (350개 중 105위)

여기서 반드시 물어야 할 두 가지:

  [Q1] 350개를 뒤져 얻은 +0.534 는 진짜인가, 탐색이 만든 거품인가?
       → 종속변수를 **클러스터 단위로 순열**한 가짜 데이터에 동일 탐색을 반복해
         '탐색 최고 LOCO-R²'의 귀무분포를 만든다. 클러스터 단위 순열은
         중분류 내 상관구조를 보존하므로 보수적이다.

  [Q2] PADI 는 '전반기 낙폭'이라는 지속성 기준선 위에 무엇을 더하는가?
       → 지속성만 쓴 모형이 이미 LOCO +0.482 다. 정직한 질문은
         "PADI 가 예측력이 있는가"가 아니라 "**지속성을 알고도 PADI 가 더 주는가**"이다.
         이것이 이 연구의 진짜 기준선이며, 지금까지 우리는 이 기준선과 겨루지 않았다.
"""
import os, json, itertools, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
SPLITS = list(range(2013, 2020))
rng = np.random.default_rng(777)
log = lambda m: print(m, flush=True)
out = {}

ALL = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str, "k3": str, "mid": str})
CAND = ["padi_e", "padi_trend_e", "padi_vol_e", "padi_last_e",
        "cn_share_e", "cn_growth_e", "kr_share_e", "kr_rel_e", "cn_trend_e", "dd_prod_early"]
SPECS = [c for r in (1, 2, 3) for c in itertools.combinations(CAND, r)]


def loco_r2(X, y, grp):
    uq = np.unique(grp)
    pred = np.full(len(y), np.nan)
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


def search_max(perm_y=None):
    """전 사양 탐색 후 '분할연도 평균 LOCO-R²'의 최댓값을 반환."""
    best = -np.inf
    rec = {}
    for tag in ("prod", "ship"):
        ycol = f"dd_{tag}_late"
        for sp in SPECS:
            cols = list(sp); vals = []
            for s in SPLITS:
                d = ALL[ALL.split == s].dropna(subset=[ycol] + cols)
                if len(d) < 15 or d.mid.nunique() < 6: vals = []; break
                y = d[ycol].values.astype(float)
                if perm_y is not None:
                    y = perm_y[(tag, s)][d.index.values - d.index.values.min()] \
                        if False else perm_y[(tag, s)][:len(d)]
                r = loco_r2(d[cols].values.astype(float), y, d.mid.values)
                if np.isnan(r): vals = []; break
                vals.append(r)
            if len(vals) == len(SPLITS):
                m = float(np.mean(vals))
                rec[(tag, "+".join(cols))] = m
                best = max(best, m)
    return best, rec


# ══════════════════════════════════════════════════════════════
log("[Q1] 승자의 저주 — 탐색 최고값의 귀무분포")
obs_best, obs_rec = search_max()
log(f"  관측 최고 LOCO평균 = {obs_best:+.4f}")

NPERM = 200
log(f"  클러스터 단위 순열 {NPERM}회로 귀무분포 생성 (중분류 라벨을 통째로 섞는다)")
nulls = []
for it in range(NPERM):
    py = {}
    for tag in ("prod", "ship"):
        ycol = f"dd_{tag}_late"
        for s in SPLITS:
            d = ALL[ALL.split == s].dropna(subset=[ycol])
            # 중분류 블록 단위 순열 (블록 내부 순서 보존 → 클러스터 상관 유지)
            mids = d.mid.values; uq = np.unique(mids)
            perm = dict(zip(uq, rng.permutation(uq)))
            src = d.set_index(d.index)
            newy = np.empty(len(d))
            pos = {m: np.where(mids == m)[0] for m in uq}
            for m in uq:
                tgt = pos[m]; donor = pos[perm[m]]
                take = np.resize(d[ycol].values[donor], len(tgt))
                newy[tgt] = take
            py[(tag, s)] = newy
    b, _ = search_max(perm_y=py)
    nulls.append(b)
    if (it + 1) % 50 == 0:
        log(f"    {it+1}/{NPERM} … 귀무 최고 중앙 {np.median(nulls):+.4f}")
nulls = np.array(nulls)
pval = float((nulls >= obs_best).mean())
log(f"\n  귀무분포: 중앙 {np.median(nulls):+.4f}  95%분위 {np.percentile(nulls,95):+.4f}  "
    f"최대 {nulls.max():+.4f}")
log(f"  ★ 관측 {obs_best:+.4f} 의 순열 p = {pval:.4f}  "
    f"→ {'탐색 거품 아님 — 진짜 신호' if pval < .05 else '탐색 거품과 구분 불가'}")
out["winners_curse"] = {"obs": obs_best, "null_median": float(np.median(nulls)),
                        "null_p95": float(np.percentile(nulls, 95)), "p": pval}

# ══════════════════════════════════════════════════════════════
log("\n[Q2] ★ 지속성 기준선 위에서 PADI 의 증분값")
log("  기준선: 후반기 낙폭 ~ 전반기 낙폭 (지속성만)")
log("  이것이 진짜 겨뤄야 할 상대다. 지금까지 우리는 상수모형과만 겨뤘다.\n")

rows = []
for tag in ("prod", "ship"):
    ycol = f"dd_{tag}_late"
    for s in SPLITS:
        d = ALL[ALL.split == s].dropna(subset=[ycol, "padi_e", "cn_share_e", "dd_prod_early"])
        y = d[ycol].values.astype(float); g = d.mid.values
        base = loco_r2(d[["dd_prod_early"]].values.astype(float), y, g)
        pad = loco_r2(d[["dd_prod_early", "padi_e"]].values.astype(float), y, g)
        full = loco_r2(d[["dd_prod_early", "padi_e", "cn_share_e"]].values.astype(float), y, g)
        cur = loco_r2(d[["padi_e", "cn_growth_e"]].values.astype(float), y, g)
        f = smf.ols(f"{ycol} ~ dd_prod_early + padi_e", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.mid})
        rows.append({"dv": tag, "split": s, "loco_base": base, "loco_padi": pad,
                     "loco_full": full, "loco_current": cur, "d_padi": pad - base,
                     "beta_padi": f.params.padi_e, "p_padi": f.pvalues.padi_e})
V = pd.DataFrame(rows)
log(f"  {'DV':5s} {'분할':>5s} {'지속성만':>9s} {'+PADI':>9s} {'증분':>8s} {'+중국비중':>9s} "
    f"{'현행모형':>9s} {'β_PADI':>9s} {'p':>7s}")
for r in V.itertuples():
    log(f"  {r.dv:5s} {r.split:5d} {r.loco_base:+9.3f} {r.loco_padi:+9.3f} {r.d_padi:+8.3f} "
        f"{r.loco_full:+9.3f} {r.loco_current:+9.3f} {r.beta_padi:+9.2f} {r.p_padi:7.4f}")
log(f"\n  평균: 지속성만 {V.loco_base.mean():+.3f} → +PADI {V.loco_padi.mean():+.3f} "
    f"(증분 {V.d_padi.mean():+.3f}) → +중국비중 {V.loco_full.mean():+.3f}")
log(f"  PADI 증분이 양수인 분할연도: {(V.d_padi>0).sum()}/{len(V)}")
log(f"  β_PADI 부호 음수 비율: {(V.beta_padi<0).mean():.0%}  |  p<0.05: {(V.p_padi<.05).sum()}/{len(V)}")
out["increment"] = {"base": float(V.loco_base.mean()), "padi": float(V.loco_padi.mean()),
                    "delta": float(V.d_padi.mean()), "full": float(V.loco_full.mean()),
                    "current": float(V.loco_current.mean()),
                    "pos_frac": float((V.d_padi > 0).mean())}

log("\n[Q2-b] 증분의 순열검정 — PADI 를 산업 간 무작위 재배치")
for tag in ("prod", "ship"):
    ycol = f"dd_{tag}_late"
    obs = V[V.dv == tag].d_padi.mean()
    nd = []
    for _ in range(1000):
        ds = []
        for s in SPLITS:
            d = ALL[ALL.split == s].dropna(subset=[ycol, "padi_e", "dd_prod_early"]).copy()
            d["padi_e"] = rng.permutation(d.padi_e.values)
            y = d[ycol].values.astype(float); g = d.mid.values
            b = loco_r2(d[["dd_prod_early"]].values.astype(float), y, g)
            p = loco_r2(d[["dd_prod_early", "padi_e"]].values.astype(float), y, g)
            ds.append(p - b)
        nd.append(np.mean(ds))
    nd = np.array(nd)
    pv = float((nd >= obs).mean())
    log(f"  {tag}: 관측 증분 {obs:+.4f} | 귀무 중앙 {np.median(nd):+.4f} "
        f"95%분위 {np.percentile(nd,95):+.4f} | ★p={pv:.4f} "
        f"{'유의' if pv<.05 else '비유의'}")
    out[f"perm_incr_{tag}"] = {"obs": float(obs), "p": pv}

V.to_csv(os.path.join(FUS, "model_search_validate.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(FUS, "model_search_validate.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")
