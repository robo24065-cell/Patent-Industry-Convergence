# -*- coding: utf-8 -*-
"""
라벨잡음 민감도 — 분류 신뢰도가 β_PADI 에 실제로 무엇을 하는가
────────────────────────────────────────────────────────────────
앞선 검증에서 산업별 라벨 신뢰도(융합데이터 ↔ 공식 IPC 연계표 일치율)가
0.1% ~ 92.0% 로 크게 달랐다. 그렇다면 검정해야 할 것은 두 가지다.

  [검정A] 라벨이 지저분한 산업을 빼면 결과가 어떻게 되는가?
          → 강해지면 라벨잡음이 우리를 0쪽으로 밀고 있었다는 뜻(보수적 추정)
          → 약해지면 잡음 산업이 결과를 만들고 있었다는 뜻(치명적)

  [검정B] 라벨이 깨끗한 산업에서 PADI 효과가 더 강한가? (상호작용)
          측정오차 감쇠 이론의 직접 예측이다. 이게 맞으면 감쇠편의가 실재한다는
          이론적 가정이 아니라 데이터로 확인된 사실이 된다.

정직 원칙: 어느 쪽이 나오든 그대로 보고한다. 사전에 방향을 정해두지 않는다.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
rng  = np.random.default_rng(77)
log  = lambda m: print(m, flush=True)
out  = {}

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
P = P.dropna(subset=["dd_late", "padi_e", "cn_growth_e"])
R = pd.read_csv(os.path.join(FUS, "cls_cc_target39.csv"), encoding="utf-8-sig",
                dtype={"ksic": str})
D = P.merge(R.rename(columns={"n": "n_pat_cmp"}), on="ksic", how="left")
log(f"결합 {len(D)}산업 | 라벨신뢰도 결측 {D.cc_agree.isna().sum()}")
D = D.dropna(subset=["cc_agree"])
log(f"  신뢰도 분포: 최소 {D.cc_agree.min():.3f} 중앙 {D.cc_agree.median():.3f} "
    f"최대 {D.cc_agree.max():.3f}")


def wild_boot(df, formula, var, B=2999):
    """wild cluster bootstrap-t (Rademacher, mid 클러스터). effective_n_check.py 와 동일 구현."""
    f = smf.ols(formula, data=df).fit()
    t_obs = f.params[var] / f.bse[var]
    terms = [t for t in formula.split("~")[1].split("+") if var not in t]
    lhs = formula.split("~")[0].strip()
    f0 = smf.ols(lhs + "~" + "+".join(terms), data=df).fit()
    resid0, fitted0 = f0.resid.values, f0.fittedvalues.values
    g = df["mid"].values
    uq = np.unique(g)
    ts = []
    for _ in range(B):
        wmap = dict(zip(uq, rng.choice([-1.0, 1.0], size=len(uq))))
        d2 = df.copy()
        d2[lhs] = fitted0 + resid0 * np.array([wmap[x] for x in g])
        try:
            fb = smf.ols(formula, data=d2).fit()
            ts.append(fb.params[var] / fb.bse[var])
        except Exception:
            pass
    ts = np.array(ts)
    return t_obs, float((np.abs(ts) >= abs(t_obs)).mean())


def fit(df, tag):
    if df.mid.nunique() < 4 or len(df) < 8:
        log(f"  {tag:26s} 표본 부족 (n={len(df)}, G={df.mid.nunique()}) — 건너뜀")
        return None
    f = smf.ols("dd_late ~ padi_e + cn_growth_e", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df.mid})
    t, wp = wild_boot(df, "dd_late ~ padi_e + cn_growth_e", "padi_e")
    log(f"  {tag:26s} n={len(df):3d} G={df.mid.nunique():2d} "
        f"β_PADI={f.params.padi_e:+8.2f} R²={f.rsquared:.3f} wild-p={wp:.4f} "
        f"{'유의' if wp < .05 else '비유의'}")
    return {"tag": tag, "n": len(df), "G": int(df.mid.nunique()),
            "beta": float(f.params.padi_e), "r2": float(f.rsquared), "wild_p": wp}


# ══════════════════════════════════════════════════════════════
log("\n[검정A] 라벨 신뢰도 하위 산업을 순차 제거")
log("  가설 방향을 사전에 정하지 않는다. 강해지든 약해지든 그대로 본다.")
rows = []
for k in (0, 3, 5, 8, 10, 13):
    sub = D.nlargest(len(D) - k, "cc_agree") if k else D
    r = fit(sub, f"하위 {k}개 제거")
    if r:
        r["dropped"] = k
        r["min_reliab"] = float(sub.cc_agree.min())
        rows.append(r)
A = pd.DataFrame(rows)
out["testA"] = rows
if len(A) >= 3:
    trend = np.polyfit(A.dropped, np.abs(A.beta), 1)[0]
    log(f"\n  |β| 추세: 하위 1개 제거당 {trend:+.2f}")
    log("  → " + ("라벨이 깨끗해질수록 효과가 커진다 = 라벨잡음이 우리를 0쪽으로 밀고 있었다"
                  if trend > 0 else
                  "라벨이 깨끗해질수록 효과가 작아진다 = 잡음 산업이 결과에 기여하고 있었다"))

log("\n[검정A-2] 임계값 기준 제거 (신뢰도 20%/30% 미만 산업 배제)")
for th in (0.10, 0.20, 0.30):
    fit(D[D.cc_agree >= th], f"신뢰도 ≥ {th:.0%}")

# ══════════════════════════════════════════════════════════════
log("\n[검정B] 라벨 신뢰도 × PADI 상호작용")
log("  감쇠 이론의 직접 예측: 라벨이 깨끗한 산업에서 |β_PADI| 가 더 커야 한다.")
d = D.copy()
d["rel_c"] = d.cc_agree - d.cc_agree.mean()
d["padi_c"] = d.padi_e - d.padi_e.mean()
f = smf.ols("dd_late ~ padi_c * rel_c + cn_growth_e", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": d.mid})
b_i, p_i = f.params["padi_c:rel_c"], f.pvalues["padi_c:rel_c"]
t_i, wp_i = wild_boot(d, "dd_late ~ padi_c + rel_c + padi_c:rel_c + cn_growth_e", "padi_c:rel_c")
log(f"  주효과 β_PADI  = {f.params.padi_c:+8.2f}  (p={f.pvalues.padi_c:.4f})")
log(f"  상호작용       = {b_i:+8.2f}  (p={p_i:.4f}, wild-p={wp_i:.4f})")
log(f"  → 예측 부호는 음수(신뢰도↑ → β 더 음수). 실제 부호: {'음수 ✔' if b_i < 0 else '양수 ✘'}")
out["interaction"] = {"beta": float(b_i), "p": float(p_i), "wild_p": wp_i}

log("\n  [신뢰도 상·하위 절반 분할 추정]")
med = d.cc_agree.median()
fit(d[d.cc_agree >= med], "신뢰도 상위 절반")
fit(d[d.cc_agree < med], "신뢰도 하위 절반")

# ══════════════════════════════════════════════════════════════
log("\n[검정C] 신뢰도 가중 최소제곱 (WLS)")
log("  라벨이 정확한 산업일수록 관측치를 더 신뢰한다.")
w = smf.wls("dd_late ~ padi_e + cn_growth_e", data=d, weights=d.cc_agree).fit(
    cov_type="cluster", cov_kwds={"groups": d.mid})
log(f"  WLS β_PADI = {w.params.padi_e:+.2f} (p={w.pvalues.padi_e:.4f}) "
    f"vs OLS {smf.ols('dd_late ~ padi_e + cn_growth_e', data=d).fit().params.padi_e:+.2f}")
out["wls_beta"] = float(w.params.padi_e)

# ══════════════════════════════════════════════════════════════
log("\n[검정D] k3 집계 실질표본(n=23)에서 동일 검정")
d["cc_w"] = d.cc_agree * d.n_pat_cmp
K = d.groupby("k3").apply(lambda g: pd.Series({
    "dd_late": g.dd_late.iloc[0],
    "padi_e": np.average(g.padi_e, weights=g.n_pat_cmp),
    "cn_growth_e": np.average(g.cn_growth_e, weights=g.n_pat_cmp),
    "cc_agree": np.average(g.cc_agree, weights=g.n_pat_cmp),
    "mid": g.mid.iloc[0]})).reset_index()
log(f"  k3 집계 n={len(K)}")
fit(K, "k3 전체")
fit(K[K.cc_agree >= K.cc_agree.median()], "k3 신뢰도 상위 절반")

D.to_csv(os.path.join(FUS, "labelnoise_panel.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(FUS, "labelnoise_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")
