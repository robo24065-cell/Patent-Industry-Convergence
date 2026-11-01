# -*- coding: utf-8 -*-
"""
적대적 검증 — diag_globalcycle.py 의 결론('H-B 메커니즘 기각')을 반증한다.

검증 렌즈: 대안설명. 특히 (a) 종속변수 자체의 성질(k3 해상도 / 중복행),
(b) 공통분모 R² 라는 지표의 타당성, (c) '무차이' 주장의 검정력,
(d) 신규8 의 이질(bimodal) 구성.

기존 스크립트는 건드리지 않는다. 평가 함수 loco 는 지시대로 그대로 사용.
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
A = os.path.join(BASE, "분석"); FUS = os.path.join(A, "fusion")
IND = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
SPLITS = list(range(2013, 2020))
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SEMI5 = [26111, 26121, 26211, 26212, 26519]      # H-B 가 실제로 지목한 반도체·디스플레이류
log = lambda m: print(m, flush=True)


def loco(X, y, g):
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try: b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception: return np.nan
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pr)
    sse = ((y[ok]-pr[ok])**2).sum(); sst = ((y[ok]-y[ok].mean())**2).sum()
    return float(1 - sse/sst) if sst > 0 else np.nan


def loco_pred(X, y, g):
    """loco 와 동일 절차의 예측값(진단용)."""
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return None
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    return pr


def loco_pred_null(y, g):
    """절편만 있는 모형(=훈련폴드 평균)으로 같은 LOCO 예측."""
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < 4: return None
        pr[te] = y[tr].mean()
    return pr


def ols(X, y):
    Xt = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xt, y, rcond=None)
    r = y - Xt @ b; n, p = Xt.shape; dof = max(n - p, 1)
    s2 = (r**2).sum()/dof
    try: se = np.sqrt(np.diag(s2*np.linalg.inv(Xt.T @ Xt)))
    except Exception: se = np.full(p, np.nan)
    return b, se, r, Xt


def ols_cluster(X, y, cl):
    """클러스터 강건 SE (cl = 클러스터 라벨). CR1 보정."""
    b, se_h, r, Xt = ols(X, y)
    n, p = Xt.shape
    XtX_inv = np.linalg.pinv(Xt.T @ Xt)
    meat = np.zeros((p, p))
    uq = np.unique(cl)
    for c in uq:
        m = cl == c
        u = Xt[m].T @ r[m]
        meat += np.outer(u, u)
    G = len(uq)
    adj = (G/(G-1)) * ((n-1)/(n-p)) if G > 1 else 1.0
    V = adj * XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V)), se_h, G


# ═══════════════════════════════ 자료 ═══════════════════════════════
P = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"))
P["ksic"] = P.ksic.astype(int)
P["k3s"] = P.k3.astype(int).astype(str).str.zfill(3)
P["mids"] = P["mid"].astype(int).astype(str).str.zfill(2)

IX = pd.read_csv(os.path.join(IND, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
IX.columns = [c.strip() for c in IX.columns]
IX["code"] = IX["산업코드"].astype(str).str.strip(); IX["ym"] = IX["연월"].astype(str)
IX["yr"] = IX.ym.str[:4].astype(int); IX["v"] = pd.to_numeric(IX["값"], errors="coerce")
DS = pd.read_csv(os.path.join(IND, "kosis_내수수출출하지수.csv"), encoding="utf-8-sig")
DS["ym"] = DS.PRD_DE.astype(str); DS["yr"] = DS.ym.str[:4].astype(int)
DS["v"] = pd.to_numeric(DS.DT, errors="coerce")


def wide(df, item_col, item, code_col):
    s = df[df[item_col] == item]
    W = s.pivot_table(index="ym", columns=code_col, values="v", aggfunc="mean").sort_index()
    L = np.log(W.where(W > 0)); return L - L.shift(12)


G_PROD = wide(IX, "항목명", "생산지수(원지수)", "code")
G_SHIP = wide(IX, "항목명", "출하지수(원지수)", "code")
G_DOM = wide(DS, "ITM_NM", "내수출하지수(원지수)", "C1")
G_EXP = wide(DS, "ITM_NM", "수출출하지수(원지수)", "C1")
DSCODES = set(G_DOM.columns) & set(G_EXP.columns)
dsmap = lambda k3s: "C"+k3s if ("C"+k3s) in DSCODES else "C"+k3s[:2]
K3LIST = sorted(P.k3s.unique())


def corr(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 24: return np.nan
    x, y = a[m].values, b[m].values
    if np.std(x) < 1e-9 or np.std(y) < 1e-9: return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def sync_table(y_end, y_start=2006):
    def win(M):
        yrs = M.index.astype(str).str[:4].astype(int)
        return M[(yrs >= y_start) & (yrs < y_end)]
    gp, gs, gd, ge = win(G_PROD), win(G_SHIP), win(G_DOM), win(G_EXP)
    rows = []
    for k in K3LIST:
        ck = "C"+k
        mfgp = gp["C"] if "C" in gp.columns else pd.Series(dtype=float)
        mfgs = gs["C"] if "C" in gs.columns else pd.Series(dtype=float)
        own_p = gp[ck] if ck in gp.columns else pd.Series(index=gp.index, dtype=float)
        own_s = gs[ck] if ck in gs.columns else pd.Series(index=gs.index, dtype=float)
        dc = dsmap(k)
        dom = gd[dc] if dc in gd.columns else pd.Series(index=gd.index, dtype=float)
        exp = ge[dc] if dc in ge.columns else pd.Series(index=ge.index, dtype=float)
        a1, b1 = own_s.align(dom, join="inner"); a2, b2 = own_s.align(exp, join="inner")
        rows.append({"k3s": k, "sync_mfg_prod": corr(own_p, mfgp), "sync_mfg_ship": corr(own_s, mfgs),
                     "corr_dom": corr(a1, b1), "corr_exp": corr(a2, b2),
                     "vol_ship": float(np.nanstd(own_s.values)) if own_s.notna().sum() >= 24 else np.nan})
    T = pd.DataFrame(rows)
    T["exp_pref"] = T.corr_exp - T.corr_dom
    T["sync_mfg"] = T[["sync_mfg_prod", "sync_mfg_ship"]].mean(axis=1)
    return T


SYS = {s: sync_table(s) for s in SPLITS}
SY17 = sync_table(2017)

rows = []
for f in glob.glob(os.path.join(BASE, "공모전데이터", "외부데이터", "산업분류별", "*.csv")):
    d = pd.read_csv(f, encoding="cp949")
    if "[B]산업별" not in d.columns: continue
    d = d[d["항목"].astype(str).str.startswith("출하액")]
    for c in [c for c in d.columns if c.strip().endswith("년")]:
        rows.append(pd.DataFrame({"code": d["[B]산업별"].astype(str), "yr": int(c.strip()[:4]),
                                  "ship_val": pd.to_numeric(d[c], errors="coerce")}))
SV = pd.concat(rows, ignore_index=True).drop_duplicates(["code", "yr"])
SV = SV[SV.code.str.match(r"^C\d{5}$", na=False)].copy(); SV["ksic"] = SV.code.str[1:].astype(int)
SZ = SV[(SV.yr >= 2008) & (SV.yr <= 2016)].groupby("ksic").ship_val.mean().rename("ship_val_e").reset_index()
SZ["ln_ship_val"] = np.log(SZ.ship_val_e.where(SZ.ship_val_e > 0))

DVCOL = "dd_ship_late"


def panel_for(s):
    d = P[P.split == s].merge(SYS[s], on="k3s", how="left").merge(SZ[["ksic", "ln_ship_val"]], on="ksic", how="left")
    d["over1"] = (d.padi_e > 1.0).astype(float)
    return d.dropna(subset=[DVCOL, "dd_prod_early", "over1", "sync_mfg", "ln_ship_val"])


PANELS = {s: panel_for(s) for s in SPLITS}
SYNCVARS = ["sync_mfg", "sync_mfg_prod", "sync_mfg_ship", "exp_pref", "vol_ship"]

log("="*80)
log("[R0] 재현 확인 — 기준 사양 LOCO 평균")
log("="*80)
base_loco = [loco(PANELS[s][["dd_prod_early", "over1"]].values, PANELS[s][DVCOL].values, PANELS[s].mids.values) for s in SPLITS]
log(f"  기준(n=47) LOCO 평균 {np.nanmean(base_loco):+.3f} 최소 {np.nanmin(base_loco):+.3f}  (보고서 +0.476 / +0.346)")

# ══════════════════════════════════════════════════════════════════
log("\n"+"="*80)
log("[R1] ★ 종속변수의 성질 — y 와 dd_prod_early 는 k3 수준, over1 만 ksic 수준")
log("="*80)
d17 = PANELS[2017]
log(f"  split=2017: 행 {len(d17)} / 고유 k3 {d17.k3s.nunique()} / 고유 mid {d17.mids.nunique()}")
log(f"  k3 내 dd_ship_late 고유값 최대 {d17.groupby('k3s')[DVCOL].nunique().max()} "
    f"| dd_prod_early 고유값 최대 {d17.groupby('k3s').dd_prod_early.nunique().max()}")
log("  → y 와 주 설명변수는 k3 내 완전 중복. 47행 회귀의 실효 관측치는 25(k3) 수준이다.")
log("\n  분할연도 간 y 상관 (같은 47산업, late 창만 다름) — 7개 split 이 독립 반복인가?")
YM = pd.concat([PANELS[s][["ksic", DVCOL]].set_index("ksic")[DVCOL].rename(s) for s in SPLITS], axis=1)
C = YM.corr()
log("       " + " ".join(f"{s:>6d}" for s in SPLITS))
for s in SPLITS:
    log(f"  {s}: " + " ".join(f"{C.loc[s,t]:+6.3f}" for t in SPLITS))
off = C.values[np.triu_indices(len(SPLITS), 1)]
log(f"  비대각 평균 상관 {off.mean():+.3f} (최소 {off.min():+.3f}) → '7분할 평균/최솟값'은 사실상 1개 관측의 재표기")

# ══════════════════════════════════════════════════════════════════
log("\n"+"="*80)
log("[R2] ★★ 결정적 반증 — '공통분모 R²' 는 예측력이 아니라 층의 y 분산을 재는 지표다")
log("="*80)
log("  같은 분모(전체 47표본 y분산)로 (a) 기준 모형 (b) 절편만 모형(=훈련폴드 평균) 을 나란히 계산")
log(f"  {'지표':14s} {'층':4s} {'모형 공통R²':>11s} {'절편만 공통R²':>13s} {'모형이 더한 몫':>13s} {'층내 LOCO R²':>12s}")
R2TAB = []
for sv in SYNCVARS:
    for half, lab in [(1, "상위"), (0, "하위")]:
        mv, nv, wv = [], [], []
        for s in SPLITS:
            d = PANELS[s]; med = d[sv].median()
            dd = d[d[sv] >= med] if half else d[d[sv] < med]
            if len(dd) < 12 or dd.mids.nunique() < 4: continue
            X = dd[["dd_prod_early", "over1"]].values; y = dd[DVCOL].values; g = dd.mids.values
            pm = loco_pred(X, y, g); pn = loco_pred_null(y, g)
            if pm is None or pn is None or np.isnan(pm).any(): continue
            den = np.var(d[DVCOL].values)                     # 전체 47표본 분산 = 공통분모
            mv.append(1 - np.mean((y-pm)**2)/den)
            nv.append(1 - np.mean((y-pn)**2)/den)
            wv.append(1 - np.mean((y-pm)**2)/np.var(y))       # 층내 분모 = 통상 LOCO R²
        if mv:
            log(f"  {sv:14s} {lab:4s} {np.mean(mv):+11.3f} {np.mean(nv):+13.3f} {np.mean(mv)-np.mean(nv):+13.3f} {np.mean(wv):+12.3f}")
            R2TAB.append({"sv": sv, "half": lab, "model": np.mean(mv), "null": np.mean(nv),
                          "gain": np.mean(mv)-np.mean(nv), "within": np.mean(wv)})
log("\n  해석: '모형이 더한 몫' 이 0 이하이면 그 층에서 PADI+지속성 모형은 층평균 예측보다 못하다.")

# ══════════════════════════════════════════════════════════════════
log("\n"+"="*80)
log("[R3] ★ '두 층의 β_over1 이 사실상 같다' 는 주장의 검정력")
log("="*80)
log("  층 더미와의 완전 상호작용 모형을 단일 회귀로 적합하고 β 차이의 CI 를 낸다.")
log("  SE 는 (a) 통상 OLS, (b) k3 클러스터 강건 (y 가 k3 내 중복이므로 이쪽이 옳다)")
for sv in SYNCVARS:
    bd, sd_ols, sd_cl = [], [], []
    for s in SPLITS:
        d = PANELS[s].copy(); med = d[sv].median()
        d["hi"] = (d[sv] >= med).astype(float)
        X = np.column_stack([d.dd_prod_early, d.over1, d.hi, d.hi*d.over1, d.hi*d.dd_prod_early])
        y = d[DVCOL].values
        b, se_c, se_h, G = ols_cluster(X, y, d.k3s.values)
        bd.append(b[4]); sd_ols.append(se_h[4]); sd_cl.append(se_c[4])
    bd = np.array(bd); sc = np.array(sd_cl); so = np.array(sd_ols)
    lo, hi = bd.mean()-1.96*sc.mean(), bd.mean()+1.96*sc.mean()
    log(f"  {sv:14s} β차이(상위-하위) {bd.mean():+7.2f} | OLS se {so.mean():5.2f}(t {bd.mean()/so.mean():+5.2f}) "
        f"| k3클러스터 se {sc.mean():5.2f}(t {bd.mean()/sc.mean():+5.2f}) | 95%CI [{lo:+6.1f}, {hi:+6.1f}]")
log("\n  전체표본 β_over1 의 SE 비교 (기준 사양)")
for s in [2015, 2017, 2019]:
    d = PANELS[s]
    b, se_c, se_h, G = ols_cluster(d[["dd_prod_early", "over1"]].values, d[DVCOL].values, d.k3s.values)
    b2, se_c2, se_h2, G2 = ols_cluster(d[["dd_prod_early", "over1"]].values, d[DVCOL].values, d.mids.values)
    log(f"    split={s} β_over1={b[2]:+7.2f} | OLS t {b[2]/se_h[2]:+5.2f} | k3클러스터 t {b[2]/se_c[2]:+5.2f} (G={G})"
        f" | mid클러스터 t {b2[2]/se_c2[2]:+5.2f} (G={G2})")

# ══════════════════════════════════════════════════════════════════
log("\n"+"="*80)
log("[R4] ★ 제3의 설명 — 층 사이 '중복행 구조(k3당 산업 수)' 차이가 LOCO 를 만든다")
log("="*80)
log(f"  {'지표':14s} {'층':4s} {'행 n':>5s} {'고유k3':>6s} {'고유mid':>7s} {'행/k3':>6s} {'최대k3블록':>9s}")
for sv in SYNCVARS:
    for half, lab in [(1, "상위"), (0, "하위")]:
        d = PANELS[2017]; med = d[sv].median()
        dd = d[d[sv] >= med] if half else d[d[sv] < med]
        log(f"  {sv:14s} {lab:4s} {len(dd):5d} {dd.k3s.nunique():6d} {dd.mids.nunique():7d} "
            f"{len(dd)/dd.k3s.nunique():6.2f} {dd.groupby('k3s').size().max():9d}")
log("\n  ★ k3 수준으로 붕괴(중복 제거, over1 = k3 내 PADI>1 산업 비중의 중앙값 기준)해 재검정")
log("     — 중복행이 사라지면 층 간 LOCO 격차가 남는가?")
log(f"  {'지표':14s} {'층':4s} {'n(k3)':>6s} {'G':>3s} {'LOCO':>7s} {'RMSE':>6s} {'sd(y)':>6s} {'β_over1':>8s}")
for sv in SYNCVARS:
    for half, lab in [(1, "상위"), (0, "하위")]:
        lo_v, rm_v, sd_v, bo_v = [], [], [], []
        for s in SPLITS:
            d = PANELS[s]
            k = d.groupby("k3s").agg(y=(DVCOL, "first"), x=("dd_prod_early", "first"),
                                     ov=("over1", "mean"), sv_=(sv, "first"),
                                     mids=("mids", "first")).reset_index()
            k["over1"] = (k.ov >= 0.5).astype(float)
            med = k.sv_.median()
            kk = k[k.sv_ >= med] if half else k[k.sv_ < med]
            if len(kk) < 8 or kk.mids.nunique() < 4 or kk.over1.nunique() < 2: continue
            X = kk[["x", "over1"]].values; y = kk.y.values
            L = loco(X, y, kk.mids.values)
            if np.isnan(L): continue
            b, se, r, Xt = ols(X, y)
            lo_v.append(L); rm_v.append(np.sqrt((1-L)*np.var(y))); sd_v.append(np.std(y, ddof=1)); bo_v.append(b[2])
        if lo_v:
            log(f"  {sv:14s} {lab:4s} {len(kk):6d} {kk.mids.nunique():3d} {np.mean(lo_v):+7.3f} "
                f"{np.mean(rm_v):6.1f} {np.mean(sd_v):6.1f} {np.mean(bo_v):+8.2f}  (splits={len(lo_v)})")

# ══════════════════════════════════════════════════════════════════
log("\n"+"="*80)
log("[R5] ★ 신규8 동조도 '차이 없음' 은 이봉(bimodal) 구성의 평균 착시인가")
log("="*80)
b17 = PANELS[2017][["ksic", "k3s", "sync_mfg", "exp_pref", "vol_ship", "padi_e"]].drop_duplicates("ksic")
b17["grp"] = np.where(b17.ksic.isin(SEMI5), "신규-반도체디스플레이5",
              np.where(b17.ksic.isin(NEW8), "신규-기타3", "기존39"))
for g, dd in b17.groupby("grp"):
    pct = b17.sync_mfg.rank(pct=True)*100
    log(f"  {g:22s} n={len(dd):2d} sync_mfg 평균 {dd.sync_mfg.mean():+.3f} "
        f"중위백분위 {pct[dd.index].median():5.1f} | 고유k3 {dd.k3s.nunique()}")
log(f"\n  신규8 sync_mfg 값: " + ", ".join(f"{int(r.ksic)}={r.sync_mfg:.3f}" for _, r in
                                        b17[b17.ksic.isin(NEW8)].sort_values("sync_mfg", ascending=False).iterrows()))
med17 = b17.sync_mfg.median()
log(f"  중앙값 {med17:.3f} 기준: 신규8 중 상위층 {int((b17[b17.ksic.isin(NEW8)].sync_mfg>=med17).sum())}개 "
    f"(그중 반도체디스플레이5 = {int((b17[b17.ksic.isin(SEMI5)].sync_mfg>=med17).sum())}/5)")
log("  → 평균이 같은 것은 26x(고동조)와 271/273/281(저동조)를 한 그룹으로 묶었기 때문이다.")

log("\n  H-B 를 실제 지목 대상(반도체·디스플레이 5개)으로 좁혀 재검정 — 제거 효과")
rng = np.random.default_rng(20260813)
allk = sorted(PANELS[2017].ksic.unique())


def loco_mean_excl(ex):
    v = []
    for s in SPLITS:
        d = PANELS[s]; dd = d[~d.ksic.isin(ex)]
        v.append(loco(dd[["dd_prod_early", "over1"]].values, dd[DVCOL].values, dd.mids.values))
    return float(np.nanmean(v))


null5 = np.array([loco_mean_excl(set(rng.choice(allk, 5, replace=False).tolist())) for _ in range(1000)])
m5 = loco_mean_excl(set(SEMI5))
log(f"    반도체디스플레이5 제거(n=42) LOCO 평균 {m5:+.3f} | 무작위 5개 제거 귀무 중위 {np.median(null5):+.3f} "
    f"(5% {np.percentile(null5,5):+.3f}, 95% {np.percentile(null5,95):+.3f}) → 상위 {(null5>=m5).mean()*100:.1f}%")
m3 = loco_mean_excl({27199, 27301, 28111})
null3 = np.array([loco_mean_excl(set(rng.choice(allk, 3, replace=False).tolist())) for _ in range(1000)])
log(f"    신규-기타3 제거(n=44)      LOCO 평균 {m3:+.3f} | 무작위 3개 제거 귀무 중위 {np.median(null3):+.3f} "
    f"→ 상위 {(null3>=m3).mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════
log("\n"+"="*80)
log("[R6] 상호작용 부호 '가설과 정반대' 주장 점검 — 부호 규약 확인")
log("="*80)
log("  y = 최대낙폭(음수, 클수록 완만). over1 효과 β_over1<0 = PADI>1 이면 낙폭 심화 = 신호 작동.")
log("  H-B: 동조도↑ 면 신호 약화 → 상호작용 β_ix 는 +(0쪽으로) 여야 한다. 관측 β_ix=-15.65 = 신호 강화 방향.")
for s in SPLITS[:1]:
    pass
for sv in ["sync_mfg"]:
    bs, ts = [], []
    for s in SPLITS:
        d = PANELS[s].copy(); d["_sc"] = d[sv]-d[sv].mean(); d["_ix"] = d._sc*d.over1
        X = d[["dd_prod_early", "over1", "_sc", "_ix"]].values; y = d[DVCOL].values
        b, se_c, se_h, G = ols_cluster(X, y, d.k3s.values)
        bs.append(b[4]); ts.append(b[4]/se_c[4])
    log(f"  {sv}: β_ix 평균 {np.mean(bs):+.2f} | k3클러스터 t̄ {np.mean(ts):+.2f} "
        f"(보고서 OLS t̄ -1.11) | 부호 일관성 {sum(1 for x in bs if x<0)}/7 split 음수")

log("\n  ★ 동조도 상위층 안에서 over1 의 '한계 예측 기여' — 상위층에서 신호가 죽었는가")
log(f"  {'지표':14s} {'층':4s} {'LOCO(x만)':>10s} {'LOCO(x+over1)':>14s} {'Δ':>7s}")
for sv in SYNCVARS:
    for half, lab in [(1, "상위"), (0, "하위")]:
        a_, b_ = [], []
        for s in SPLITS:
            d = PANELS[s]; med = d[sv].median()
            dd = d[d[sv] >= med] if half else d[d[sv] < med]
            if len(dd) < 12 or dd.mids.nunique() < 4 or dd.over1.nunique() < 2: continue
            y = dd[DVCOL].values; g = dd.mids.values
            L1 = loco(dd[["dd_prod_early"]].values, y, g)
            L2 = loco(dd[["dd_prod_early", "over1"]].values, y, g)
            if np.isnan(L1) or np.isnan(L2): continue
            a_.append(L1); b_.append(L2)
        if a_:
            log(f"  {sv:14s} {lab:4s} {np.mean(a_):+10.3f} {np.mean(b_):+14.3f} {np.mean(b_)-np.mean(a_):+7.3f}")
log("\n완료")
