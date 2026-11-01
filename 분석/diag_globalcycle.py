# -*- coding: utf-8 -*-
"""
각도 D — '대형 전자는 글로벌 사이클에 지배되므로 PADI 가 안 먹는다' 가설 직접 검정
────────────────────────────────────────────────────────────────────────────
[가설 H-B-mech]
  반도체·디스플레이류 산업은 국내 기업의 기술자산 이탈(PADI)과 무관하게
  글로벌 수요 사이클이 낙폭을 결정한다 → 동조도가 높을수록 PADI 효과가 약하다.

[검정 설계]
  (1) 산업별 '글로벌/공통 사이클 동조도' 를 전반기 자료로만 산출
      - 제조업 전체 지수와의 상관 (공통 사이클 동조도)
      - 수출출하 vs 내수출하 어느 쪽과 더 붙는가 (수출지향성)
      - 변동성/진폭
  (2) 동조도 조절변수 상호작용 모형 + 동조도 상·하위 분할 (β_over1, LOCO, n)
  (3) 신규 8개 산업의 동조도 백분위
  (4) 규모(출하액·등록특허수) 통제 후에도 상호작용이 남는가
  (5) 판정

[사전 고정 사항 — 결과를 보고 바꾸지 않는다]
  기준 사양 : dd_ship_late ~ dd_prod_early + over1,  over1 = (padi_e > 1.0)
  평가      : LOCO-CV R² (클러스터=mid), 분할연도 2013~2019 평균과 최솟값 동시 보고
  탐색 규모 : 동조도 지표 5종 × 모형 4종 = 20 사양 (아래 SPECS 에 전부 명시, 전부 보고)

[주의] 표본 크기가 다른 결과를 나란히 비교하지 않는다. 모든 표에 실제 사용 n 을 적는다.
"""
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
A = os.path.join(BASE, "분석")
FUS = os.path.join(A, "fusion")
IND = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
SPLITS = list(range(2013, 2020))
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
log = lambda m: print(m, flush=True)


# ══════════════════════════════════════════════════════════════════════
# 평가 함수 — 지시받은 그대로 사용 (수정 금지)
# ══════════════════════════════════════════════════════════════════════
def loco(X, y, g):
    """leave-one-cluster-out 예측 R². g는 클러스터 라벨(보통 mid)."""
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


def ols(X, y):
    Xt = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xt, y, rcond=None)
    r = y - Xt @ b
    n, p = Xt.shape
    dof = max(n - p, 1)
    s2 = (r ** 2).sum() / dof
    try:
        cov = s2 * np.linalg.inv(Xt.T @ Xt)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(p, np.nan)
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / sst if sst > 0 else np.nan
    return b, se, r2


def maxdd(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 12 or np.nanmax(v) <= 0: return np.nan
    return float(np.min(v / np.maximum.accumulate(v) - 1.0) * 100)


# ══════════════════════════════════════════════════════════════════════
log("=" * 78)
log("[0] 자료 적재")
log("=" * 78)

P = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"))
P["ksic"] = P.ksic.astype(int)
P["k3s"] = P.k3.astype(int).astype(str).str.zfill(3)
P["mids"] = P["mid"].astype(int).astype(str).str.zfill(2)
log(f"  패널 {len(P)}행 | 산업 {P.ksic.nunique()} | 소분류 {P.k3s.nunique()} | 중분류 {P.mids.nunique()} | split {sorted(P.split.unique())}")

IX = pd.read_csv(os.path.join(IND, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
IX.columns = [c.strip() for c in IX.columns]
IX["code"] = IX["산업코드"].astype(str).str.strip()
IX["ym"] = IX["연월"].astype(str)
IX["yr"] = IX.ym.str[:4].astype(int)
IX["v"] = pd.to_numeric(IX["값"], errors="coerce")
log(f"  광공업지수 {len(IX):,}행 | {IX.yr.min()}~{IX.yr.max()} | 항목 {list(IX['항목명'].unique())}")

DS = pd.read_csv(os.path.join(IND, "kosis_내수수출출하지수.csv"), encoding="utf-8-sig")
DS["ym"] = DS.PRD_DE.astype(str)
DS["yr"] = DS.ym.str[:4].astype(int)
DS["v"] = pd.to_numeric(DS.DT, errors="coerce")
log(f"  내수/수출 출하지수 {len(DS):,}행 | 코드 {DS.C1.nunique()}개 | {DS.yr.min()}~{DS.yr.max()}")


def wide(df, item_col, item, code_col):
    """월별 지수를 code × ym 행렬로. 원지수 → 12개월 로그차분(YoY)으로 계절성 제거."""
    s = df[df[item_col] == item]
    W = s.pivot_table(index="ym", columns=code_col, values="v", aggfunc="mean").sort_index()
    L = np.log(W.where(W > 0))
    return L - L.shift(12)   # YoY 로그성장률


G_PROD = wide(IX, "항목명", "생산지수(원지수)", "code")
G_SHIP = wide(IX, "항목명", "출하지수(원지수)", "code")
G_DOM = wide(DS, "ITM_NM", "내수출하지수(원지수)", "C1")
G_EXP = wide(DS, "ITM_NM", "수출출하지수(원지수)", "C1")
log(f"  YoY 성장률 행렬: 생산 {G_PROD.shape} 출하 {G_SHIP.shape} 내수 {G_DOM.shape} 수출 {G_EXP.shape}")

# k3 → 내수/수출 지수 코드 매핑 (수출지수는 2자리 해상도, 단 C261~C265 는 3자리 존재)
DSCODES = set(G_DOM.columns) & set(G_EXP.columns)
def dsmap(k3s):
    return "C" + k3s if ("C" + k3s) in DSCODES else "C" + k3s[:2]

K3LIST = sorted(P.k3s.unique())
log(f"  대상 소분류 {len(K3LIST)}개 | 내수/수출 매핑: " +
    ", ".join(f"{k}->{dsmap(k)}" for k in K3LIST[:6]) + " ...")


# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[1] 산업별 글로벌/공통 사이클 동조도 — 전반기 자료만 사용")
log("=" * 78)

def corr(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 24: return np.nan
    x, y = a[m].values, b[m].values
    if np.std(x) < 1e-9 or np.std(y) < 1e-9: return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def sync_table(y_end, y_start=2006):
    """[y_start, y_end) 구간의 월별 YoY 성장률로 동조도 산출."""
    def win(M):
        idx = M.index.astype(str)
        yrs = idx.str[:4].astype(int)
        return M[(yrs >= y_start) & (yrs < y_end)]
    gp, gs, gd, ge = win(G_PROD), win(G_SHIP), win(G_DOM), win(G_EXP)
    rows = []
    for k in K3LIST:
        ck = "C" + k
        mfgp = gp["C"] if "C" in gp.columns else pd.Series(dtype=float)
        mfgs = gs["C"] if "C" in gs.columns else pd.Series(dtype=float)
        own_p = gp[ck] if ck in gp.columns else pd.Series(index=gp.index, dtype=float)
        own_s = gs[ck] if ck in gs.columns else pd.Series(index=gs.index, dtype=float)
        dc = dsmap(k)
        dom = gd[dc] if dc in gd.columns else pd.Series(index=gd.index, dtype=float)
        exp = ge[dc] if dc in ge.columns else pd.Series(index=ge.index, dtype=float)
        # 출하 계열은 인덱스가 다른 표에서 오므로 정렬
        own_s_al, dom_al = own_s.align(dom, join="inner")
        own_s_al2, exp_al = own_s.align(exp, join="inner")
        # 진폭·변동성 (전반기 출하지수 수준 기준 최대낙폭 / YoY 성장률 표준편차)
        lev = IX[(IX["항목명"] == "출하지수(원지수)") & (IX.code == ck) &
                 (IX.yr >= y_start) & (IX.yr < y_end)].sort_values("ym").v.values
        rows.append({
            "k3s": k,
            "sync_mfg_prod": corr(own_p, mfgp),
            "sync_mfg_ship": corr(own_s, mfgs),
            "corr_dom": corr(own_s_al, dom_al),
            "corr_exp": corr(own_s_al2, exp_al),
            "vol_ship": float(np.nanstd(own_s.values)) if own_s.notna().sum() >= 24 else np.nan,
            "amp_ship_early": maxdd(lev),
        })
    T = pd.DataFrame(rows)
    T["exp_pref"] = T.corr_exp - T.corr_dom            # +면 수출 쪽에 더 붙음
    T["sync_mfg"] = T[["sync_mfg_prod", "sync_mfg_ship"]].mean(axis=1)
    return T


SY17 = sync_table(2017)   # 지시대로 전반기(<2017) 고정창 — 서술·백분위용
log(f"\n  [고정창 2006~2016] 소분류 {len(SY17)}개 동조도")
log(f"  {'k3':>4s} {'제조업동조(생산)':>10s} {'제조업동조(출하)':>10s} {'sync_mfg':>9s} {'내수상관':>8s} {'수출상관':>8s} {'수출선호':>8s} {'변동성':>7s} {'전반기낙폭':>9s}")
K2NAME = (IX[IX.code.str.match(r"^C\d{3}$", na=False)][["code", "산업명"]]
          .drop_duplicates().set_index("code")["산업명"].to_dict())
for _, r in SY17.sort_values("sync_mfg", ascending=False).iterrows():
    log(f"  {r.k3s:>4s} {r.sync_mfg_prod:10.3f} {r.sync_mfg_ship:10.3f} {r.sync_mfg:9.3f} "
        f"{r.corr_dom:8.3f} {r.corr_exp:8.3f} {r.exp_pref:8.3f} {r.vol_ship:7.3f} {r.amp_ship_early:9.1f}  "
        f"{K2NAME.get('C'+r.k3s,'')}")

# 분할연도별 동조도 (누수 방지: year < split 자료만)
SYS = {s: sync_table(s).assign(split=s) for s in SPLITS}
log("\n  분할연도별 동조도 산출 완료 (year < split 자료만 사용, 누수 없음)")
for s in SPLITS:
    t = SYS[s]
    log(f"    split={s}: sync_mfg 평균 {t.sync_mfg.mean():.3f} "
        f"(min {t.sync_mfg.min():.3f} / max {t.sync_mfg.max():.3f}), exp_pref 평균 {t.exp_pref.mean():+.3f}")


# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[1b] 규모 변수 — 출하액(광업제조업조사) · 등록특허수")
log("=" * 78)
rows = []
for f in glob.glob(os.path.join(BASE, "공모전데이터", "외부데이터", "산업분류별", "*.csv")):
    d = pd.read_csv(f, encoding="cp949")
    if "[B]산업별" not in d.columns: continue
    d = d[d["항목"].astype(str).str.startswith("출하액")]
    ycols = [c for c in d.columns if c.strip().endswith("년")]
    for c in ycols:
        yy = int(c.strip()[:4])
        rows.append(pd.DataFrame({"code": d["[B]산업별"].astype(str), "yr": yy,
                                  "ship_val": pd.to_numeric(d[c], errors="coerce")}))
SV = pd.concat(rows, ignore_index=True).drop_duplicates(["code", "yr"])
SV = SV[SV.code.str.match(r"^C\d{5}$", na=False)].copy()
SV["ksic"] = SV.code.str[1:].astype(int)
SZ = (SV[(SV.yr >= 2008) & (SV.yr <= 2016)].groupby("ksic").ship_val.mean().rename("ship_val_e").reset_index())
SZ["ln_ship_val"] = np.log(SZ.ship_val_e.where(SZ.ship_val_e > 0))
log(f"  출하액(2008~2016 평균) 확보 산업 {SZ.ksic.nunique()}개 | 패널 47개 중 매칭 "
    f"{P.ksic.drop_duplicates().isin(SZ.ksic).sum()}개")


# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[3] 신규 8개 산업의 동조도 위치 (47개 중 백분위) — 고정창 2006~2016")
log("=" * 78)
base = P[P.split == 2017][["ksic", "k3s", "mids", "padi_e", "R"]].drop_duplicates("ksic").merge(SY17, on="k3s", how="left")
base = base.merge(SZ[["ksic", "ln_ship_val", "ship_val_e"]], on="ksic", how="left")
base["is_new"] = base.ksic.isin(NEW8).astype(int)
base["ln_R"] = np.log(base.R.clip(lower=1))
METRICS = ["sync_mfg", "sync_mfg_prod", "sync_mfg_ship", "exp_pref", "vol_ship", "corr_exp"]
for m in METRICS:
    base[m + "_pct"] = base[m].rank(pct=True) * 100

log(f"  {'ksic':>6s} {'sync_mfg':>9s} {'%ile':>6s} {'exp_pref':>9s} {'%ile':>6s} {'vol':>6s} {'%ile':>6s} {'PADI':>6s}")
for _, r in base[base.is_new == 1].sort_values("sync_mfg", ascending=False).iterrows():
    log(f"  {int(r.ksic):>6d} {r.sync_mfg:9.3f} {r.sync_mfg_pct:6.1f} {r.exp_pref:9.3f} {r.exp_pref_pct:6.1f} "
        f"{r.vol_ship:6.3f} {r.vol_ship_pct:6.1f} {r.padi_e:6.3f}")
log("")
for m in METRICS:
    a = base.loc[base.is_new == 1, m]; b = base.loc[base.is_new == 0, m]
    log(f"  {m:14s} 신규8 평균 {a.mean():+.3f} (중위백분위 {base.loc[base.is_new==1, m+'_pct'].median():5.1f}) | "
        f"기존39 평균 {b.mean():+.3f} | 차이 {a.mean()-b.mean():+.3f}")
log(f"\n  규모: 신규8 ln(출하액) 평균 {base.loc[base.is_new==1,'ln_ship_val'].mean():.2f} "
    f"vs 기존39 {base.loc[base.is_new==0,'ln_ship_val'].mean():.2f} "
    f"| ln(등록특허) {base.loc[base.is_new==1,'ln_R'].mean():.2f} vs {base.loc[base.is_new==0,'ln_R'].mean():.2f}")
log(f"  동조도와 규모의 상관 (n={base[['sync_mfg','ln_ship_val']].dropna().shape[0]}): "
    f"sync_mfg~ln(출하액) r={base[['sync_mfg','ln_ship_val']].corr().iloc[0,1]:+.3f} | "
    f"sync_mfg~ln(등록특허) r={base[['sync_mfg','ln_R']].corr().iloc[0,1]:+.3f}")


# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[2] 동조도 조절 모형 — 상호작용 · 상하위 분할")
log("=" * 78)

def panel_for(split):
    d = P[P.split == split].merge(SYS[split].drop(columns=["split"]), on="k3s", how="left")
    d = d.merge(SZ[["ksic", "ln_ship_val"]], on="ksic", how="left")
    d["over1"] = (d.padi_e > 1.0).astype(float)
    d["ln_R"] = np.log(d.R.clip(lower=1))
    return d


DVCOL, YSPEC = "dd_ship_late", ["dd_prod_early", "over1"]

# --- 탐색 규모 사전 고정 -------------------------------------------------
SYNCVARS = ["sync_mfg", "sync_mfg_prod", "sync_mfg_ship", "exp_pref", "vol_ship"]   # 5종
MODELS = ["base", "+sync", "+sync×over1", "+sync×over1+size"]                        # 4종
log(f"  탐색 규모: 동조도 지표 {len(SYNCVARS)}종 × 모형 {len(MODELS)}종 = {len(SYNCVARS)*len(MODELS)} 사양 (전부 보고)")
log(f"  기준 사양: {DVCOL} ~ {' + '.join(YSPEC)}\n")


def run(dsub, cols, dv=DVCOL):
    d = dsub.dropna(subset=[dv] + cols)
    if len(d) < 12 or d.mids.nunique() < 4: return None
    X = d[cols].values.astype(float); y = d[dv].values.astype(float)
    if np.any(np.std(X, axis=0) < 1e-12): return None
    b, se, r2 = ols(X, y)
    L = loco(X, y, d.mids.values)
    sd_y = float(np.std(y, ddof=1))
    rmse = float(np.sqrt((1 - L) * np.var(y)) ) if L is not None and not np.isnan(L) else np.nan
    return {"n": len(d), "ng": d.mids.nunique(), "b": b, "se": se, "r2": r2,
            "loco": L, "cols": cols, "sd_y": sd_y, "rmse": rmse}


def summarize(tag, results, keys=("over1",)):
    ok = [r for r in results if r]
    if not ok:
        log(f"  {tag:34s}  (추정 불가)"); return None
    ns = sorted(set(r["n"] for r in ok))
    ngs = sorted(set(r["ng"] for r in ok))
    lo = np.array([r["loco"] for r in ok], float)
    out = {"tag": tag, "n": ns, "ng": ngs, "splits": len(ok),
           "loco_mean": np.nanmean(lo), "loco_min": np.nanmin(lo),
           "r2_mean": np.mean([r["r2"] for r in ok]),
           "sd_y": float(np.mean([r["sd_y"] for r in ok])),
           "rmse": float(np.mean([r["rmse"] for r in ok]))}
    txt = (f"  {tag:34s} n={str(ns):>10s} G={str(ngs):>9s} "
           f"LOCO 평균 {np.nanmean(lo):+.3f} 최소 {np.nanmin(lo):+.3f} R² {out['r2_mean']:.3f} "
           f"sd(y) {out['sd_y']:5.1f} LOCO-RMSE {out['rmse']:5.1f}")
    for k in keys:
        bs = [r["b"][1 + r["cols"].index(k)] for r in ok if k in r["cols"]]
        ts = [r["b"][1 + r["cols"].index(k)] / r["se"][1 + r["cols"].index(k)] for r in ok if k in r["cols"]]
        if bs:
            txt += f" | β_{k}={np.mean(bs):+.3f}(t̄={np.mean(ts):+.2f})"
            out["b_" + k] = float(np.mean(bs)); out["t_" + k] = float(np.mean(ts))
    log(txt)
    return out


# ── 2-A. 전체 표본: 기준 vs 동조도 추가 vs 상호작용 ─────────────────────
log("─" * 78)
log("[2-A] 전체 표본 (47산업) — 동조도 상호작용")
log("─" * 78)
SUM = []
SUM.append(summarize("기준: dd_prod_early+over1", [run(panel_for(s), ["dd_prod_early", "over1"]) for s in SPLITS]))

for sv in SYNCVARS:
    res_m, res_i, res_is = [], [], []
    for s in SPLITS:
        d = panel_for(s)
        d["_sc"] = d[sv] - d[sv].mean()
        d["_ix"] = d["_sc"] * d["over1"]
        res_m.append(run(d, ["dd_prod_early", "over1", "_sc"]))
        res_i.append(run(d, ["dd_prod_early", "over1", "_sc", "_ix"]))
        d["_lsz"] = d["ln_ship_val"] - d["ln_ship_val"].mean()
        d["_lix"] = d["_lsz"] * d["over1"]
        res_is.append(run(d, ["dd_prod_early", "over1", "_sc", "_ix", "_lsz", "_lix"]))
    summarize(f"+{sv}", res_m, keys=("over1", "_sc"))
    summarize(f"+{sv}×over1", res_i, keys=("over1", "_sc", "_ix"))
    summarize(f"+{sv}×over1+규모통제", res_is, keys=("over1", "_ix", "_lix"))
    log("")


# ── 2-B. 동조도 상·하위 분할 ────────────────────────────────────────────
log("─" * 78)
log("[2-B] 동조도 상·하위 분할 — 각 층에서 기준 사양 재적합")
log("─" * 78)
SPLITRES = []
for sv in SYNCVARS:
    for half, lab in [(1, "상위(동조도 高)"), (0, "하위(동조도 低)")]:
        res = []
        for s in SPLITS:
            d = panel_for(s)
            med = d[sv].median()
            dd = d[d[sv] >= med] if half else d[d[sv] < med]
            res.append(run(dd, ["dd_prod_early", "over1"]))
        r = summarize(f"{sv:14s} {lab}", res)
        if r: r["sv"] = sv; r["half"] = lab; SPLITRES.append(r)
    log("")

# ── 2-C. 신규8 제외 / 신규8 이질성 직접 확인 ────────────────────────────
log("─" * 78)
log("[2-C] 참고 — 신규8 제외 시 (n=39, 표본 다름! 47 결과와 직접 비교 금지)")
log("─" * 78)
summarize("기준 (신규8 제외)", [run(panel_for(s)[~panel_for(s).ksic.isin(NEW8)], ["dd_prod_early", "over1"]) for s in SPLITS])
d17 = panel_for(2017)
for lab, sel in [("신규8", d17.ksic.isin(NEW8)), ("기존39", ~d17.ksic.isin(NEW8))]:
    dd = d17[sel].dropna(subset=[DVCOL, "dd_prod_early", "over1"])
    log(f"  [split=2017] {lab:6s} n={len(dd):2d} | 후반기낙폭 평균 {dd[DVCOL].mean():+7.1f} | "
        f"over1=1 평균낙폭 {dd.loc[dd.over1==1,DVCOL].mean():+7.1f} / over1=0 {dd.loc[dd.over1==0,DVCOL].mean():+7.1f} "
        f"(차이 {dd.loc[dd.over1==1,DVCOL].mean()-dd.loc[dd.over1==0,DVCOL].mean():+6.1f}, n_1={int(dd.over1.sum())})")


# ── 4. 대안설명 배제: 동조도 vs 규모 ────────────────────────────────────
log("\n" + "=" * 78)
log("[4] 대안 설명 배제 — 동조도는 단순히 '산업 규모'의 대리변수인가")
log("=" * 78)
d17 = panel_for(2017).dropna(subset=["sync_mfg", "ln_ship_val", DVCOL, "dd_prod_early"])
log(f"  (split=2017, n={len(d17)})  sync_mfg ~ ln(출하액) 회귀")
b, se, r2 = ols(d17[["ln_ship_val"]].values, d17.sync_mfg.values)
log(f"    β={b[1]:+.4f} (se {se[1]:.4f}, t {b[1]/se[1]:+.2f}), R²={r2:.3f}")
b2, se2, r22 = ols(d17[["ln_R"]].values, d17.sync_mfg.values)
log(f"  sync_mfg ~ ln(등록특허수): β={b2[1]:+.4f} (t {b2[1]/se2[1]:+.2f}), R²={r22:.3f}")

log("\n  규모 잔차화 동조도(sync_resid = sync_mfg 에서 ln출하액·ln등록특허 설명분 제거)로 상호작용 재적합")
res_r, res_size_only = [], []
for s in SPLITS:
    d = panel_for(s).dropna(subset=["sync_mfg", "ln_ship_val", "ln_R"])
    Z = np.column_stack([np.ones(len(d)), d.ln_ship_val.values, d.ln_R.values])
    bb, *_ = np.linalg.lstsq(Z, d.sync_mfg.values, rcond=None)
    d["_sc"] = d.sync_mfg.values - Z @ bb          # 규모와 직교인 동조도
    d["_ix"] = d["_sc"] * d["over1"]
    res_r.append(run(d, ["dd_prod_early", "over1", "_sc", "_ix"]))
    # 비교군: 규모 자체를 조절변수로
    d["_sz"] = d.ln_ship_val - d.ln_ship_val.mean()
    d["_szix"] = d["_sz"] * d["over1"]
    res_size_only.append(run(d, ["dd_prod_early", "over1", "_sz", "_szix"]))
summarize("동조도(규모 잔차화)×over1", res_r, keys=("over1", "_sc", "_ix"))
summarize("규모(ln출하액)×over1", res_size_only, keys=("over1", "_sz", "_szix"))

log("\n  규모 상·하위 분할 (동조도 분할과 대조)")
for half, lab in [(1, "대형(ln출하액 상위)"), (0, "소형(ln출하액 하위)")]:
    res = []
    for s in SPLITS:
        d = panel_for(s).dropna(subset=["ln_ship_val"])
        med = d.ln_ship_val.median()
        dd = d[d.ln_ship_val >= med] if half else d[d.ln_ship_val < med]
        res.append(run(dd, ["dd_prod_early", "over1"]))
    summarize(f"{lab}", res)


# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[5] 반론 검정 — 분할 층의 LOCO 붕괴는 '동조도' 때문인가, 그냥 절반으로 쪼개서인가")
log("=" * 78)
log("  플라시보: 47산업을 무작위 절반으로 나눠 기준 사양 LOCO 재계산 (500회 × 7 split)")
rng = np.random.default_rng(20260813)
PANELS = {s: panel_for(s).dropna(subset=[DVCOL, "dd_prod_early", "over1", "sync_mfg", "ln_ship_val"]) for s in SPLITS}
null_ind, null_mid = [], []
for it in range(500):
    v1, v2 = [], []
    for s in SPLITS:
        d = PANELS[s]
        # (a) 산업 단위 무작위 절반
        idx = rng.permutation(len(d)); half = len(d) // 2
        for sel in (d.iloc[idx[:half]], d.iloc[idx[half:]]):
            r = run(sel, ["dd_prod_early", "over1"])
            if r and not np.isnan(r["loco"]): v1.append(r["loco"])
        # (b) 중분류(mid) 단위 무작위 절반 — 클러스터 구조 유지
        ms = d.mids.unique(); rng.shuffle(ms); h = len(ms) // 2
        for grp in (ms[:h], ms[h:]):
            r = run(d[d.mids.isin(grp)], ["dd_prod_early", "over1"])
            if r and not np.isnan(r["loco"]): v2.append(r["loco"])
    if v1: null_ind.append(np.mean(v1))
    if v2: null_mid.append(np.mean(v2))
null_ind, null_mid = np.array(null_ind), np.array(null_mid)
log(f"    (a) 산업 무작위 절반  : LOCO 평균의 귀무분포 중위 {np.median(null_ind):+.3f}, "
    f"5%분위 {np.percentile(null_ind,5):+.3f}, 95%분위 {np.percentile(null_ind,95):+.3f} (반복 {len(null_ind)})")
log(f"    (b) 중분류 무작위 절반: LOCO 평균의 귀무분포 중위 {np.median(null_mid):+.3f}, "
    f"5%분위 {np.percentile(null_mid,5):+.3f}, 95%분위 {np.percentile(null_mid,95):+.3f} (반복 {len(null_mid)})")
for sv in SYNCVARS:
    res = []
    for s in SPLITS:
        d = PANELS[s]; med = d[sv].median()
        r = run(d[d[sv] >= med], ["dd_prod_early", "over1"])
        if r and not np.isnan(r["loco"]): res.append(r["loco"])
    if res:
        m = np.mean(res)
        p = float((null_ind <= m).mean())
        log(f"    {sv:14s} 상위층 LOCO 평균 {m:+.3f} → 귀무분포(a) 하위 {p*100:5.1f}% 위치")

log("\n  붕괴 주범 식별 — 산업 1개씩 제외했을 때 기준 사양 LOCO 평균 변화 (n=46, 모두 동일 표본크기)")
full = []
for s in SPLITS:
    r = run(PANELS[s], ["dd_prod_early", "over1"]); full.append(r["loco"])
full_m = np.nanmean(full)
log(f"    전체 47산업 기준 LOCO 평균 = {full_m:+.3f}")
loo = []
for k in sorted(PANELS[2017].ksic.unique()):
    v = []
    for s in SPLITS:
        r = run(PANELS[s][PANELS[s].ksic != k], ["dd_prod_early", "over1"])
        if r: v.append(r["loco"])
    loo.append({"ksic": int(k), "loco46": np.nanmean(v), "delta": np.nanmean(v) - full_m,
                "is_new": int(k in NEW8)})
LOO = pd.DataFrame(loo).sort_values("delta", ascending=False)
log(f"    {'ksic':>6s} {'신규':>4s} {'LOCO(n=46)':>11s} {'Δ':>7s}   (Δ 큰 순 = 제외하면 좋아지는 = 붕괴 기여)")
for _, r in LOO.head(10).iterrows():
    log(f"    {int(r.ksic):>6d} {'★' if r.is_new else '':>4s} {r.loco46:11.3f} {r.delta:+7.3f}")
log(f"    ... 하위 3개: " + " | ".join(f"{int(r.ksic)}{'★' if r.is_new else ''} Δ{r.delta:+.3f}" for _, r in LOO.tail(3).iterrows()))
log(f"    신규8 평균 Δ {LOO[LOO.is_new==1].delta.mean():+.4f} vs 기존39 평균 Δ {LOO[LOO.is_new==0].delta.mean():+.4f}")
log(f"    Δ 상위 8개 중 신규8 개수: {int(LOO.head(8).is_new.sum())}/8")

log("\n  조절변수 경합 — '동조도' vs '신규8 더미' vs '규모' 중 무엇이 이질성을 잡는가 (전부 n=47)")
for lab, mk in [("sync_mfg", lambda d: d.sync_mfg - d.sync_mfg.mean()),
                ("신규8더미", lambda d: d.ksic.isin(NEW8).astype(float)),
                ("ln출하액", lambda d: d.ln_ship_val - d.ln_ship_val.mean()),
                ("전자대분류더미(mid26)", lambda d: (d.mids == "26").astype(float))]:
    res = []
    for s in SPLITS:
        d = PANELS[s].copy(); d["_m"] = mk(d); d["_mix"] = d["_m"] * d["over1"]
        res.append(run(d, ["dd_prod_early", "over1", "_m", "_mix"]))
    summarize(f"조절변수={lab}", res, keys=("over1", "_m", "_mix"))

LOO.to_csv(os.path.join(FUS, "diag_globalcycle_loo.csv"), index=False, encoding="utf-8-sig")

# ── 5-c. 같은 n=39 끼리의 정당한 비교 ────────────────────────────────────
log("\n" + "─" * 78)
log("[5-C] ★ 같은 표본크기(n=39)끼리 비교 — '8개 제거' 라는 행위 자체의 효과를 제거")
log("─" * 78)


def loco_mean_excl(excl):
    v = []
    for s in SPLITS:
        d = PANELS[s]
        r = run(d[~d.ksic.isin(excl)], ["dd_prod_early", "over1"])
        v.append(r["loco"] if r else np.nan)
    return float(np.nanmean(v)), float(np.nanmin(v))


allk = sorted(PANELS[2017].ksic.unique())
m_new, mn_new = loco_mean_excl(NEW8)
sync_rank = base.set_index("ksic").sync_mfg.to_dict()
exp_rank = base.set_index("ksic").exp_pref.to_dict()
sz_rank = base.set_index("ksic").ln_ship_val.to_dict()
top8_sync = sorted(allk, key=lambda k: -sync_rank.get(k, -9))[:8]
top8_exp = sorted(allk, key=lambda k: -exp_rank.get(k, -9))[:8]
top8_sz = sorted(allk, key=lambda k: -sz_rank.get(k, -9))[:8]
null39 = []
for it in range(2000):
    ex = rng.choice(allk, 8, replace=False)
    m, _ = loco_mean_excl(set(ex.tolist()))
    if not np.isnan(m): null39.append(m)
null39 = np.array(null39)
log(f"  기준: 47산업 전체 LOCO 평균 = {full_m:+.3f} (n=47)")
log(f"  무작위 8개 제거(n=39) 귀무분포: 중위 {np.median(null39):+.3f}, "
    f"5% {np.percentile(null39,5):+.3f}, 95% {np.percentile(null39,95):+.3f} (반복 {len(null39)})")
for lab, ex in [("신규8 제거", set(NEW8)), ("동조도 상위8 제거", set(top8_sync)),
                ("수출선호 상위8 제거", set(top8_exp)), ("출하액 상위8 제거", set(top8_sz))]:
    m, mn = loco_mean_excl(ex)
    p = float((null39 >= m).mean())
    log(f"    {lab:16s} LOCO 평균 {m:+.3f} 최소 {mn:+.3f} (n=39) → 무작위8제거 분포에서 상위 {p*100:5.1f}%  "
        f"[제거대상 중 신규8 {len(ex & set(NEW8))}개]")

log("\n  동조도 상·하위 층의 구성 (split=2017 기준)")
d17 = PANELS[2017]
for sv in ["sync_mfg", "exp_pref", "vol_ship"]:
    med = d17[sv].median()
    hi = d17[d17[sv] >= med]; lo = d17[d17[sv] < med]
    log(f"    {sv:10s} 상위 n={len(hi):2d} (신규8 중 {hi.ksic.isin(NEW8).sum()}개, mid={sorted(hi.mids.unique())}) "
        f"sd(y)={hi[DVCOL].std():5.1f}")
    log(f"    {'':10s} 하위 n={len(lo):2d} (신규8 중 {lo.ksic.isin(NEW8).sum()}개, mid={sorted(lo.mids.unique())}) "
        f"sd(y)={lo[DVCOL].std():5.1f}")

log("\n  [해석 보조] 상·하위 층 LOCO 를 '동일 분모'로 재계산 — 전체표본 y분산 기준 R²")
for sv in SYNCVARS:
    for half, lab in [(1, "상위"), (0, "하위")]:
        vals = []
        for s in SPLITS:
            d = PANELS[s]; med = d[sv].median()
            dd = d[d[sv] >= med] if half else d[d[sv] < med]
            r = run(dd, ["dd_prod_early", "over1"])
            if r and not np.isnan(r["loco"]):
                mse = (1 - r["loco"]) * np.var(dd[DVCOL].values)
                vals.append(1 - mse / np.var(d[DVCOL].values))   # 분모 = 전체 47 표본 분산
        if vals:
            log(f"    {sv:14s} {lab} : 공통분모 R² 평균 {np.mean(vals):+.3f}")

log("\n  [해석 보조] 층별 β_over1 과 층 내 y 분산 (split=2017, 단일 분할연도라 참고용)")
for sv in SYNCVARS:
    med = d17[sv].median()
    for half, lab in [(1, "상위"), (0, "하위")]:
        dd = d17[d17[sv] >= med] if half else d17[d17[sv] < med]
        dd = dd.dropna(subset=[DVCOL, "dd_prod_early", "over1"])
        if dd.over1.nunique() < 2: continue
        b, se, r2 = ols(dd[["dd_prod_early", "over1"]].values, dd[DVCOL].values)
        log(f"    {sv:14s} {lab} n={len(dd):2d} (over1=1 {int(dd.over1.sum())}개) "
            f"β_over1={b[2]:+7.2f} (se {se[2]:5.2f}, t {b[2]/se[2]:+5.2f}) sd(y)={dd[DVCOL].std():5.1f}")

# ── 5. 저장 ─────────────────────────────────────────────────────────────
OUT = os.path.join(FUS, "diag_globalcycle_sync.csv")
SY17.merge(base[["k3s", "ksic", "is_new", "padi_e", "ln_ship_val", "ln_R"]], on="k3s", how="left").to_csv(
    OUT, index=False, encoding="utf-8-sig")
log(f"\n  동조도 지표 저장: {OUT}")
pd.DataFrame([r for r in SPLITRES]).to_csv(
    os.path.join(FUS, "diag_globalcycle_halves.csv"), index=False, encoding="utf-8-sig")
log("완료")
