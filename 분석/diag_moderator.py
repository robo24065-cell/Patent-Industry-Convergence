# -*- coding: utf-8 -*-
"""
각도 B — 무엇이 PADI 효과를 조절하는가 (H-B 대형전자 이질성 검정)
────────────────────────────────────────────────────────────────
[질문] 39→47 산업 확장 시 예측력이 떨어진 것이 '신규 8개(대형 전자·기계)의 이질성'
       때문이라면, 그 이질성을 만드는 **관측 가능한 산업 특성 Z** 가 있어야 한다.
       Z 를 찾아 상호작용 over1:Z 로 모형에 넣으면 LOCO 가 회복되어야 한다.

[주 기준] LOCO-CV R² 개선 (표본내 R²·p 는 보조)
[사양 고정] 기준 M0 : dd_ship_late ~ dd_prod_early + over1
            M1     : + Z
            M2     : + Z + over1:Z
[탐색 규모] 사전 고정 후보 15개 (규모4 · 글로벌3 · 집중도2 · 자본집약6). BH-FDR 적용.
            참고용으로 '신규8 더미'를 상한 벤치마크로 별도 보고(FDR 가족에서 제외).
[표본] 모든 표에 실제 사용 n 명기. 결측이 있는 Z 는 M0 도 같은 행으로 재계산해 비교.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
A    = os.path.join(BASE, "분석")
FUS  = os.path.join(A, "fusion")
DAT  = os.path.join(BASE, "공모전데이터", "작업데이터")
OUT  = FUS
PANEL_START = 2008
SPLITS = list(range(2013, 2020))
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
log = lambda m: print(m, flush=True)


# ══════════════════════════════════════════════════════════════
# 평가 함수 — 지시된 것을 그대로 사용 (수정 금지)
# ══════════════════════════════════════════════════════════════
def loco(X, y, g):
    """leave-one-cluster-out 예측 R². g는 클러스터 라벨(보통 mid)."""
    import numpy as np
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


def cluster_ols(X, y, cl, names):
    """상수항 포함 OLS + 클러스터 로버스트 SE(CR1). X는 상수항 제외."""
    Xd = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    e = y - Xd @ b
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    uq = np.unique(cl)
    for c in uq:
        m = cl == c
        s = Xd[m].T @ e[m]
        meat += np.outer(s, s)
    G, n, k = len(uq), len(y), Xd.shape[1]
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = XtX_inv @ meat @ XtX_inv * adj
    se = np.sqrt(np.maximum(np.diag(V), 0))
    from scipy import stats
    t = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
    p = 2 * stats.t.sf(np.abs(t), df=max(G - 1, 1))
    return pd.DataFrame({"var": ["const"] + list(names), "b": b, "se": se, "t": t, "p": p})


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    pv = p[idx]; m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[idx[order]] = np.minimum(ranked, 1.0)
    return q


# ══════════════════════════════════════════════════════════════
log("=" * 78)
log("[1] 재료 적재")
P = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig")
P["ksic"] = P.ksic.astype(int)
KS = sorted(P.ksic.unique())
log(f"  분석패널 {P.shape} | 산업 {len(KS)} | split {sorted(P.split.unique())}")

# --- 등록특허 원부 (규모·글로벌성·집중도 원천)
PT = pd.read_parquet(os.path.join(DAT, "kr_registered.parquet"),
                     columns=["출원연도", "세세분류코드", "패밀리출원번호", "출원인명"])
PT = PT[PT.세세분류코드.isin(KS)].copy()
PT["fam"] = PT.패밀리출원번호.notna().astype(float)
PT["app"] = PT.출원인명.fillna("(무명)").astype(str).str.replace(r"\s+", "", regex=True)
log(f"  등록특허 {len(PT):,}건 (47산업 한정) | 패밀리보유율 전체 {PT.fam.mean():.3f}")

# --- 광업제조업조사 (규모)
KO = pd.read_csv(os.path.join(DAT, "kosis_광업제조업_50개산업.csv"), encoding="utf-8-sig")
KO = KO[KO.세세분류코드.isin(KS)].copy()
ITEM = {"T01": "estab", "T02": "emp", "T04": "ship"}
KO = KO[KO.항목코드.isin(ITEM)].copy()
KO["item"] = KO.항목코드.map(ITEM)
log(f"  광업제조업조사 {len(KO):,}행 | 연도 {KO.연도.min()}~{KO.연도.max()}")

# --- 경쟁 연패널 (글로벌성)  ※ kr_share_e / cn_share_e 는 분석패널에 이미 존재
# --- 산업구조 (자본집약) — 시점불변(2008-2016). 전반기 한정 아님 → 한계로 명기
ST = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig")
ST["ksic"] = ST.ksic.astype(int)
ST = ST[ST.ksic.isin(KS)][["ksic", "ln_str_capital", "ln_str_capital_pw", "str_va_ratio",
                           "ln_str_estab_size", "ln_str_labor_prod", "str_wage_share"]].copy()
log(f"  산업구조 지표 {ST.shape} (2008-2016 시점불변)")


# ══════════════════════════════════════════════════════════════
log("\n[2] 후보 조절변수 구축 — 전반기(year < split) 정보만, 5자리 해상도")

def build_moderators(split):
    """split 이전 정보만으로 산업별 조절변수 후보를 만든다."""
    rows = []
    e = PT[(PT.출원연도 < split) & (PT.출원연도 >= 2000)]
    for k, g in e.groupby("세세분류코드"):
        n = len(g)
        sh = g.groupby("app").size().values / n
        rows.append({"ksic": k,
                     "ln_npat": np.log(n),
                     "fam_share": g.fam.mean(),
                     "hhi_app": float((sh ** 2).sum()),
                     "top5_share": float(np.sort(sh)[::-1][:5].sum())})
    M = pd.DataFrame(rows)

    ko = KO[(KO.연도 < split) & (KO.연도 >= PANEL_START)]
    kw = (ko.groupby(["세세분류코드", "item"])["값"].mean().unstack()
            .rename_axis(None, axis=1).reset_index().rename(columns={"세세분류코드": "ksic"}))
    for c in ["estab", "emp", "ship"]:
        kw["ln_" + c] = np.log(kw[c].where(kw[c] > 0))
    M = M.merge(kw[["ksic", "ln_estab", "ln_emp", "ln_ship"]], on="ksic", how="left")
    M = M.merge(ST, on="ksic", how="left")
    M["split"] = split
    return M

MOD = pd.concat([build_moderators(s) for s in SPLITS], ignore_index=True)
D = P.merge(MOD, on=["ksic", "split"], how="left")
D["over1"] = (D.padi_e > 1.0).astype(float)
D["new8"] = D.ksic.isin(NEW8).astype(float)
log(f"  결합 패널 {D.shape} | over1 비율 {D.over1.mean():.3f} | 신규8 행수 {int(D.new8.sum())}")

# 후보 목록 — 사전 고정 (탐색 규모 = 15). new8 더미는 참고용 상한, FDR 가족 제외.
CANDS = [
    ("ln_npat",           "규모",     "등록특허 보유건수(로그)"),
    ("ln_estab",          "규모",     "사업체수(로그)"),
    ("ln_emp",            "규모",     "종사자수(로그)"),
    ("ln_ship",           "규모",     "출하액(로그)"),
    ("fam_share",         "글로벌성", "해외패밀리 보유 특허 비중"),
    ("kr_share_e",        "글로벌성", "5개국 출원 중 한국 비중"),
    ("cn_share_e",        "글로벌성", "5개국 출원 중 중국 비중"),
    ("hhi_app",           "집중도",   "출원인 HHI"),
    ("top5_share",        "집중도",   "상위5 출원인 점유율"),
    ("ln_str_capital",    "자본집약", "유형자산(로그)"),
    ("ln_str_capital_pw", "자본집약", "1인당 유형자산(로그)"),
    ("str_va_ratio",      "자본집약", "부가가치율"),
    ("ln_str_estab_size", "자본집약", "사업체 평균규모(로그)"),
    ("ln_str_labor_prod", "자본집약", "노동생산성(로그)"),
    ("str_wage_share",    "자본집약", "임금몫"),
]
NCAND = len(CANDS)
log(f"  탐색 후보 개수 = {NCAND} (BH-FDR 적용 대상). 참고 벤치마크 'new8 더미' 1개는 가족 제외.")

miss = D[[c for c, _, _ in CANDS]].isna().sum()
log("  후보별 결측(329행 중):\n" + "    " + miss[miss > 0].to_string().replace("\n", "\n    ")
    if miss.sum() else "  후보 결측 없음 (329행 전부 사용 가능)")


# ══════════════════════════════════════════════════════════════
log("\n[3] 상호작용 모형 평가 — 주 기준 LOCO-CV R²")
Y = "dd_ship_late"

def zscore_within_split(df, col):
    """split 내에서 47산업 기준 표준화 (스케일이 split마다 다르므로)."""
    return df.groupby("split")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0))

def eval_cand(col):
    """M0/M1/M2 를 '동일 행'으로 split별 LOCO 평가 + 풀링 클러스터OLS."""
    sub = D.dropna(subset=[Y, "dd_prod_early", "over1", col]).copy()
    sub["Z"] = zscore_within_split(sub, col)
    sub["ZX"] = sub.Z * sub.over1
    res = {"var": col, "n_rows": len(sub), "n_ind": sub.ksic.nunique()}
    l0, l1, l2 = [], [], []
    for s in SPLITS:
        t = sub[sub.split == s]
        if len(t) < 15: continue
        y = t[Y].values; g = t["mid"].values
        l0.append(loco(t[["dd_prod_early", "over1"]].values, y, g))
        l1.append(loco(t[["dd_prod_early", "over1", "Z"]].values, y, g))
        l2.append(loco(t[["dd_prod_early", "over1", "Z", "ZX"]].values, y, g))
    for tag, v in [("M0", l0), ("M1", l1), ("M2", l2)]:
        v = np.array(v, float)
        res[f"loco_{tag}_mean"] = np.nanmean(v)
        res[f"loco_{tag}_min"] = np.nanmin(v)
    res["d_M2_M0"] = res["loco_M2_mean"] - res["loco_M0_mean"]
    res["d_M1_M0"] = res["loco_M1_mean"] - res["loco_M0_mean"]
    res["n_splits"] = len(l2)

    # 풀링 회귀 (split 더미 포함, 클러스터 SE = mid)
    dm = pd.get_dummies(sub.split, prefix="s", drop_first=True).astype(float)
    Xn = ["dd_prod_early", "over1", "Z", "ZX"] + list(dm.columns)
    Xm = np.column_stack([sub[["dd_prod_early", "over1", "Z", "ZX"]].values, dm.values])
    tab = cluster_ols(Xm, sub[Y].values, sub["mid"].values, Xn)
    r = tab[tab["var"] == "ZX"].iloc[0]
    res["b_int"] = r.b; res["se_int"] = r.se; res["p_int"] = r.p
    rz = tab[tab["var"] == "Z"].iloc[0]
    res["b_Z"] = rz.b; res["p_Z"] = rz.p
    return res

R = pd.DataFrame([eval_cand(c) for c, _, _ in CANDS])
R["cat"] = [c[1] for c in CANDS]; R["desc"] = [c[2] for c in CANDS]
R["q_int"] = bh_fdr(R.p_int.values)
R = R.sort_values("d_M2_M0", ascending=False).reset_index(drop=True)

log(f"\n  ── 후보 15개 결과 (종속 {Y}, 클러스터 SE = mid, G=13) ──")
hdr = f"{'조절변수':<20}{'범주':<10}{'n행':>5}{'n산업':>6}{'LOCO_M0':>9}{'LOCO_M2':>9}{'ΔM2-M0':>9}{'ΔM1-M0':>9}{'β_int':>9}{'p_int':>8}{'q(BH)':>8}"
log("  " + hdr)
for _, r in R.iterrows():
    log(f"  {r['var']:<20}{r['cat']:<10}{r.n_rows:>5}{r.n_ind:>6}"
        f"{r.loco_M0_mean:>9.3f}{r.loco_M2_mean:>9.3f}{r.d_M2_M0:>+9.3f}{r.d_M1_M0:>+9.3f}"
        f"{r.b_int:>+9.3f}{r.p_int:>8.3f}{r.q_int:>8.3f}")

log("\n  ── 최솟값(7분할 중 최악) 병기 ──")
for _, r in R.iterrows():
    log(f"  {r['var']:<20} M0 mean/min = {r.loco_M0_mean:+.3f}/{r.loco_M0_min:+.3f}   "
        f"M2 mean/min = {r.loco_M2_mean:+.3f}/{r.loco_M2_min:+.3f}")

# 참고 벤치마크 — 관측특성이 아니라 '신규8 더미' 자체를 조절변수로
log("\n  ── [참고·FDR 가족 제외] 상한 벤치마크: new8 더미를 조절변수로 ──")
bench = eval_cand("new8")
log(f"  new8 더미: n행={bench['n_rows']} n산업={bench['n_ind']} "
    f"LOCO_M0={bench['loco_M0_mean']:+.3f} LOCO_M1={bench['loco_M1_mean']:+.3f} "
    f"LOCO_M2={bench['loco_M2_mean']:+.3f} ΔM2-M0={bench['d_M2_M0']:+.3f} "
    f"β_int={bench['b_int']:+.3f} p={bench['p_int']:.3f}")
log(f"    (M2 min={bench['loco_M2_min']:+.3f})")

# 39산업 기준선 재현 — 표본크기 다른 비교 금지 원칙상 별도 블록으로만 보고
log("\n  ── [별도] 39산업 파일에서의 M0 기준선 (n이 다르므로 위 표와 병렬 비교 금지) ──")
P39 = pd.read_csv(os.path.join(FUS, "model_search_panel_39ind.csv"), encoding="utf-8-sig")
P39["over1"] = (P39.padi_e > 1.0).astype(float)
for tag, dd in [("47산업", D), ("39산업", P39)]:
    v = []
    for s in SPLITS:
        t = dd[dd.split == s].dropna(subset=[Y, "dd_prod_early", "over1"])
        v.append(loco(t[["dd_prod_early", "over1"]].values, t[Y].values, t["mid"].values))
    v = np.array(v, float)
    log(f"    {tag}: n행={len(dd)} n산업={dd.ksic.nunique()} "
        f"LOCO_M0 mean={np.nanmean(v):+.3f} min={np.nanmin(v):+.3f}")


# ══════════════════════════════════════════════════════════════
# [3b] 조절변수가 왜 안 듣는가 — '기울기 이질성'인가 '예측 난이도'인가
#      조절변수(상호작용)는 기울기 이질성만 고칠 수 있다. 신규8이 단지 예측이
#      어려운(잔차 큰) 산업이라면 어떤 Z 를 넣어도 LOCO 는 회복되지 않는다.
log("\n[3b] 오차 분해 — 신규8은 '기울기가 다른가' 아니면 '그냥 예측이 안 되는가'")
sub = D.dropna(subset=[Y, "dd_prod_early", "over1"]).copy()
rows_a, rows_b = [], []
for s in SPLITS:
    t = sub[sub.split == s].copy()
    y = t[Y].values; g = t["mid"].values
    Xm = t[["dd_prod_early", "over1"]].values
    # (a) 47산업 LOCO 예측오차를 신규8 / 기존39 로 분해
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        Xt = np.column_stack([np.ones(tr.sum()), Xm[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), Xm[te]]) @ b
    isnew = t.new8.values == 1
    rows_a.append({"split": s,
                   "rmse_new8": np.sqrt(np.nanmean((y[isnew] - pr[isnew]) ** 2)),
                   "rmse_old39": np.sqrt(np.nanmean((y[~isnew] - pr[~isnew]) ** 2)),
                   "sd_y_new8": y[isnew].std(), "sd_y_old39": y[~isnew].std()})
    # (b) 동일 평가행(기존39)에서 훈련표본만 바꿔 '오염' 여부 확인
    ev = ~isnew
    e_all, e_39 = [], []
    for c in uq:
        te = (g == c) & ev
        if te.sum() == 0: continue
        tr_all = (g != c)
        tr_39 = (g != c) & (~isnew)
        for trm, box in [(tr_all, e_all), (tr_39, e_39)]:
            Xt = np.column_stack([np.ones(trm.sum()), Xm[trm]])
            b = np.linalg.lstsq(Xt, y[trm], rcond=None)[0]
            box.append(y[te] - (np.column_stack([np.ones(te.sum()), Xm[te]]) @ b))
    e_all = np.concatenate(e_all); e_39 = np.concatenate(e_39)
    yy = y[ev]; sst = ((yy - yy.mean()) ** 2).sum()
    rows_b.append({"split": s, "n_eval": int(ev.sum()),
                   "R2_train47": 1 - (e_all ** 2).sum() / sst,
                   "R2_train39": 1 - (e_39 ** 2).sum() / sst})
Adf = pd.DataFrame(rows_a); Bdf = pd.DataFrame(rows_b)
log(f"  (a) 47산업 M0 의 LOCO 예측오차 RMSE (전체 329행, 신규8=56행 / 기존39=273행)")
log(f"      신규8  RMSE 평균 = {Adf.rmse_new8.mean():.2f}  (종속변수 표준편차 {Adf.sd_y_new8.mean():.2f})")
log(f"      기존39 RMSE 평균 = {Adf.rmse_old39.mean():.2f}  (종속변수 표준편차 {Adf.sd_y_old39.mean():.2f})")
log(f"      → 신규8 오차/기존39 오차 = {Adf.rmse_new8.mean()/Adf.rmse_old39.mean():.2f}배")
log(f"  (b) 평가행을 기존39(=행 {Bdf.n_eval.iloc[0]}개)로 고정하고 훈련표본만 교체")
log(f"      훈련=47산업 → LOCO R² 평균 {Bdf.R2_train47.mean():+.3f} (최소 {Bdf.R2_train47.min():+.3f})")
log(f"      훈련=39산업 → LOCO R² 평균 {Bdf.R2_train39.mean():+.3f} (최소 {Bdf.R2_train39.min():+.3f})")
log(f"      차이(오염분) = {Bdf.R2_train47.mean()-Bdf.R2_train39.mean():+.3f}")


# ══════════════════════════════════════════════════════════════
log("\n[4] 신규 8개는 그 조절변수에서 실제로 극단인가 (split=2016 단면, 47산업)")
snap = D[D.split == 2016].copy()
pc_rows = []
for c, cat, desc in CANDS:
    s = snap.dropna(subset=[c])
    if len(s) < 40: continue
    rk = s[c].rank(pct=True) * 100
    new = rk[s.ksic.isin(NEW8)]
    old = rk[~s.ksic.isin(NEW8)]
    pc_rows.append({"var": c, "cat": cat, "n_ind": len(s),
                    "신규8_중위백분위": new.median(), "신규8_최소": new.min(), "신규8_최대": new.max(),
                    "기존39_중위백분위": old.median(),
                    "신규8_상위25%내_개수": int((new >= 75).sum()),
                    "신규8_하위25%내_개수": int((new <= 25).sum())})
PC = pd.DataFrame(pc_rows).sort_values("신규8_중위백분위", ascending=False)
log(f"  {'조절변수':<20}{'범주':<10}{'신규8중위%':>11}{'기존39중위%':>12}{'신규8범위%':>16}{'상위25%수':>10}{'하위25%수':>10}")
for _, r in PC.iterrows():
    rng = f"{r['신규8_최소']:.0f}~{r['신규8_최대']:.0f}"
    log(f"  {r['var']:<20}{r['cat']:<10}{r['신규8_중위백분위']:>11.1f}{r['기존39_중위백분위']:>12.1f}"
        f"{rng:>16}{r['신규8_상위25%내_개수']:>10}{r['신규8_하위25%내_개수']:>10}")

top = R.iloc[0]["var"]
log(f"\n  ── 최유망(ΔLOCO 최대) 조절변수 '{top}' 의 47산업 분포와 신규8 위치 (split=2016) ──")
s = snap.dropna(subset=[top]).sort_values(top).reset_index(drop=True)
s["pct"] = s[top].rank(pct=True) * 100
for _, r in s.iterrows():
    mark = " ★신규" if r.ksic in NEW8 else ""
    log(f"    {int(r.ksic):>6}  {top}={r[top]:>10.3f}  백분위={r.pct:>5.1f}{mark}")


# ══════════════════════════════════════════════════════════════
log("\n[5] 판정")
THR = 0.02
win = R[R.d_M2_M0 >= THR]
log(f"  기준: ΔLOCO(M2-M0) >= +{THR:.2f} 를 '실질 개선'으로 사전 정의")
if len(win):
    log(f"  → 실질 개선 후보 {len(win)}개: {', '.join(win['var'])}")
    for _, r in win.iterrows():
        log(f"     {r['var']}: Δ={r.d_M2_M0:+.3f}, β_int={r.b_int:+.3f}, p={r.p_int:.3f}, q(BH)={r.q_int:.3f}, n={r.n_rows}")
else:
    log(f"  → 실질 개선 후보 없음. 최대 Δ = {R.d_M2_M0.max():+.3f} ({R.iloc[0]['var']})")
log(f"  BH-FDR q<0.10 인 상호작용: {int((R.q_int < 0.10).sum())}개 / {NCAND}개")
log(f"  참고 상한(new8 더미 상호작용) Δ = {bench['d_M2_M0']:+.3f}")

R.to_csv(os.path.join(OUT, "diag_moderator_results.csv"), index=False, encoding="utf-8-sig")
PC.to_csv(os.path.join(OUT, "diag_moderator_percentile.csv"), index=False, encoding="utf-8-sig")
log(f"\n  저장: {os.path.join(OUT, 'diag_moderator_results.csv')}")
log(f"  저장: {os.path.join(OUT, 'diag_moderator_percentile.csv')}")
log("=" * 78)
