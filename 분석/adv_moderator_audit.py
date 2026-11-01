# -*- coding: utf-8 -*-
"""
적대적 검증 — diag_moderator.py 의 결론('H-B 지지: 하락은 신규8에 국지화')에 대한 반증 시도
검증 렌즈: 대안설명 (제3의 설명이 있는가)

A. 처치변수 재측정 효과: padi_e 는 표본 확장 시 기존 39개에서도 값이 바뀐다(E_adj 풀링 추정).
B. 클러스터 구조 효과: mid=26 이 8→13 산업으로 커짐. LOCO 폴드 난이도 자체가 변함.
C. 위약(placebo): 기존 39개에서 무작위 8개를 뽑아도 '집단내 R2≈0.06'이 흔한가.
D. 범위축소/신호분산: 신규8 내부에서 예측변수(dd_prod_early, over1)의 분산이 얼마나 되는가.
E. 종속변수 성질: 신규8의 dd_ship_late 분포·시계열 상관.
F. H-A 직접 검정: 규모 층별 관계 강도.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
DAT  = os.path.join(BASE, "공모전데이터", "작업데이터")
SPLITS = list(range(2013, 2020))
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
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
    """LOCO 예측값 벡터 반환 (분해용)."""
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    return pr


P47 = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig")
P39 = pd.read_csv(os.path.join(FUS, "model_search_panel_39ind.csv"), encoding="utf-8-sig")
for d in (P47, P39):
    d["ksic"] = d.ksic.astype(int)
    d["over1"] = (d.padi_e > 1.0).astype(float)
P47["new8"] = P47.ksic.isin(NEW8).astype(int)
Y = "dd_ship_late"; XC = ["dd_prod_early", "over1"]

log("=" * 84)
log("[A] 처치변수 재측정 — '39→47 하락'이 표본구성이 아니라 padi_e 값 변화 탓일 가능성")
m = P47.merge(P39, on=["ksic", "split"], suffixes=("_47", "_39"))
log(f"  공유 273행: padi_e 최대절대차 {np.abs(m.padi_e_47-m.padi_e_39).max():.4f}, "
    f"over1 뒤바뀐 행 {(m.over1_47!=m.over1_39).sum()}개 (전부 0→1)")
log(f"  dd_ship_late / dd_prod_early 는 완전 동일(최대차 0) → 종속·주설명변수는 재측정 없음")
# 동일 39산업·동일 평가행에서 over1 판본만 교체
for tag, col, src in [("over1_39판본", "over1_39", m), ("over1_47판본", "over1_47", m)]:
    v = []
    for s in SPLITS:
        t = src[src.split == s]
        v.append(loco(t[["dd_prod_early_47", col]].values, t[Y+"_47"].values, t["mid_47"].values))
    v = np.array(v, float)
    log(f"  39산업 273행, {tag}: LOCO 평균 {np.nanmean(v):+.3f} 최소 {np.nanmin(v):+.3f}")
log("  → 재측정 기여분은 위 두 줄의 차이로 읽는다.")

log("\n" + "=" * 84)
log("[B] 클러스터 구조 효과 — 하락이 '신규8'이 아니라 'mid=26/27/28 폴드'에 있는 것 아닌가")
c47 = P47.drop_duplicates("ksic").mid.value_counts().sort_index()
c39 = P39.drop_duplicates("ksic").mid.value_counts().sort_index()
log("  산업수/클러스터  " + "  ".join(f"{int(k)}:{c39.get(k,0)}→{c47.get(k,0)}" for k in c47.index))
log(f"  최대 클러스터 비중: 39산업 {c39.max()}/39={c39.max()/39:.1%}  →  47산업 {c47.max()}/47={c47.max()/47:.1%}")

# 47 패널 M0 LOCO 예측오차를 (신규8) x (mid그룹) 으로 교차 분해
rows = []
for s in SPLITS:
    t = P47[P47.split == s].dropna(subset=[Y]+XC).copy()
    y = t[Y].values; g = t["mid"].values
    pr = loco_pred(t[XC].values, y, g)
    t = t.assign(err=y-pr, pred=pr)
    rows.append(t)
E = pd.concat(rows)
E["grp26"] = E.mid.isin([26, 27, 28])


def blk(mask, name):
    d = E[mask]
    rmse = np.sqrt((d.err ** 2).mean()); sd = d[Y].std()
    return {"블록": name, "n행": len(d), "n산업": d.ksic.nunique(),
            "RMSE": rmse, "y_SD": sd, "집단내R2": 1 - rmse ** 2 / sd ** 2,
            "y_평균": d[Y].mean(), "편향(err평균)": d.err.mean()}


tab = pd.DataFrame([
    blk(E.new8 == 1, "신규8 (전부 mid26/27/28)"),
    blk((E.new8 == 0) & E.grp26, "기존39 중 mid26/27/28"),
    blk((E.new8 == 0) & ~E.grp26, "기존39 중 그 외 mid"),
    blk(E.new8 == 0, "기존39 전체"),
])
log("\n  ── 47산업 M0 의 LOCO 오차 교차분해 ──")
log(tab.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
log("  ※ 기존39 중 mid26/27/28 이 신규8과 비슷하게 나쁘면 '신규8 이질성'이 아니라 '클러스터 난이도'다.")

# 클러스터를 mid 대신 ksic(산업)로 바꾼 LOCO — 폴드 크기 효과 제거
log("\n  ── 클러스터 라벨 교체: mid(13개) vs ksic(=leave-one-industry-out) ──")
for tag, d in [("47산업", P47), ("39산업", P39)]:
    for gl in ["mid", "ksic"]:
        v = []
        for s in SPLITS:
            t = d[d.split == s].dropna(subset=[Y]+XC)
            v.append(loco(t[XC].values, t[Y].values, t[gl].values))
        v = np.array(v, float)
        log(f"    {tag} 클러스터={gl:5s}: LOCO 평균 {np.nanmean(v):+.3f} 최소 {np.nanmin(v):+.3f}  (n행={len(d)})")

log("\n" + "=" * 84)
log("[C] 위약검정 — 기존39에서 무작위 8개를 뽑아도 '집단내 R2≈0.06'이 흔한가")
rng = np.random.default_rng(20260813)
old_ind = sorted(set(P47.ksic) - set(NEW8))
obs = blk(E.new8 == 1, "x")["집단내R2"]
stat = []
for _ in range(4000):
    pick = rng.choice(old_ind, 8, replace=False)
    d = E[E.ksic.isin(pick)]
    rmse = np.sqrt((d.err ** 2).mean()); sd = d[Y].std()
    stat.append(1 - rmse ** 2 / sd ** 2 if sd > 0 else np.nan)
stat = np.array(stat, float)
log(f"  신규8 실측 집단내 R2 = {obs:+.3f}")
log(f"  무작위 8개(기존39에서, 4000회) 집단내 R2 분포: "
    f"평균 {np.nanmean(stat):+.3f}  중위 {np.nanmedian(stat):+.3f}  "
    f"5%={np.nanpercentile(stat,5):+.3f}  25%={np.nanpercentile(stat,25):+.3f}  "
    f"75%={np.nanpercentile(stat,75):+.3f}")
log(f"  P(무작위8 ≤ 신규8 관측치) = {np.nanmean(stat <= obs):.3f}   "
    f"(=위약 p값. 크면 '신규8이 특별히 나쁘다'는 주장 약함)")

# RMSE 비만으로도 동일 위약
obs_r = blk(E.new8 == 1, "x")["RMSE"]
rs = []
for _ in range(4000):
    pick = rng.choice(old_ind, 8, replace=False)
    d = E[E.ksic.isin(pick)]
    rs.append(np.sqrt((d.err ** 2).mean()))
rs = np.array(rs)
log(f"  RMSE 기준: 신규8 {obs_r:.2f} vs 무작위8 중위 {np.median(rs):.2f}, "
    f"P(무작위 ≥ 신규8) = {np.mean(rs >= obs_r):.3f}")

log("\n" + "=" * 84)
log("[D] 신호 분산 — 신규8 내부에서 설명변수가 애초에 움직이는가 (범위축소)")
for name, mask in [("신규8", E.new8 == 1), ("기존39", E.new8 == 0)]:
    d = E[mask]
    log(f"  {name}: dd_prod_early SD={d.dd_prod_early.std():6.2f}  범위 {d.dd_prod_early.min():.1f}~{d.dd_prod_early.max():.1f}")
    log(f"        over1 평균={d.over1.mean():.3f} (분산={d.over1.var():.4f})  "
        f"dd_ship_late SD={d[Y].std():6.2f}")
    log(f"        corr(dd_prod_early, dd_ship_late) = {d.dd_prod_early.corr(d[Y]):+.3f}")
# 신규8 내부 산업별 고유 정보량: 산업 8개 각각의 (x,y) 를 split 평균으로 축약
log("\n  ── 산업단위(split 평균) 상관: 반복측정 중복 제거 ──")
for name, mask in [("신규8(n=8)", E.new8 == 1), ("기존39(n=39)", E.new8 == 0)]:
    a = E[mask].groupby("ksic")[["dd_prod_early", Y, "over1"]].mean()
    log(f"  {name}: corr(x,y)={a.dd_prod_early.corr(a[Y]):+.3f}  "
        f"x_SD={a.dd_prod_early.std():.2f}  y_SD={a[Y].std():.2f}  over1평균={a.over1.mean():.2f}")

log("\n" + "=" * 84)
log("[E] 종속변수 성질 — 신규8의 y 가 다른 척도/다른 구조인가")
a = E.groupby(["new8", "split"])[Y].agg(["mean", "std"]).reset_index()
log(a.pivot(index="split", columns="new8", values=["mean", "std"]).to_string(float_format=lambda x: f"{x:7.2f}"))
log("\n  산업별 y 평균 (신규8):")
log(E[E.new8 == 1].groupby("ksic")[[Y, "dd_prod_early", "over1", "padi_e"]].mean().to_string(float_format=lambda x: f"{x:8.2f}"))

log("\n" + "=" * 84)
log("[F] H-A 직접 검정 — 규모 층별로 관계 강도가 다른가 (저자가 검정하지 않은 축)")
KO = pd.read_csv(os.path.join(DAT, "kosis_광업제조업_50개산업.csv"), encoding="utf-8-sig")
KO = KO[(KO.항목코드 == "T04") & (KO.연도.between(2008, 2012))]
size = KO.groupby("세세분류코드")["값"].mean().rename("ship").reset_index().rename(columns={"세세분류코드": "ksic"})
S = P47.merge(size, on="ksic", how="left")
S["ln_ship"] = np.log(S.ship.where(S.ship > 0))
med = S.drop_duplicates("ksic").ln_ship.median()
S["small"] = (S.ln_ship <= med).astype(int)
log(f"  47산업 출하액(2008-12 평균) 중위 로그값 {med:.2f}. 소형/대형 각 산업수: "
    f"{S[S.small==1].ksic.nunique()}/{S[S.small==0].ksic.nunique()}")
log(f"  신규8 중 소형에 속한 수: {S[(S.new8==1)&(S.small==1)].ksic.nunique()} / 8")
for lab, msk in [("소형층", S.small == 1), ("대형층", S.small == 0)]:
    v = []
    for s in SPLITS:
        t = S[(S.split == s) & msk].dropna(subset=[Y]+XC)
        v.append(loco(t[XC].values, t[Y].values, t["mid"].values))
    v = np.array(v, float)
    d = S[msk]
    log(f"  {lab}: n산업={d.ksic.nunique()} 층내 LOCO 평균 {np.nanmean(v):+.3f} "
        f"(유효 split {np.sum(~np.isnan(v))}/7)  corr(x,y)={d.dd_prod_early.corr(d[Y]):+.3f}")
# 층별 기울기 (풀링, split더미)
for lab, msk in [("소형층", S.small == 1), ("대형층", S.small == 0)]:
    d = S[msk].dropna(subset=[Y]+XC)
    dm = pd.get_dummies(d.split, drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), d[XC].values, dm.values])
    b = np.linalg.lstsq(X, d[Y].values, rcond=None)[0]
    log(f"  {lab} 기울기: dd_prod_early={b[1]:+.3f}  over1={b[2]:+.3f}  (n행={len(d)})")

log("\n" + "=" * 84)
log("[G] R2 하락의 산술 분해 — 분모(SST) 변화 vs 분자(SSE) 변화")
for s in [2016]:
    pass
r = []
for s in SPLITS:
    t = E[E.split == s]
    y = t[Y].values; e = t.err.values; nw = t.new8.values == 1
    sst_all = ((y - y.mean()) ** 2).sum(); sse_all = (e ** 2).sum()
    yo = y[~nw]; eo = e[~nw]
    sst_o = ((yo - yo.mean()) ** 2).sum(); sse_o = (eo ** 2).sum()
    r.append({"split": s, "R2_47": 1 - sse_all / sst_all,
              "R2_47평가는기존39만": 1 - sse_o / sst_o,
              "SST비중_신규8": 1 - sst_o / sst_all,
              "SSE비중_신규8": 1 - sse_o / sse_all})
r = pd.DataFrame(r)
log(r.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
log(f"  평균: 신규8은 SST의 {r.SST비중_신규8.mean():.1%} 를 차지하지만 SSE의 {r.SSE비중_신규8.mean():.1%} 를 차지")
log("=" * 84)
