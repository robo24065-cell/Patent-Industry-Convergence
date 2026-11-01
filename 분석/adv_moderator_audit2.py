# -*- coding: utf-8 -*-
"""
적대적 검증 2 — '하락은 신규8에 국지화' / '신규8은 신호 부재' 두 주장의 반증
제3의 설명 후보: (X) 전자블록(mid 26/27/28) 자체가 원래 예측 불가였고,
                확장이 그 블록의 표본 비중을 30.8% → 42.6% 로 키웠을 뿐.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
DAT = os.path.join(BASE, "공모전데이터", "작업데이터")
SPLITS = list(range(2013, 2020))
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
ELEC = [26, 27, 28]
Y = "dd_ship_late"; XC = ["dd_prod_early", "over1"]
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
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    return pr


def panel_loco(d, cols=XC, gl="mid"):
    v = []
    for s in SPLITS:
        t = d[d.split == s].dropna(subset=[Y]+list(cols))
        v.append(loco(t[list(cols)].values, t[Y].values, t[gl].values))
    v = np.array(v, float)
    return np.nanmean(v), np.nanmin(v)


P47 = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig")
P39 = pd.read_csv(os.path.join(FUS, "model_search_panel_39ind.csv"), encoding="utf-8-sig")
for d in (P47, P39):
    d["ksic"] = d.ksic.astype(int); d["over1"] = (d.padi_e > 1.0).astype(float)
    d["elec"] = d.mid.isin(ELEC).astype(int)
P47["new8"] = P47.ksic.isin(NEW8).astype(int)

# 47패널 LOCO 오차
rows = []
for s in SPLITS:
    t = P47[P47.split == s].dropna(subset=[Y]+XC).copy()
    t["err"] = t[Y].values - loco_pred(t[XC].values, t[Y].values, t["mid"].values)
    rows.append(t)
E = pd.concat(rows)
# 39패널 LOCO 오차
rows = []
for s in SPLITS:
    t = P39[P39.split == s].dropna(subset=[Y]+XC).copy()
    t["err"] = t[Y].values - loco_pred(t[XC].values, t[Y].values, t["mid"].values)
    rows.append(t)
E39 = pd.concat(rows)

log("=" * 86)
log("[1] 전자블록(mid 26/27/28)은 확장 '이전'에도 이미 예측 불가였는가")
for tag, D in [("39산업 패널", E39), ("47산업 패널", E)]:
    for lab, msk in [("전자블록", D.elec == 1), ("비전자블록", D.elec == 0)]:
        d = D[msk]
        rmse = np.sqrt((d.err**2).mean()); sd = d[Y].std()
        log(f"  {tag} | {lab:6s}: n산업={d.ksic.nunique():>2} n행={len(d):>3} "
            f"행비중={len(d)/len(D):5.1%}  RMSE={rmse:6.2f}  y_SD={sd:6.2f}  "
            f"집단내R2={1-rmse**2/sd**2:+.3f}  편향={d.err.mean():+6.2f}")
log("  → 39산업에서도 전자블록이 이미 나빴다면 '신규8 이질성'은 원인이 아니다.")

log("\n[2] 47패널 전자블록 내부를 '신규8' vs '기존12' 로 다시 쪼갠다")
for lab, msk in [("전자블록·신규8", (E.elec == 1) & (E.new8 == 1)),
                 ("전자블록·기존12", (E.elec == 1) & (E.new8 == 0))]:
    d = E[msk]
    rmse = np.sqrt((d.err**2).mean()); sd = d[Y].std()
    log(f"  {lab:16s}: n산업={d.ksic.nunique():>2} n행={len(d):>3} RMSE={rmse:6.2f} "
        f"y_SD={sd:6.2f} 집단내R2={1-rmse**2/sd**2:+.3f}")

log("\n[3] 조건부 위약 — 전자블록 20개 중 무작위 8개를 뽑아도 신규8만큼 나쁜가")
rng = np.random.default_rng(7)
elec_ind = sorted(E[E.elec == 1].ksic.unique())
d = E[E.new8 == 1]; obs = np.sqrt((d.err**2).mean()); obs_r2 = 1 - obs**2/d[Y].std()**2
st = []
for _ in range(4000):
    pk = rng.choice(elec_ind, 8, replace=False)
    dd = E[E.ksic.isin(pk)]
    r = np.sqrt((dd.err**2).mean()); st.append(1 - r**2/dd[Y].std()**2)
st = np.array(st, float)
log(f"  신규8 집단내R2 = {obs_r2:+.3f} (RMSE {obs:.2f})")
log(f"  전자블록 20개에서 무작위8: 중위 {np.nanmedian(st):+.3f}  "
    f"25%={np.nanpercentile(st,25):+.3f} 75%={np.nanpercentile(st,75):+.3f}")
log(f"  P(무작위8 ≤ 신규8) = {np.nanmean(st <= obs_r2):.3f}  "
    f"← 앞선 '기존39 전체에서 뽑기' 위약(p=0.050)과 대조")

log("\n[4] 8개 제거 실험 — 47산업에서 8개를 빼면 LOCO 는 얼마나 오르는가")
base_m, _ = panel_loco(P47)
log(f"  47산업 전체 LOCO = {base_m:+.3f}")
res_new8, _ = panel_loco(P47[P47.new8 == 0])
log(f"  신규8 제거(=기존39) LOCO = {res_new8:+.3f}   (상승 {res_new8-base_m:+.3f})")
all_ind = sorted(P47.ksic.unique())
draws = []
for _ in range(600):
    pk = rng.choice(all_ind, 8, replace=False)
    m, _ = panel_loco(P47[~P47.ksic.isin(pk)])
    draws.append(m)
draws = np.array(draws, float)
log(f"  무작위 8개 제거(600회): 중위 {np.nanmedian(draws):+.3f}  "
    f"90%={np.nanpercentile(draws,90):+.3f}  95%={np.nanpercentile(draws,95):+.3f}  "
    f"최대={np.nanmax(draws):+.3f}")
log(f"  P(무작위 8개 제거가 신규8 제거보다 더 좋음) = {np.nanmean(draws >= res_new8):.3f}")
draws2 = []
for _ in range(600):
    pk = rng.choice(elec_ind, 8, replace=False)
    m, _ = panel_loco(P47[~P47.ksic.isin(pk)])
    draws2.append(m)
draws2 = np.array(draws2, float)
log(f"  전자블록에서 무작위 8개 제거: 중위 {np.nanmedian(draws2):+.3f}  "
    f"P(≥ 신규8 제거) = {np.nanmean(draws2 >= res_new8):.3f}")

log("\n[5] '신호 부재'인가 — 신규8 내부에 관계가 실제로 존재하는가")
d8 = E[E.new8 == 1]
for lab, dd in [("신규8(56행/8산업)", d8), ("기존39(273행)", E[E.new8 == 0])]:
    X = np.column_stack([np.ones(len(dd)), dd[XC].values])
    b = np.linalg.lstsq(X, dd[Y].values, rcond=None)[0]
    e = dd[Y].values - X @ b
    r2 = 1 - (e**2).sum()/((dd[Y]-dd[Y].mean())**2).sum()
    log(f"  {lab:20s} 표본내 OLS: b_x={b[1]:+.3f} b_over1={b[2]:+.3f} R2={r2:+.3f}")
# 신규8 내부 leave-one-industry-out (자기 집단으로 훈련)
y = d8[Y].values; Xm = d8[XC].values; g = d8.ksic.values
pr = np.full(len(y), np.nan)
for c in np.unique(g):
    te = g == c; tr = ~te
    Xt = np.column_stack([np.ones(tr.sum()), Xm[tr]])
    b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
    pr[te] = np.column_stack([np.ones(te.sum()), Xm[te]]) @ b
r2 = 1 - ((y-pr)**2).sum()/((y-y.mean())**2).sum()
log(f"  신규8 내부 leave-one-industry-out 예측 R2 = {r2:+.3f}  "
    f"(자기 집단 계수로 훈련하면 예측되는가)")
# 오차 분해: 편향(수준이동) vs 분산
bias = d8.err.mean(); mse = (d8.err**2).mean()
log(f"  신규8 LOCO 오차 분해: MSE={mse:.1f} 중 편향제곱={bias**2:.1f}({bias**2/mse:.1%}) "
    f"잔여분산={mse-bias**2:.1f}")

log("\n[6] 오라클(new8 더미) 실패는 '틀의 실패'인가 '추정력 부족'인가")
D = P47.copy(); D["nx"] = D.new8 * D.over1
log("  신규8 산업은 mid 26(5개)/27(2개)/28(1개) 에만 존재 → mid 클러스터를 빼면 훈련표본에서")
log("  new8 산업이 최대 5개까지 통째로 사라진다. 클러스터를 ksic 로 바꿔 추정력을 회복시켜 본다.")
for gl in ["mid", "ksic"]:
    m0, n0 = panel_loco(D, XC, gl)
    m1, n1 = panel_loco(D, XC + ["new8"], gl)
    m2, n2 = panel_loco(D, XC + ["new8", "nx"], gl)
    log(f"  클러스터={gl:5s}: M0={m0:+.3f} M1(+더미)={m1:+.3f} M2(+더미×over1)={m2:+.3f} "
        f"ΔM1-M0={m1-m0:+.3f} ΔM2-M0={m2-m0:+.3f}")
# 전자블록 더미도 같이
D["elecx"] = D.elec * D.over1
for gl in ["mid", "ksic"]:
    m0, _ = panel_loco(D, XC, gl)
    m1, _ = panel_loco(D, XC + ["elec"], gl)
    m2, _ = panel_loco(D, XC + ["elec", "elecx"], gl)
    log(f"  [전자블록 더미] 클러스터={gl:5s}: M0={m0:+.3f} M1={m1:+.3f} M2={m2:+.3f} "
        f"ΔM1-M0={m1-m0:+.3f} ΔM2-M0={m2-m0:+.3f}")

log("\n[7] H-B 전제 점검 — 신규8은 정말 '전부 대형'인가 (출하액 기준)")
KO = pd.read_csv(os.path.join(DAT, "kosis_광업제조업_50개산업.csv"), encoding="utf-8-sig")
KO = KO[(KO.항목코드 == "T04") & (KO.연도.between(2008, 2012))]
sz = KO.groupby("세세분류코드")["값"].mean().rename("ship").reset_index().rename(columns={"세세분류코드": "ksic"})
S = P47.drop_duplicates("ksic")[["ksic", "mid", "new8"]].merge(sz, on="ksic", how="left")
S["pct"] = S.ship.rank(pct=True)*100
log(f"  47산업 출하액 백분위 — 신규8 중위 {S[S.new8==1].pct.median():.1f}, "
    f"기존39 중위 {S[S.new8==0].pct.median():.1f}")
log("  신규8 개별: " + "  ".join(f"{int(r.ksic)}:{r.pct:.0f}%" for _, r in S[S.new8 == 1].sort_values('pct').iterrows()))
log(f"  기존39가 '소형 편향'인가: 47산업 중 하위50%에 기존39가 {int((S[S.new8==0].pct<=50).sum())}/39, "
    f"신규8이 {int((S[S.new8==1].pct<=50).sum())}/8")
log("=" * 86)
