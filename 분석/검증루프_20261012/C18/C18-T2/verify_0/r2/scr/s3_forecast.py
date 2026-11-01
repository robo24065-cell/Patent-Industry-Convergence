# [C18-T2 verify_0 r2 copy] HERE, MC changed
# -*- coding: utf-8 -*-
r"""split34 3단계: 예측 관점 — 연계표 해상도(그룹 신호) vs 제공 5자리 신호, 전진 검증.
(3a) 특허 미래: 목표 = 세세분류 k의 향후 3년 출원(평균 t0+1..t0+3) 대비 기준(평균 t0-2..t0) 로그 성장.
     특징은 출원연도 e=t0-L까지만(L=2 주: 공개 18개월 지연 반영, L=0 민감도). 학습 원점 s ≤ t0-3-L(그 목표가 e 시점에 관측 완료).
     G 모형 = 그룹 합계 특징만(연계표 해상도 분석자가 가질 수 있는 최선), M 모형 = G + 자기 5자리 특징.
     학습은 50개 KSIC 전부(풀링 OLS+약한 릿지), 평가는 34개(그룹 공유 KSIC).
     개발: 목표 끝 ≤2017 → t0 ≤ 2014. 사후 분석: t0 2015–2019(목표 ≤2022, 5개국 완결 연도).
(3b) 외부 미래(KOSIS 5자리 종사자·출하·사업체): 정보시점 o, 목표 = log X(o+2) − log X(o−1).
     신호 = 자기 특허 모멘텀(e=o−2) − 그룹 평균. 연계표 해상도는 그룹 안 순위를 낼 수 없다(쌍 순서 정확도 0.5).
     KOSIS 자기 과거 성장(외부 5자리, 연계표 분석자도 가짐)을 함께 넣었을 때 특허 신호의 추가 기여도 본다.
     개발 o ∈ {2009,2010,2012,2013,2014,2015}(목표 ≤2017), 사후 o ∈ {2016,2017,2018}(10차 2018–2020)."""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.environ['C18_OUT']
MC = 'C:\\Users\\PC\\Downloads\\특허-산업 융합데이터 활용 아이디어 공모전\\분석\\모델경쟁_20260927'
BASE = os.path.abspath(os.path.join(MC, r"..\.."))
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group
T0 = time.time()
rng = np.random.default_rng(7)
P = pd.read_parquet(os.path.join(HERE, "panel_ksic_cc_year.parquet"))
P = P[P.year <= 2022]  # 2023+ 절단
K50 = sorted(P.ksic.unique()); G = pd.Series({k: ksic_to_group(k) for k in K50}); gs = G.value_counts()
K34 = [k for k in K50 if gs[G[k]] > 1]
YR = list(range(2000, 2023))
ser = {"5국": P.groupby(["ksic", "year"]).n.sum().unstack()}
for c in ["KR", "US", "JP", "CN", "EP"]:
    ser[c] = P[P.cc == c].groupby(["ksic", "year"]).n.sum().unstack()
ser["KR기원"] = P.groupby(["ksic", "year"]).n_kro.sum().unstack()
ser = {k: v.reindex(index=K50, columns=YR, fill_value=0).fillna(0) for k, v in ser.items()}
grp = {k: v.groupby(G).sum() for k, v in ser.items()}


def lmean(A, ys):
    return np.log(A[ys].mean(axis=1) + 1)


def feats(e):
    """출원연도 e까지의 정보로 만든 특징(KSIC 50행)."""
    f = pd.DataFrame(index=K50)
    for nm in ["5국", "KR", "US", "JP", "CN", "KR기원"]:
        A, Ag = ser[nm], grp[nm].reindex(G.values).set_axis(K50)
        f[f"own_mom_{nm}"] = lmean(A, [e - 1, e]) - lmean(A, [e - 3, e - 2])
        f[f"grp_mom_{nm}"] = lmean(Ag, [e - 1, e]) - lmean(Ag, [e - 3, e - 2])
    s5, g5 = ser["5국"][[e - 2, e - 1, e]].sum(axis=1), grp["5국"][[e - 2, e - 1, e]].sum(axis=1).reindex(G.values).set_axis(K50)
    f["own_krsh"] = ser["KR기원"][[e - 2, e - 1, e]].sum(axis=1) / s5
    f["grp_krsh"] = grp["KR기원"][[e - 2, e - 1, e]].sum(axis=1).reindex(G.values).set_axis(K50) / g5
    f["own_cnsh"] = ser["CN"][[e - 2, e - 1, e]].sum(axis=1) / s5
    f["grp_cnsh"] = grp["CN"][[e - 2, e - 1, e]].sum(axis=1).reindex(G.values).set_axis(K50) / g5
    f["grp_lsize"] = np.log(g5 + 1)
    f["own_lshare_in_grp"] = np.log((s5 + 1) / (g5 + 1))
    return f


def target(nm, t0):
    A = ser[nm]
    return lmean(A, [t0 + 1, t0 + 2, t0 + 3]) - lmean(A, [t0 - 2, t0 - 1, t0])


GF = [c for c in feats(2010).columns if c.startswith("grp_")]
MF = GF + [c for c in feats(2010).columns if c.startswith("own_")]


def fit_pred(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    A = ((Xtr - mu) / sd).values; B = ((Xte - mu) / sd).values
    A1 = np.c_[np.ones(len(A)), A]; B1 = np.c_[np.ones(len(B)), B]
    R = lam * np.eye(A1.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A1.T @ A1 + R, A1.T @ ytr.values)
    return B1 @ w


def pair_acc(pred, y, g):
    hit, n = 0.0, 0
    for x in np.unique(g):
        ix = np.where(g == x)[0]
        for i in range(len(ix)):
            for j in range(i + 1, len(ix)):
                a, b = ix[i], ix[j]
                if y[a] == y[b]:
                    continue
                dp = pred[a] - pred[b]
                hit += 0.5 if dp == 0 else float(np.sign(dp) == np.sign(y[a] - y[b])); n += 1
    return hit / n if n else np.nan, n


def wcorr(pred, y, g):
    d = pd.DataFrame({"p": pred, "y": y, "g": g})
    dp, dy = d.p - d.groupby("g").p.transform("mean"), d.y - d.groupby("g").y.transform("mean")
    return float(np.corrcoef(dp, dy)[0, 1]) if dp.std() > 1e-12 else 0.0


rows, preds = [], []
for tgt in ["KR", "5국", "KR기원"]:
    for L in (2, 0):
        first_s = 2003 + L
        for t0 in range(2006, 2020):
            e = t0 - L
            tr_orig = [s for s in range(first_s, t0 - 3 - L + 1)]
            if len(tr_orig) < 2:
                continue
            Xtr = pd.concat([feats(s - L) for s in tr_orig]); ytr = pd.concat([target(tgt, s) for s in tr_orig])
            Xte = feats(e); yte = target(tgt, t0)
            out = {"zero": np.zeros(len(K50))}
            for nm, F in (("G_연계표해상도", GF), ("M_제공5자리", MF)):
                out[nm] = fit_pred(Xtr[F], ytr, Xte[F])
            ix = [K50.index(k) for k in K34]
            y34, g34 = yte.values[ix], G[K34].values
            for nm, p in out.items():
                p34 = p[ix]
                pa, npair = pair_acc(p34, y34, g34)
                rows.append({"목표": tgt, "L": L, "t0": t0, "구간": "개발" if t0 + 3 <= 2017 else "사후분석", "모형": nm, "n학습원점": len(tr_orig),
                             "RMSE_34": float(np.sqrt(np.mean((p34 - y34) ** 2))), "MAE_34": float(np.mean(np.abs(p34 - y34))),
                             "RMSE_50": float(np.sqrt(np.mean((p - yte.values) ** 2))),
                             "그룹안_쌍순서정확도": pa, "n쌍": npair, "그룹안_r": wcorr(p34, y34, g34)})
                preds += [{"목표": tgt, "L": L, "t0": t0, "모형": nm, "ksic": k, "pred": float(p34[i]), "y": float(y34[i])} for i, k in enumerate(K34)]
R = pd.DataFrame(rows); PR = pd.DataFrame(preds)
R.to_csv(os.path.join(HERE, "t3a_forecast_by_origin.csv"), index=False, encoding="utf-8-sig")
PR.to_parquet(os.path.join(HERE, "t3a_forecast_preds.parquet"), index=False)

# 요약 + KSIC 군집 부트스트랩(원점 묶음 유지) CI: RMSE 차이(M−G), 쌍정확도
summ = []
for (tgt, L, seg), d in R.groupby(["목표", "L", "구간"]):
    pv = d.pivot_table(index="t0", columns="모형", values=["RMSE_34", "그룹안_쌍순서정확도", "그룹안_r"])
    q = PR[(PR.목표 == tgt) & (PR.L == L) & PR.t0.isin(d.t0.unique())]
    W = q.pivot_table(index=["t0", "ksic"], columns="모형", values="pred").join(q.drop_duplicates(["t0", "ksic"]).set_index(["t0", "ksic"]).y)
    W = W.reset_index()
    ks = np.array(K34); bs = []
    for _ in range(1000):
        pick = rng.choice(ks, len(ks))
        Z = pd.concat([W[W.ksic == k] for k in pick])
        bs.append(np.sqrt(((Z["M_제공5자리"] - Z.y) ** 2).mean()) - np.sqrt(((Z["G_연계표해상도"] - Z.y) ** 2).mean()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    summ.append({"목표": tgt, "L": L, "구간": seg, "원점": f"{d.t0.min()}–{d.t0.max()} ({d.t0.nunique()})",
                 "RMSE_zero": round(pv["RMSE_34"]["zero"].mean(), 4), "RMSE_G": round(pv["RMSE_34"]["G_연계표해상도"].mean(), 4), "RMSE_M": round(pv["RMSE_34"]["M_제공5자리"].mean(), 4),
                 "RMSE차이(M−G)": round(pv["RMSE_34"]["M_제공5자리"].mean() - pv["RMSE_34"]["G_연계표해상도"].mean(), 4), "차이_95CI(KSIC부트)": f"[{lo:.4f}, {hi:.4f}]",
                 "M우세원점": f"{int((pv['RMSE_34']['M_제공5자리'] < pv['RMSE_34']['G_연계표해상도']).sum())}/{len(pv)}",
                 "쌍순서_G": 0.5, "쌍순서_M": round(pv["그룹안_쌍순서정확도"]["M_제공5자리"].mean(), 3),
                 "쌍순서_M>0.5원점": f"{int((pv['그룹안_쌍순서정확도']['M_제공5자리'] > 0.5).sum())}/{len(pv)}",
                 "그룹안r_M": round(pv["그룹안_r"]["M_제공5자리"].mean(), 3)})
S = pd.DataFrame(summ)
S.to_csv(os.path.join(HERE, "t3a_forecast_summary.csv"), index=False, encoding="utf-8-sig")
print(S.to_string()); print(round(time.time() - T0), "s")

# ---- (3b) 외부 5자리 미래 ----
ko = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
k10 = ko[(ko.기준 == "10차") & (ko.마스킹 == 0)]
XS = {nm: k10[k10.항목코드 == c].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(K50)
      for c, nm in (("T02", "종사자수"), ("T04", "출하액"), ("T01", "사업체수"))}


def kos_past(X, o):
    for k in (3, 4, 2, 5, 1):  # o−1에서 거꾸로 가장 가까운 3년 창(2010 없음)
        a = o - 1 - k
        if a in X.columns and a >= 2008:
            return (np.log(X[o - 1]) - np.log(X[a])) / k
    return pd.Series(np.nan, index=X.index)


rows, pool = [], []
for xn, X in XS.items():
    for o in [2009, 2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        if (o - 1) not in X.columns or (o + 2) not in X.columns:
            continue
        z = np.log(X[o + 2]) - np.log(X[o - 1])
        f = feats(o - 2)
        d = pd.DataFrame({"z": z, "kp": kos_past(X, o), "g": G}).loc[K34]
        for nm in ["KR", "KR기원", "5국"]:
            d[f"pm_{nm}"] = f[f"own_mom_{nm}"].loc[K34]
        d = d.dropna(subset=["z"])
        g = d.g.values
        rec = {"외부지표": xn, "o": o, "목표": f"{o-1}→{o+2}", "구간": "개발" if o + 2 <= 2017 else "사후분석", "n": len(d)}
        for s in ["pm_KR", "pm_KR기원", "pm_5국", "kp"]:
            if d[s].notna().sum() < len(d):
                rec[f"쌍순서_{s}"] = np.nan; rec[f"그룹안r_{s}"] = np.nan; continue
            rec[f"쌍순서_{s}"], rec["n쌍"] = pair_acc(d[s].values, d.z.values, g)
            rec[f"그룹안r_{s}"] = wcorr(d[s].values, d.z.values, g)
        rows.append(rec)
        dd = d.copy(); dd["xn"] = xn; dd["o"] = o; dd["seg"] = rec["구간"]; pool.append(dd)
RB = pd.DataFrame(rows)
RB.to_csv(os.path.join(HERE, "t3b_external_by_origin.csv"), index=False, encoding="utf-8-sig")
PO = pd.concat(pool)
# 풀링 검정: 그룹×원점 안에서 특허 신호를 섞는 순열(5000회)
out = []
for (xn, seg), d in PO.groupby(["xn", "seg"]):
    d = d.copy(); d["cell"] = d.g + "_" + d.o.astype(str)
    c = pd.factorize(d.cell)[0]
    def dmv(v):
        m = np.bincount(c, weights=v) / np.bincount(c); return v - m[c]
    zz = dmv(d.z.values.astype(float))
    for s in ["pm_KR", "pm_KR기원", "pm_5국"]:
        sv = d[s].values.astype(float); r = np.corrcoef(dmv(sv), zz)[0, 1]
        cells = [np.where(c == x)[0] for x in np.unique(c)]
        sims = []
        for _ in range(5000):
            sp = sv.copy()
            for ix in cells:
                sp[ix] = sv[rng.permutation(ix)]
            sims.append(np.corrcoef(dmv(sp), zz)[0, 1])
        pa = np.nanmean([pair_acc(d[d.o == o][s].values, d[d.o == o].z.values, d[d.o == o].g.values)[0] for o in d.o.unique()])
        # KOSIS 과거성장 통제 후 부분상관(그룹×원점 고정효과)
        ok = d.kp.notna().values
        if ok.sum() > 20:
            c2 = pd.factorize(d.cell[ok])[0]
            def dm2(v):
                m = np.bincount(c2, weights=v) / np.bincount(c2); return v - m[c2]
            Z = np.c_[dm2(d.kp.values[ok].astype(float))]
            a, b = dm2(sv[ok]), dm2(d.z.values[ok].astype(float))
            ra = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]; rb = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
            pr = float(np.corrcoef(ra, rb)[0, 1]); nk = int(ok.sum())
            rk = float(np.corrcoef(Z[:, 0], b)[0, 1])
        else:
            pr, nk, rk = np.nan, 0, np.nan
        out.append({"외부지표": xn, "구간": seg, "특허신호": s, "원점": ",".join(map(str, sorted(d.o.unique()))), "n(KSIC×원점)": len(d),
                    "그룹안_r(풀링)": round(r, 3), "순열p(단측)": round(float(np.mean(np.array(sims) >= r - 1e-12)), 4),
                    "쌍순서정확도(원점평균)": round(pa, 3), "연계표해상도_쌍순서": 0.5,
                    "KOSIS과거성장_그룹안r": round(rk, 3), "KOSIS과거통제_부분r": round(pr, 3), "n(통제)": nk})
OB = pd.DataFrame(out)
OB.to_csv(os.path.join(HERE, "t3b_external_summary.csv"), index=False, encoding="utf-8-sig")
print(OB.to_string()); print("done", round(time.time() - T0), "s")
