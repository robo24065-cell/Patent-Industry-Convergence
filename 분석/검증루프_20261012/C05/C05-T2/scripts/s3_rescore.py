# -*- coding: utf-8 -*-
"""s3: 연결 편향 (c) 표 + 동결 예측 층별·IPW 재채점 (N4 t0=2023, N7 t0=2022). 재학습 없음.
출력: selection_bias.csv, rescore_strata.csv(N4·N7 행), cache/s3_results.json"""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
PRED = {"N4": (os.path.join(N4, "final_test_pred_r4.parquet"), "p_champ", "p_M0", 0.802867665495882, 0.760477643184264),
        "N7": (os.path.join(N7, "final_test_pred_r5.parquet"), "p_champ", "p_M0", 0.7460584938575965, 0.6445159723382918)}
SB, RS, RES = [], [], {}


def addsb(track, comp, metric, a, b, note=""):
    SB.append(dict(track=track, comparison=comp, metric=metric, value=round(a, 4) if isinstance(a, float) else a, extra=b, note=note))


def X_t0(U):
    X = pd.DataFrame(index=U.index)
    for c in ["log_app", "has_live", "log_live", "live_ratio", "lapse_2y"]:
        X[c] = U[c].astype(float)
    X["first_ay"] = U.first_ay.fillna(U.first_ay.median())
    yb = pd.cut(U.yrs_since_last, [-1, 0, 2, 5, 10, 100], labels=["y0", "y1_2", "y3_5", "y6_10", "y10p"]).astype(str)
    X = X.join(pd.get_dummies(yb, prefix="ys", dtype=float))
    top = U.ksic2.value_counts().index[:15]
    X = X.join(pd.get_dummies(U.ksic2.where(U.ksic2.isin(top), "oth"), prefix="k", dtype=float))
    X = X.join(pd.get_dummies(U.sido, prefix="sd", dtype=float))
    return X


def X_fut(U):
    X = X_t0(U)
    X["M"] = U.M.astype(float)
    lb = pd.cut(U.last_year.fillna(0), [-1, 2005, 2010, 2015, 2020, 2100], labels=["l05", "l10", "l15", "l20", "l25"]).astype(str)
    return X.join(pd.get_dummies(lb, prefix="ly", dtype=float))


def prop(X, L):
    sc = StandardScaler().fit(X.values)
    m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X.values), L)
    p = m.predict_proba(sc.transform(X.values))[:, 1]
    from sklearn.metrics import roc_auc_score
    return p, float(roc_auc_score(L, p))


for tr, (fp, pc, p0, auc_c_off, auc_0_off) in PRED.items():
    U = pd.read_parquet(os.path.join(CACHE, f"U_{tr}.parquet"))
    P = pd.read_parquet(fp)
    P["applicantCd"] = P.applicantCd.astype(str)
    D = P.merge(U, on="applicantCd", how="left", suffixes=("", "_u"))
    assert D.M.notna().all(), "시험 모집단 일부가 U에 없음"
    # 공식 재현
    a1, a0 = wauc(D.y, D[pc]), wauc(D.y, D[p0])
    RES[tr] = dict(n=len(D), pos=int(D.y.sum()), auc=a1, auc_M0=a0, skill=skill(a1, a0),
                   repro_diff=[a1 - auc_c_off, a0 - auc_0_off])
    log(tr, "재현", RES[tr])
    S_off = skill(a1, a0)
    # ---------- (c) 연결 편향: U 안 L=1 vs L=0 ----------
    L1, L0 = U[U.L == 1], U[U.L == 0]
    for v, lab in [("log_app", "규모: log(1+출원 수)"), ("log_live", "규모: log(1+살아있는 등록특허)"), ("has_live", "살아있는 등록특허 보유"),
                   ("first_ay", "업력: 첫 출원연도"), ("yrs_since_last", "최근 알려진 문서 후 경과 연수"), ("lapse_2y", "최근 2년 불납 소멸 비율"),
                   ("M", "[진단·미래] 기준일 후 20개월 활동 M"), ("M_all", "[진단·미래] 기준일 후 데이터 끝까지 활동"), ("last_year", "[진단·미래] 마지막 활동 연도")]:
        addsb(tr, "U: 연결(L=1) vs 비연결(L=0)", f"SMD {lab}", smd(L1[v], L0[v]),
              f"{L1[v].mean():.3f} / {L0[v].mean():.3f} (n {len(L1)} / {len(L0)})")
    addsb(tr, "U: 연결 vs 비연결", "TVD 최빈 KSIC 앞2자리", tvd(L1.ksic2, L0.ksic2), "")
    addsb(tr, "U: 연결 vs 비연결", "TVD 시도(등록원부 주소, t0 이전 최근 등록)", tvd(L1.sido, L0.sido),
          f"수도권(서울·경기·인천) {L1.sido.isin(['서울','경기','인천']).mean():.3f} / {L0.sido.isin(['서울','경기','인천']).mean():.3f}; 미상·해외 {L1.sido.isin(['미상','기타/해외']).mean():.3f} / {L0.sido.isin(['미상','기타/해외']).mean():.3f}")
    # 사업자번호 보유율 × 마지막 활동 연도 / M (C04 재현)
    lb = pd.cut(U.last_year, [0, 2005, 2010, 2015, 2020, 2026], labels=["~2005", "2006-10", "2011-15", "2016-20", "2021-25"])
    addsb(tr, "U: 연결률(L) × 마지막 활동 연도[진단]", "연결률", json.dumps(U.groupby(lb, observed=True).L.mean().round(3).to_dict(), ensure_ascii=False), "C04 0.14→0.71과 같은 방향인지")
    addsb(tr, "U: 연결률 × M[진단]", "연결률 M=1 / M=0", f"{U[U.M==1].L.mean():.3f} / {U[U.M==0].L.mean():.3f}", "")
    # 기준일 변수를 통제해도 M이 연결에 남는가(조건부 의존): 로지스틱 계수
    Xf = X_fut(U); sc = StandardScaler().fit(Xf.values)
    mm = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xf.values), U.L)
    coefM = float(mm.coef_[0][list(Xf.columns).index("M")] / sc.scale_[list(Xf.columns).index("M")])
    addsb(tr, "U: P(L | X_t0, M, 마지막 활동)[진단]", "M 로지스틱 계수(원척도, 오즈비)", round(coefM, 3), f"OR {np.exp(coefM):.2f}")
    # 공식 모집단 vs U\L
    off = U.applicantCd.isin(set(D.applicantCd))
    for v in ["log_app", "has_live", "first_ay", "yrs_since_last", "M"]:
        addsb(tr, "공식 모집단 vs 비연결(L=0)", f"SMD {v}", smd(U.loc[off, v], L0[v]), f"{U.loc[off, v].mean():.3f} / {L0[v].mean():.3f}")
    # 공식 모집단 안 M 층 구성
    addsb(tr, "공식 모집단", "M=1 비율(공식) / U 전체 / L=0", f"{D.M.mean():.3f} / {U.M.mean():.3f} / {L0.M.mean():.3f}", "")
    # ---------- 층별 재채점 ----------
    cl = D.bizno.astype(str)
    def row(stratum, mask, w=None, note=""):
        r = score(D.y[mask].values, D[pc][mask].values, D[p0][mask].values, cl[mask].values, None if w is None else w[mask], NB)
        r.update(track=tr, stratum=stratum, d_skill=r["skill"] - S_off, note=note)
        RS.append(r); log(tr, stratum, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("n", "pos", "rate", "auc", "auc_base", "skill", "skill_ci")})
        return r
    allm = np.ones(len(D), bool)
    row("공식(재현)", allm)
    rM0 = row("S1 M=0(기준일 후 20개월 활동 없음: 인지 가능성 높음)", (D.M == 0).values)
    rM1 = row("S1 M=1(기준일 후 활동 있음)", (D.M == 1).values)
    row("S1' M_all=0(데이터 끝까지 활동 없음)", (D.M_all == 0).values, note="민감도")
    row("S1' M_all=1", (D.M_all == 1).values, note="민감도")
    rA1 = row("S2 as-of 확인(NPS t0 활성·NTIS≤t0·DART≤t0 중 하나)", (D.S2 == 1).values)
    rA0 = row("S2 as-of 미확인(2026 목록에만 의존)", (D.S2 == 0).values)
    row("S2a NPS t0 활성 연결", (D.asof_nps == 1).values, note="참고")
    row("S2b NTIS≤t0 또는 DART≤t0", ((D.asof_ntis + D.asof_dart) > 0).values, note="참고")
    row("S3 출처 CA", (D.src == "CA").values, note=f"권리목록만 {int((D.src=='RL').sum())}개")
    RES[tr]["rate_ratio_S1"] = rM0["rate"] / rM1["rate"]
    RES[tr]["rate_ratio_S2"] = rA0["rate"] / rA1["rate"]
    # ---------- IPW ----------
    Xa = X_t0(U); pa, auc_a = prop(Xa, U.L.values)
    Xb = X_fut(U); pb_, auc_b = prop(Xb, U.L.values)
    U["p1"] = pa; U["p2"] = pb_
    # 38셀(VERIFY 방식) 반박용
    U["cell"] = pd.cut(U.yrs_since_last, [-1, 0, 2, 5, 10, 100]).astype(str) + pd.cut(U.n_app, [0, 1, 3, 10, 1e9]).astype(str) + U.has_live.astype(str)
    U["p3"] = U.groupby("cell").L.transform("mean")
    D = D.drop(columns=[c for c in ["p1", "p2", "p3"] if c in D]).merge(U[["applicantCd", "p1", "p2", "p3"]], on="applicantCd", how="left")
    for k, nm, a_ in [("p1", "W1 IPW(기준일 변수 로지스틱)", auc_a), ("p2", "W2 IPW 진단(+M·마지막 활동연도, 미래 정보)", auc_b), ("p3", "W3 IPW 38셀(VERIFY 방식, 반박용)", np.nan)]:
        w = 1.0 / D[k].clip(lower=0.05).values
        r = row(nm, allm, w, note=f"성향 AUC {a_:.3f}; 가중치 평균 {w.mean():.3f} 최대 {w.max():.2f}; 가중 양성률 {np.average(D.y, weights=w):.4f}" if a_ == a_ else f"가중치 평균 {w.mean():.3f} 최대 {w.max():.2f}")
    RES[tr]["prop_auc"] = [auc_a, auc_b]
    # 가중 후 M 구성 = U 전체 구성 근사인지
    w2 = 1.0 / D.p2.clip(lower=0.05)
    addsb(tr, "W2 가중 공식 모집단", "가중 M=1 비율(대 U 전체)", f"{np.average(D.M, weights=w2):.3f} / {U.M.mean():.3f}", "")
    # 위 상위 1% 적중(참고): 공식 대 W2 가중
    k = max(1, int(round(0.01 * len(D)))); top = np.argsort(-D[pc].values, kind="stable")[:k]
    RES[tr]["top1pct_prec"] = float(D.y.values[top].mean())
    RES[tr]["top1pct_prec_w2"] = float(np.average(D.y.values[top], weights=w2.values[top]))
    RES[tr]["base_rate_w2"] = float(np.average(D.y, weights=w2))
    RES[tr]["top1pct_M1_share"] = float(D.M.values[top].mean())

pd.DataFrame(SB).to_csv(os.path.join(OUT, "selection_bias.csv"), index=False, encoding="utf-8-sig")
cols = ["track", "stratum", "n", "pos", "rate", "auc", "auc_base", "skill", "skill_ci", "auc_ci", "auc_base_ci", "d_skill", "n_boot", "note"]
pd.DataFrame(RS)[cols].to_csv(os.path.join(CACHE, "rescore_N4N7.csv"), index=False, encoding="utf-8-sig")
json.dump(RES, open(os.path.join(CACHE, "s3_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done", json.dumps(RES, ensure_ascii=False, default=float))
