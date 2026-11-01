# -*- coding: utf-8 -*-
r"""이미 채점된 시험 예측(동결 재현 orig = 공식, 별칭 보정 alias)으로:
1) 위약 A: 시험 레이블 섞기 1,000회 × 3방식(R5 층 = NPS 가입×출원 규모 5분위, 주 KSIC 2자리×NPS 규모, 완전 무작위)
2) NPS 오분류 구간 분해: 'NPS 미가입 AUC 0.806'이 숨은 가입 기업(별칭으로 찾은 기업)에서 나오는지
3) 하위집단·대형 출원인 제외·상위 k·PR-AUC·보정
재학습 없음(예측 파일만 사용)."""
from w_common import *
from sklearn.metrics import average_precision_score, brier_score_loss
import statsmodels.api as sm
O = pd.read_parquet(os.path.join(VC, "preds_orig.parquet")); A = pd.read_parquet(os.path.join(VC, "preds_alias.parquet"))
PO = pd.read_parquet(os.path.join(VC, "final_panel_orig.parquet")); PO = PO[PO.t0 == 2022].reset_index(drop=True)
PA = pd.read_parquet(os.path.join(VC, "final_panel_alias.parquet")); PA = PA[PA.t0 == 2022].reset_index(drop=True)
assert (O.applicantCd.values == PO.applicantCd.values).all() and (A.applicantCd.values == PO.applicantCd.values).all()
y = O.y.values.astype(int); n = len(y)
# 보조 변수
AF = pd.read_parquet(os.path.join(VC, "app_firm.parquet")); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
W = n7lib.ksic_weights(PO, 2022, AF).sort_values(["applicantCd", "w", "ksic"], ascending=[True, False, True]).drop_duplicates("applicantCd")
k2 = PO.applicantCd.map(W.set_index("applicantCd").ksic).astype(str).str[:2].values
emp = np.expm1(PO.log_emp.values)
size_o = np.where(PO.nps_present.values == 0, "NPS없음", pd.cut(emp, [-1, 9, 49, 299, 1e9], labels=["1-9", "10-49", "50-299", "300+"]).astype(str))
hidden = (PO.nps_present.values == 0) & (PA.nps_present.values == 1)
R = {}

# ---------- 1) 위약 A ----------
def perm_within(groups, rng):
    ys = y.copy()
    for ix in pd.Series(np.arange(n)).groupby(groups).indices.values():
        ys[ix] = rng.permutation(ys[ix])
    return ys
q5 = pd.qcut(PO.log_app.rank(method="first"), 5, labels=False).values
STR = {"R5층(NPS가입×출원규모5분위)": PO.nps_present.astype(int).astype(str).values + "_" + q5.astype(str),
       "주KSIC2자리×NPS규모": np.char.add(np.char.add(k2.astype(str), "_"), size_o.astype(str)),
       "완전무작위": np.zeros(n)}
R["placebo_A"] = {}
for src, P in [("orig", O), ("alias", A)]:
    obs = skill(auc_rs(y, P.p_champ), auc_rs(y, P.p_M0))
    for nm, g in STR.items():
        rng = np.random.default_rng(42)
        sk, ac, a0 = [], [], []
        for _ in range(1000):
            ys = perm_within(g, rng)
            c, b = auc_rs(ys, P.p_champ), auc_rs(ys, P.p_M0)
            sk.append((c - b) / (1 - b)); ac.append(c); a0.append(b)
        sk = np.array(sk)
        R["placebo_A"][f"{src}|{nm}"] = dict(n_rep=1000, n_strata=int(len(np.unique(g))), observed=float(obs), mean=float(sk.mean()),
                                            q025=float(np.percentile(sk, 2.5)), q975=float(np.percentile(sk, 97.5)), n_ge_obs=int((sk >= obs).sum()),
                                            auc_champ_mean=float(np.mean(ac)), auc_M0_mean=float(np.mean(a0)))
        log("placeboA", src, nm, R["placebo_A"][f"{src}|{nm}"])

# ---------- 2) NPS 오분류 분해 ----------
m0 = PO.nps_present.values == 0
rest = m0 & ~hidden
def sub(mask, P, nm):
    yy = y[mask]
    d = dict(n=int(mask.sum()), pos=int(yy.sum()), rate=float(yy.mean()))
    if 0 < yy.sum() < len(yy):
        d.update(auc_champ=float(auc(yy, P.p_champ.values[mask])), auc_M0=float(auc(yy, P.p_M0.values[mask])))
        if "p_EXT_ens" in P:
            d["auc_EXT"] = float(auc(yy, P.p_EXT_ens.values[mask]))
        d["skill"] = float(skill(d["auc_champ"], d["auc_M0"])) if d["auc_M0"] < 1 else None
    return d
R["nps_misclass"] = {
    "frozen|원래 NPS 미가입 전체(공식 0.806)": sub(m0, O, ""),
    "frozen|그중 숨은 가입(별칭으로 t0 가입 확인)": sub(hidden, O, ""),
    "frozen|그중 나머지(진짜 미가입)": sub(rest, O, ""),
    "frozen|NPS 가입(원래)": sub(~m0, O, ""),
    "alias재학습|보정 후 NPS 미가입": sub(PA.nps_present.values == 0, A, ""),
    "alias재학습|보정 후 NPS 가입": sub(PA.nps_present.values == 1, A, ""),
    "alias재학습|숨은 가입 기업": sub(hidden, A, ""),
}
# 숨은 가입 기업이 원래 미가입 구간에서 차지하는 순위 위치
r0 = pd.Series(O.p_champ.values[m0]).rank(pct=True).values; h_in = hidden[m0]
R["nps_misclass"]["frozen|미가입 구간 안 챔피언 평균 순위(숨은/나머지)"] = [float(r0[h_in].mean()), float(r0[~h_in].mean())]
r0m = pd.Series(O.p_M0.values[m0]).rank(pct=True).values
R["nps_misclass"]["frozen|미가입 구간 안 M0 평균 순위(숨은/나머지)"] = [float(r0m[h_in].mean()), float(r0m[~h_in].mean())]
# 숨은 가입 기업을 빼고 전체 skill(동결 예측)
keep = ~hidden
R["nps_misclass"]["frozen|숨은 가입 기업 제외 전체 skill"] = dict(n=int(keep.sum()), auc_champ=float(auc(y[keep], O.p_champ[keep])), auc_M0=float(auc(y[keep], O.p_M0[keep])),
                                                          skill=float(skill(auc(y[keep], O.p_champ[keep]), auc(y[keep], O.p_M0[keep]))))
log("misclass", json.dumps(R["nps_misclass"], ensure_ascii=False))

# ---------- 3) 하위집단 ----------
def subtab(P, panel):
    out = {}
    emp_p = np.expm1(panel.log_emp.values)
    sz = np.where(panel.nps_present.values == 0, "NPS없음", pd.cut(emp_p, [-1, 9, 49, 299, 1e9], labels=["1-9", "10-49", "50-299", "300+"]).astype(str))
    for v in ["NPS없음", "1-9", "10-49", "50-299", "300+"]:
        out[f"NPS규모 {v}"] = sub(sz == v, P, "")
    for v in ["20", "26", "27", "28", "29", "21", "24", "25", "30", "33"]:
        if (k2 == v).sum() > 200:
            out[f"주KSIC {v}xxx"] = sub(k2 == v, P, "")
    na = PO.n_app.values
    for lo, hi, nm in [(1, 2, "출원 1-2건"), (3, 9, "출원 3-9건"), (10, 1e9, "출원 10건+")]:
        out[nm] = sub((na >= lo) & (na <= hi), P, "")
    gp = PO.g_part_any.values == 1
    out["NTIS 성과 연결 특허 있음(g_part_any=1)"] = sub(gp, P, ""); out["NTIS 성과 연결 없음"] = sub(~gp, P, "")
    out["최근 1년 신규 공개·등록 있음"] = sub(PO.known_1y.values > 0, P, ""); out["최근 1년 신규 없음(휴면)"] = sub(PO.known_1y.values == 0, P, "")
    return out
R["subgroups_orig"] = subtab(O, PO); R["subgroups_alias"] = subtab(A, PA)
# 대형 출원인 제외
R["exclude_large"] = {}
for q in [0.01, 0.05, 0.10]:
    thr = np.quantile(PO.n_app.values, 1 - q); k = PO.n_app.values < thr
    for src, P in [("orig", O), ("alias", A)]:
        R["exclude_large"][f"{src}|출원 수 상위 {int(q*100)}% 제외"] = sub(k, P, "")
k = ~(emp >= 300)
R["exclude_large"]["orig|NPS 300인+ 제외"] = sub(k, O, "")

# ---------- 상위 k ----------
R["topk"] = {}
for src, P in [("orig", O), ("alias", A)]:
    for kk in [50, 100, 123, 300, 614, 1229, 2457]:
        row = {}
        for m in ["champ", "M0", "SB", "EXT_ens"]:
            o = np.argsort(-P["p_" + m].values, kind="mergesort")[:kk]
            row[m] = int(y[o].sum())
        row["lift_champ"] = row["champ"] / kk / y.mean(); row["recall_champ"] = row["champ"] / y.sum()
        R["topk"][f"{src}|{kk}"] = row
    R["topk"][f"{src}|PR-AUC"] = {m: float(average_precision_score(y, P["p_" + m])) for m in ["champ", "M0", "SB", "EXT_ens"]}

# ---------- 보정 ----------
def calib(p, yy):
    p = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))
    f = sm.Logit(yy, sm.add_constant(lg)).fit(disp=0)
    f2 = sm.GLM(yy, np.ones((len(yy), 1)), offset=lg, family=sm.families.Binomial()).fit()
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    return dict(mean_pred=float(p.mean()), base=float(yy.mean()), brier=float(brier_score_loss(yy, p)), slope=float(f.params[1]),
                intercept_in_large=float(f2.params[0]), dec_pred=[round(float(v), 4) for v in pd.Series(p).groupby(dec).mean()],
                dec_obs=[round(float(v), 4) for v in pd.Series(yy).groupby(dec).mean()])
R["calibration"] = {}
for src, P in [("orig", O), ("alias", A)]:
    R["calibration"][f"{src}|챔피언 XGB 성분(47피처, 확률)"] = calib(P.p_champ_xgb.values, y)
    R["calibration"][f"{src}|M0 XGB"] = calib(P.p_M0.values, y)
    dec = pd.qcut(P.p_champ.rank(method="first"), 10, labels=False)
    R["calibration"][f"{src}|챔피언(순위 평균) 10분위 실제율"] = [round(float(v), 4) for v in pd.Series(y).groupby(dec).mean()]
json.dump(R, open(os.path.join(HERE, "w4_analysis.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("saved w4_analysis.json")
