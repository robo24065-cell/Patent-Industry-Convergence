# -*- coding: utf-8 -*-
r"""(2) 외부 결과 B 평가 — 국민연금 등록 업종 변화. 사후 분석(NPS가 2015-11 이후만 존재, T1 최종시험 구간과 겹침/이후).
B1 업종 변경 여부(y_chg 6자리, y_maj 앞2자리):
  (i) 시간 분할: 평가 원점 o∈{2019,2020,2021}, 학습 원점 ≤ o−4(결과 관측 완료분만). 로지스틱(L2, 표준화).
  (ii) 보조: 전 원점 묶음, 기업 단위 5겹 교차검증(시간 분리 없음).
  (iii) 무학습 점수: 등록업종 그룹의 포트폴리오 몫이 낮을수록 전환(−sh_rg), 최빈 그룹≠등록(top_ne).
  팔: NONE(라벨 무관 기반) / P(+제공 라벨 구성) / C(+연계표 구성) / P5(+제공 5자리 몫) / PC.
B2 전환 방향: 그룹 이동(rg0≠rg1, rg1∈27그룹) 기업에서 후보 26그룹을 포트폴리오 몫으로 순위 → 도착 그룹 AUC·top-1·top-3.
B2b 무조건: rg0가 알려진 모든 기업 × 후보 그룹(≠rg0), y = 도착 그룹 여부. 점수 = 몫.
CI: 기업 군집 Poisson 부트스트랩.
사용: python b_eval.py → res/B_eval.json
"""
from nc_common import *
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import SPACE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

RES = os.path.join(OUT, "res")
B = pd.concat([pd.read_parquet(os.path.join(CACHE, f"B_o{o}.parquet")) for o in range(2015, 2022)], ignore_index=True)
log("B rows", len(B), "firms", B.applicantCd.nunique())
BASEF = ["f_n", "f_rec3", "f_age", "f_last", "f_nsc", "f_scent", "f_newsc_rec", "f_emp", "f_rg_known", "f_rg_space"]
LF = ["sh_rg", "sh_rg_rec", "top_ne", "ent", "maxoth", "newg_rec", "cov"]
ARMS = {"NONE": BASEF, "P": BASEF + [f"P_{x}" for x in LF], "C": BASEF + [f"C_{x}" for x in LF],
        "P5": BASEF + [f"P_{x}" for x in LF] + ["P5_sh_rc"], "PC": BASEF + [f"P_{x}" for x in LF] + [f"C_{x}" for x in LF]}
B["norec"] = B.P_sh_rg_rec.isna().astype(np.int8)
for L in ("P", "C"):
    B[f"{L}_sh_rg_rec"] = B[f"{L}_sh_rg_rec"].fillna(B[f"{L}_sh_rg"])
    B[f"{L}_top_ne"] = B[f"{L}_top_ne"].fillna(1); B[f"{L}_newg_rec"] = B[f"{L}_newg_rec"].fillna(0); B[f"{L}_ent"] = B[f"{L}_ent"].fillna(0)
BASEF.append("norec")
for a in ARMS:
    ARMS[a] = ARMS[a] + ["norec"]


def fitpred(tr, te, cols, y):
    sc = StandardScaler().fit(tr[cols].values)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr[cols].values), tr[y].values)
    return m.predict_proba(sc.transform(te[cols].values))[:, 1]


def boot_pair(y, s1, s2, cl, B_=500, seed=0):
    lo, hi, pr = cluster_boot_delta(y, s1, s2, cl, B=B_, seed=seed)
    return {"d": auc_w(y, s1) - auc_w(y, s2), "ci": [lo, hi], "p_gt0": pr}


out = {"note": "사후 분석: NPS 스냅샷이 2015-11 이후만 있어 개발 기간(≤2014) 평가 불가", "n_rows": len(B), "n_firms": int(B.applicantCd.nunique())}
CMP = (("P", "C"), ("P5", "C"), ("P", "NONE"), ("C", "NONE"), ("PC", "C"), ("PC", "P"), ("P5", "P"))
for y in ("y_chg", "y_maj"):
    o_y = {}
    # (i) 시간 분할
    preds = []
    for to in (2019, 2020, 2021):
        tr = B[B.o <= to - 4]; te = B[B.o == to].copy()
        r = {"n_train": len(tr), "pos_train": int(tr[y].sum()), "n_test": len(te), "pos_test": int(te[y].sum())}
        for a, cols in ARMS.items():
            te[a] = fitpred(tr, te, cols, y)
            r[a] = auc_w(te[y], te[a])
        o_y[f"time_o{to}"] = r; log(y, to, r)
        preds.append(te[["applicantCd", "o", y] + list(ARMS)])
    T = pd.concat(preds, ignore_index=True)
    # 원점별 순위 정규화 후 묶음 AUC
    for a in ARMS:
        T[a] = T.groupby("o")[a].rank(pct=True)
    o_y["time_pooled"] = {a: auc_w(T[y], T[a]) for a in ARMS}
    o_y["time_pooled_delta"] = {f"{a}-{b}": boot_pair(T[y].values, T[a].values, T[b].values, T.applicantCd.values, seed=1) for a, b in CMP}
    o_y["time_pooled_n"] = [len(T), int(T[y].sum())]
    log(y, "time pooled", o_y["time_pooled"], {k: (round(v["d"], 4), [round(x, 4) for x in v["ci"]]) for k, v in o_y["time_pooled_delta"].items()})
    # (ii) 기업 단위 5겹 CV(전 원점)
    S = pd.DataFrame({"applicantCd": B.applicantCd, y: B[y]})
    for a, cols in ARMS.items():
        s = np.zeros(len(B))
        for tri, tei in GroupKFold(5).split(B, groups=B.applicantCd):
            s[tei] = fitpred(B.iloc[tri], B.iloc[tei], cols, y)
        S[a] = s
    o_y["cv"] = {a: auc_w(S[y], S[a]) for a in ARMS}
    o_y["cv_delta"] = {f"{a}-{b}": boot_pair(S[y].values, S[a].values, S[b].values, S.applicantCd.values, seed=2) for a, b in CMP}
    log(y, "cv", o_y["cv"])
    # (iii) 무학습 점수(전 원점, 등록업종 그룹이 27그룹 안인 기업)
    Z = B[B.f_rg_space == 1]
    zs = {"P_mis": -Z.P_sh_rg, "C_mis": -Z.C_sh_rg, "P5_mis": -Z.P5_sh_rc, "P_mis_rec": -Z.P_sh_rg_rec, "C_mis_rec": -Z.C_sh_rg_rec,
          "P_top_ne": Z.P_top_ne, "C_top_ne": Z.C_top_ne}
    o_y["zero_param"] = {"n": len(Z), "pos": int(Z[y].sum()), **{k: auc_w(Z[y], v) for k, v in zs.items()}}
    o_y["zero_param_delta"] = {f"{a}-{b}": boot_pair(Z[y].values, zs[a].values, zs[b].values, Z.applicantCd.values, seed=3)
                               for a, b in (("P_mis", "C_mis"), ("P5_mis", "C_mis"), ("P_mis_rec", "C_mis_rec"), ("P_top_ne", "C_top_ne"))}
    log(y, "zero", o_y["zero_param"])
    out[y] = o_y

# ---- B2 전환 방향
G = np.array(SPACE)
known = B[B.rg0.notna() & B.rg1.notna()].copy()
known["move"] = (known.rg0 != known.rg1) & known.rg1.isin(SPACE)
out["B2_counts"] = {"rg_known_both": len(known), "moves_into_space": int(known.move.sum()),
                    "moves_unique_firms": int(known[known.move].applicantCd.nunique()),
                    "rg0_space_share_of_movers": float(known[known.move].rg0.isin(SPACE).mean())}


def long_pairs(D):
    rows = []
    for L in ("P", "C"):
        Mg = D[[f"{L}g_{g}" for g in G]].values; Mr = D[[f"{L}r_{g}" for g in G]].values
        rows.append((L, Mg, Mr))
    rg0 = D.rg0.values; rg1 = D.rg1.values
    mask = (G[None, :] != rg0[:, None])           # 후보 = rg0 제외
    Y = (G[None, :] == rg1[:, None]).astype(np.int8)
    fi, gi = np.where(mask)
    res = {"applicantCd": D.applicantCd.values[fi], "y": Y[fi, gi]}
    for L, Mg, Mr in rows:
        res[f"{L}"] = Mg[fi, gi] + 1e-3 * Mr[fi, gi]   # 전체 몫, 동점은 최근 몫으로
        res[f"{L}_rec"] = Mr[fi, gi]
    return pd.DataFrame(res), fi


def topk(D, L, k):
    Mg = D[[f"{L}g_{g}" for g in G]].values + 1e-3 * D[[f"{L}r_{g}" for g in G]].values
    Mg = np.where(G[None, :] == D.rg0.values[:, None], -1, Mg)
    o = np.argsort(-Mg, axis=1)[:, :k]
    hit = (G[o] == D.rg1.values[:, None]).any(1)
    return float(hit.mean())


MV = known[known.move]
Lp, _ = long_pairs(MV)
out["B2_movers"] = {"n_firm_origin": len(MV), "n_pairs": len(Lp),
                    "auc_P": auc_w(Lp.y, Lp.P), "auc_C": auc_w(Lp.y, Lp.C), "auc_P_rec": auc_w(Lp.y, Lp.P_rec), "auc_C_rec": auc_w(Lp.y, Lp.C_rec),
                    "top1_P": topk(MV, "P", 1), "top1_C": topk(MV, "C", 1), "top3_P": topk(MV, "P", 3), "top3_C": topk(MV, "C", 3),
                    "delta_P-C": boot_pair(Lp.y.values, Lp.P.values, Lp.C.values, Lp.applicantCd.values, seed=4)}
# 기업별 첫 이동만(중복 원점 제거)
MV1 = MV.sort_values("o").drop_duplicates("applicantCd")
L1, _ = long_pairs(MV1)
out["B2_movers_first"] = {"n_firm": len(MV1), "auc_P": auc_w(L1.y, L1.P), "auc_C": auc_w(L1.y, L1.C),
                          "top1_P": topk(MV1, "P", 1), "top1_C": topk(MV1, "C", 1), "top3_P": topk(MV1, "P", 3), "top3_C": topk(MV1, "C", 3),
                          "delta_P-C": boot_pair(L1.y.values, L1.P.values, L1.C.values, L1.applicantCd.values, seed=5)}
# 원점별
out["B2_by_origin"] = {}
for o, d in MV.groupby("o"):
    lo, _ = long_pairs(d)
    out["B2_by_origin"][int(o)] = {"n": len(d), "auc_P": auc_w(lo.y, lo.P), "auc_C": auc_w(lo.y, lo.C), "top1_P": topk(d, "P", 1), "top1_C": topk(d, "C", 1)}
log("B2", out["B2_movers"], out["B2_movers_first"])
# B2b 무조건
K2 = B[B.rg0.notna() & B.rg1.notna()]
L2, _ = long_pairs(K2)
out["B2b_all"] = {"n_firm_origin": len(K2), "n_pairs": len(L2), "pos": int(L2.y.sum()),
                  "auc_P": auc_w(L2.y, L2.P), "auc_C": auc_w(L2.y, L2.C),
                  "delta_P-C": boot_pair(L2.y.values, L2.P.values, L2.C.values, L2.applicantCd.values, seed=6)}
log("B2b", out["B2b_all"])
json.dump(out, open(os.path.join(RES, "B_eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done")
