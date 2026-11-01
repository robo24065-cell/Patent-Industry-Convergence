# -*- coding: utf-8 -*-
"""챔피언/도전자 실험 실행기(dev 원점만). 사용: python n4_03_exp.py E00 E01 E02 ...
상태: champ.json(현재 챔피언 피처·모형·원점별 AUC), 기록: experiments.csv(추가), 예측: cache/preds_<exp>.parquet"""
import os, sys, json
import pandas as pd, numpy as np
from n4lib import *
from n4feat import BUNDLES

D = pd.read_parquet(os.path.join(CACHE, "base_dev.parquet"))
X = pd.read_parquet(os.path.join(CACHE, "feat_dev.parquet"))
D = pd.concat([D, X], axis=1)
assert D.t0.max() + 2 < F_YEAR and D.y.notna().all()
V = DEV_ORIGINS
SEEDS3 = (0, 1, 2)
CH = os.path.join(OUT, "champ.json")
EXPF = os.path.join(OUT, "experiments.csv")

M1 = M0 + M1_ADD
PLAN = {  # exp: (parent 규칙, 설명, 피처 추가, 모형, 시드)
    "E00": ("ref", "규모 1변수(log_emp) 참고선", None, ["log_emp"], "xgb", (0,)),
    "E01": ("base", "M0 기준선: NPS 속성+미가입 여부+사업자번호 유형(파일럿 재현, 시드0)", None, M0, "xgb", (0,)),
    "E02": ("pilot", "M1 = M0+특허 행태 12개(파일럿 재현, 시드0)", None, M1, "xgb", (0,)),
    "E03": ("rescore", "채점 규칙 변경: XGB 3시드 평균으로 M0·M1 재채점(기준 행)", None, None, "xgb", SEEDS3),
}
ORDER = ["C01_nps_exit", "C02_ntis_project", "C03_ntis_patent", "C04_pay_delay", "C05_portfolio_age", "C06_rl_patent_oos",
         "C07_design_utility", "C08_trademark", "C09_dart_fin", "C10_ksic_peer_closure", "C11_disposal"]
for b in ORDER:
    PLAN[b[:3]] = ("chal", b, BUNDLES[b], None, None, SEEDS3)
PLAN["C13"] = ("chal", "C13_nps_flow", BUNDLES["C13_nps_flow"], None, None, SEEDS3)
# 시점 정합 실패(누수) 판정: 권리목록은 '현재 권리자' 기준 스냅샷이라 원점 이후 양도·합병으로 권리가 목록에서 빠진다
# (n4_04_rl_leak_check.py → rl_leak_check.csv). 이 데이터로 만든 묶음은 수치와 무관하게 ROLLBACK(누수).
LEAK = {"C06", "C07", "C08", "V678"}
PLAN["C14"] = ("chal", "C14_foreign_family", BUNDLES["C14_foreign_family"], None, None, SEEDS3)
PLAN["C15"] = ("chal_decay", "최근 원점 가중 학습(0.8^(v-2-t0))", [], None, None, SEEDS3)
PLAN["C16"] = ("chal_swap", "nps_gone → 창 고정판 nps_gone3 교체(t0-3..t0-1)", ["nps_gone3"], ["nps_gone"], None, SEEDS3)
PLAN["C12"] = ("chal_model", "모형 변경: XGB(3시드)+로지스틱 순위 평균", [], None, "ens", SEEDS3)
PLAN["V02"] = ("chal", "C02 축소판(ntis_any5, ntis_n3)", ["ntis_any5", "ntis_n3"], None, None, SEEDS3)
PLAN["V04"] = ("chal", "C04 축소판(overdue_share)", ["overdue_share"], None, None, SEEDS3)
PLAN["V09"] = ("chal", "C09 축소판(dart_has, dart_impair, dart_neg_eq)", ["dart_has", "dart_impair", "dart_neg_eq"], None, None, SEEDS3)
PLAN["V678"] = ("chal", "권리목록 최근 2년 활동 3개(rl_pat_2y, rl_des_2y, rl_tm_2y)", ["rl_pat_2y", "rl_des_2y", "rl_tm_2y"], None, None, SEEDS3)


def score(feats, model, seeds, decay=None):
    res, preds = {}, {}
    for v in V:
        tr = D[(D.t0 >= TRAIN_MIN_T0) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        sw = (decay ** (v - 2 - tr.t0)).values if decay else None
        p = predict(tr[feats].astype(float), tr.y, te[feats].astype(float), model=model, seeds=seeds, sw=sw)
        preds[v] = p; res[v] = auc(te.y, p)
    return res, preds


def save_preds(tag, preds):
    rows = []
    for v, p in preds.items():
        te = D[D.t0 == v]
        rows.append(pd.DataFrame({"applicantCd": te.applicantCd.values, "bizno": te.bizno.values, "t0": v, "y": te.y.values, "p": p}))
    pd.concat(rows).to_parquet(os.path.join(CACHE, f"preds_{tag}.parquet"), index=False)


def append(row):
    df = pd.DataFrame([row])
    if os.path.exists(EXPF):
        df = pd.concat([pd.read_csv(EXPF, encoding="utf-8-sig"), df], ignore_index=True)
    df.to_csv(EXPF, index=False, encoding="utf-8-sig")


def st_load():
    return json.load(open(CH, encoding="utf-8")) if os.path.exists(CH) else {}


def st_save(s):
    json.dump(s, open(CH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


for exp in sys.argv[1:]:
    kind, desc, add, feats, model, seeds = PLAN[exp]
    S = st_load()
    row = dict(exp_id=exp, change=desc)
    if kind in ("ref", "base", "pilot"):
        r, pr = score(feats, model, seeds); save_preds(exp, pr)
        row.update(parent="-" if kind != "pilot" else "E01", model=f"{model}(seed{list(seeds)})", n_feats=len(feats),
                   **{f"auc_{v}": round(r[v], 5) for v in V})
        if kind == "base":
            S["M0_auc"] = r
        if kind == "pilot":
            b = S["M0_auc"]
            sk = [skill(r[v], b[str(v)] if str(v) in b else b[v]) for v in V]
            row.update(mean_skill_vs_M0=round(np.mean(sk), 4), mean_skill_vs_champ=round(np.mean(sk), 4),
                       win_vs_champ=np.mean([r[v] > (b[str(v)] if str(v) in b else b[v]) for v in V]), decision="ACCEPT(출발 챔피언)")
            S.update(champ_tag="E02", feats=feats, model="xgb", seeds=list(seeds), champ_auc=r)
        row["note"] = "파일럿 experiments_I2b.csv와 대조"
    elif kind == "rescore":
        r0, p0 = score(M0, "xgb", seeds); save_preds("E03_M0", p0)
        r1, p1 = score(M1, "xgb", seeds); save_preds("E03_M1", p1)
        sk = [skill(r1[v], r0[v]) for v in V]
        row.update(parent="E02", model="xgb(seed[0,1,2])", n_feats=len(M1), **{f"auc_{v}": round(r1[v], 5) for v in V},
                   mean_skill_vs_M0=round(np.mean(sk), 4), win_vs_M0=np.mean([r1[v] > r0[v] for v in V]), decision="BASE(규칙 변경)",
                   note="M0 3시드 AUC " + "/".join(f"{r0[v]:.4f}" for v in V))
        S.update(M0_auc=r0, champ_tag="E03", feats=M1, model="xgb", seeds=list(seeds), champ_auc=r1)
    else:
        champ = {int(k): v for k, v in S["champ_auc"].items()}
        m0 = {int(k): v for k, v in S["M0_auc"].items()}
        nf = S["feats"] + [f for f in add if f not in S["feats"]]
        if kind == "chal_swap":
            nf = [f for f in nf if f not in feats]   # feats 자리에 '제거할 피처' 목록
        md = model or S["model"]
        dc = 0.8 if kind == "chal_decay" else S.get("decay")
        r, pr = score(nf, md, seeds, dc); save_preds(exp, pr)
        sk = np.mean([skill(r[v], champ[v]) for v in V]); win = np.mean([r[v] > champ[v] for v in V])
        dec = "ACCEPT" if (sk >= 0.003 and win >= 0.6) else "ROLLBACK"
        if exp in LEAK:
            dec = "ROLLBACK(누수: 권리목록=현재 권리자 스냅샷)"
        row.update(parent=S["champ_tag"], model=f"{md}(seed{list(seeds)})", n_feats=len(nf), **{f"auc_{v}": round(r[v], 5) for v in V},
                   mean_skill_vs_champ=round(sk, 4), win_vs_champ=win, decision=dec,
                   mean_skill_vs_M0=round(np.mean([skill(r[v], m0[v]) for v in V]), 4),
                   note="+" + ",".join(add) if add else md)
        if dec == "ACCEPT":
            S.update(champ_tag=exp, feats=nf, model=md, champ_auc=r, decay=dc)
    st_save(S); append(row)
    log(exp, {k: row.get(k) for k in ["mean_skill_vs_champ", "win_vs_champ", "mean_skill_vs_M0", "decision"]},
        [row.get(f"auc_{v}") for v in V])
