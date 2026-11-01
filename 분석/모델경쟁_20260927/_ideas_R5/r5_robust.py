# -*- coding: utf-8 -*-
"""사후 점검(채택 판단에 쓰지 않음). 사용: python r5_robust.py J2 | J1 | J3"""
import sys, json, time
import numpy as np, pandas as pd
from r5_common import *
import r5_eval_spec as SP

idea = sys.argv[1]
S = SP.SPEC[idea]
t = time.time()
D = pd.read_parquet(f"{CACHE}/{S['panel']}.parquet").reset_index(drop=True)
PR = pd.read_parquet(f"{CACHE}/preds_{idea}_xgb.parquet")
champ = json.load(open(f"{CACHE}/champ_{idea}_xgb.json", encoding="utf-8"))
rng = np.random.default_rng(0)
nf = S.get("neg_frac")
keep = (D.y.values == 1) | (rng.random(len(D)) < nf) if nf else np.ones(len(D), bool)
wfull = np.where(D.y.values == 1, 1.0, 1.0 / nf) if nf else np.ones(len(D))


def run(feats):
    res = {}
    for v in S["dev"]:
        msk = (D.t0 <= v - S["gap"]).values & keep
        if S.get("min_train"):
            msk &= (D.t0 >= S["min_train"]).values
        tr = D[msk]; te = D[D.t0 == v]
        res[v] = fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), w=wfull[msk])
    return res


rows = []
def rec(name, pa, pb, sub=None, note=""):
    sks, wins, aa, ab = [], [], [], []
    for v in S["dev"]:
        te = D[D.t0 == v]; m = np.ones(len(te), bool) if sub is None else sub(te).values
        y = te.y.values[m]
        if y.sum() < 5 or y.sum() == len(y):
            continue
        a1 = auc(y, pa[v][m]); a0 = auc(y, pb[v][m]); aa.append(a1); ab.append(a0)
        sks.append(skill(a1, a0)); wins.append(a1 > a0)
    rows.append(dict(check=name, n_origins=len(sks), auc_new=np.mean(aa), auc_ref=np.mean(ab), mean_skill=np.mean(sks), win=np.mean(wins),
                     by_origin=" ".join(f"{s:.3f}" for s in sks), note=note))
    print(rows[-1], round(time.time() - t), flush=True)


pc = {v: PR[PR.t0 == v].p_champ.values for v in S["dev"]}
p0 = {v: PR[PR.t0 == v].p_M0.values for v in S["dev"]}
base, steps = S["base"], dict(S["steps"])

if idea == "J2":
    hist = ["own_disp", "own_trial", "owner_disp5"]
    m1 = steps["M1_이력"]; m2 = steps["M2_외부단위"]
    fresh = lambda te: (te.own_disp == 0) & (te.own_trial == 0)
    rec("챔피언 vs M0 — 과거 분쟁·심판 없는 특허만(신규 분쟁)", pc, p0, sub=fresh)
    p_strong = run(base + hist)
    rec("강한 기준선(M0+자기·권리자 과거 분쟁) 대비 챔피언", pc, p_strong)
    rec("강한 기준선 대비 챔피언 — 신규 분쟁만", pc, p_strong, sub=fresh)
    p_nohist = run(base + [f for f in m1 if f not in hist])
    rec("M1에서 분쟁 이력 3개 뺀 모형 vs M0", p_nohist, p0)
    p_m2 = run(base + m2)
    rec("M0+M2(외부만) vs M0", p_m2, p0)
    p_full = run(base + m1 + m2)
    rec("M0+M1+M2 vs 챔피언(M0+M1)", p_full, pc)
    # 대형 권리자 제외: 개발 원점 양성 상위 5개 권리자
    top = D[D.y == 1].owner.value_counts().head(5).index.tolist()
    rec("챔피언 vs M0 — 양성 상위 5개 권리자 제외", pc, p0, sub=lambda te: ~te.owner.isin(top), note="제외: 상위 5개 권리자(이름 비공개)")
    share = D[D.y == 1].owner.isin(top).mean()
    rows.append(dict(check="양성 중 상위 5개 권리자 비중", mean_skill=share))
    rec("챔피언 vs M0 — 실용신안 제외", pc, p0, sub=lambda te: te.utility == 0)
    rec("챔피언 vs M0 — 국내 출원인만", pc, p0, sub=lambda te: te.foreign == 0)
elif idea == "J1":
    m1 = steps["M1_과제내용"]; m2 = steps["M2_제공50산업이력"]
    rec("챔피언 vs M0 — 기업 수행 과제만", pc, p0, sub=lambda te: te.firm == 1)
    rec("챔피언 vs M0 — 대학 과제만", pc, p0, sub=lambda te: te.perform_grp.astype(str) == "대학")
    rec("챔피언 vs M0 — 출연연·국공립연 과제만", pc, p0, sub=lambda te: te.perform_grp.astype(str).isin(["출연연", "국공립연"]))
    rec("챔피언 vs M0 — 과거 3년 과제가 없는 신규 수행기관", pc, p0, sub=lambda te: te.org_nproj == 0)
    rec("챔피언 vs M0 — 선행과제 없음", pc, p0, sub=lambda te: te.has_pre == 0)
    p_strong = run(base + m2)
    rec("강한 기준선(M0+제공 50산업 이력) 대비 챔피언(=과제 내용 몫)", pc, p_strong)
    p_noassign = run(base + [f for f in m1 if not f.endswith("assigned")] + m2)
    rec("배정 플래그(임계값이 ≤2014 성과로 정해짐) 뺀 모형 vs 챔피언", p_noassign, pc)
    rec("배정 플래그 뺀 모형 vs M0", p_noassign, p0)
    p_text = run(base + ["b1_tfidf_score", "b2_dict_score"])
    rec("M0+KSIC50 텍스트 점수 2개만 vs M0", p_text, p0)
    L2 = lambda fs: [f + "_L2" if f in ("org_kr_pat", "org_kr_rate", "pre_kr_pat", "org_p50", "org_p50_rate", "org_p50_share", "pre_p50") else f for f in fs]
    p0_l2 = run(L2(base)); pc_l2 = run(L2(champ["feats"]))
    rec("보고 지연 1년 가정(이력 출원연도 ≤ t−2): 챔피언 vs M0 (둘 다 지연판)", pc_l2, p0_l2)
    rec("보고 지연 1년 가정: 챔피언 지연판 vs 원래 챔피언", pc_l2, pc)
elif idea == "J3":
    pass
pd.DataFrame(rows).to_csv(f"{OUT}/{idea}_robust.csv", index=False, encoding="utf-8-sig")
print("sec", round(time.time() - t))
