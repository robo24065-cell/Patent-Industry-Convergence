# -*- coding: utf-8 -*-
"""사후 점검(채택 판단에 쓰지 않음, dev 원점만): 최종 챔피언 vs 기준선 M0(XGB 3시드 / 같은 모형족 ens)
원점별 AUC·skill·사업자번호 군집 부트스트랩 CI·PR-AUC·상위 1% 적중, NPS 미등록 하위집단, 묶음 하나씩 빼기(leave-one-bundle-out)."""
import os, json
import pandas as pd, numpy as np
from n4lib import *
from n4feat import BUNDLES

D = pd.concat([pd.read_parquet(os.path.join(CACHE, "base_dev.parquet")), pd.read_parquet(os.path.join(CACHE, "feat_dev.parquet"))], axis=1)
assert D.t0.max() + 2 < F_YEAR
S = json.load(open(os.path.join(OUT, "champ.json"), encoding="utf-8"))
FE, MD = S["feats"], S["model"]; SE = (0, 1, 2)
V = DEV_ORIGINS
rows, sub = [], []
P = {}
for v in V:
    tr = D[(D.t0 >= TRAIN_MIN_T0) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
    f = lambda feats, md: predict(tr[feats].astype(float), tr.y, te[feats].astype(float), model=md, seeds=SE)
    pc = f(FE, MD); p0 = f(M0, "xgb"); p0e = f(M0, "ens"); p1 = f(M0 + M1_ADD, "xgb")
    P[v] = (te, pc, p0)
    y = te.y.values; k = max(1, int(0.01 * len(te)))
    top = lambda p: y[np.argsort(-p)[:k]].mean()
    ci = cluster_boot(y, pc, p0, te.bizno.values, n=500, seed=v)
    ce = cluster_boot(y, pc, p0e, te.bizno.values, n=500, seed=v)
    a_c, a0, a0e, a1 = auc(y, pc), auc(y, p0), auc(y, p0e), auc(y, p1)
    rows.append(dict(v=v, n=len(te), pos=int(y.sum()), base_rate=y.mean(), auc_M0_xgb=a0, auc_M0_ens=a0e, auc_M1_xgb=a1, auc_champ=a_c,
                     skill_vs_M0=skill(a_c, a0), ci_lo=ci["skill_ci"][0], ci_hi=ci["skill_ci"][1],
                     skill_vs_M0ens=skill(a_c, a0e), ci_ens_lo=ce["skill_ci"][0], ci_ens_hi=ce["skill_ci"][1],
                     prauc_champ=prauc(y, pc), prauc_M0=prauc(y, p0), top1_champ=top(pc), top1_M0=top(p0), top1_M1=top(p1)))
    m = te.nps_present.values == 0
    sub.append(dict(v=v, n_nonnps=int(m.sum()), rate_nonnps=y[m].mean(), rate_nps=y[~m].mean(),
                    auc_M0_nonnps=auc(y[m], p0[m]), auc_champ_nonnps=auc(y[m], pc[m]), auc_M0_nps=auc(y[~m], p0[~m]), auc_champ_nps=auc(y[~m], pc[~m])))
    log(v, round(a0, 4), round(a_c, 4))
R = pd.DataFrame(rows); SB = pd.DataFrame(sub)
print(R.round(4).to_string(index=False)); print(SB.round(4).to_string(index=False))
R.to_csv(os.path.join(OUT, "posthoc_champ_by_origin.csv"), index=False, encoding="utf-8-sig")
SB.to_csv(os.path.join(OUT, "posthoc_subgroup_nonnps.csv"), index=False, encoding="utf-8-sig")
# leave-one-bundle-out (XGB 단일 시드 0으로 빠르게, 같은 모형 ens)
groups = {"M1_patent_behavior": M1_ADD, "C01_nps_exit": BUNDLES["C01_nps_exit"], "C02_ntis_project": BUNDLES["C02_ntis_project"],
          "C04_pay_delay": BUNDLES["C04_pay_delay"], "C11_disposal": BUNDLES["C11_disposal"], "V09_dart": ["dart_has", "dart_impair", "dart_neg_eq"]}
lo = []
for g, fs in groups.items():
    if not set(fs) & set(FE):
        continue
    feats = [x for x in FE if x not in fs]
    r = {}
    for v in V:
        tr = D[(D.t0 >= TRAIN_MIN_T0) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        r[v] = auc(te.y, predict(tr[feats].astype(float), tr.y, te[feats].astype(float), model=MD, seeds=SE))
    full = {v: auc(P[v][0].y, P[v][1]) for v in V}
    lo.append(dict(drop=g, **{f"auc_{v}": r[v] for v in V}, mean_skill_loss=np.mean([skill(full[v], r[v]) for v in V]),
                   n_origins_worse=int(sum(r[v] < full[v] for v in V))))
    log("drop", g)
L = pd.DataFrame(lo); print(L.round(4).to_string(index=False))
L.to_csv(os.path.join(OUT, "posthoc_leave_one_bundle_out.csv"), index=False, encoding="utf-8-sig")
