# -*- coding: utf-8 -*-
"""사후 점검(채택 판단 무관): 최종 챔피언에서 '특허 유래 피처 전부'(M1 특허 행태 + C04 납부 지연 + C11 권리 처분)를 뺀 외부 전용 모형과 비교.
같은 ens 모형. 특허 신호가 외부 데이터(NPS 탈퇴·NTIS·DART)를 다 넣은 뒤에도 남는지 본다."""
import os, json
import pandas as pd, numpy as np
from n4lib import *
from n4feat import BUNDLES
D = pd.concat([pd.read_parquet(os.path.join(CACHE, "base_dev.parquet")), pd.read_parquet(os.path.join(CACHE, "feat_dev.parquet"))], axis=1)
assert D.t0.max() + 2 < F_YEAR
S = json.load(open(os.path.join(OUT, "champ.json"), encoding="utf-8")); FE = S["feats"]
PAT = set(M1_ADD + BUNDLES["C04_pay_delay"] + BUNDLES["C11_disposal"])
EXTF = [f for f in FE if f not in PAT]
rows = []
for v in DEV_ORIGINS:
    tr = D[(D.t0 >= TRAIN_MIN_T0) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
    pe = predict(tr[EXTF].astype(float), tr.y, te[EXTF].astype(float), model="ens", seeds=(0, 1, 2))
    pc = pd.read_parquet(os.path.join(CACHE, "preds_C12.parquet")).query("t0 == @v").p.values
    ci = cluster_boot(te.y.values, pc, pe, te.bizno.values, n=300, seed=v)
    m = te.nps_present.values == 0
    rows.append(dict(v=v, auc_external_only=auc(te.y, pe), auc_champ=auc(te.y, pc), skill_patent_on_top=skill(auc(te.y, pc), auc(te.y, pe)),
                     ci_lo=ci["skill_ci"][0], ci_hi=ci["skill_ci"][1],
                     auc_ext_nonnps=auc(te.y.values[m], pe[m]), auc_champ_nonnps=auc(te.y.values[m], pc[m])))
    log(v)
R = pd.DataFrame(rows); print(R.round(4).to_string(index=False)); print("mean skill", R.skill_patent_on_top.mean().round(4))
R.to_csv(os.path.join(OUT, "posthoc_patent_value.csv"), index=False, encoding="utf-8-sig")
