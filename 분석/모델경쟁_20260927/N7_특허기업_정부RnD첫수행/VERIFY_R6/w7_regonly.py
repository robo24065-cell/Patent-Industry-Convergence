# -*- coding: utf-8 -*-
r"""민감도(L5 후속): 제공 KR 출원→출원인코드 연결에서 'CORP_APPLICANT 2026 상호 일치' 경로(src=name, 사후 목록)를 빼고 등록원부 정확 코드만 쓴 패널.
NPS는 원래 키(별칭 없음)로 두어 상호 경로 효과만 분리. 동결 스펙으로 재학습 → 공통 기업에서 원래 패널과 비교. 결과 cache/final_panel_regonly.parquet, w7_regonly.json"""
from w_common import *
_orig_af = n7lib.load_app_firm
def af_reg():
    a = _orig_af(); return a[a.src == "registry"].reset_index(drop=True)
n7lib.load_app_firm = af_reg; n7feat.load_app_firm = af_reg
D = n7lib.build_panel("final"); ctx = n7feat.load_ctx("final")
X = n7feat.build_features(D, ctx, SPEC["feature_bundles_needed"]); D = pd.concat([D, X], axis=1)
D.to_parquet(os.path.join(VC, "final_panel_regonly.parquet"), index=False)
tr = D[D.t0 <= 2020]; te = D[D.t0 == 2022].reset_index(drop=True)
pc = predict(tr[FE].astype(float), tr.y, te[FE].astype(float), "ens", SEEDS)
p0 = predict(tr[M0F].astype(float), tr.y, te[M0F].astype(float), "xgb", SEEDS)
pe = predict(tr[EXT_ALL].astype(float), tr.y, te[EXT_ALL].astype(float), "ens", SEEDS)
y = te.y.values
O = pd.read_parquet(os.path.join(VC, "preds_orig.parquet"))
common = te.applicantCd.isin(set(O.applicantCd)).values
Oc = O[O.applicantCd.isin(set(te.applicantCd))]
R = dict(n_test=len(te), n_orig=len(O), n_common=int(common.sum()), pos=int(y.sum()),
         auc_champ=float(auc(y, pc)), auc_M0=float(auc(y, p0)), auc_EXT=float(auc(y, pe)), skill=float(skill(auc(y, pc), auc(y, p0))),
         skill_vs_EXT=float(skill(auc(y, pc), auc(y, pe))),
         common_regonly=dict(auc_champ=float(auc(y[common], pc[common])), auc_M0=float(auc(y[common], p0[common]))),
         common_orig=dict(auc_champ=float(auc(Oc.y, Oc.p_champ)), auc_M0=float(auc(Oc.y, Oc.p_M0))),
         rows_by_t0={int(t): int(len(g)) for t, g in D.groupby("t0")})
json.dump(R, open(os.path.join(HERE, "w7_regonly.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(R)
