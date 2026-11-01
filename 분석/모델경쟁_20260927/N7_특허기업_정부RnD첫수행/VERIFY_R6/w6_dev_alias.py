# -*- coding: utf-8 -*-
r"""개발 원점(2017–2020)에서 NPS 별칭 보정 영향 + 시험 하위집단 AUC의 부트스트랩 CI.
- 학습 원점 피처는 final/dev 모드가 같다(R5 감사 100% 일치, 별칭도 스냅샷 s 값만 씀) → final 패널의 t0 ≤ 2020 행만 사용(결과 ≤2022 < F).
- 원점 v: 학습 t0 2015..v−2 → 평가 v (동결 실행기 dev 모드와 같은 분할). 챔피언 ens·M0 xgb 3시드.
- 시험 하위집단 CI: 이미 저장된 시험 예측(preds_orig/alias)으로 사업자번호 군집 부트스트랩 1,000회."""
from w_common import *
R = {"dev": {}}
for nm in ["orig", "alias"]:
    D = pd.read_parquet(os.path.join(VC, f"final_panel_{nm}.parquet")); D = D[D.t0 <= 2020]
    for v in [2017, 2018, 2019, 2020]:
        tr = D[D.t0 <= v - 2]; te = D[D.t0 == v]
        pc = predict(tr[FE].astype(float), tr.y, te[FE].astype(float), "ens", SEEDS)
        p0 = predict(tr[M0F].astype(float), tr.y, te[M0F].astype(float), "xgb", SEEDS)
        y = te.y.values; m = te.nps_present.values == 0
        R["dev"][f"{nm}|{v}"] = dict(auc_champ=float(auc(y, pc)), auc_M0=float(auc(y, p0)), skill=float(skill(auc(y, pc), auc(y, p0))),
                                     nonnps_n=int(m.sum()), nonnps_auc_champ=float(auc(y[m], pc[m])), nonnps_auc_M0=float(auc(y[m], p0[m])))
        log(nm, v, R["dev"][f"{nm}|{v}"])
    R["dev"][f"{nm}|mean_skill"] = float(np.mean([R["dev"][f"{nm}|{v}"]["skill"] for v in [2017, 2018, 2019, 2020]]))
    R["dev"][f"{nm}|mean_nonnps_auc_champ"] = float(np.mean([R["dev"][f"{nm}|{v}"]["nonnps_auc_champ"] for v in [2017, 2018, 2019, 2020]]))
R["dev"]["frozen_spec_dev_auc"] = SPEC["dev"]["champ_auc"]

# 시험 하위집단 CI
O = pd.read_parquet(os.path.join(VC, "preds_orig.parquet")); A = pd.read_parquet(os.path.join(VC, "preds_alias.parquet"))
PA = pd.read_parquet(os.path.join(VC, "final_panel_alias.parquet")); PA = PA[PA.t0 == 2022].reset_index(drop=True)
hidden = (O.nps_present.values == 0) & (PA.nps_present.values == 1)
def boot_auc(y, p, cl, n=1000, seed=0):
    rng = np.random.default_rng(seed); u, inv = np.unique(cl.astype(str), return_inverse=True)
    idx = pd.Series(np.arange(len(y))).groupby(inv).apply(lambda s: s.values).values
    v = []
    for _ in range(n):
        i = np.concatenate([idx[j] for j in rng.integers(0, len(u), len(u))])
        if 0 < y[i].sum() < len(i):
            v.append(auc_rs(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
R["sub_ci"] = {}
for lab, P, mask in [("frozen|원래 미가입", O, O.nps_present.values == 0), ("frozen|진짜 미가입(숨은 가입 제외)", O, (O.nps_present.values == 0) & ~hidden),
                     ("frozen|숨은 가입", O, hidden), ("alias|보정 후 미가입", A, PA.nps_present.values == 0)]:
    y = P.y.values[mask].astype(int)
    R["sub_ci"][lab] = dict(n=int(mask.sum()), pos=int(y.sum()), auc_champ=auc_rs(y, P.p_champ.values[mask]),
                            ci=boot_auc(y, P.p_champ.values[mask], P.bizno.values[mask], seed=11), auc_M0=auc_rs(y, P.p_M0.values[mask]))
    log(lab, R["sub_ci"][lab])
json.dump(R, open(os.path.join(HERE, "w6_dev_alias.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("saved")
