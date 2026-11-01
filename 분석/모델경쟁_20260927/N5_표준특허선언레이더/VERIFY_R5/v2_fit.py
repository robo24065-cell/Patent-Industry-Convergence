# -*- coding: utf-8 -*-
"""R5 레드팀 2단계: 새로 만든 패널로 동결 스펙(E21) 재학습 → 동결 예측(final_preds_r4.parquet)과 대조.
  part=main : 챔피언·M0·M1·로지스틱 2종 (공식 재현)
  part=sens : 강화 기준선·묶음 제거·NTIS 실시간 근사 (민감도/누수 영향)"""
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
from n5lib import FEATS, XGB_PARAMS, fit_pred, auc, skill
import numpy as np, pandas as pd
part = sys.argv[1]
SPEC = json.load(open(os.path.join(TR, "final_spec_r4.json"), encoding="utf-8"))
FE = SPEC["champion_features"]; F0 = SPEC["baseline_M0"]; F1 = FEATS["M0"] + FEATS["M1"]
tr = pd.read_parquet(os.path.join(HERE, "cache", "panel_train.parquet"))
te = pd.read_parquet(os.path.join(HERE, "cache", "panel_final.parquet"))
assert tr.t0.max() + 2 < 2023
t = time.time()
def fp(f, model="xgb", tr_=tr, te_=te):
    return fit_pred(tr_[f].astype(float), tr_.y, te_[f].astype(float), model=model, seeds=(0, 1, 2) if model == "xgb" else (0,))
P = {}
if part == "main":
    P["champ"] = fp(FE); P["M0"] = fp(F0); P["M1_pilot"] = fp(F1)
    P["champ_logit"] = fp(FE, "logit"); P["M0_logit"] = fp(F0, "logit")
    print("fit", round(time.time() - t), flush=True)
    out = pd.DataFrame({"appl_no": te.appl_no.values, "app": te.app.values, "y": te.y.values, **{f"p_{k}": v for k, v in P.items()}})
    out.to_parquet(os.path.join(HERE, "cache", "repro_preds.parquet"), index=False)
else:
    def drop(g):
        return [f for f in FE if f not in FEATS[g] and f != g]
    P["M0plus"] = fp(F0 + ["sg_rate", "sg_logn", "app_rec_rate", "mg_rec_rate"])   # 강화 기준선: 최신 선언률을 기준선에 포함
    P["M0_M2p"] = fp(F0 + FEATS["M2p"])                                              # 파일럿 M2 참고 피처 포함 기준선
    P["champ_noNTIS"] = fp(drop("ntis")); P["champ_noRoS"] = fp(drop("ros")); P["champ_noTitle"] = fp(drop("title"))
    print("fit", round(time.time() - t), flush=True)
    # NTIS 실시간 근사: 보고 지연 때문에 t0 시점에는 최근 출원의 NTIS 연결이 덜 보였을 것.
    # 현재 스냅샷의 출원연도별 연결률(KR 전체 제공 특허)로 '경과 연수별 보고 비율'을 추정해 시험 피처를 무작위로 지운다.
    fl = pd.read_parquet(os.path.join(TR, "..", "_shared", "ext", "ntis_patent_flag.parquet"))
    fl = fl[fl.cc == "KR"].drop_duplicates("appl_no")
    fl["ay"] = pd.to_numeric(fl.appl_year, errors="coerce")
    cov = fl.groupby("ay").gov_rnd.mean()
    steady = float(cov.loc[2015:2020].mean())
    ret = {2022: float(cov.loc[2025] / steady), 2021: float(cov.loc[2024] / steady), 2020: float(cov.loc[2023] / steady)}
    ret = {k: min(1.0, v) for k, v in ret.items()}
    print("coverage", cov.loc[2014:2025].round(4).to_dict(), "steady", round(steady, 4), "retention", ret, flush=True)
    from xgboost import XGBClassifier
    ms = [XGBClassifier(random_state=s, **XGB_PARAMS).fit(tr[FE].astype(float), tr.y) for s in (0, 1, 2)]
    pred = lambda X: np.mean([mm.predict_proba(X)[:, 1] for mm in ms], axis=0)
    base = pred(te[FE].astype(float))
    sims = []
    nt = ["ntis_gov", "ntis_gri", "ntis_firm", "ntis_univ"]
    for s in range(20):
        rng = np.random.default_rng(s); X = te[FE].astype(float).copy()
        keep = np.array([rng.random() < ret.get(int(a), 1.0) for a in te.ay])
        X.loc[~keep, nt] = 0.0
        sims.append(pred(X))
    P["champ_ntis_rt_mean"] = np.mean(sims, axis=0)
    X0 = te[FE].astype(float).copy(); X0[nt] = 0.0; P["champ_ntis_zero_test"] = pred(X0)
    sim_auc = [float(auc(te.y, p)) for p in sims]
    json.dump(dict(coverage_by_ay={int(k): float(v) for k, v in cov.loc[2010:2025].items()}, steady_2015_2020=steady, retention=ret,
                   auc_base=float(auc(te.y, base)), auc_sims=sim_auc), open(os.path.join(HERE, "v2_ntis_realtime.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    out = pd.DataFrame({"appl_no": te.appl_no.values, **{f"p_{k}": v for k, v in P.items()}})
    out.to_parquet(os.path.join(HERE, "cache", "sens_preds.parquet"), index=False)
print({k: round(float(auc(te.y, v)), 5) for k, v in P.items()}, round(time.time() - t), flush=True)
