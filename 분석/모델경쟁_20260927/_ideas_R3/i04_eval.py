# -*- coding: utf-8 -*-
"""사전 등록된 순차 실험(M0→M1→M2→M3)만 채점. 전진 검증: 원점 v 평가, 학습 t0 ≤ v-2 (결과창 2년이 v 이전에 끝남).
채택: 평균 skill(챔피언 대비) ≥ +0.003 AND 원점 60% 이상 우세. 아니면 롤백하고 다음 묶음은 챔피언 위에 얹는다."""
import os, sys, json, time
import pandas as pd, numpy as np
from i00_common import *

F_NPS = ["log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "n_wkpl", "div2", "sido"]
SETS = {
    "I1": dict(panel="I1_panel.parquet", y="y", dev=[2017, 2018, 2019], base=F_NPS,
               steps=[("M1_patent_penetration", ["pen_c6", "ent_c6", "pen_c4", "ent_c4", "pen_sgg", "ent_sgg", "pen_sggd", "ent_sggd"]),
                      ("M2_ksic_patent_momentum", ["x_s_kr", "x_s_fr"]),
                      ("M3_external_sci_price", ["x_s_sci", "x_s_ppi"])]),
    "I2": dict(panel="I23_panel.parquet", y="y_exit", dev=[2017, 2018, 2019, 2020, 2021], base=F_NPS + ["mfg"],
               steps=[("M1_patent_behavior", ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y", "log_live", "lapse_1y", "lapse_2y",
                                              "pay_1y", "pay_trend", "trf_2y", "live_ratio"]),
                      ("M2_ksic_patent_momentum", ["x_s_kr", "x_s_fr"]),
                      ("M3_external_sci_price", ["x_s_sci", "x_s_ppi"])]),
    "I3": dict(panel="I23_panel.parquet", y="y_hgf", dev=[2017, 2018, 2019, 2020, 2021], base=F_NPS + ["mfg"],
               steps=[("M1_patent_behavior", ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y", "log_live", "lapse_1y", "lapse_2y",
                                              "pay_1y", "pay_trend", "trf_2y", "live_ratio"]),
                      ("M2_ksic_patent_momentum", ["x_s_kr", "x_s_fr"]),
                      ("M3_external_sci_price", ["x_s_sci", "x_s_ppi"])]),
    "I4": dict(panel="I4_panel.parquet", y="y", dev=[2012, 2013, 2014, 2015, 2016, 2017, 2018],
               base=["log_app", "pat_age", "yrs_since_last", "app_trend", "known_2y", "n_ksic", "claims", "reg_share"], size="log_app", base_label="KR 출원이력만(기업)",
               steps=[("M1_5country_fusion", ["x_s_abroad_rate", "x_peer_first_abroad", "x_s_fr", "x_s_kr"]),
                      ("M2_external_export_sci_price", ["x_s_exp", "x_s_explvl", "x_s_sci", "x_s_ppi"]),
                      ("M3_nps_size", ["nps_log_emp"])]),
    "I5": dict(panel="I5_panel.parquet", y="y", dev=[2014, 2015, 2016, 2017, 2018, 2019, 2020],
               base=["app_age", "reg_t0", "claims_t0", "app_rate", "app_logn", "mg_rate", "mg_logn"], size="app_rate", base_label="출원인·IPC 선언성향+출원속성",
               steps=[("M1_5country_family", ["fam_US", "fam_EP", "fam_JP", "fam_n"]),
                      ("M2_etsi_family_transfer_trend", ["fam_declared", "app_recent_dec", "mg_recent_dec"])]),
    "I2b": dict(panel="I2b_panel.parquet", y="y", dev=[2017, 2018, 2019, 2020, 2021],
                base=["nps_present", "log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "corp_type"], size="log_emp",
                base_label="외부 기준선(NPS 속성·미가입 여부·사업자번호 유형)",
                steps=[("M1_patent_behavior", ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y", "log_live", "lapse_1y", "lapse_2y",
                                               "pay_1y", "pay_trend", "trf_2y", "live_ratio"]),
                       ("M2_ksic_patent_momentum", ["x_s_kr", "x_s_fr"]),
                       ("M3_external_sci_price", ["x_s_sci", "x_s_ppi"])]),
}


def run(idea, model="xgb"):
    S = SETS[idea]
    D = pd.read_parquet(os.path.join(CACHE, S["panel"]))
    D = D[D[S["y"]] >= 0].copy()
    D["mfg"] = D.get("mfg", 0).astype(float) if "mfg" in D else 0.0
    preds = {}

    def score(feats, tag):
        res = {}
        for v in S["dev"]:
            tr = D[(D.t0 <= v - 2)]; te = D[D.t0 == v]
            p = fit_pred(tr[feats].astype(float), tr[S["y"]], te[feats].astype(float), model=model)
            preds[(tag, v)] = p
            res[v] = auc(te[S["y"]], p)
        return res

    rows = []
    # 참고 기준선: 규모만
    r_size = score([S.get("size", "log_emp")], "B_size")
    champ_feats = list(S["base"]); champ = score(champ_feats, "M0"); champ_tag = "M0"
    base = dict(champ)
    rows.append(dict(idea=idea, model=model, exp="B_size", change="규모 1변수(" + S.get("size", "log_emp") + ")", **{f"auc_{v}": r_size[v] for v in S["dev"]}))
    rows.append(dict(idea=idea, model=model, exp="M0", change=S.get("base_label", "NPS 기업속성 기준선"), **{f"auc_{v}": champ[v] for v in S["dev"]}, decision="BASE"))
    for name, add in S["steps"]:
        feats = champ_feats + add
        r = score(feats, name)
        sk = np.mean([skill(r[v], champ[v]) for v in S["dev"]])
        win = np.mean([r[v] > champ[v] for v in S["dev"]])
        dec = "ACCEPT" if (sk >= 0.003 and win >= 0.6) else "ROLLBACK"
        rows.append(dict(idea=idea, model=model, exp=name, change="+" + ",".join(add), **{f"auc_{v}": r[v] for v in S["dev"]},
                         d_skill_vs_champ=round(sk, 4), win=win, decision=dec))
        if dec == "ACCEPT":
            champ_feats, champ, champ_tag = feats, r, name
    # 최종 챔피언 vs M0 기준선 skill, 원점별 부트스트랩 CI
    out = []
    for v in S["dev"]:
        te = D[D.t0 == v]
        lo, hi = boot_ci_skill(te[S["y"]].values, preds[(champ_tag, v)], preds[("M0", v)], n=200)
        out.append(dict(v=v, n=len(te), pos=int(te[S["y"]].sum()), auc_M0=base[v], auc_champ=champ[v], auc_size=r_size[v],
                        skill=skill(champ[v], base[v]), ci_lo=lo, ci_hi=hi))
    # 상위 1% 적중률(챔피언)
    for o in out:
        te = D[D.t0 == o["v"]]; p = preds[(champ_tag, o["v"])]; yv = te[S["y"]].values
        k = max(1, int(0.01 * len(te)))
        o["top1pct_prec"] = yv[np.argsort(-p)[:k]].mean(); o["base_rate"] = yv.mean()
        pb = preds[("M0", o["v"])]; o["top1pct_prec_M0"] = yv[np.argsort(-pb)[:k]].mean()
    return pd.DataFrame(rows), pd.DataFrame(out), champ_tag, champ_feats


if __name__ == "__main__":
    ideas = sys.argv[1:] or ["I1", "I2", "I3"]
    allr, allo = [], []
    for idea in ideas:
        for model in ["xgb", "logit"]:
            t = time.time()
            r, o, tag, feats = run(idea, model)
            o.insert(0, "idea", idea); o.insert(1, "model", model); o.insert(2, "champ", tag)
            allr.append(r); allo.append(o)
            print(r.to_string(index=False)); print(o.round(4).to_string(index=False)); print("champion", tag, round(time.time() - t), flush=True)
    R = pd.concat(allr); O = pd.concat(allo)
    tag = "_".join(ideas)
    R.to_csv(os.path.join(OUT, f"experiments_{tag}.csv"), index=False, encoding="utf-8-sig")
    O.to_csv(os.path.join(OUT, f"pilot_by_origin_{tag}.csv"), index=False, encoding="utf-8-sig")
