# -*- coding: utf-8 -*-
"""N6 챔피언/도전자 순차 실험 (PLAN.md 사전 선언 순서). dev 원점 2014–2019만. 학습 t0 ∈ [2012, v-2].
채택: 평균 skill(M0 대비) 차 >= +0.003 AND 원점 60% 이상에서 AUC 우세.
출력: experiments.csv, champion_by_origin.csv, cache/preds_champion.parquet
사용: python n6_exp.py main | retest <C..> [<C..>]"""
import sys, json, warnings
import numpy as np, pandas as pd
from n6_lib import *
warnings.filterwarnings("ignore")

M0 = ["n_proj", "funds", "n_yrs", "rec", "sbiz_sh", "cont_sh", "sme", "career", "n_proj_all", "corp_type"]
M1 = ["ph_dev", "ph_app"] + ["t6_" + x for x in ["IT", "BT", "NT", "ET", "ST", "CT"]] \
    + ["cl_" + x for x in ["기계", "전기전자", "정보통신", "화공", "재료", "보건의료", "농림수산", "생명과학"]] + ["k50_assigned", "k50_score", "b1_score"]
M2 = ["nps_present", "log_emp", "g_emp1", "nps_age"]
BUNDLES = {
    "C01": ("텍스트 대응 고도화 A(최댓값·배정 수·원점 내 순위·t0년 과제)", ["k50_max", "k50_n", "b1_assigned", "b1_max", "k50_rank", "b1_rank", "k50_cur"]),
    "C02": ("과제↔KSIC50 특허 제목 TF-IDF 중심 대응(제공 KR ≤2010)", ["pt_mean", "pt_max", "pt_margin", "pt_all"]),
    "C03": ("적용분야·부처 비중", ["ap_mfg", "ap_ict", "ap_health", "ap_agri", "ap_energy", "ap_prosvc", "mn_sme", "mn_motie", "mn_msit", "mn_agri", "mn_env", "mn_health"]),
    "C04": ("과거 전환율 인코딩(사업명·과학기술 중분류, t0<=T-2 레이블)", ["te_prog_max", "te_prog_mean", "te_clsM_max", "te_clsM_mean"]),
    "C05": ("NTIS 성과 IP(범위 밖 특허·디자인·상표·해외, 성과연도<=t0-1)", ["o_pat", "o_pat_rg", "o_des", "o_util", "o_tm", "o_frgn", "o_pat_rec"]),
    "C06": ("국민연금 고도화(채용·이탈·임금·사업장·2년 증감)", ["hire_rate", "loss_rate", "wage", "n_wkpl", "g_emp2"]),
    "C07": ("기업 산업(NPS 업종)의 제공범위 특허 침투율", ["ind_pen6", "ind_pen3", "ind_k50"]),
    "C08": ("국세청 폐업(원점까지)", ["nts_closed"]),
    "C09": ("과제 시점 구조(종료 임박·신규 수주·연구비 집중)", ["end_soon", "n_new_t0", "fund_t0_sh", "n_prog", "max_fund", "tot_funds"]),
    "C10": ("6T 코드 교정·기술수명주기", ["s6_01", "s6_02", "s6_03", "s6_04", "s6_05", "s6_06", "lc_intro", "lc_growth", "lc_mature"]),
}
MODEL_STEPS = {
    "C11": ("모형: XGB+로지스틱 순위 평균(0.5/0.5)", {"model": "ens", "seeds": [0], "w_xgb": 0.5}),
    "C12": ("모형: XGB 3시드 평균", {"seeds": [0, 1, 2]}),
    "C13": ("최근 원점 가중 0.8^(최신 학습 원점−t0)", {"recent_decay": 0.8}),
}
ORDER = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13"]
# 2차(PLAN.md 변경 기록, 1차 결과를 본 뒤 실행 전 선언)
BUNDLES["C14"] = ("(R) C04 재시험: " + BUNDLES["C04"][0], BUNDLES["C04"][1])
BUNDLES["C15"] = ("(R) C01 재시험: " + BUNDLES["C01"][0], BUNDLES["C01"][1])
BUNDLES["C16"] = ("NTIS 성과 특허 중 제공 KR(50 KSIC) 연결분(성과연도<=t0-1)", ["o_prov", "o_prov_any"])
MODEL_STEPS["C17"] = ("앙상블 XGB 성분 3시드 평균", {"seeds": [0, 1, 2]})
ORDER += ["C14", "C15", "C16", "C17"]
DEV = DEV_ORIGINS


def load_panel():
    D = pd.read_parquet(f"{CACHE}/panel_dev.parquet")
    assert D.t0.max() + 2 <= F_YEAR - 1
    return D[D.t0 >= MIN_TRAIN].reset_index(drop=True)


def score(D, feats, spec=None):
    aucs, preds = {}, {}
    for v in DEV:
        tr = D[(D.t0 <= v - 2) & (D.t0 >= MIN_TRAIN)]; te = D[D.t0 == v]
        p = predict(tr, te, feats, spec=spec)
        preds[v] = p; aucs[v] = auc(te.y.values, p)
    return aucs, preds


def topk_stats(D, preds):
    out = {}
    for fr in (0.01, 0.05, 0.10):
        out[f"top{int(fr * 100)}"] = float(np.mean([topk(D[D.t0 == v].y.values, preds[v], fr) for v in DEV]))
    out["base_rate"] = float(np.mean([D[D.t0 == v].y.mean() for v in DEV]))
    return out


def row(exp_id, parent, change, aucs, m0, champ, preds, D, decision="", note=""):
    sk = {v: skill(aucs[v], m0[v]) for v in DEV}
    r = dict(exp_id=exp_id, parent=parent, change=change, **{f"auc_{v}": round(aucs[v], 4) for v in DEV},
             mean_auc=round(float(np.mean(list(aucs.values()))), 4), **{f"skill_{v}": round(sk[v], 4) for v in DEV},
             mean_skill_vs_M0=round(float(np.mean(list(sk.values()))), 4))
    if champ is not None:
        csk = np.mean([skill(champ[v], m0[v]) for v in DEV])
        r["d_skill_vs_champ"] = round(float(np.mean(list(sk.values())) - csk), 4)
        r["d_skill_rel_champ"] = round(float(np.mean([skill(aucs[v], champ[v]) for v in DEV])), 4)
        r["win"] = f"{sum(aucs[v] > champ[v] for v in DEV)}/{len(DEV)}"
    ts = topk_stats(D, preds)
    r.update({k: round(v, 4) for k, v in ts.items()})
    r["decision"] = decision; r["note"] = note
    return r


def decide(aucs, m0, champ):
    d = np.mean([skill(aucs[v], m0[v]) for v in DEV]) - np.mean([skill(champ[v], m0[v]) for v in DEV])
    wins = sum(aucs[v] > champ[v] for v in DEV) / len(DEV)
    return ("ACCEPT" if (d >= 0.003 and wins >= 0.6) else "ROLLBACK"), d, wins


def main():
    D = load_panel()
    rows = []
    a, p = score(D, ["n_proj"]); rows.append(row("E00", "", "규모 1변수(n_proj) 참고", a, a, None, p, D, "INFO"))
    m0, pm0 = score(D, M0)
    rows.append(row("E01", "", "M0 재현(NTIS 과제 이력 10개, XGB)", m0, m0, None, pm0, D, "BASE",
                    "파일럿 XGB M0 AUC 0.6403/0.6719/0.7046/0.7028/0.7162/0.7078"))
    feats = M0 + M1 + M2; spec = {"model": "xgb", "seeds": [0]}
    champ, pc = score(D, feats, spec)
    rows.append(row("E02", "E01", "파일럿 챔피언 재현 M0+M1+M2(XGB) = 출발 챔피언", champ, m0, None, pc, D, "CHAMP0", "파일럿 skill 평균 0.122"))
    lm0, plm0 = score(D, M0, {"model": "logit"}); lm1, plm1 = score(D, M0 + M1, {"model": "logit"})
    rows.append(row("E03a", "", "로지스틱 M0 재현(INFO)", lm0, m0, None, plm0, D, "INFO"))
    r = row("E03b", "", "로지스틱 M0+M1 재현(INFO, skill은 XGB M0 대비)", lm1, m0, None, plm1, D, "INFO",
            "로지스틱 M0 대비 skill " + "/".join(f"{skill(lm1[v], lm0[v]):.3f}" for v in DEV))
    rows.append(r)
    log("E02 champ", round(np.mean([skill(champ[v], m0[v]) for v in DEV]), 4))
    champ_id = "E02"
    for cid in ORDER:
        if cid in BUNDLES:
            desc, add = BUNDLES[cid]; nf = feats + add; ns = spec
            change = f"+{desc}: " + ",".join(add)
        else:
            desc, sp = MODEL_STEPS[cid]; nf = feats; ns = {**spec, **sp}
            if cid == "C13" and spec.get("model") == "ens":
                ns = {**spec, **sp}
            change = desc
        a, p = score(D, nf, ns)
        dec, d, w = decide(a, m0, champ)
        rows.append(row(cid, champ_id, change, a, m0, champ, p, D, dec))
        log(cid, dec, round(d, 4), w)
        if dec == "ACCEPT":
            feats, spec, champ, pc, champ_id = nf, ns, a, p, cid
    pd.DataFrame(rows).to_csv(f"{OUT}/experiments.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(champ_id=champ_id, feats=feats, spec=spec), open(f"{CACHE}/champion_state.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    by_origin(D, feats, spec, champ_id, m0, pm0, pc)


def by_origin(D, feats, spec, champ_id, m0, pm0, pc):
    out = []
    for v in DEV:
        te = D[D.t0 == v]; y = te.y.values
        ci = boot_ci(y, pc[v], pm0[v], n=1000, seed=v, groups=te.bz.values)
        r = dict(champ=champ_id, v=v, n=len(te), pos=int(y.sum()), base_rate=round(y.mean(), 4), auc_M0=round(m0[v], 4),
                 auc_champ=round(auc(y, pc[v]), 4), skill=round(skill(auc(y, pc[v]), m0[v]), 4),
                 skill_lo=round(ci["skill_ci"][0], 4), skill_hi=round(ci["skill_ci"][1], 4),
                 auc_lo=round(ci["auc_ci"][0], 4), auc_hi=round(ci["auc_ci"][1], 4))
        for fr in (0.01, 0.05, 0.10):
            r[f"top{int(fr * 100)}_champ"] = round(topk(y, pc[v], fr), 4); r[f"top{int(fr * 100)}_M0"] = round(topk(y, pm0[v], fr), 4)
        out.append(r)
    O = pd.DataFrame(out); O.to_csv(f"{OUT}/champion_by_origin.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); print(O.to_string(index=False))
    P = pd.concat([pd.DataFrame({"bz": D[D.t0 == v].bz.values, "t0": v, "y": D[D.t0 == v].y.values, "p_champ": pc[v], "p_m0": pm0[v]}) for v in DEV])
    P.to_parquet(f"{CACHE}/preds_champion.parquet", index=False)


if __name__ == "__main__":
    main()
