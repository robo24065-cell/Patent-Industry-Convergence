# -*- coding: utf-8 -*-
"""사전 선언(PLAN.md) 순차 실험 실행기. 사용: python e_run.py <family xgb|logit> <EXP_ID> [<EXP_ID> ...]
챔피언 상태는 cache/champ_<family>.json, 기록은 experiments.csv(누적), 예측은 cache/preds_<family>_<EXP>.parquet."""
import sys, json, os, time
sys.dont_write_bytecode = True
from n8lib import *

M0 = ["age_rg", "age_ap", "rem_life", "claims", "n_ipc", "n_inv", "utility", "atype", "foreign", "n_appl", "ksic_i", "ipc_sec", "last_annl", "paid_ahead"]
M1 = ["own_disp", "own_trial", "tech_eval", "n_trf", "trf_2y", "merger_trf", "lic", "owner_live", "owner_disp5", "mg_disp5", "ksic_disp5", "fam_offices", "fam_us"]
M2 = ["us_granted", "us_sci", "us_papercit", "kpss_xi", "etsi", "gov", "gov_firm", "gov_univ", "gov_gri"]
EXPS = {
    "E00": dict(change="M0 기준선(IDEAS 14개)", base=True),
    "E01": dict(change="+M1 파일럿 이력 13개", add=M1),
    "E02": dict(change="권리자 이력 6개 → owner∪utilityModelOwner 판", swap={"n_trf": "n_trf_v2", "trf_2y": "trf_2y_v2", "merger_trf": "merger_trf_v2", "owner_live": "owner_live_v2", "owner_disp5": "owner_disp5_v2"}),
    "E03": dict(change="챔피언 last_annl·paid_ahead → 수치 max 판", swap={"last_annl": "last_annl_v2", "paid_ahead": "paid_ahead_v2"}),
    "E04": dict(change="+셀 분쟁 당사자 수·최근 2년 분쟁률", add=["mg_parties5", "ksic_parties5", "mg_disp2"]),
    "E05": dict(change="+자기 분쟁 이력 세부", add=["yrs_last_disp", "own_decided", "own_appeal", "own_corr"]),
    "E06": dict(change="+권리자 심판 경험·매입 집중", add=["owner_trial5", "owner_acq_share", "owner_n_acq"]),
    "E07": dict(change="+실시권 세부", add=["lic_active", "lic_excl", "lic_2y", "n_licensees"]),
    "E08": dict(change="+M2 외부 단위 9개(RoS·KPSS·ETSI·NTIS) 재시험", add=M2),
    "E09": dict(change="+KSIC 광업제조업조사(출하 성장·사업체)", add=["ks_ship_g2", "ks_estab_log", "ks_estab_g2"]),
    "E10": dict(change="+KSIC 무역(HS6 수입 성장·중국 비중)", add=["imp_g3", "imp_cn_share"]),
    "E11": dict(change="XGB+로지스틱 순위 평균 앙상블", set={"ens": True}),
    "E12": dict(change="학습 음성 표본 5%→20%", set={"neg_frac": 0.2}),
    "E13": dict(change="최근 원점 가중 0.85^(v-2-t0)", set={"decay": 0.85}),
    "E14": dict(change="음성 표본 시드 배깅(0·1·2) 확률 평균", set={"bag": [0, 1, 2]}),
    "E15": dict(change="(재시험) 권리자 이력 6개 → owner∪utilityModelOwner 판", swap={"n_trf": "n_trf_v2", "trf_2y": "trf_2y_v2", "merger_trf": "merger_trf_v2", "owner_live": "owner_live_v2", "owner_disp5": "owner_disp5_v2"}),
    "E16": dict(change="(재시험) +M2 외부 단위 9개", add=M2),
    "E17": dict(change="(재시험) last_annl·paid_ahead → 수치 max 판", swap={"last_annl": "last_annl_v2", "paid_ahead": "paid_ahead_v2"}),
}
COLS = ["rgst_no", "t0", "y", "owner_v2", "own_disp", "own_trial", "ksic"]


def load_panel(feats_needed):
    D = pd.read_parquet(f"{CACHE}/panel_dev.parquet", columns=list(dict.fromkeys(COLS + feats_needed)))
    return D.reset_index(drop=True)


def score(D, feats, cfg, family, seed=0, ytrain=None):
    """원점별 예측. cfg: neg_frac, decay, ens, bag(시드 목록). ytrain: 학습 레이블 대체(위약용)."""
    if cfg.get("bag"):
        c2 = {k: v for k, v in cfg.items() if k != "bag"}
        ps = [score(D, feats, c2, family, seed=s, ytrain=ytrain) for s in cfg["bag"]]
        return {v: np.mean([p[v] for p in ps], axis=0) for v in DEV}
    nf = cfg.get("neg_frac", 0.05)
    rng = np.random.default_rng(seed)
    y = D.y.values if ytrain is None else ytrain
    keep = (y == 1) | (rng.random(len(D)) < nf)
    w0 = np.where(y == 1, 1.0, 1.0 / nf)
    preds = {}
    for v in DEV:
        msk = (D.t0.values <= v - 2) & (D.t0.values >= TRAIN_MIN) & keep
        w = w0[msk].copy()
        if cfg.get("decay"):
            w = w * cfg["decay"] ** (v - 2 - D.t0.values[msk])
        Xtr = D.loc[msk, feats].astype(float); ytr = y[msk]
        te = D.t0.values == v; Xte = D.loc[te, feats].astype(float)
        if cfg.get("ens"):
            p1 = fit_pred(Xtr, ytr, Xte, "xgb", w=w, seed=seed); p2 = fit_pred(Xtr, ytr, Xte, "logit", w=w)
            preds[v] = 0.5 * rank01(p1) + 0.5 * rank01(p2)
        else:
            preds[v] = fit_pred(Xtr, ytr, Xte, family, w=w, seed=seed)
    return preds


def aucs(D, preds):
    return {v: auc(D.y.values[D.t0.values == v], preds[v]) for v in DEV}


def save_preds(D, preds, family, eid):
    rows = []
    for v in DEV:
        te = D[D.t0 == v]
        rows.append(pd.DataFrame({"rgst_no": te.rgst_no.values, "t0": v, "p": preds[v]}))
    pd.concat(rows).to_parquet(f"{CACHE}/preds_{family}_{eid}.parquet", index=False)


def append_row(row):
    f = f"{HERE}/experiments.csv"
    df = pd.DataFrame([row])
    if os.path.exists(f):
        old = pd.read_csv(f, encoding="utf-8-sig")
        old = old[~((old.exp_id == row["exp_id"]) & (old.family == row["family"]))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(f, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    family = sys.argv[1]
    allf = M0 + M1 + M2 + sorted({x for e in EXPS.values() for x in e.get("add", [])} | {x for e in EXPS.values() for x in e.get("swap", {}).values()})
    D = load_panel(list(dict.fromkeys(allf)))
    cf = f"{CACHE}/champ_{family}.json"
    for eid in sys.argv[2:]:
        t = time.time(); e = EXPS[eid]
        if e.get("base"):
            feats, cfg = list(M0), {}
            p = score(D, feats, cfg, family); a = aucs(D, p)
            save_preds(D, p, family, eid)
            st = dict(exp=eid, feats=feats, cfg=cfg, auc=a, m0=a)
            json.dump(st, open(cf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            append_row(dict(exp_id=eid, family=family, parent="", change=e["change"], **{f"auc_{v}": round(a[v], 4) for v in DEV},
                            mean_auc=round(np.mean(list(a.values())), 4), skill_vs_M0=0.0, d_skill_vs_champ="", win="", decision="BASE", n_feats=len(feats), sec=round(time.time() - t)))
            log(family, eid, "M0", {v: round(x, 4) for v, x in a.items()}); continue
        st = json.load(open(cf, encoding="utf-8")); st["auc"] = {int(k): v for k, v in st["auc"].items()}; st["m0"] = {int(k): v for k, v in st["m0"].items()}
        feats, cfg = list(st["feats"]), dict(st["cfg"])
        if "add" in e:
            feats = feats + [x for x in e["add"] if x not in feats]
        if "swap" in e:
            feats = [e["swap"].get(x, x) for x in feats]
        if "set" in e:
            cfg.update(e["set"])
        p = score(D, feats, cfg, family); a = aucs(D, p)
        save_preds(D, p, family, eid)
        sk = np.mean([skill(a[v], st["auc"][v]) for v in DEV]); win = np.mean([a[v] > st["auc"][v] for v in DEV])
        sk0 = np.mean([skill(a[v], st["m0"][v]) for v in DEV])
        dec = "ACCEPT" if (sk >= 0.003 and win >= 0.6) else "ROLLBACK"
        append_row(dict(exp_id=eid, family=family, parent=st["exp"], change=e["change"], **{f"auc_{v}": round(a[v], 4) for v in DEV},
                        mean_auc=round(np.mean(list(a.values())), 4), skill_vs_M0=round(sk0, 4), d_skill_vs_champ=round(sk, 4), win=round(win, 3),
                        decision=dec, n_feats=len(feats), sec=round(time.time() - t)))
        log(family, eid, e["change"], "Δskill", round(sk, 4), "win", round(win, 3), dec, "skill_vs_M0", round(sk0, 4))
        if dec == "ACCEPT":
            json.dump(dict(exp=eid, feats=feats, cfg=cfg, auc=a, m0=st["m0"]), open(cf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
