# -*- coding: utf-8 -*-
"""N9 챔피언/도전자 실험기. 사용: python n9_exp.py R01 R02 ...  |  python n9_exp.py C01  |  python n9_exp.py report
실험 정의는 PLAN.md와 같다(실행 전에 고정). 결과: experiments.csv(행 단위 덮어쓰기), cache/preds_{id}.parquet, cache/champ_state.json"""
import sys, json
import numpy as np, pandas as pd
from n9_lib import *

D = pd.read_parquet(f"{CACHE}/panel_dev.parquet").reset_index(drop=True)
assert (D.y >= 0).all() and D.t0.max() <= 2020
_rng = np.random.default_rng(0)
KEEP = (D.y.values == 1) | (_rng.random(len(D)) < NEG_FRAC)   # 파일럿 run_steps와 같은 마스크
W = np.where(D.y.values == 1, 1.0, 1.0 / NEG_FRAC)
STATE = f"{CACHE}/champ_state.json"
EXPCSV = f"{OUT}/experiments.csv"
M1_ = list(M1)
CHAMP0 = M0_L2 + M1_ + M2_L2

REF = {  # 재현 단계
    "R01": ("파일럿 M0(원판)", M0_ORIG, {}),
    "R02": ("M0+M1(원판)", M0_ORIG + M1_, {}),
    "R03": ("M0+M1+M2(원판) = 파일럿 챔피언", M0_ORIG + M1_ + M2_ORIG, {}),
    "R04": ("M0_L2 지연판 기준선", M0_L2, {}),
    "R05": ("M0_L2+M1+M2_L2 = 6R 출발 챔피언(지연판)", CHAMP0, {}),
    "R06": ("강한 기준선 지연판 M0_L2+M2_L2", M0_L2 + M2_L2, {}),
    "R07": ("강한 기준선 원판 M0+M2", M0_ORIG + M2_ORIG, {}),
    "R08": ("로지스틱 M0_L2", M0_L2, {"model": "logit"}),
    "R09": ("로지스틱 챔피언 지연판", CHAMP0, {"model": "logit"}),
}
CH = {  # 도전자(순서 고정): (설명, 추가 열, cfg 변경)
    "C01": ("텍스트 대응 점수 원점 내 순위", ["b1_rank", "b2_rank"], {}),
    "C02": ("부처×수행주체 교호", ["minper_c"], {}),
    "C03": ("선행과제 사슬", ["chain_len", "chain_p50_L2", "chain_kr_L2"], {}),
    "C04": ("텍스트 대응 KSIC 정체성·과거 양성률", ["b1_top", "top_ksic", "ksic_prior_b1", "ksic_prior_top"], {}),
    "C05": ("기관-KSIC 적합도", ["org_ksic_fit_L2"], {}),
    "C06": ("과제 텍스트↔제공 KR 특허 제목 KSIC 중심점", ["pt_max", "pt_top", "pt_margin", "pt_ent"], {}),
    "C07": ("제공 KR 출원 이력(기관명 매칭)", ["org_krapp", "org_krapp3", "org_krapp_topk"], {}),
    "C08": ("산업 추세(제공 KR 출원 성장)", ["ind_growth"], {}),
    "C09": ("KOSIS 출하·종사자 성장", ["kosis_ship_g", "kosis_emp_g"], {}),
    "C10": ("과제 텍스트 분류기 스태킹", ["txt_clf"], {}),
    "C11": ("학습 음성 표본추출 제거", [], {"neg_frac": 1.0}),
    "C12": ("시드 배깅 3", [], {"seeds": [0, 1, 2]}),
    "C13": ("최근 학습창 t >= v-8", [], {"window": 8}),
}


def score(feats, cfg):
    cfg = dict(cfg); model = cfg.get("model", "xgb"); seeds = cfg.get("seeds", [0])
    full = cfg.get("neg_frac", NEG_FRAC) >= 1.0
    preds = {}
    for v in DEV_ORIGINS:
        X = D
        if "txt_clf" in feats:
            tc = pd.read_parquet(f"{CACHE}/txtclf_dev_{v}.parquet")
            X = D.copy(); X["txt_clf"] = np.nan; X.loc[tc.row_id.values, "txt_clf"] = tc.txt_clf.values
        msk = (X.t0 <= v - GAP).values
        if cfg.get("window"):
            msk &= (X.t0 >= v - cfg["window"]).values
        if not full:
            msk &= KEEP
        w = None if full else W[msk]
        tr = X[msk]; te = X[X.t0 == v]
        p = np.mean([fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), model=model, seed=s, w=w) for s in seeds], axis=0)
        preds[v] = p
    return preds


def save_preds(eid, preds):
    pd.concat([pd.DataFrame({"v": v, "row_id": D.index[D.t0 == v], "p": p}) for v, p in preds.items()]).to_parquet(f"{CACHE}/preds_{eid}.parquet", index=False)


def load_preds(eid):
    q = pd.read_parquet(f"{CACHE}/preds_{eid}.parquet")
    return {v: g.p.values for v, g in q.groupby("v")}


def summarize(eid, preds):
    r = {}
    for v in DEV_ORIGINS:
        y = D.y.values[D.t0.values == v]; p = preds[v]
        r[v] = dict(auc=auc(y, p), top1=topk(y, p, 0.01)[0], top5=topk(y, p, 0.05)[0], hit100=topk(y, p, k=100)[1])
    return r


def write_row(row):
    df = pd.read_csv(EXPCSV, encoding="utf-8-sig") if os.path.exists(EXPCSV) else pd.DataFrame()
    if len(df):
        df = df[df.exp_id != row["exp_id"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EXPCSV, index=False, encoding="utf-8-sig")


def base_row(eid, parent, change, s, ref=None):
    row = dict(exp_id=eid, parent=parent, change=change)
    for v in DEV_ORIGINS:
        row[f"auc_{v}"] = round(s[v]["auc"], 4)
    row["mean_auc"] = round(np.mean([s[v]["auc"] for v in DEV_ORIGINS]), 4)
    if ref is not None:
        sk = [skill(s[v]["auc"], ref[v]["auc"]) for v in DEV_ORIGINS]
        row["skill_vs_M0L2"] = round(np.mean(sk), 4); row["skill_vs_M0L2_by_origin"] = " ".join(f"{x:.3f}" for x in sk)
    row["top1_mean"] = round(np.mean([s[v]["top1"] for v in DEV_ORIGINS]), 4)
    row["top5_mean"] = round(np.mean([s[v]["top5"] for v in DEV_ORIGINS]), 4)
    row["hit100_mean"] = round(np.mean([s[v]["hit100"] for v in DEV_ORIGINS]), 1)
    return row


def run_ref(eid):
    desc, feats, cfg = REF[eid]
    p = score(feats, cfg); save_preds(eid, p); s = summarize(eid, p)
    ref = summarize("R04", load_preds("R04")) if os.path.exists(f"{CACHE}/preds_R04.parquet") else None
    row = base_row(eid, "-", desc, s, ref); row["decision"] = "REF"; row["note"] = json.dumps(cfg, ensure_ascii=False) if cfg else ""
    write_row(row); log(eid, row["mean_auc"], row.get("skill_vs_M0L2"))
    if eid == "R05":
        json.dump(dict(tag="R05", feats=CHAMP0, cfg={}), open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def run_ch(eid):
    st = json.load(open(STATE, encoding="utf-8"))
    desc, add, cfgc = CH[eid]
    feats = st["feats"] + [a for a in add if a not in st["feats"]]; cfg = dict(st["cfg"]); cfg.update(cfgc)
    p = score(feats, cfg); save_preds(eid, p); s = summarize(eid, p)
    sc = summarize(st["tag"], load_preds(st["tag"])); ref = summarize("R04", load_preds("R04"))
    d = [skill(s[v]["auc"], sc[v]["auc"]) for v in DEV_ORIGINS]; win = np.mean([s[v]["auc"] > sc[v]["auc"] for v in DEV_ORIGINS])
    dec = "ACCEPT" if (np.mean(d) >= 0.003 and win >= 0.6) else "ROLLBACK"
    row = base_row(eid, st["tag"], desc + (" +" + ",".join(add) if add else "") + (" " + json.dumps(cfgc) if cfgc else ""), s, ref)
    row.update(d_skill_vs_champ=round(float(np.mean(d)), 4), d_by_origin=" ".join(f"{x:.3f}" for x in d), win=round(float(win), 3), decision=dec)
    write_row(row); log(eid, dec, row["d_skill_vs_champ"], win, row["mean_auc"])
    if dec == "ACCEPT":
        json.dump(dict(tag=eid, feats=feats, cfg=cfg), open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    for eid in sys.argv[1:]:
        (run_ref if eid in REF else run_ch)(eid)
