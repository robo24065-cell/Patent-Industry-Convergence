# -*- coding: utf-8 -*-
"""챔피언/도전자 실험 실행기(dev 원점만). 사용: python n7_03_exp.py E00 E01 ... C01 ...
상태: champ.json(현재 챔피언 피처·모형·원점별 AUC, M0·SB 기준 AUC), 기록: experiments.csv(추가), 예측: cache/preds_<exp>.parquet"""
import os, sys, json
import pandas as pd, numpy as np
from n7lib import *
from n7feat import BUNDLES

D = pd.read_parquet(os.path.join(CACHE, "dev_panel.parquet"))
X = pd.read_parquet(os.path.join(CACHE, "feat_dev.parquet"))
D = pd.concat([D, X], axis=1)
assert D.t0.max() + 2 < F_YEAR and D.y.notna().all()
V = DEV_ORIGINS
SEEDS3 = (0, 1, 2)
CH = os.path.join(OUT, "champ.json")
EXPF = os.path.join(OUT, "experiments.csv")
PILOT = M0 + M1_ADD + M2_ADD + M3_ADD
LEAK = {"C07"}   # n7_04_rl_leak_check.py: 권리목록 보유율 y1−y0 차이 2.6–12.6%p, 조회실패 y0/y1 1.7–2.7배 → PLAN §3 규칙상 누수

PLAN = {  # exp: (종류, 설명, 추가 피처, 모형, 시드)
    "E00": ("ref", "규모 1변수(log_emp) 참고선(시드0)", ["log_emp"], "xgb", (0,)),
    "E01": ("ref", "M0 기준선: NPS 속성+사업자번호 유형(파일럿 재현, 시드0)", M0, "xgb", (0,)),
    "E02": ("ref", "M1 = M0+특허 행태 12개(파일럿 재현, 시드0)", M0 + M1_ADD, "xgb", (0,)),
    "E03": ("ref", "M2 = M1+NTIS 참여 신호 8개(파일럿 재현, 시드0)", M0 + M1_ADD + M2_ADD, "xgb", (0,)),
    "E04": ("ref", "M3 = M2+KSIC 정부연구비 2개 = 파일럿 챔피언(재현, 시드0)", PILOT, "xgb", (0,)),
    "SB0": ("ref", "강한 기준선 M0+특허 규모 3개(log_app, log_live, n_ksic)(시드0)", STRONG, "xgb", (0,)),
    "E05": ("rescore", "채점 규칙 변경: XGB 3시드 평균으로 M0·SB·파일럿 챔피언 재채점(기준 행) → 출발 챔피언", PILOT, "xgb", SEEDS3),
}
for b in ["C01_pay_delay", "C02_disposal", "C03_portfolio_spend", "C04_coapplicant", "C05_ksic_firm_rnd", "C06_peer_entry_ksic",
          "C07_design_utility", "C08_foreign_family", "C09_peer_entry_region", "C10_nps_ext", "C11_ksic_mix"]:
    PLAN[b[:3]] = ("chal", b, BUNDLES[b], None, SEEDS3)
PLAN["V04"] = ("chal", "C04 축소판(co_pub_any, co_ntis_any)", ["co_pub_any", "co_ntis_any"], None, SEEDS3)
PLAN["V05"] = ("chal", "V05 C05의 KSIC 연결을 텍스트 대응(b2_dict_soft)으로", BUNDLES["V05_ksic_firm_rnd_text"], None, SEEDS3)
PLAN["V06"] = ("chal", "C06 축소판(peer_entry_k)", ["peer_entry_k"], None, SEEDS3)
PLAN["C12"] = ("chal_model", "모형 변경: XGB(3시드)+로지스틱 순위 평균", [], "ens", SEEDS3)
PLAN["C13"] = ("chal_decay", "최근 원점 가중 학습 0.8^(v-2-t0)", [], None, SEEDS3)
# 2차 선언(PLAN §6)
PLAN["D01"] = ("chal", "D01_inventor_size", BUNDLES["D01_inventor_size"], None, SEEDS3)
PLAN["D02"] = ("chal", "D02_inventor_gov_exp", BUNDLES["D02_inventor_gov_exp"], None, SEEDS3)
PLAN["R01"] = ("chal", "재시험 C09_peer_entry_region(이미 본 결과로 고른 재시험)", BUNDLES["C09_peer_entry_region"], None, SEEDS3)
PLAN["R02"] = ("chal", "재시험 C02_disposal(이미 본 결과로 고른 재시험)", BUNDLES["C02_disposal"], None, SEEDS3)
PLAN["R03"] = ("chal", "재시험 C01_pay_delay(이미 본 결과로 고른 재시험)", BUNDLES["C01_pay_delay"], None, SEEDS3)
PLAN["R04"] = ("chal", "재시험 C05_ksic_firm_rnd(이미 본 결과로 고른 재시험)", BUNDLES["C05_ksic_firm_rnd"], None, SEEDS3)


def score(feats, model, seeds, decay=None, ycol="y"):
    res, preds = {}, {}
    for v in V:
        tr = D[(D.t0 >= TRAIN_MIN_T0) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        sw = (decay ** (v - 2 - tr.t0)).values if decay else None
        p = predict(tr[feats].astype(float), tr[ycol], te[feats].astype(float), model=model, seeds=seeds, sw=sw)
        preds[v] = p; res[v] = auc(te[ycol], p)
    return res, preds


def topstats(preds):
    o = {}
    for fr, nm in [(0.01, "top1"), (0.05, "top5"), (0.10, "top10")]:
        o[nm] = np.mean([topk(D[D.t0 == v].y.values, preds[v], fr)[0] for v in V])
    o["rec10"] = np.mean([topk(D[D.t0 == v].y.values, preds[v], 0.10)[1] for v in V])
    return o


def save_preds(tag, preds):
    rows = []
    for v, p in preds.items():
        te = D[D.t0 == v]
        rows.append(pd.DataFrame({"applicantCd": te.applicantCd.values, "bizno": te.bizno.values, "t0": v, "y": te.y.values, "p": p}))
    pd.concat(rows).to_parquet(os.path.join(CACHE, f"preds_{tag}.parquet"), index=False)


def record(row):
    df = pd.DataFrame([row])
    if os.path.exists(EXPF):
        old = pd.read_csv(EXPF, encoding="utf-8-sig")
        old = old[old.exp_id != row["exp_id"]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(EXPF, index=False, encoding="utf-8-sig")


def msk(r, b):
    return float(np.mean([skill(r[v], b[v]) for v in V]))


def run(exp):
    kind, desc, add, model, seeds = PLAN[exp]
    st = json.load(open(CH, encoding="utf-8")) if os.path.exists(CH) else {}
    row = dict(exp_id=exp, change=desc)
    if kind == "ref":
        r, p = score(add, model, seeds)
        save_preds(exp, p)
        row.update(parent="-", model=f"{model}(seed{list(seeds)})", n_feats=len(add), **{f"auc_{v}": round(r[v], 5) for v in V})
        if exp == "E01":
            st["m0_s0"] = r
        if exp != "E01" and "m0_s0" in st:
            m0 = {int(k): v for k, v in st["m0_s0"].items()}
            row["mean_skill_vs_M0"] = round(msk(r, m0), 4); row["win_vs_M0"] = np.mean([r[v] > m0[v] for v in V])
        row.update({k: round(v, 4) for k, v in topstats(p).items()}); row["mean_auc"] = round(np.mean(list(r.values())), 4)
        row["note"] = "파일럿 experiments_I7.csv와 대조"; row["decision"] = "REF" if exp != "E04" else "REF(파일럿 챔피언)"
    elif kind == "rescore":
        r0, p0 = score(M0, "xgb", seeds); save_preds("E05_M0", p0)
        rs, ps = score(STRONG, "xgb", seeds); save_preds("E05_SB", ps)
        r, p = score(add, "xgb", seeds); save_preds("E05", p)
        st.update(m0=r0, sb=rs, champ=dict(exp="E05", feats=add, model="xgb", decay=None, auc=r))
        row.update(parent="E04", model=f"xgb(seed{list(seeds)})", n_feats=len(add), **{f"auc_{v}": round(r[v], 5) for v in V},
                   mean_skill_vs_M0=round(msk(r, r0), 4), win_vs_M0=np.mean([r[v] > r0[v] for v in V]),
                   mean_skill_vs_SB=round(msk(r, rs), 4), win_vs_SB=np.mean([r[v] > rs[v] for v in V]),
                   note="M0 3시드 AUC " + "/".join(f"{r0[v]:.4f}" for v in V) + "; SB 3시드 AUC " + "/".join(f"{rs[v]:.4f}" for v in V)
                   + "; M0 상위1/5/10% " + "/".join(f"{x:.3f}" for x in list(topstats(p0).values())[:3])
                   + "; SB 상위1/5/10% " + "/".join(f"{x:.3f}" for x in list(topstats(ps).values())[:3]),
                   decision="BASE(규칙 변경)")
        row.update({k: round(v, 4) for k, v in topstats(p).items()}); row["mean_auc"] = round(np.mean(list(r.values())), 4)
    else:
        ch = st["champ"]; cf = list(ch["feats"]); cm = ch["model"]; cd = ch.get("decay")
        feats, mdl, dec = cf, cm, cd
        if kind == "chal":
            feats = cf + [f for f in add if f not in cf]
        elif kind == "chal_model":
            mdl = model
        elif kind == "chal_decay":
            dec = 0.8
        r, p = score(feats, mdl, seeds, decay=dec)
        save_preds(exp, p)
        champ_auc = {int(k): v for k, v in ch["auc"].items()}
        m0 = {int(k): v for k, v in st["m0"].items()}; sb = {int(k): v for k, v in st["sb"].items()}
        dsk = msk(r, champ_auc); win = float(np.mean([r[v] > champ_auc[v] for v in V]))
        ok = dsk >= 0.003 and win >= 0.6
        dec_s = "ACCEPT" if ok else "ROLLBACK"
        note = ""
        if exp in LEAK:
            note = "시점 정합 실패(누수): 권리목록 현재 권리자 스냅샷(rl_leak_check.csv) — 수치와 무관하게 ROLLBACK"
            if ok:
                dec_s = "ROLLBACK(누수)"
            else:
                dec_s = "ROLLBACK(누수·규칙 미달)"
            ok = False
        row.update(parent=ch["exp"], model=f"{mdl}(seed{list(seeds)})" + (f",decay{dec}" if dec else ""), n_feats=len(feats),
                   **{f"auc_{v}": round(r[v], 5) for v in V}, mean_auc=round(np.mean(list(r.values())), 4),
                   mean_skill_vs_M0=round(msk(r, m0), 4), win_vs_M0=np.mean([r[v] > m0[v] for v in V]),
                   mean_skill_vs_SB=round(msk(r, sb), 4), win_vs_SB=np.mean([r[v] > sb[v] for v in V]),
                   mean_skill_vs_champ=round(dsk, 4), win_vs_champ=win, decision=dec_s, note=note)
        row.update({k: round(v, 4) for k, v in topstats(p).items()})
        if ok:
            st["champ"] = dict(exp=exp, feats=feats, model=mdl, decay=dec, auc=r)
    json.dump(st, open(CH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    record(row)
    print({k: row.get(k) for k in ["exp_id", "mean_auc", "mean_skill_vs_M0", "mean_skill_vs_SB", "mean_skill_vs_champ", "win_vs_champ", "decision", "top1", "top5", "top10"]},
          "AUC", [row.get(f"auc_{v}") for v in V], flush=True)


if __name__ == "__main__":
    for e in sys.argv[1:]:
        run(e)
        log("done", e)
