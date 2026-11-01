# -*- coding: utf-8 -*-
"""챔피언/도전자 실험 실행기. 사용: python n5_eval.py E00 E01 ...  (PLAN.md 순서)
상태(state.json): 현재 챔피언 스펙·원점별 AUC. 예측은 cache/preds_<exp>.parquet. 기록은 experiments.csv에 추가."""
import os, sys, json, time
import numpy as np, pandas as pd
from n5lib import *

DEV = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
STATE = os.path.join(OUT, "state.json")
EXPCSV = os.path.join(OUT, "experiments.csv")
D = pd.read_parquet(os.path.join(CACHE, "panel_dev_26429.parquet"))
_D22 = None


def D22():
    global _D22
    if _D22 is None:
        _D22 = pd.read_parquet(os.path.join(CACHE, "panel_dev_26422.parquet"))
    return _D22


def feats_of(groups):
    return [f for g in groups for f in FEATS[g]]


def score(spec, seeds=(0, 1, 2), model="xgb"):
    feats = feats_of(spec["groups"]) + spec.get("extra", [])
    feats = [f for f in feats if f not in spec.get("drop", [])]
    res, P = {}, []
    for v in DEV:
        tr = D[D.t0 <= v - 2]
        if spec.get("expand"):
            tr = pd.concat([tr, D22()[D22().t0 <= v - 2]], ignore_index=True)
        te = D[D.t0 == v]
        w = None
        if spec.get("decay"):
            w = (spec["decay"] ** (v - 2 - tr.t0)).values
        p = fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), model=model, seeds=seeds, w=w)
        res[v] = auc(te.y, p)
        P.append(pd.DataFrame({"t0": v, "appl_no": te.appl_no.values, "app": te.app.values, "y": te.y.values, "p": p}))
    return res, pd.concat(P, ignore_index=True), feats


def load_state():
    return json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}


def save_state(s):
    json.dump(s, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


COLS = ["exp_id", "parent", "change"] + [f"auc_{v}" for v in DEV] + ["mean_skill_vs_M0", "d_skill_vs_champ", "win_rate", "wins", "decision", "note"]


def record(row):
    df = pd.DataFrame([row]).reindex(columns=COLS)
    df.to_csv(EXPCSV, mode="a", header=not os.path.exists(EXPCSV), index=False, encoding="utf-8-sig")


BASE_SPEC = {"groups": ["M0"]}
M1_SPEC = {"groups": ["M0", "M1"]}
PLAN = {
    "E03": ("+ros (RoS family US 과학 인용)", {"add_groups": ["ros"]}),
    "E04": ("+ntis (정부R&D 성과·수행주체)", {"add_groups": ["ntis"]}),
    "E05": ("+etsig (ETSI 글로벌 기업 선언 동향)", {"add_groups": ["etsig"]}),
    "E06": ("+oa (OpenAlex 메인그룹 논문)", {"add_groups": ["oa"]}),
    "E07": ("최근 원점 가중 0.7^(v-2-t0)", {"decay": 0.7}),
    "E08": ("+regime (출원인 선언 시기 관행)", {"add_groups": ["regime"]}),
    "E09": ("+title (최근 선언 제목 유사도)", {"add_groups": ["title"]}),
    "E10": ("+famstrat (출원인 해외 패밀리 비율)", {"add_groups": ["famstrat"]}),
    "E11": ("+pct (PCT 국내단계·외국 우선권)", {"add_groups": ["pct"]}),
    "E12": ("+sg (IPC 서브그룹 선언률)", {"add_groups": ["sg"]}),
    "E13": ("+appmg (출원인×메인그룹 선언률)", {"add_groups": ["appmg"]}),
    "E14": ("학습에 KSIC 26422 단위 추가", {"expand": True}),
    "E15": ("[E03 변형] +ros_sci만", {"add_extra": ["ros_sci"]}),
    "E16": ("[E05 변형] +etsi_co_rec2만", {"add_extra": ["etsi_co_rec2"]}),
    "E17": ("[E07 변형] 가중 0.85^(v-2-t0)", {"decay": 0.85}),
    "E18": ("[E08 변형] +app_rec_rate만", {"add_extra": ["app_rec_rate"]}),
    "E19": ("[E09 변형] +title_sim_top10만", {"add_extra": ["title_sim_top10"]}),
    "E20": ("[재시험] +ntis", {"add_groups": ["ntis"]}),
    "E21": ("[재시험] +pct", {"add_groups": ["pct"]}),
    "E22": ("+fam_declared만 (해외 패밀리 ETSI 선언 전이)", {"add_extra": ["fam_declared"]}),
}


def row_of(exp, parent, change, r, ref=None, decision="", note="", base=None):
    row = dict(exp_id=exp, parent=parent, change=change)
    for v in DEV:
        row[f"auc_{v}"] = round(r[v], 5)
    if base is not None:
        row["mean_skill_vs_M0"] = round(np.mean([skill(r[v], base[v]) for v in DEV]), 4)
    if ref is not None:
        d = [skill(r[v], ref[v]) for v in DEV]
        row["d_skill_vs_champ"] = round(np.mean(d), 4)
        row["win_rate"] = round(np.mean([r[v] > ref[v] for v in DEV]), 3)
        row["wins"] = f"{sum(r[v] > ref[v] for v in DEV)}/{len(DEV)}"
    row["decision"] = decision
    row["note"] = note
    return row


def run(exp):
    t = time.time()
    S = load_state()
    if exp == "E00":
        r, P, f = score(BASE_SPEC, seeds=(0,))
        record(row_of("E00", "-", "M0 재현(시드0): 출원 경과·등록·청구항·출원인/메인그룹 선언률·규모", r, decision="REPRO", note="파일럿 experiments_I5.csv M0와 대조"))
        S["E00"] = r
    elif exp == "E01":
        r, P, f = score(M1_SPEC, seeds=(0,))
        base = {int(k): v for k, v in S["E00"].items()}
        record(row_of("E01", "E00", "M1 재현(시드0): +fam_US,fam_EP,fam_JP,fam_n", r, ref=base, base=base, decision="REPRO", note="파일럿 skill 0.0936(7/7)과 대조"))
    elif exp == "E02":
        r0, P0, f0 = score(BASE_SPEC)
        r1, P1, f1 = score(M1_SPEC)
        P0.to_parquet(os.path.join(CACHE, "preds_M0.parquet"), index=False)
        P1.to_parquet(os.path.join(CACHE, "preds_E02.parquet"), index=False)
        record(row_of("E02-M0", "-", "M0 시드3 평균 = 공식 기준선", r0, decision="BASE"))
        record(row_of("E02", "E02-M0", "M1 시드3 평균 = 출발 챔피언", r1, ref=r0, base=r0, decision="CHAMP_START"))
        S.update(base=r0, champ=r1, champ_id="E02", champ_spec=dict(M1_SPEC), history=["E02"])
    else:
        change, mod = PLAN[exp]
        base = {int(k): v for k, v in S["base"].items()}
        ref = {int(k): v for k, v in S["champ"].items()}
        spec = json.loads(json.dumps(S["champ_spec"]))
        if "add_groups" in mod:
            spec["groups"] = spec["groups"] + mod["add_groups"]
        if "add_extra" in mod:
            spec["extra"] = spec.get("extra", []) + mod["add_extra"]
        if "decay" in mod:
            spec["decay"] = mod["decay"]
        if "expand" in mod:
            spec["expand"] = True
        r, P, f = score(spec)
        P.to_parquet(os.path.join(CACHE, f"preds_{exp}.parquet"), index=False)
        sk = np.mean([skill(r[v], ref[v]) for v in DEV]); win = np.mean([r[v] > ref[v] for v in DEV])
        dec = "ACCEPT" if (sk >= 0.003 and win >= 0.6) else "ROLLBACK"
        record(row_of(exp, S["champ_id"], change, r, ref=ref, base=base, decision=dec, note=f"n_feats={len(f)}"))
        if dec == "ACCEPT":
            S.update(champ=r, champ_id=exp, champ_spec=spec)
            S["history"].append(exp)
        S.setdefault("results", {})[exp] = dict(sk=round(sk, 4), win=round(win, 3), dec=dec)
    save_state(S)
    print(exp, "done", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    for e in sys.argv[1:]:
        run(e)
    if os.path.exists(EXPCSV):
        X = pd.read_csv(EXPCSV)
        print(X[["exp_id", "parent", "mean_skill_vs_M0", "d_skill_vs_champ", "wins", "decision"]].to_string())
