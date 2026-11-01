# -*- coding: utf-8 -*-
"""T4 묶음 채점. r2lib.combine_xgb·ensemble·final_run_r2.score_final의 계산을 그대로 옮기되 배깅 구성원 번호만 묶음의 k(3b..3b+2)로 바꾼다.
로지스틱 구성요소·기준선은 결정적이므로 묶음 0에서 다시 학습한 예측을 모든 묶음이 쓴다. HGB는 묶음마다(random_state=b).
M0 대비 CI: final_run_r2.cluster_boot(출원인 군집, B=1000, seed 0, CI 4자리 반올림 저장)와 같은 코드(아래 cboot, r2boot 복사본 사용).
강화 기준선 CI: VERIFY_R5 v3_analysis.boot(B=300, seed 3, 가중 roc_auc_score)와 같은 코드, 강화 예측은 VERIFY_R5/cache/strong_full.npy(결정적).
사용: python t4_score.py 0,1 → cache/t4/b{b}.json"""
import os, sys, json
sys.dont_write_bytecode = True
os.environ["T4R2_MODE"] = "final"; os.environ.pop("T4R2_SAVE", None)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from c13util import *
sys.path.insert(0, os.path.join(HERE, "t4mirror", "R2"))
import numpy as np, pandas as pd
import r2lib as R
from r2boot import wauc_prep, wauc, wap
from sklearn.metrics import roc_auc_score

FS = json.load(open(os.path.join(HERE, "t4mirror", "R2", "final_spec_r2.json"), encoding="utf-8")); SPEC = FS["spec"]
TD = os.path.join(CACHE, "t4")
ev = pd.read_parquet(os.path.join(TD, "frames", "ev.parquet"), columns=["pid", "L", "ksic", "acd", "atype", "y"])
row_h = pd.read_parquet(os.path.join(TD, "frames", "ev_haz.parquet"), columns=["row"]).row.values
N = len(ev)
m = ev.y.notna().values; assert m.all()
y = ev.y.values.astype(float)
SB = np.load(os.path.join(MC, "T4_특허포기예측", "VERIFY_R5", "cache", "strong_full.npy"))
FROZ = os.path.join(MC, "T4_특허포기예측", "R2", "runs_final", "F22", "pred")
key = {c: R.comp_key(SPEC, c) for c in ("cls", "haz", "logit", "hgb")}
bkey = [f for f in os.listdir(os.path.join(TD, "b0", "pred")) if f.startswith("base_")][0][:-4]


def comb(b, comp):
    ks = [3 * b, 3 * b + 1, 3 * b + 2]
    z = np.mean([np.load(os.path.join(TD, f"b{b}", "pred", f"{key[comp]}_b{k}.npy")) for k in ks], 0)
    p = 1 / (1 + np.exp(-z))
    if comp == "haz":
        p = R.HZ.combine(pd.DataFrame({"row": row_h}), p, N)
    return p          # cls는 float32 그대로(공식 score_final은 결합 파일이 없을 때 계산값을 바로 써서 float32), haz는 combine이 float64


def cboot(y, p, pb, acd, pid, B, seed=0):
    cl = np.where(acd == "NA", "P" + pid.astype(str), acd)
    codes, uniq = pd.factorize(cl); K = len(uniq)
    pm, pbb = wauc_prep(y, p), wauc_prep(y, pb)
    rng = np.random.default_rng(seed)
    S = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        a, ab = wauc(pm, w), wauc(pbb, w)
        S.append((a - ab) / (1 - ab))
    return [float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))], K


def sboot(pm, pb_, B=300, seed=3):
    cl = np.where(ev.acd.values == "NA", "P" + ev.pid.astype(str).values, ev.acd.values)
    codes, uniq = pd.factorize(cl); K = len(uniq); rng = np.random.default_rng(seed); out = []
    yy = ev.y.values.astype(int)
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        a = roc_auc_score(yy, pm, sample_weight=w); bb = roc_auc_score(yy, pb_, sample_weight=w); out.append((a - bb) / (1 - bb))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for b in [int(x) for x in sys.argv[1].split(",")]:
    fo = os.path.join(TD, f"b{b}.json")
    if os.path.exists(fo):
        tlog("skip", b); continue
    ps = {"cls": comb(b, "cls"), "haz": comb(b, "haz"),
          "logit": np.load(os.path.join(TD, "b0", "pred", f"{key['logit']}.npy")),
          "hgb": np.load(os.path.join(TD, f"b{b}", "pred", f"{key['hgb']}.npy"))}
    pb = np.load(os.path.join(TD, "b0", "pred", f"{bkey}.npy"))
    ws = SPEC["ensemble"]["weights"]
    z = sum(w * lg(ps[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
    p = 1 / (1 + np.exp(-z))                       # 공식 final_result 경로(cls float32)
    ps64 = dict(ps); ps64["cls"] = ps["cls"].astype("float64")
    z64 = sum(w * lg(ps64[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
    p64 = 1 / (1 + np.exp(-z64))                   # VERIFY_R5 강화 기준선 경로(저장 npy float64)
    a, ab, asb = float(roc_auc_score(y, p)), float(roc_auc_score(y, pb)), float(roc_auc_score(y, SB))
    a64 = float(roc_auc_score(y, p64))
    ci, K = cboot(y, p, pb, ev.acd.values, ev.pid.values, 1000, seed=0)
    tlog("boot M0", b)
    cs = sboot(p64, SB)
    t1 = top1(y, p)
    top = np.argsort(-p)[:int(0.1 * len(p))]
    extra = dict(strong_name="13변수 로지스틱(VERIFY_R5)", n_clusters=K, precision_top10_official_rule=float(y[top].mean()),
                 comps_auc={k: float(roc_auc_score(y, v)) for k, v in ps.items()}, xgb_bags=[3 * b, 3 * b + 1, 3 * b + 2], hgb_random_state=b,
                 hit100=int(y[np.argsort(-p, kind="stable")[:100]].sum()))
    if b == 0:
        fr = {"cls": np.load(os.path.join(FROZ, f"{key['cls']}.npy")), "haz": np.load(os.path.join(FROZ, f"{key['haz']}.npy")),
              "logit": np.load(os.path.join(FROZ, f"{key['logit']}.npy")), "hgb": np.load(os.path.join(FROZ, f"{key['hgb']}.npy"))}
        extra["max_pred_diff_vs_frozen"] = {k: float(np.abs(ps[k] - fr[k]).max()) for k in fr}
        extra["max_pred_diff_vs_frozen"]["base"] = float(np.abs(pb - np.load(os.path.join(FROZ, f"{bkey}.npy"))).max())
        bag_diff = {}
        for comp in ("cls", "haz"):
            for k in range(3):
                bag_diff[f"{comp}_b{k}"] = float(np.abs(np.load(os.path.join(TD, "b0", "pred", f"{key[comp]}_b{k}.npy")) - np.load(os.path.join(FROZ, f"{key[comp]}_b{k}.npy"))).max())
        extra["max_bag_logit_diff_vs_frozen"] = bag_diff
    extra["auc_champ_float64path"] = a64
    r = row("T4", b, dict(xgb_bags=[3 * b, 3 * b + 1, 3 * b + 2], hgb_random_state=b), a, ab, (a - ab) / (1 - ab), ci, asb, (a64 - asb) / (1 - asb), cs, t1, len(y), y.sum(), extra=extra)
    pd.DataFrame({"acd": ev.acd.values, "pid": ev.pid.values, "y": y, "p_champ": p, "p_M0": pb, "p_strong": SB}).to_parquet(os.path.join(TD, f"preds_b{b}.parquet"), index=False)
    dump(r, fo)
    tlog("done", b, round(r["skill"], 4), [round(x, 4) for x in ci], "strong", round(r["skill_vs_strong"], 4), [round(x, 4) for x in cs])
