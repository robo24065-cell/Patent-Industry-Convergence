# -*- coding: utf-8 -*-
"""사전 등록 순차 실험(이어달리기 가능). 사용: python r6_eval.py K1 xgb   — 종료코드 3이면 같은 명령을 다시 실행(예측은 캐시됨).
원점 v 평가, 학습 min_train ≤ t0 ≤ v−2(결과창 2년이 v 이전에 끝남). 평가는 원점 전체, 학습만 음성 표본추출(가중치 1/비율)."""
import sys, os, json, time, hashlib, numpy as np, pandas as pd
from r6_common import CACHE, OUT, fit_pred, auc, skill, topk, boot_ci_skill
from r6_eval_spec import SPEC

T_START = time.time(); BUDGET = 470
task, model = sys.argv[1], sys.argv[2]
S = SPEC[task]
D = pd.read_parquet(os.path.join(CACHE, S["panel"]))
D = D[D.t0 >= S["min_train"]].reset_index(drop=True)
y = D[S["y"]].values
rng = np.random.default_rng(0)
keep = (y == 1) | (rng.random(len(D)) < S["neg_frac"])
wfull = np.where(y == 1, 1.0, 1.0 / S["neg_frac"])
PD = os.path.join(CACHE, f"preds_{task}"); os.makedirs(PD, exist_ok=True)
dev = S["dev"]


def tcheck():
    if time.time() - T_START > BUDGET:
        print("TIME BUDGET — 다시 실행하세요", flush=True); sys.exit(3)


def score(feats):
    res, P = {}, {}
    for v in dev:
        key = hashlib.md5(f"{model}|{','.join(sorted(feats))}|{v}|{S['neg_frac']}|{S['min_train']}".encode()).hexdigest()[:16]
        f = os.path.join(PD, key + ".npy")
        te = (D.t0 == v).values
        if os.path.exists(f):
            p = np.load(f)
        else:
            tcheck()
            tr = (D.t0 <= v - 2).values & keep
            p = fit_pred(D.loc[tr, feats].astype(float), y[tr], D.loc[te, feats].astype(float), model=model, w=wfull[tr])
            np.save(f, p.astype(np.float32))
        P[v] = p; res[v] = auc(y[te], p)
    return res, P


rows = []
champ_feats = list(S["base"]); champ, Pc = score(champ_feats); b0, P0 = dict(champ), Pc
rows.append(dict(task=task, model=model, exp="M0", change="기준선", **{f"auc_{v}": round(champ[v], 5) for v in dev}, decision="BASE"))
print(model, "M0", {v: round(champ[v], 4) for v in dev}, flush=True)
champ_tag = "M0"; Pchamp = P0
for name, add in S["steps"]:
    feats = champ_feats + add; r, P = score(feats)
    sk = np.mean([skill(r[v], champ[v]) for v in dev]); win = np.mean([r[v] > champ[v] for v in dev])
    dec = "ACCEPT" if (sk >= 0.003 and win >= 0.6) else "ROLLBACK"
    rows.append(dict(task=task, model=model, exp=name, change="+" + ",".join(add), **{f"auc_{v}": round(r[v], 5) for v in dev},
                     d_skill_vs_champ=round(sk, 4), win=round(win, 3), decision=dec))
    print(model, name, round(sk, 4), win, dec, flush=True)
    if dec == "ACCEPT":
        champ_feats, champ, champ_tag, Pchamp = feats, r, name, P
SBr = {}
for nm, fl in S.get("sb", {}).items():
    r, P = score(fl); SBr[nm] = (r, P)
    rows.append(dict(task=task, model=model, exp=nm, change="(강한 기준선, 채택 판단 무관)", **{f"auc_{v}": round(r[v], 5) for v in dev}, decision="INFO"))
# 챔피언 vs 강한 기준선 평균
out = []
for v in dev:
    tcheck()
    te = (D.t0 == v).values; yv = y[te]; g = D.loc[te, S["group"]].values
    p, pb = Pchamp[v], P0[v]
    lo, hi = boot_ci_skill(yv, p, pb, groups=g, n=200) if champ_tag != "M0" else (np.nan, np.nan)
    row = dict(task=task, model=model, champ=champ_tag, v=v, n=int(te.sum()), pos=int(yv.sum()), base_rate=round(yv.mean(), 5),
               auc_M0=round(b0[v], 4), auc_champ=round(champ[v], 4), skill=round(skill(champ[v], b0[v]), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4))
    for fr, nm in ((0.01, "top1pct"), (0.05, "top5pct"), (0.10, "top10pct")):
        row[nm] = round(topk(yv, p, fr)[0], 4); row[nm + "_M0"] = round(topk(yv, pb, fr)[0], 4)
    row["hit100"] = topk(yv, p, k=100)[1]; row["hit100_M0"] = topk(yv, pb, k=100)[1]
    row["recall_top10pct"] = round(topk(yv, p, 0.10)[1] / max(1, yv.sum()), 4); row["recall_top10pct_M0"] = round(topk(yv, pb, 0.10)[1] / max(1, yv.sum()), 4)
    for nm, (r, P) in SBr.items():
        ps = P[v]
        row["auc_SB"] = round(r[v], 4); row["skill_vs_SB"] = round(skill(champ[v], r[v]), 4)
        l2, h2 = boot_ci_skill(yv, p, ps, groups=g, n=100)
        row["ci_lo_vs_SB"] = round(l2, 4); row["ci_hi_vs_SB"] = round(h2, 4)
        row["top1pct_SB"] = round(topk(yv, ps, 0.01)[0], 4); row["top10pct_SB"] = round(topk(yv, ps, 0.10)[0], 4)
    out.append(row)
E = pd.DataFrame(rows); O = pd.DataFrame(out)
E.to_csv(os.path.join(OUT, f"experiments_{task}_{model}.csv"), index=False, encoding="utf-8-sig")
O.to_csv(os.path.join(OUT, f"pilot_by_origin_{task}_{model}.csv"), index=False, encoding="utf-8-sig")
json.dump(dict(task=task, model=model, champ=champ_tag, feats=champ_feats), open(os.path.join(CACHE, f"champ_{task}_{model}.json"), "w", encoding="utf-8"), ensure_ascii=False)
# 챔피언·M0·SB 예측 저장(사후 점검용)
pp = []
for v in dev:
    te = (D.t0 == v).values
    dd = D.loc[te, ["appl_no", "cc", "t0", "ksic", S["group"], S["y"]]].copy()
    dd["p_champ"] = Pchamp[v]; dd["p_M0"] = P0[v]
    for nm, (r, P) in SBr.items():
        dd["p_SB"] = P[v]
    pp.append(dd)
pd.concat(pp).to_parquet(os.path.join(CACHE, f"preds_{task}_{model}.parquet"), index=False)
print(O[["v", "n", "pos", "auc_M0", "auc_champ", "skill", "ci_lo", "ci_hi", "top1pct", "top1pct_M0", "auc_SB", "skill_vs_SB"]].to_string())
print("mean skill", O.skill.mean().round(4), "wins", (O.skill > 0).mean(), "vs SB", O.skill_vs_SB.mean().round(4), (O.skill_vs_SB > 0).mean(), round(time.time() - T_START), "s")
