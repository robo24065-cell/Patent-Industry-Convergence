# -*- coding: utf-8 -*-
"""사후 점검(INFO, 채택 판단에 쓰지 않음): I1 로지스틱 판, I2 원점별 출원인 군집 부트스트랩 CI, I3 묶음 분해, I4 상위100 적중·국면별 skill."""
import os, json
import numpy as np, pandas as pd
from n5lib import *
import n5_eval as ev

S = ev.load_state()
spec = S["champ_spec"]; base = {int(k): v for k, v in S["base"].items()}; champ = {int(k): v for k, v in S["champ"].items()}
print("champion", S["champ_id"], spec)
P0 = pd.read_parquet(os.path.join(CACHE, "preds_M0.parquet"))
PC = pd.read_parquet(os.path.join(CACHE, f"preds_{S['champ_id']}.parquet"))
P1 = pd.read_parquet(os.path.join(CACHE, "preds_E02.parquet"))
rows = []
# I2·I4
for v in ev.DEV:
    a = P0[P0.t0 == v]; b = PC[PC.t0 == v]; c = P1[P1.t0 == v]
    assert (a.appl_no.values == b.appl_no.values).all()
    (lo, hi), mb = cluster_boot_ci(b.y.values, b.p.values, a.p.values, b.app.values, n=300, seed=v)
    top = lambda d: int(d.sort_values("p", ascending=False).y.head(100).sum())
    rows.append(dict(check="I2_I4", t0=v, n=len(b), pos=int(b.y.sum()), auc_M0=base[v], auc_M1=float(auc(c.y, c.p)), auc_champ=champ[v],
                     skill_champ_vs_M0=skill(champ[v], base[v]), ci_lo=lo, ci_hi=hi, skill_M1_vs_M0=skill(auc(c.y, c.p), base[v]),
                     top100_M0=top(a), top100_M1=top(c), top100_champ=top(b)))
R = pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
early = R[R.t0 <= 2016].skill_champ_vs_M0.mean(); late = R[R.t0 >= 2018].skill_champ_vs_M0.mean()
print("국면별 skill(M0 대비) 2014–16", round(early, 4), "2018–20", round(late, 4),
      "| M1:", round(R[R.t0 <= 2016].skill_M1_vs_M0.mean(), 4), round(R[R.t0 >= 2018].skill_M1_vs_M0.mean(), 4))
# I1 로지스틱
rl, _, _ = ev.score(spec, model="logit", seeds=(0,))
r0l, _, _ = ev.score(ev.BASE_SPEC, model="logit", seeds=(0,))
I1 = dict(check="I1_logit", **{f"auc_champ_{v}": rl[v] for v in ev.DEV}, **{f"auc_M0_{v}": r0l[v] for v in ev.DEV})
skl = [skill(rl[v], r0l[v]) for v in ev.DEV]
print("I1 로지스틱 champion vs M0 skill", np.round(skl, 4), "평균", round(np.mean(skl), 4), "우세", sum(s > 0 for s in skl))
# I3 분해(챔피언에서 하나씩 빼기)
added = [g for g in spec["groups"] if g not in ("M0", "M1")] + spec.get("extra", [])
abl = []
for item in ["M1"] + added:
    s2 = json.loads(json.dumps(spec))
    if item in s2["groups"]:
        s2["groups"] = [g for g in s2["groups"] if g != item]
    else:
        s2["extra"] = [e for e in s2.get("extra", []) if e != item]
    r2, _, _ = ev.score(s2)
    d = [skill(champ[v], r2[v]) for v in ev.DEV]
    abl.append(dict(check="I3_ablate", removed=item, mean_loss_skill=np.mean(d), wins_champ=sum(x > 0 for x in d), **{f"auc_{v}": r2[v] for v in ev.DEV}))
    print("drop", item, round(np.mean(d), 4), sum(x > 0 for x in d), flush=True)
A = pd.DataFrame(abl)
out = os.path.join(OUT, "info_checks.csv")
pd.concat([R, pd.DataFrame([I1 | {"skill_mean": float(np.mean(skl)), "wins": int(sum(s > 0 for s in skl))}]), A], ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
print("saved", out)
