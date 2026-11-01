# -*- coding: utf-8 -*-
"""필수 정직 보고: 현재 챔피언 vs '같은 모형에서 특허 피처만 제거' — Δρ·승률·KSIC 군집 부트스트랩 1,000회 CI. R2/increment_log.csv 누적."""
from engine import *
st = champ_state(); F = st["F"]; kw = dict(alpha=st.get("alpha", 5.0), train_tg=st.get("train_tg"))
oF = preds(F, **kw); FT = [c for c in F if not is_patent(c)]; oT = preds(FT, **kw); o0 = b0()
sF, sT = score(oF), score(oT); d = sF - sT
r = boot(oF, oT, n=1000); rb = boot(oF, o0, n=1000)
row = {"champ": st["id"], "F": "|".join(F), "rho": round(sF.mean(), 4), "rho_ci": r["ci"], "rho_noPatent": round(sT.mean(), 4),
       "pat_inc": round(d.mean(), 4), "pat_inc_win": round((d > 0).mean(), 3), "pat_inc_ci": r["dci"], "p_inc_le0": round(r["p_le0"], 3),
       "skill_vs_B0": round(sF.mean() - score(o0).mean(), 4), "skill_ci": rb["dci"]}
print(row)
f = f"{R2}/increment_log.csv"
L = pd.read_csv(f, encoding="utf-8-sig") if os.path.exists(f) else pd.DataFrame()
if len(L): L = L[L.champ != st["id"]]
pd.concat([L, pd.DataFrame([row])]).to_csv(f, index=False, encoding="utf-8-sig")
