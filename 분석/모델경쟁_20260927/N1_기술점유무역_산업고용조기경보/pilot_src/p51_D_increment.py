# -*- coding: utf-8 -*-
"""아이디어 D 챔피언/도전자: 챔피언 = 무역만(cnsh,dcnsh,과거 수출성장). 특허 피처 순차 추가, 채택 규칙 Δρ≥0.003 & 승률≥60%. 산업 부트스트랩 CI."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from ilib import *
P = pd.read_parquet(f"{CACHE}/panel_krexp_ksic.parquet"); VS = range(2008, 2019)
def preds(F, a=5.0):
    Q = pd.concat([g.assign(**{c: g[c].rank(pct=True).fillna(.5) for c in F+["y"]}) for _, g in P.groupby("t0")]); o = {}
    for v in VS:
        tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]
        o[v] = pd.DataFrame({"ksic": te.ksic.values, "p": Ridge(alpha=a).fit(tr[F], tr.y).predict(te[F]), "y": P[P.t0 == v].y.values})
    return o
sc = lambda o: pd.Series({v: spearmanr(d.y, d.p)[0] for v, d in o.items()})
F = ["cnsh", "dcnsh", "mom"]; ch = sc(preds(F)); log = [{"exp_id": "D0", "change": "무역만: 대중수입점유 수준·변화 + 과거 3년 수출성장", "mean_rho": ch.mean(), "decision": "BASE"}]
print(f"D0 챔피언 ρ {ch.mean():.3f}")
for eid, lab, add in [("D1", "+dsh_KR(한국 특허 세계점유 변화)", ["dsh_KR"]), ("D2", "+mom_KR", ["mom_KR"]), ("D3", "+frgn_mom(해외 특허 모멘텀)", ["frgn_mom"]),
                      ("D4", "+lvl_KR", ["lvl_KR"]), ("D5", "+sh_KR", ["sh_KR"]), ("D6", "+xlv(수출 규모)", ["xlv"]), ("D7", "+mom_CN", ["mom_CN"]),
                      ("D8", "+newmom_KR", ["newmom_KR"]), ("D9", "+gap_CN_KR", ["gap_CN_KR"])]:
    s = sc(preds(F+add)); d = s - ch; acc = d.mean() >= 0.003 and (d > 0).mean() >= 0.6
    print(f"{eid} {lab:30s} ρ {s.mean():.3f} Δ {d.mean():+.3f} 승률 {(d>0).mean():.2f} {'ACCEPT' if acc else 'ROLLBACK'}")
    log.append({"exp_id": eid, "change": lab, **{f"rho_{v}": round(s[v], 4) for v in VS}, "mean_rho": round(s.mean(), 4), "d_vs_champ": round(d.mean(), 4), "win": round((d > 0).mean(), 3), "decision": "ACCEPT" if acc else "ROLLBACK"})
    if acc: F, ch = F+add, s
print("최종", F, round(ch.mean(), 3))
pd.DataFrame(log).to_csv(f"{HERE}/experiments_D.csv", index=False, encoding="utf-8-sig")
T = ["cnsh", "dcnsh", "mom"]; oT, oF = preds(T), preds(F); rng = np.random.default_rng(0); ks = P.ksic.unique(); bd, bf = [], []
for b in range(500):
    s_ = rng.choice(ks, len(ks)); a, c = [], []
    for v in VS:
        x, z = oF[v].set_index("ksic"), oT[v].set_index("ksic"); kk = [k for k in s_ if k in x.index]
        a.append(spearmanr(x.loc[kk].y, x.loc[kk].p)[0]); c.append(spearmanr(z.loc[kk].y, z.loc[kk].p)[0])
    bf.append(np.nanmean(a)); bd.append(np.nanmean(a)-np.nanmean(c))
print("최종 ρ CI", np.percentile(bf, [2.5, 97.5]).round(3), "특허 증분 CI", np.percentile(bd, [2.5, 97.5]).round(3))
base = pd.Series({v: spearmanr(g.y, g.mom, nan_policy="omit")[0] for v, g in P.groupby("t0") if v >= 2008}); print("모멘텀 기준선 ρ", round(base.mean(), 3), "최종−기준선 승률", round(((ch-base) > 0).mean(), 2))
