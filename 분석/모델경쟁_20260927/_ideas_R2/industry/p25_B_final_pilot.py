# -*- coding: utf-8 -*-
"""아이디어 B 파일럿 정리: 챔피언/도전자 순차 적용 기록(experiments_B.csv) + 기준선 비교 + 산업 군집 부트스트랩 CI.
채택 규칙(PROTOCOL_R2 §3 준용): 평균 Δρ ≥ +0.003 이고 원점 승률 ≥ 60%."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ilib import *
P = pd.read_parquet(f"{CACHE}/panel_kosis3y_ext.parquet"); pv = load_pat()
L2 = []
for t0 in sorted(P.t0.unique()):
    f = pat_feats(pv, t0, lag=2)[["mom_KR", "dsh_KR"]].add_suffix("_L2"); f["t0"] = t0; L2.append(f.reset_index())
P = P.merge(pd.concat(L2), on=["ksic", "t0"], how="left")
TG = "y_emp"; VS = list(range(2011, 2019))
def preds(F, alpha=5.0, data=P):
    cols = F+[TG]; D = data.dropna(subset=[TG])
    Q = pd.concat([g.assign(**{c: g[c].rank(pct=True).fillna(.5) for c in cols}) for _, g in D.groupby("t0")])
    out = {}
    for v in VS:
        tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]
        p = Ridge(alpha=alpha).fit(tr[F], tr[TG]).predict(te[F])
        out[v] = pd.DataFrame({"ksic": te.ksic.values, "p": p, "y": D[D.t0 == v].set_index("ksic").loc[te.ksic, TG].values})
    return out
def score(o): return pd.Series({v: spearmanr(d.y, d.p)[0] for v, d in o.items()})
def auc(o): return pd.Series({v: roc_auc_score((d.y.rank(pct=True) >= 2/3).astype(int), d.p) for v, d in o.items()})
# 기준선
mom = {v: pd.DataFrame({"ksic": d.ksic, "p": P[P.t0 == v].set_index("ksic").loc[d.ksic, "g3_emp"].fillna(0).values, "y": d.y}) for v, d in preds(["mom_KR"]).items()}
rows = []; log = []
def rec(eid, parent, change, o, champ):
    s = score(o); d = s - champ if champ is not None else s*0
    acc = champ is not None and d.mean() >= 0.003 and (d > 0).mean() >= 0.6
    log.append({"exp_id": eid, "parent": parent, "change": change, **{f"rho_{v}": round(s[v], 4) for v in VS},
                "mean_rho": round(s.mean(), 4), "auc_top3": round(auc(o).mean(), 4),
                "d_vs_champ": round(d.mean(), 4), "win": round((d > 0).mean(), 3),
                "decision": "BASE" if champ is None else ("ACCEPT" if acc else "ROLLBACK")})
    print(f"{eid:4s} {change:34s} ρ {s.mean():.3f} AUC {auc(o).mean():.3f} Δ {d.mean():+.3f} 승률 {(d>0).mean():.2f} {log[-1]['decision']}")
    return acc, s
rec("B0", "-", "기준선: 과거3년 고용성장 순위(모멘텀)", mom, None)
_, sB = rec("B1", "-", "무역만: 대중수입점유 수준+변화", preds(["cnsh", "dcnsh"]), None)
_, sP = rec("M0", "-", "특허만: KR 모멘텀+KR 점유변화", preds(["mom_KR", "dsh_KR"]), None)
F = ["mom_KR", "dsh_KR"]; ch = sP; par = "M0"
steps = [("M1", "+cnsh(대중 수입점유 수준)", lambda F: F+["cnsh"]),
         ("M2", "+dcnsh(대중 수입점유 3년변화)", lambda F: F+["dcnsh"]),
         ("M3", "특허 시차1→시차2", lambda F: [c+"_L2" if c in ("mom_KR", "dsh_KR") else c for c in F]),
         ("M4", "+xg3(한국 수출 3년성장, 결측多)", lambda F: F+["xg3"]),
         ("M5", "+lvl_KR(특허 규모)", lambda F: F+["lvl_KR"]),
         ("M6", "+sh_KR(KR 점유 수준)", lambda F: F+["sh_KR"]),
         ("M7", "+frgn_mom(US·JP·EP 모멘텀)", lambda F: F+["frgn_mom"]),
         ("M8", "+newmom_KR(신규출원인)", lambda F: F+["newmom_KR"])]
for eid, ch_name, fn in steps:
    F2 = fn(F); acc, s = rec(eid, par, ch_name, preds(F2), ch)
    if acc: F, ch, par = F2, s, eid
print("최종 피처", F)
L = pd.DataFrame(log); L.to_csv(f"{HERE}/experiments_B.csv", index=False, encoding="utf-8-sig")
# 산업 군집 부트스트랩(원점별 ρ 평균, 모형 − 모멘텀 기준선)
o_fin = preds(F); rng = np.random.default_rng(0); ks = P.ksic.unique(); bm, bd, b0 = [], [], []
for b in range(1000):
    samp = rng.choice(ks, len(ks), replace=True); cnt = pd.Series(samp).value_counts()
    rr, r0 = [], []
    for v in VS:
        d = o_fin[v].set_index("ksic").loc[[k for k in samp if k in o_fin[v].ksic.values]]
        m = mom[v].set_index("ksic").loc[d.index]
        rr.append(spearmanr(d.y, d.p)[0]); r0.append(spearmanr(m.y, m.p)[0])
    bm.append(np.nanmean(rr)); b0.append(np.nanmean(r0)); bd.append(np.nanmean(rr)-np.nanmean(r0))
print("최종 ρ", round(score(o_fin).mean(), 3), "CI", np.percentile(bm, [2.5, 97.5]).round(3), "| 모멘텀 기준선 ρ", round(score(mom).mean(), 3),
      "| 차이 CI", np.percentile(bd, [2.5, 97.5]).round(3))
