# -*- coding: utf-8 -*-
"""N3 강건성·절제 — 최종 챔피언(G2) 기준.
1) 절제: 챔피언 − 괴리(GAP) / − 특허 전체(PAT_LVL+GAP) → 특허·외부무역 기여를 강화된 자기이력 위에서 측정
2) KSIC 군집 부트스트랩(1,000회) 95% CI: 챔피언 vs B1, 챔피언 vs 특허 제거판
3) 해외점유 상위 3 KSIC(26212·26422·26211) 평가 제외
4) 5분위표: p_outshare, gap_kr_h, gap_cn 분위별 y 평균(개발 원점)
결과: R2/robust_X.json, R2/quintiles_X.csv"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n3_lib import *
FLOW = sys.argv[1] if len(sys.argv) > 1 else "X"
D = load_panel(FLOW); B1 = get_b1(D, FLOW); ch = champion(FLOW)
feats, spec = ch["feats"], ch["spec"]
GAPS = [c for c in feats if c.startswith("gap_")]
PATS = [c for c in feats if c in PAT_LVL]
out = {"champion": ch["exp_id"], "feats": feats}
R, P = evaluate(D, feats, spec, return_preds=True)
_, PB = evaluate(D, OWN[FLOW], {}, return_preds=True)
P["eM"] = (P.y - P.pred).abs(); P["eB1"] = (P.y - PB.pred.values).abs()
abl = {}
for lab, rm in [("minus_gap", GAPS), ("minus_patent_all", GAPS + PATS)]:
    f2 = [c for c in feats if c not in rm]
    R2_, P2 = evaluate(D, f2, spec, return_preds=True)
    P["e_" + lab] = (P.y - P2.pred.values).abs()
    sk = 1 - R.mae.values / R2_.mae.values
    abl[lab] = {"removed": rm, "skill_champ_vs_ablated": round(float(sk.mean()), 4),
                "win": round(float((R.mae.values < R2_.mae.values).mean()), 2),
                "per_origin": [round(float(x), 4) for x in sk],
                "ablated_skill_vsB1": round(float((1 - R2_.mae.values / B1.mae.values).mean()), 4)}
out["ablation"] = abl
rng = np.random.default_rng(0)
agg = P.groupby("k5")[["eM", "eB1", "e_minus_gap", "e_minus_patent_all"]].sum(); ks = agg.index.values
def boot(a, b):
    bs = []
    for _ in range(1000):
        s = agg.loc[rng.choice(ks, len(ks))].sum(); bs.append(1 - s[a] / s[b])
    s = agg.sum()
    return {"pooled": round(float(1 - s[a] / s[b]), 4), "ci95": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]}
out["boot_champ_vs_B1"] = boot("eM", "eB1")
out["boot_champ_vs_minus_gap"] = boot("eM", "e_minus_gap")
out["boot_champ_vs_minus_patent"] = boot("eM", "e_minus_patent_all")
ex = P[~P.k5.isin(["26212", "26422", "26211"])]
g = ex.groupby("t0")[["eM", "eB1", "e_minus_patent_all"]].mean()
out["excl_top3"] = {"skill_vsB1": round(float((1 - g.eM / g.eB1).mean()), 4), "win_vsB1": round(float((g.eM < g.eB1).mean()), 2),
                    "skill_vs_minus_patent": round(float((1 - g.eM / g.e_minus_patent_all).mean()), 4),
                    "win_vs_minus_patent": round(float((g.eM < g.e_minus_patent_all).mean()), 2)}
k = P.groupby("k5")[["eM", "eB1"]].sum(); out["share_ksic_improved_vsB1"] = round(float((k.eB1 > k.eM).mean()), 3)
# 5분위
E = D[D.t0.isin(DEV_V)]
rows = []
for c in ["p_outshare", "gap_kr_h", "gap_kr_lr", "gap_cn", "p_krshare5"]:
    q = E.groupby("t0")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    for qq, grp in E.groupby(q):
        rows.append({"feature": c, "quintile": int(qq) + 1, "n": len(grp), "y_mean": round(grp.y.mean(), 4),
                     "y_median": round(grp.y.median(), 4), "top20_rate": round(float((grp.y >= E.loc[grp.index].groupby('t0').y.transform(lambda s: s.quantile(0.8))).mean()), 3),
                     "feat_mean": round(grp[c].mean(), 4)})
Q = pd.DataFrame(rows); Q.to_csv(os.path.join(R2, f"quintiles_{FLOW}.csv"), index=False, encoding="utf-8-sig")
print(Q.to_string())
json.dump(out, open(os.path.join(R2, f"robust_{FLOW}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "feats"}, ensure_ascii=False, indent=1))
P.to_parquet(os.path.join(CACHE, f"champ_preds_{FLOW}.parquet"), index=False)
