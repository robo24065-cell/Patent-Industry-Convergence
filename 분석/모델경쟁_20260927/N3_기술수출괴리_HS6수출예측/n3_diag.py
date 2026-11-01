# -*- coding: utf-8 -*-
"""N3 진단 — 현재 챔피언의 원점별 skill, 피처 중요도(gain), 세그먼트 오차, 괴리 피처 5분위별 상대성장.
사용: python n3_diag.py X"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n3_lib import *
FLOW = sys.argv[1] if len(sys.argv) > 1 else "X"
D = load_panel(FLOW); B1 = get_b1(D, FLOW); ch = champion(FLOW)
feats, spec = ch["feats"], ch["spec"]
R, P = evaluate(D, feats, spec, return_preds=True)
_, PB = evaluate(D, OWN[FLOW], {}, return_preds=True)
P["eM"] = (P.y - P.pred).abs(); P["eB"] = (P.y - PB.pred.values).abs()
print("챔피언", ch["exp_id"], "원점별 skill", np.round(1 - R.mae.values / B1.mae.values, 4))
# gain 중요도(학습 전체 ≤2015, 시드 0)
tr = D[D.t0 <= 2015]
p = dict(BASE_PARAMS); p.update(spec.get("params", {}))
mdl = xgb.XGBRegressor(random_state=0, **p).fit(tr[feats], tr.y)
g = pd.Series(mdl.get_booster().get_score(importance_type="total_gain")).reindex(feats).fillna(0)
print((g / g.sum()).sort_values(ascending=False).round(3).head(20).to_string())
# KSIC별 개선(챔피언 − B1) 상하위
k = P.groupby("k5")[["eM", "eB"]].sum(); k["d"] = k.eB - k.eM; k["n"] = P.groupby("k5").size()
print("KSIC별 오차 감소(합, +가 개선) 상위/하위\n", k.sort_values("d").iloc[[*range(5), *range(-5, 0)]].round(2).to_string())
print("개선 KSIC 비율", round((k.d > 0).mean(), 3))
# 5분위
E = D[D.t0.isin(DEV_V)]
for c in ["gap_kr_lr", "gap_kr_h", "p_outshare", "gap_kr"]:
    if c in E:
        q = E.groupby("t0")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        print(c, "5분위별 y 평균", E.groupby(q).y.mean().round(3).to_dict(), "중앙값", E.groupby(q).y.median().round(3).to_dict())
P.to_parquet(os.path.join(CACHE, f"diag_preds_{FLOW}.parquet"), index=False)
