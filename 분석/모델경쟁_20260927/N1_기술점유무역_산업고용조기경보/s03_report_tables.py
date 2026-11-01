# -*- coding: utf-8 -*-
"""보고서용 표(개발 원점만): ① 원점별 ρ 비교(B0·무역만·특허만·챔피언·동일모형 무특허) ② 피처 제거(ablation) ③ 시계·항목 민감도(참고용)
④ 전체 개발 원점 적합 계수 ⑤ 개발 원점 2018(→2021) 상·하위 예시. 최종시험 구간 미접근."""
from engine import *
from sklearn.linear_model import Ridge
st = champ_state(); F = st["F"]; A = st["alpha"]; nm = ksic_names()
o0 = b0(); oC = preds(F, alpha=A); oT = preds([c for c in F if not is_patent(c)], alpha=A); oP = preds([c for c in F if is_patent(c)], alpha=A)
oB1 = preds(["cnsh", "dcnsh"]); oM0 = preds(["mom_KR", "dsh_KR"]); oM7 = preds(["mom_KR_L2", "dsh_KR_L2", "cnsh", "dcnsh", "lvl_KR", "frgn_mom"])
T = pd.DataFrame({"B0 과거고용모멘텀": score(o0), "B1 무역만(참고)": score(oB1), "M0 특허만(참고)": score(oM0), "C0 출발챔피언(M7)": score(oM7),
                  "최종 챔피언": score(oC), "동일모형 특허 제거": score(oT), "동일모형 특허만": score(oP)})
T.loc["평균"] = T.mean(); T.loc["양수 원점"] = (T.iloc[:-1] > 0).sum()
Aa = pd.DataFrame({"B0": auc(o0), "챔피언": auc(oC), "특허 제거": auc(oT)}); Aa.loc["평균"] = Aa.mean()
T.round(3).to_csv(f"{R2}/per_origin_final.csv", encoding="utf-8-sig"); Aa.round(3).to_csv(f"{R2}/per_origin_auc.csv", encoding="utf-8-sig")
print(T.round(3).to_string()); print(Aa.round(3).to_string())
print("동일모형 특허만 CI", boot(oP, o0, n=500))
# ② ablation
ab = []
for c in F:
    s = score(preds([x for x in F if x != c], alpha=A)); d = score(oC) - s
    ab.append({"removed": c, "rho_without": round(s.mean(), 4), "drop_in_rho": round(d.mean(), 4), "win_with": round((d > 0).mean(), 3), "patent": is_patent(c)})
AB = pd.DataFrame(ab).sort_values("drop_in_rho", ascending=False); AB.to_csv(f"{R2}/ablation.csv", index=False, encoding="utf-8-sig"); print(AB.to_string())
# ③ 시계·항목 민감도(피처는 h=3 종사자로 골랐으므로 참고용)
hs = []
for item in ["emp", "wage", "shp", "va", "est"]:
    for h in [2, 3, 4]:
        tg = f"y_{item}_h{h}"; vs = [v for v in VS if v + h <= DEV_MAX and v - h >= 2008]
        o = preds(F, alpha=A, tg=tg, gap=h, vs=vs); ob = b0(tg=tg, mom=f"g3_{item}", vs=vs)
        s, sb = score(o), score(ob)
        hs.append({"item": item, "h": h, "n_orig": len(vs), "rho_model": round(s.mean(), 3), "pos": int((s > 0).sum()), "rho_B0_g3": round(sb.mean(), 3), "win_vs_B0": round(((s - sb) > 0).mean(), 2)})
HS = pd.DataFrame(hs); HS.to_csv(f"{R2}/aux_horizon_target.csv", index=False, encoding="utf-8-sig"); print(HS.to_string())
# ④ 계수(개발 원점 전체 t0 ≤ 2018 적합 = 최종 모형과 동일 학습집합)
D = P.dropna(subset=[TG]); Q = rank_within(D, F + [TG]); m = Ridge(alpha=A).fit(Q[F], Q[TG])
CF = pd.DataFrame({"feature": F, "coef": m.coef_.round(4), "patent": [is_patent(c) for c in F]}); CF.to_csv(f"{R2}/coef_final.csv", index=False, encoding="utf-8-sig"); print(CF.to_string())
# ⑤ 개발 원점 2018 예시
d = oC[2018].copy(); d["name"] = d.ksic.map(nm); d["rank_pred"] = d.p.rank(ascending=False); d["rank_true"] = d.y.rank(ascending=False)
d.sort_values("p", ascending=False).to_csv(f"{R2}/dev2018_ranking.csv", index=False, encoding="utf-8-sig")
print(d.sort_values("p", ascending=False).head(8)[["ksic", "name", "rank_pred", "rank_true"]].to_string()); print(d.sort_values("p").head(8)[["ksic", "name", "rank_pred", "rank_true"]].to_string())
