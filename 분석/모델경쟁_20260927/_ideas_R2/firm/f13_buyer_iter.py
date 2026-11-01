# -*- coding: utf-8 -*-
"""아이디어 3(기술 구매기업) 재정식화.
R1: 로지스틱 + 소수 피처 챔피언/도전자(기준선에 한 묶음씩).
R2: 타깃을 '대학·연구기관·개인 → 기업' 기술이전(공공·개인 기술 수요)으로 한정: y_tt.
개발 v=2015..2020, 학습 t0 ≤ v−2. 기준선 = 로지스틱[log 생존특허, log 최근3년 등록, 과거3년 양수 여부, 업력]."""
import os, sys, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalkit import walk, skill
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
B = pd.read_parquet(os.path.join(C, "panel3.parquet"))
E = pd.read_parquet(os.path.join(C, "buy_events.parquet")); assert E.y.max() <= 2022
def stype(s):
    s = str(s or "")
    if re.search(r"대학|산학협력|연구원|연구소|재단|진흥원|공사|기술원|과학기술|국가|대한민국", s): return "pub"
    if re.search(r"회사|법인|inc|corp|ltd|co$|gmbh|ag$|llc|주$|㈜", s, re.I): return "corp"
    if len(s) <= 4: return "ind"
    return "corp"
E["stype"] = E.seller.map(stype)
D = E.drop_duplicates(["corp_code", "seller", "d"])
tt = D[D.stype.isin(["pub", "ind"])]
print("deal seller types", D.stype.value_counts().to_dict())
def cnt(df, lo, hi): return df[df.y.between(lo, hi)].groupby("corp_code").size()
parts = []
for t0, g in B.groupby("t0"):
    g = g.copy()
    g["y_tt"] = (g.corp_code.map(cnt(tt, t0 + 1, t0 + 2)).fillna(0) > 0).astype(float)
    g["tt_past3"] = g.corp_code.map(cnt(tt, t0 - 2, t0)).fillna(0)
    parts.append(g)
B = pd.concat(parts)
B["buy_past3_any"] = (B.buy_past3 > 0).astype(float)
B["log_buy_all"] = np.log1p(B.buy_past_all)
B["tt_past3_any"] = (B.tt_past3 > 0).astype(float)
B["log_lic"] = np.log1p(B.n_lic3)
print("y_tt rate by t0", B.groupby("t0").y_tt.agg(["mean", "sum"]).round(4).to_dict())
DEV = list(range(2015, 2021)); H = 2
base_f = ["log_alive", "log_reg3", "buy_past3_any", "firm_age"]
cands = [("reg_mom", ["reg_mom"]), ("buy_hist", ["log_buy_all"]), ("newksic", ["n_newksic3"]), ("lap_rate", ["lap_rate"]),
         ("lic", ["log_lic"]), ("hhi", ["hhi", "n_ksic"]), ("pat_age", ["pat_age", "firm_pat_age"]), ("tr_rate", ["tr_rate"]),
         ("x_pmom", ["x_pmom_foreign", "x_pmom_KR"]), ("x_kr_gap", ["x_kr_gap"]), ("x_g_ship", ["x_g_ship"]), ("x_g_imp", ["x_g_imp"])]
out = []
for Y, extra_base in [("y_buy", []), ("y_tt", ["tt_past3_any"])]:
    bf = base_f + extra_base
    ab, _ = walk(B, Y, bf, DEV, H, "logit"); champ = list(bf); ac = dict(ab)
    out.append({"target": Y, "exp": "B0", "change": "baseline", "mean_auc": round(np.mean(list(ab.values())), 4), "decision": "BASE"})
    for name, f in cands:
        am, _ = walk(B, Y, champ + f, DEV, H, "logit")
        s_new = np.mean(list(skill(am, ab).values())); s_ch = np.mean(list(skill(ac, ab).values()))
        win = np.mean([am[v] > ac[v] for v in DEV])
        dec = "ACCEPT" if (s_new - s_ch >= 0.003 and win >= 0.6) else "ROLLBACK"
        if dec == "ACCEPT": champ += f; ac = am
        out.append({"target": Y, "exp": name, "change": "+" + ",".join(f), "mean_auc": round(np.mean(list(am.values())), 4),
                    "skill_vs_base": round(s_new, 4), "d_skill": round(s_new - s_ch, 4), "win": round(win, 2), "decision": dec})
    am, _ = walk(B, Y, champ, DEV, H, "xgb", n_estimators=200, max_depth=3)
    out.append({"target": Y, "exp": "xgb_champ", "change": "xgb(깊이3) on champion", "mean_auc": round(np.mean(list(am.values())), 4),
                "skill_vs_base": round(np.mean(list(skill(am, ab).values())), 4), "decision": "INFO"})
    out.append({"target": Y, "exp": "CHAMP", "change": ",".join(champ), "mean_auc": round(np.mean(list(ac.values())), 4),
                "skill_vs_base": round(np.mean(list(skill(ac, ab).values())), 4), "decision": "FINAL",
                "per_origin": str({v: round(ac[v], 3) for v in ac}), "base_per_origin": str({v: round(ab[v], 3) for v in ab})})
L = pd.DataFrame(out); L.to_csv(os.path.join(HERE, "iter_buyer.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(L.drop(columns=[c for c in ["per_origin", "base_per_origin"] if c in L]).to_string())
print(L[L.exp == "CHAMP"][["target", "per_origin", "base_per_origin"]].to_string())
