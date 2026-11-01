# -*- coding: utf-8 -*-
"""N1 추가 실험(챔피언 = OLS[g_idx, gd_w_full_f]) + 보조 지표 + KSIC 군집 부트스트랩.
E13 최근 4년 이동창 학습 / E14 +DART 자산성장 배분 / E15 +DART 영업이익률 변화 배분
보조: 중앙절대오차, 방향 적중률, Spearman. 부트스트랩: KSIC 50개 군집 재표본 1000회로 skill(챔피언 vs 기준선) CI.
정보용 대체 타깃: 종사자 성장, 부가가치 성장(같은 설계)."""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
Y = pd.read_parquet(os.path.join(C, "nowcast_panel2.parquet")); assert Y.t.max() <= 2023
DEV = list(range(2016, 2024))
F = pd.read_parquet(os.path.join(C, "fin.parquet"))
W = pd.read_parquet(os.path.join(C, "alloc_w.parquet"))
fr = F.merge(F.assign(year=F.year + 1)[["corp_code", "year", "rev", "assets", "op"]].rename(columns={"rev": "rev_prev", "assets": "as_prev", "op": "op_prev"}), on=["corp_code", "year"])
fr = fr[(fr.rev > 0) & (fr.rev_prev > 0) & (fr.assets > 0) & (fr.as_prev > 0)]
M = fr.merge(W, left_on=["corp_code", fr.year - 1], right_on=["corp_code", "t0"])
G = M.groupby(["ksic", "year"])
ex = pd.DataFrame({"gd_assets": np.log(G.apply(lambda x: (x.w_full * x.assets).sum() / (x.w_full * x.as_prev).sum(), include_groups=False)),
                   "gd_dopm": G.apply(lambda x: ((x.w_full * x.op).sum() / (x.w_full * x.rev).sum()) - ((x.w_full * x.op_prev).sum() / (x.w_full * x.rev_prev).sum()), include_groups=False)}).reset_index().rename(columns={"year": "t"})
Y = Y.merge(ex, on=["ksic", "t"], how="left")
# 대체 타깃
k = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv")); k = k[k.연도 <= 2023]; k["ksic"] = k.세세분류코드.astype(str)
alt = {}
for it, nm in [("T02", "g_emp_t"), ("T06", "g_va_t")]:
    rows = []
    for (ks, bs), g in k[k.항목코드 == it].groupby(["ksic", "기준"]):
        s = g.set_index("연도").값.sort_index()
        for t in s.index:
            if t - 1 in s.index and s[t - 1] > 0 and s[t] > 0 and ((bs == "10차" and t <= 2020) or (bs == "11차" and t >= 2021)):
                rows.append((ks, t, np.log(s[t] / s[t - 1])))
    Y = Y.merge(pd.DataFrame(rows, columns=["ksic", "t", nm]), on=["ksic", "t"], how="left")

def run(feats, target="g", window=None):
    P = {}
    for t in DEV:
        tr = Y[(Y.t <= t - 1) & (Y.t >= (2014 if window is None else t - window))].dropna(subset=[target])
        te = Y[Y.t == t].dropna(subset=[target])
        med = tr[feats].median()
        m = LinearRegression().fit(tr[feats].fillna(med), tr[target])
        P[t] = pd.DataFrame({"ksic": te.ksic.values, "y": te[target].values, "p": m.predict(te[feats].fillna(med))})
    return P
def mets(P):
    return {"mae": np.mean([np.mean(np.abs(P[t].y - P[t].p)) for t in DEV]),
            "medae": np.mean([np.median(np.abs(P[t].y - P[t].p)) for t in DEV]),
            "dir": np.mean([np.mean(np.sign(P[t].y - P[t].y.median()) == np.sign(P[t].p - P[t].p.median())) for t in DEV]),
            "rho": np.mean([spearmanr(P[t].y, P[t].p).statistic for t in DEV])}
base = run(["g_idx"]); champ_f = ["g_idx", "gd_w_full_f"]; ch = run(champ_f)
mb, mc = mets(base), mets(ch)
def sk(P, Q): return np.mean([1 - np.mean(np.abs(P[t].y - P[t].p)) / np.mean(np.abs(Q[t].y - Q[t].p)) for t in DEV])
def win(P, Q): return np.mean([np.mean(np.abs(P[t].y - P[t].p)) < np.mean(np.abs(Q[t].y - Q[t].p)) for t in DEV])
rows = [{"exp": "B_M1", **mb}, {"exp": "CHAMP_M3", **mc, "skill_vs_base": sk(ch, base)}]
for nm, kw in [("E13_window4", dict(feats=champ_f, window=4)), ("E14_+gd_assets", dict(feats=champ_f + ["gd_assets"])),
               ("E15_+gd_dopm", dict(feats=champ_f + ["gd_dopm"]))]:
    P = run(**kw); d = sk(P, base) - sk(ch, base); w = win(P, ch)
    rows.append({"exp": nm, **mets(P), "skill_vs_base": sk(P, base), "d_skill": d, "win_vs_champ": w,
                 "decision": "ACCEPT" if d >= 0.003 and w >= 0.6 else "ROLLBACK"})
for tg in ["g_emp_t", "g_va_t"]:
    b = run(["g_idx"], tg); c = run(champ_f, tg)
    rows.append({"exp": f"INFO_target_{tg}", **mets(c), "skill_vs_base": sk(c, b), "win_vs_champ": win(c, b), "decision": "INFO(대체타깃: 기준선=같은 타깃 idx OLS)"})
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "iter_N1_more.csv"), index=False)
pd.set_option("display.width", 250); print(R.round(4).to_string())
# KSIC 군집 부트스트랩
rng = np.random.default_rng(0); ks = sorted(Y.ksic.unique()); bs = []
eb = pd.concat([base[t].assign(t=t) for t in DEV]); ec = pd.concat([ch[t].assign(t=t) for t in DEV])
eb["e"] = (eb.y - eb.p).abs(); ec["e"] = (ec.y - ec.p).abs()
Eb = eb.groupby(["ksic", "t"]).e.sum().unstack(); Ec = ec.groupby(["ksic", "t"]).e.sum().unstack()
for _ in range(1000):
    s = rng.choice(ks, len(ks), replace=True)
    b = Eb.loc[s].sum(); c = Ec.loc[s].sum()
    bs.append(np.mean(1 - c / b))
print("champion skill vs baseline, KSIC-cluster bootstrap 95% CI:", np.round(np.percentile(bs, [2.5, 50, 97.5]), 4), "P(skill>0)=", np.mean(np.array(bs) > 0))
