# -*- coding: utf-8 -*-
"""C07-T3 verify_1: 산업 지수 순위 발견 반박 시도. n8lib/c07lib/s04 레이블 캐시 미사용.
레이블은 sqlite(mode=ro)에서 직접, 산업 집계는 future_scores_r6.parquet에서 직접 재계산."""
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
t = time.time()
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
N8 = os.path.join(BASE, r"분석\모델경쟁_20260927\N8_특허분쟁표적레이더")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
W = os.path.dirname(os.path.abspath(__file__))
C07 = os.path.dirname(W)
OUT = {}
FS = pd.read_parquet(os.path.join(N8, "future_scores_r6.parquet"))
K = pd.read_csv(os.path.join(N8, "future_ksic_r6.csv"), encoding="utf-8-sig", dtype={"ksic": str})
FP = pd.read_parquet(os.path.join(N8, "final_preds_r6.parquet"), columns=["rgst_no", "ksic", "y"])
# ---- 1. 산업 집계 재계산 대조
agg = FS.groupby("ksic").agg(n=("p", "size"), exp=("p", lambda s: float(s.astype("float64").sum())), mp=("p", lambda s: float(s.astype("float64").mean())), top=("top1pct", "sum"))
M = K.set_index("ksic").join(agg, how="outer", lsuffix="_x", rsuffix="_y")
OUT["agg_check"] = dict(n_ksic_csv=len(K), n_ksic_scores=len(agg), n_match=int((M.n_x == M.n_y).sum()),
                        max_rel_exp=float((M.exp_disputes / M.exp - 1).abs().max()), max_rel_meanp=float((M.mean_p / M.mp - 1).abs().max()),
                        top_match=int((M.n_top1pct == M.top).sum()), dup_rgst=int(FS.rgst_no.duplicated().sum()),
                        top1pct_count=int(FS.top1pct.sum()))
# ---- 2. 레이블: sqlite 직접 (이름 변형까지 LIKE로 점검)
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
names = pd.read_sql("select rgstCsName, count(*) n from right where rgstCsDate >= '20250101' and (rgstCsName like '%무효%' or rgstCsName like '%권리범위%') group by rgstCsName", c)
OUT["names_ge2025"] = names.to_dict("records")
EV = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName from right where rgstCsDate between '20250101' and '20260815' and rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록')", c)
EV["d"] = EV.rgstCsDate.astype(int)
c.close()
pos_main = set(EV[EV.d <= 20260630].rgst_no.astype(str)); pos_A = set(EV.rgst_no.astype(str))
FS["y"] = FS.rgst_no.astype(str).isin(pos_main).astype(int)
FS["yA"] = FS.rgst_no.astype(str).isin(pos_A).astype(int)
OUT["labels"] = dict(events_main=int((EV.d <= 20260630).sum()), patents_main=len(pos_main), pos_in_pop=int(FS.y.sum()), posA_in_pop=int(FS.yA.sum()))
# ---- 3. 순위상관
ks = sorted(K.ksic); Ki = K.set_index("ksic").loc[ks]
gy = FS.groupby("ksic").agg(n=("y", "size"), r=("y", "sum"), rA=("yA", "sum")).loc[ks]
rate = (gy.r / gy.n).values; cnt = gy.r.values; rateA = (gy.rA / gy.n).values
pers = FP.groupby("ksic").y.agg(["sum", "size"]).reindex(ks)
prate = (pers["sum"] / pers["size"]).values; pcnt = pers["sum"].values.astype(float)
mean_p = Ki.mean_p.values; exp_d = Ki.exp_disputes.values
sp = lambda a, b: float(spearmanr(a, b).correlation)
R = dict(rho_meanp_rate=sp(mean_p, rate), rho_exp_cnt=sp(exp_d, cnt), rho_pers_rate=sp(prate, rate), rho_perscnt_cnt=sp(pcnt, cnt),
         rho_meanp_pers=sp(mean_p, prate),
         n_zero=int((cnt == 0).sum()), pers_n_zero=int((pcnt == 0).sum()),
         kendall_meanp_rate=float(kendalltau(mean_p, rate).correlation), kendall_pers_rate=float(kendalltau(prate, rate).correlation),
         rho_meanp_rateA=sp(mean_p, rateA), rho_pers_rateA=sp(prate, rateA))
nz = cnt > 0
R["rho_meanp_rate_nonzero"] = sp(mean_p[nz], rate[nz]); R["rho_pers_rate_nonzero"] = sp(prate[nz], rate[nz])
# 가장 단순한 '크기' 기준선: 특허 수 n만으로 실현 건수 순위
R["rho_n_cnt"] = sp(gy.n.values, cnt)
# 섞은 지수: 지수와 지속의 평균 순위
from scipy.stats import rankdata
R["rho_blend_rate"] = sp(rankdata(mean_p) + rankdata(prate), rate)
OUT["rho"] = R
# 상위 5
rk = pd.DataFrame(dict(ksic=ks, exp=exp_d, mp=mean_p, cnt=cnt, rate=rate))
rk["r_exp"] = rk.exp.rank(ascending=False, method="min"); rk["r_mp"] = rk.mp.rank(ascending=False, method="min")
rk["r_rate"] = rk.rate.rank(ascending=False, method="min"); rk["r_cnt"] = rk.cnt.rank(ascending=False, method="min")
OUT["top5_exp"] = rk.sort_values("r_exp").head(5)[["ksic", "r_exp", "r_mp", "cnt", "r_cnt", "r_rate"]].to_dict("records")
# ---- 4. 부트스트랩: 권리자 군집 (owner_v2는 s02b 캐시에서 읽기만)
S = pd.read_parquet(os.path.join(C07, "cache", "s02b_scores_t2024.parquet"), columns=["rgst_no", "owner_v2"])
assert (S.rgst_no.values == FS.rgst_no.values).all()
o = S.owner_v2.fillna("").astype(str).values
g = np.where(o == "", "R" + FS.rgst_no.astype(str).values, o)
codes_g, uq = pd.factorize(g); G = len(uq)
kidx = {k: i for i, k in enumerate(ks)}; ck = np.array([kidx[k] for k in FS.ksic.astype(str)])
y = FS.y.values.astype(float)
def boot(seed, cluster=True, B=1000):
    rng = np.random.default_rng(seed); v = {"mp": [], "pers": [], "diff": [], "exp": []}
    for _ in range(B):
        if cluster:
            w = rng.multinomial(G, np.full(G, 1.0 / G))[codes_g].astype(float)
        else:
            w = rng.multinomial(len(y), np.full(len(y), 1.0 / len(y))).astype(float)
        bn = np.bincount(ck, weights=w, minlength=len(ks)); by = np.bincount(ck, weights=w * y, minlength=len(ks))
        br = np.divide(by, bn, out=np.zeros_like(by), where=bn > 0)
        a = sp(mean_p, br); b = sp(prate, br)
        v["mp"].append(a); v["pers"].append(b); v["diff"].append(a - b); v["exp"].append(sp(exp_d, by))
    out = {}
    for k2, arr in v.items():
        arr = np.array(arr); arr = arr[~np.isnan(arr)]
        out[k2] = dict(ci=[float(x) for x in np.percentile(arr, [2.5, 97.5])], mean=float(arr.mean()), frac_gt0=float((arr > 0).mean()))
    return out
OUT["boot_cluster_2027"] = boot(2027)
OUT["boot_cluster_99"] = boot(99)
OUT["boot_patent_2027"] = boot(2027, cluster=False)
OUT["n_clusters"] = G
# 큰 군집 영향: 27199 실현 15건의 권리자 분포
d27 = pd.DataFrame(dict(g=g, y=y, k=FS.ksic.astype(str).values))
top_own = d27[(d27.y == 1)].groupby("k").g.agg(lambda s: int(s.value_counts().iloc[0])).to_dict()
OUT["max_pos_per_owner_by_ksic"] = {k: v for k, v in top_own.items() if v >= 2}
OUT["sec"] = round(time.time() - t)
json.dump(OUT, open(os.path.join(W, "v1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
