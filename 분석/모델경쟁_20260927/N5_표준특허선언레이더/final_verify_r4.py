# -*- coding: utf-8 -*-
"""🔒 최종시험(t0=2022) 독립 재계산 대조 — n5lib를 쓰지 않는다.
1) 원자료(kr_patents, trade_etsi_sep_link)에서 단위 집합·레이블을 따로 만들어 final_preds_r4.parquet와 대조
2) AUC를 순위합(Mann-Whitney) 공식으로 다시 계산
3) 출원인 군집 부트스트랩을 다른 구현·다른 시드로 다시 계산
4) 상위 100 적중, 출원인 집중도(사후 INFO)
모형은 다시 학습하지 않는다(동결 예측 대조). 결과 → final_verify_r4.json"""
import os, re, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, r"분석\모델경쟁_20260927\N5_표준특허선언레이더")
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
T0 = 2022
END, OEND = 20221231, 20241231

P = pd.read_parquet(os.path.join(OUT, "final_preds_r4.parquet"))
R = json.load(open(os.path.join(OUT, "final_result_r4.json"), encoding="utf-8"))

# ---- 1) 단위·레이블 독립 재구성
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"),
                    columns=["출원번호", "출원연도", "공개일자", "등록일자", "세세분류코드"])
k = k[k["세세분류코드"].astype(str) == "26429"].copy()
k["appl_no"] = k["출원번호"].astype(str)
def d8(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9]", "", regex=True).str[:8], errors="coerce")
k["pubd"] = d8(k["공개일자"]); k["rgdd"] = d8(k["등록일자"])
k["kn"] = np.fmin(k.pubd.values, k.rgdd.values)
L = pd.read_parquet(os.path.join(EXT, "trade_etsi_sep_link.parquet"), columns=["cc", "appl_no", "first_refl"])
L = L[L.cc == "KR"]
first = L.groupby(L.appl_no.astype(str)).first_refl.min()
k["rf"] = k.appl_no.map(first)
k["rfn"] = pd.to_numeric(k.rf.dt.strftime("%Y%m%d"), errors="coerce")
yr = pd.to_numeric(k["출원연도"], errors="coerce")
unit = k[(yr >= T0 - 5) & (yr <= T0) & (k.kn <= END) & ~(k.rfn <= END)].copy()
unit["y2"] = ((unit.rfn > END) & (unit.rfn <= OEND)).astype(int)
set_ind = set(unit.appl_no); set_run = set(P.appl_no.astype(str))
ym = P.assign(appl_no=P.appl_no.astype(str)).merge(unit[["appl_no", "y2"]], on="appl_no", how="left")
chk_units = dict(n_indep=len(unit), n_run=len(P), pos_indep=int(unit.y2.sum()), pos_run=int(P.y.sum()),
                 only_indep=len(set_ind - set_run), only_run=len(set_run - set_ind),
                 label_mismatch=int((ym.y2 != ym.y).sum()))

# ---- 2) AUC 순위합 공식
def auc_mw(y, p):
    y = np.asarray(y); r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
y = P.y.values
A2 = {c[2:]: float(auc_mw(y, P[c].values)) for c in P.columns if c.startswith("p_")}
sk = lambda a, b: (a - b) / (1 - b)
chk_auc = {kk: dict(indep=A2[kk], run=R["auc"][kk], diff=A2[kk] - R["auc"][kk]) for kk in A2}
skill2 = sk(A2["champ"], A2["M0"])

# ---- 3) 출원인 군집 부트스트랩(가중치 방식: 군집 복원추출 횟수를 가중치로)
def auc_w(y, p, w):
    o = np.argsort(p, kind="mergesort"); y, p, w = y[o], p[o], w[o]
    # 동점 처리: 같은 점수 묶음은 절반 기여
    _, start = np.unique(p, return_index=True); ends = np.r_[start[1:], len(p)]
    wp = w * y; wn = w * (1 - y)
    pp = np.add.reduceat(wp, start); nn = np.add.reduceat(wn, start)
    below = np.cumsum(nn) - nn
    s = (pp * (below + 0.5 * nn)).sum()
    return s / (wp.sum() * wn.sum())
g = pd.factorize(P.app.fillna("").values)[0]; G = g.max() + 1
rng = np.random.default_rng(20260929)
pc, p0 = P.p_champ.values, P.p_M0.values
bs = []
for _ in range(1000):
    cnt = np.bincount(rng.integers(0, G, G), minlength=G)
    w = cnt[g].astype(float)
    if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
        continue
    bs.append(sk(auc_w(y, pc, w), auc_w(y, p0, w)))
ci2 = np.percentile(bs, [2.5, 97.5]).tolist()
# (사후 INFO) 더 강한 기준선: 로지스틱 M0가 XGB M0보다 AUC가 높으면 그것 대비 챔피언 skill도 본다(기준선 강화는 허용, 보고용)
pl = P.p_M0_logit.values; rng2 = np.random.default_rng(20260930); bs3 = []
for _ in range(1000):
    cnt = np.bincount(rng2.integers(0, G, G), minlength=G); w = cnt[g].astype(float)
    if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
        continue
    bs3.append(sk(auc_w(y, pc, w), auc_w(y, pl, w)))
strong = dict(skill_champ_vs_M0logit=sk(A2["champ"], A2["M0_logit"]), ci95=np.percentile(bs3, [2.5, 97.5]).tolist())

# ---- 4) 상위 100, 출원인 집중도(INFO)
top = lambda p, kk: int(y[np.argsort(-p)[:kk]].sum())
pos_by_app = P[P.y == 1].app.value_counts()
top5 = pos_by_app.head(5)
mask = ~P.app.isin(top5.index)
info = dict(pos_top5_apps_share=float(top5.sum() / y.sum()), n_apps=int(G), n_pos_apps=int(len(pos_by_app)),
            excl_top5_pos_apps=dict(n=int(mask.sum()), pos=int(P.y[mask].sum()),
                                    auc_champ=float(auc_mw(P.y[mask].values, pc[mask.values])),
                                    auc_M0=float(auc_mw(P.y[mask].values, p0[mask.values]))))
info["excl_top5_pos_apps"]["skill"] = sk(info["excl_top5_pos_apps"]["auc_champ"], info["excl_top5_pos_apps"]["auc_M0"])
out = dict(units=chk_units, auc=chk_auc, skill_indep=skill2, skill_run=R["skill_champ_vs_M0"],
           ci95_indep_weighted_boot=ci2, ci95_run=R["ci95_cluster_app"], n_boot=len(bs),
           top100_indep={kk: top(P["p_" + kk].values, 100) for kk in ["champ", "M0", "M1_pilot"]}, top100_run=R["top100_hits"],
           info_post=info, info_strong_baseline=strong)
json.dump(out, open(os.path.join(OUT, "final_verify_r4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
