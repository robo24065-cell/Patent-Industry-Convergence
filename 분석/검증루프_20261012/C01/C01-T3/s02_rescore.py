# -*- coding: utf-8 -*-
"""C01-T3 s02: 동결 점수(final_preds_r6.parquet) 재채점 — 레이블 변형별 AUC·상위1% 포착, 군집 부트스트랩 CI.
사용: python s02_rescore.py <scheme: point|owner|bundle|bundle30> [B]
재학습 없음. 공식 숫자 불변. 출력: s02_<scheme>.json"""
import os, sys, json, time
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
scheme = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; PART = int(sys.argv[3]) if len(sys.argv) > 3 else 0
t_start = time.time()
P = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no", "y", "p_champ", "p_M0", "p_SB"])
CL = pd.read_parquet(os.path.join(OUT, "_clusters_internal.parquet"))
P = P.merge(CL, on="rgst_no", how="left"); assert P.g_owner.notna().all()
L = pd.read_csv(os.path.join(OUT, "labels_rebuilt.csv"), comment="#", dtype={"rgst_no": str, "ksic": str})
assert set(L.rgst_no) == set(P.rgst_no[P.y == 1])
Li = L.set_index("rgst_no")
n = len(P); r = P.rgst_no.values
def m(col_bool):
    return P.rgst_no.map(col_bool.astype(float)).fillna(0).astype(bool).values
inv = m(Li.n_inv_win > 0); sco = m(Li.n_sco_win > 0)
act_det = m(Li.sco_dir.isin(["active", "mixed"])); pas_det = m(Li.sco_dir == "passive")
pact = P.rgst_no.map(Li.p_active).fillna(0).values
posall = P.y.values == 1
# 레이블: (y, eval_mask)
LAB = {}
LAB["L_all"] = (posall, np.ones(n, bool))
LAB["L_inv"] = (inv, ~(posall & ~inv))                       # 권리범위만 양성은 평가에서 제외(사전 등록 주 분석)
LAB["L_inv_negs"] = (inv, np.ones(n, bool))                    # 민감도: 권리범위만 양성을 음성으로
LAB["L_scope"] = (sco, ~(posall & ~sco))
y3 = inv | (sco & ~act_det & (pact < 0.5))
LAB["L_3p"] = (y3, ~(posall & ~y3))                            # 사전 등록
y3s = inv | pas_det
LAB["L_3p_strict"] = (y3s, ~(posall & ~y3s))                   # 사후 추가: 무효 + 소극 판정만
newp = P.rgst_no.map(Li.prior_disp_pre2023).fillna(1).values == 0
LAB["L_all_no_prior"] = (posall & newp, ~(posall & ~newp))     # 참고: 2023 전 분쟁 이력 없는 양성만

MODELS = ["p_champ", "p_M0", "p_SB"]
S = {k: P[k].values.astype(float) for k in MODELS}
INV_ = {}; ORD = {}
for k in MODELS:
    u, iv = np.unique(S[k], return_inverse=True); INV_[k] = (iv, len(u))
    ORD[k] = np.argsort(-S[k], kind="stable")

def wauc(y, k, w):
    iv, G = INV_[k]
    pos = np.bincount(iv, weights=w * y, minlength=G); neg = np.bincount(iv, weights=w * (1 - y), minlength=G)
    cn = np.cumsum(neg) - neg
    return float((pos * (cn + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))

def wcap(y, k, w, frac=0.01):
    o = ORD[k]; wo = w[o]; cw = np.cumsum(wo); thr = frac * cw[-1]
    top = (cw - wo) < thr
    return float((wo * y[o] * top).sum() / (w * y).sum())

def metrics(w):
    res = {}
    for lab, (y, msk) in LAB.items():
        yy = y.astype(float); ww = w * msk
        if (ww * yy).sum() == 0:
            continue
        a = {k: wauc(yy, k, ww) for k in MODELS}; c = {k: wcap(yy, k, ww) for k in MODELS}
        res[lab] = dict(auc_champ=a["p_champ"], auc_M0=a["p_M0"], auc_SB=a["p_SB"],
                        skill_M0=(a["p_champ"] - a["p_M0"]) / (1 - a["p_M0"]), skill_SB=(a["p_champ"] - a["p_SB"]) / (1 - a["p_SB"]),
                        cap_champ=c["p_champ"], cap_M0=c["p_M0"], cap_SB=c["p_SB"],
                        dcap_M0=c["p_champ"] - c["p_M0"], dcap_SB=c["p_champ"] - c["p_SB"])
    return res

if scheme == "point":
    w = np.ones(n)
    res = metrics(w)
    for lab, (y, msk) in LAB.items():
        res[lab]["n_eval"] = int(msk.sum()); res[lab]["pos"] = int((y & msk).sum())
        k = int(round(0.01 * msk.sum()))
        for mm in MODELS:
            o = ORD[mm][msk[ORD[mm]]]
            res[lab][f"hits1pct_{mm[2:]}"] = int(y[o[:k]].sum()); res[lab][f"hit100_{mm[2:]}"] = int(y[o[:100]].sum())
        res[lab]["k1pct"] = k
    json.dump(res, open(os.path.join(OUT, "s02_point.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res["L_all"], indent=1)); sys.exit()

key = {"owner": "g_owner", "bundle": "g_bundle", "bundle30": "g_bundle30"}[scheme]
codes, uniq = pd.factorize(P[key].values); G = len(uniq)
rng = np.random.default_rng(20261012 + ["owner", "bundle", "bundle30"].index(scheme) + 1000 * PART)
draws = []
for b in range(B):
    w = rng.multinomial(G, np.full(G, 1.0 / G))[codes].astype(float)
    draws.append(metrics(w))
    if time.time() - t_start > 500:
        print("time stop at", b + 1); break
pd.to_pickle(draws, os.path.join(OUT, f"_draws_{scheme}_p{PART}.pkl"))
if PART > 0 or len(sys.argv) > 4:
    draws = []
    for p_ in range(int(sys.argv[4]) if len(sys.argv) > 4 else PART + 1):
        fp = os.path.join(OUT, f"_draws_{scheme}_p{p_}.pkl")
        if os.path.exists(fp):
            draws += pd.read_pickle(fp)
out = {"scheme": scheme, "B_done": len(draws), "n_clusters": G}
for lab in LAB:
    vals = [d[lab] for d in draws if lab in d]
    out[lab] = {mt: [float(np.percentile([v[mt] for v in vals], 2.5)), float(np.percentile([v[mt] for v in vals], 97.5))] for mt in vals[0]}
    out[lab]["share_dcap_SB_le0"] = float(np.mean([v["dcap_SB"] <= 0 for v in vals]))
    out[lab]["share_skill_SB_le0"] = float(np.mean([v["skill_SB"] <= 0 for v in vals]))
    out[lab]["B_valid"] = len(vals)
json.dump(out, open(os.path.join(OUT, f"s02_{scheme}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done", scheme, len(draws), round(time.time() - t_start))

