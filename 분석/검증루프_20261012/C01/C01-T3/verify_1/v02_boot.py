# -*- coding: utf-8 -*-
"""verify_1 v02: L_all_no_prior skill vs SB / 상위1% 차이 군집 부트스트랩 재계산(독립 구현, g_bundle 키만 재사용). B=1000."""
import os, sys, sqlite3, json, time, numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__)); T = os.path.dirname(OUT)
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
t0 = time.time()
P = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no", "y", "p_champ", "p_SB"])
P["rgst_no"] = P.rgst_no.astype(str)
CL = pd.read_parquet(os.path.join(T, "_clusters_internal.parquet")); CL["rgst_no"] = CL.rgst_no.astype(str)
P = P.merge(CL[["rgst_no", "g_bundle"]], on="rgst_no", how="left")
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
h = pd.read_sql("select distinct rgst_no from right where (rgstCsName like '%무효%예고%' or rgstCsName like '%권리범위%예고%') and rgstCsDate < '20230101'", c)
pr = P.rgst_no.isin(set(h.rgst_no.astype(str))).values
y0 = P.y.values == 1
keep = ~(y0 & pr); P = P[keep].reset_index(drop=True); y = (y0 & ~pr)[keep].astype(float)
assert y.sum() == 78
gcol = P.g_bundle.fillna(P.rgst_no).values
codes, uniq = pd.factorize(gcol); G = len(uniq)
def prep(s):
    o = np.argsort(s, kind="stable"); u, inv = np.unique(s, return_inverse=True); return inv, len(u), np.argsort(-s, kind="stable")
PR = {m: prep(P[m].values) for m in ["p_champ", "p_SB"]}
def auc(w, m):
    inv, U, _ = PR[m]
    pos = np.bincount(inv, w * y, U); neg = np.bincount(inv, w * (1 - y), U)
    below = np.cumsum(neg) - neg
    return (pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum())
def cap(w, m):
    o = PR[m][2]; wo = w[o]; cw = np.cumsum(wo); top = (cw - wo) < 0.01 * cw[-1]
    return (wo * y[o] * top).sum() / (w * y).sum()
rng = np.random.default_rng(777)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
sk, dc = [], []
for b in range(B):
    w = rng.multinomial(G, np.full(G, 1 / G))[codes].astype(float)
    if (w * y).sum() == 0: continue
    ac, as_ = auc(w, "p_champ"), auc(w, "p_SB")
    sk.append((ac - as_) / (1 - as_)); dc.append(cap(w, "p_champ") - cap(w, "p_SB"))
    if time.time() - t0 > 480: break
w1 = np.ones(len(P)); a1, a2 = auc(w1, "p_champ"), auc(w1, "p_SB")
r = dict(G=G, B=len(sk), point_skill=(a1 - a2) / (1 - a2), point_dcap=cap(w1, "p_champ") - cap(w1, "p_SB"),
         skill_ci=[float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))], share_skill_le0=float(np.mean(np.array(sk) <= 0)),
         dcap_ci=[float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))], share_dcap_le0=float(np.mean(np.array(dc) <= 0)),
         sec=round(time.time() - t0))
json.dump(r, open(os.path.join(OUT, "v02_out.json"), "w", encoding="utf-8"), indent=1, default=float); print(r)
