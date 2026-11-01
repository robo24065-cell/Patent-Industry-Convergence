# -*- coding: utf-8 -*-
"""C07-T3 verify_0 v02: 보조 분석 (i)·(iv) 독립 재현(SQL 경로, s04b 미사용). → v02_out.json"""
import os, json, sqlite3
import numpy as np, pandas as pd
from scipy.stats import rankdata
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
FS = pd.read_parquet(os.path.join(H, "future_scores_r6.parquet"))
B = pd.read_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"), columns=["rgst_no", "p_M0", "p_SB"])
D = pd.read_sql("select rgst_no, rgstCsDate d, rgstCsName cs, rgstCsReason rs from right where rgstCsName like '%무효%예고%' or rgstCsName like '%권리범위%예고%'", c)
D["d"] = D.d.astype(int)
win = D[(D.d >= 20250101) & (D.d <= 20260630)]
y = FS.rgst_no.isin(set(win.rgst_no)).values
pre = FS.rgst_no.isin(set(D[D.d < 20230101].rgst_no)).values
mid = FS.rgst_no.isin(set(D[(D.d >= 20230101) & (D.d <= 20241231)].rgst_no)).values & ~pre
hist = pre | mid
k = int(round(0.01 * len(FS)))
out = {}
for nm, s in (("champ", FS.p.values), ("SB", B.p_SB.values), ("M0", B.p_M0.values)):
    o = np.argsort(-s, kind="stable")[:k]; top = np.zeros(len(s), bool); top[o] = True
    out[nm] = dict(hits=int((y & top).sum()), hist_pre=int((y & top & pre).sum()), hist_2324=int((y & top & mid).sum()), new=int((y & top & ~hist).sum()))
oc = np.argsort(-FS.p.values, kind="stable")[:k]; os_ = np.argsort(-B.p_SB.values, kind="stable")[:k]
hc = set(FS.rgst_no.values[oc][y[oc]]); hs = set(FS.rgst_no.values[os_][y[os_]])
out["champ_SB_same_hit_set"] = hc == hs; out["overlap_top1_sets"] = len(set(FS.rgst_no.values[oc]) & set(FS.rgst_no.values[os_]))
out["pos"] = int(y.sum()); out["pos_hist_pre"] = int((y & pre).sum()); out["pos_hist_2324"] = int((y & mid).sum()); out["pos_new"] = int((y & ~hist).sum())
out["rule_hist_all"] = dict(n=int(hist.sum()), hits=int((y & hist).sum()), precision=float((y & hist).sum() / hist.sum()))
# 이력 양성 전부가 상위 1%에 드는가
out["hist_pos_in_champ_top1"] = int(len([r for r in FS.rgst_no.values[oc] if r in set(FS.rgst_no[y & hist])]))
# (iv) 유형
P = set(FS.rgst_no[y])
w = win[win.rgst_no.isin(P)]
inv = set(w[w.cs.str.contains("무효")].rgst_no); sco = set(w[w.cs.str.contains("권리범위")].rgst_no)
out["types"] = dict(inv_only=len(inv - sco), scope_only=len(sco - inv), both=len(inv & sco))
dec = pd.read_sql("select rgst_no, rgstCsDate d, rgstCsName cs, rgstCsReason rs from right where rgstCsName like '%확정%' and rgstCsDate >= '20250101'", c)
dec = dec[dec.rgst_no.isin(sco) & ~dec.cs.str.contains("기술평가|이의신청")]
rs = dec.groupby("rgst_no").rs.agg(lambda s: "|".join(s.fillna("").str.replace(" ", "")))
out["scope_decided"] = dict(n_scope_pos=len(sco), with_decision=int(len(rs)), contains_속함=int(rs.str.contains("속함").sum()), contains_무속=int(rs.str.contains("무속").sum()))
json.dump(out, open(os.path.join(V, "v02_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
