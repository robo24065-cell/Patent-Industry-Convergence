# -*- coding: utf-8 -*-
"""사후 보조 S-IY: 코드 발급연도 층(≤2012 / ≥2013)별 N4·N7 동결 예측 재채점. 출력 cache/s11_out.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
import pandas as pd, numpy as np
def skill(a, b): return (a - b) / (1 - b)
def wauc(y, s, w=None):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    brk = np.r_[True, s[1:] != s[:-1]]; gid = np.cumsum(brk) - 1
    wp = np.bincount(gid, weights=w * y); wn = np.bincount(gid, weights=w * (~y)); cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
def boot(y, pm, pb, cl, n=1000, seed=0):
    rng = np.random.default_rng(seed); u, inv = np.unique(pd.Series(cl).astype(str).values, return_inverse=True); K = len(u); sk = []
    for _ in range(n):
        ww = rng.multinomial(K, np.full(K, 1.0 / K))[inv].astype(float)
        if (ww * y).sum() == 0 or (ww * (1 - y)).sum() == 0: continue
        sk.append(skill(wauc(y, pm, ww), wauc(y, pb, ww)))
    return [round(float(np.percentile(sk, 2.5)), 4), round(float(np.percentile(sk, 97.5)), 4)]
O = {}
for tr, f in [("N4", r"N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet"), ("N7", r"N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet")]:
    P = pd.read_parquet(os.path.join(MC, f)); P["applicantCd"] = P.applicantCd.astype(str); iy = P.applicantCd.str[1:5]
    for nm, m in [("le2012", iy <= "2012"), ("ge2013", iy >= "2013")]:
        d = P[m]; a1, a0 = wauc(d.y, d.p_champ), wauc(d.y, d.p_M0)
        O[f"{tr}_{nm}"] = dict(n=int(len(d)), pos=int(d.y.sum()), rate=round(float(d.y.mean()), 4), auc=round(a1, 4), auc_M0=round(a0, 4), skill=round(skill(a1, a0), 4),
                              ci=boot(d.y.values, d.p_champ.values, d.p_M0.values, d.bizno.values))
json.dump(O, open(guard_path(os.path.join(CACHE, "s11_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(O, ensure_ascii=False, indent=0))
