# -*- coding: utf-8 -*-
"""C01-T3 verify_0: '권리범위 단독 레이블 신호 약함' 반박 시도. 레이블을 sqlite(mode=ro)에서 독립 재추출, 동결 점수 재채점.
배치 효과(placement) 방식: 음성 집합이 L_inv/L_scope에서 동일(공식 y==0)하므로 양성 placement 평균 = AUC."""
import os, sqlite3, json, numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(BASE, r"분석\모델경쟁_20260927\N8_특허분쟁표적레이더")
c = sqlite3.connect(f"file:{os.path.join(BASE, r'수집파이프라인\registry_full.sqlite')}?mode=ro", uri=True)
INV, SCO = "무효(심판)예고등록", "권리범위(심판)예고등록"
rt = pd.read_sql(f"select rgst_no, rgstCsDate, rgstCsName from right where rgstCsName in ('{INV}','{SCO}')", c)
rt["d"] = pd.to_numeric(rt.rgstCsDate, errors="coerce")
w = rt[(rt.d >= 20230101) & (rt.d <= 20241231)]
print("events in window", len(w), w.rgstCsName.value_counts().to_dict())
P = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no", "y", "p_champ", "p_M0", "p_SB"])
inv_set = set(w[w.rgstCsName == INV].rgst_no) & set(P.rgst_no); sco_set = set(w[w.rgstCsName == SCO].rgst_no) & set(P.rgst_no)
P["inv"] = P.rgst_no.isin(inv_set); P["sco"] = P.rgst_no.isin(sco_set)
print("pos inv", P.inv.sum(), "sco", P.sco.sum(), "both", (P.inv & P.sco).sum(), "union", (P.inv | P.sco).sum(), "y", P.y.sum(),
      "union==y", bool(((P.inv | P.sco).astype(int) == P.y).all()))
neg = P[P.y == 0]; pos = P[P.y == 1].copy()
M = ["p_champ", "p_M0", "p_SB"]
for m in M:
    s = np.sort(neg[m].values)
    lo = np.searchsorted(s, pos[m].values, "left"); hi = np.searchsorted(s, pos[m].values, "right")
    pos["pl_" + m] = (lo + 0.5 * (hi - lo)) / len(s)
    # top1% threshold over full eval set (official k=3178 on all 317,840)
    k = int(round(0.01 * len(P))); thr = np.sort(P[m].values)[::-1][k - 1]
    pos["top_" + m] = pos[m].values >= thr
CL = pd.read_parquet(os.path.join(os.path.dirname(OUT), "_clusters_internal.parquet"))
pos = pos.merge(CL, on="rgst_no", how="left")
def stats(df):
    a = {m: df["pl_" + m].mean() for m in M}; t = {m: df["top_" + m].mean() for m in M}
    return dict(n=len(df), auc_c=a["p_champ"], auc_M0=a["p_M0"], auc_SB=a["p_SB"],
                sk_M0=(a["p_champ"] - a["p_M0"]) / (1 - a["p_M0"]), sk_SB=(a["p_champ"] - a["p_SB"]) / (1 - a["p_SB"]),
                cap_c=t["p_champ"], cap_M0=t["p_M0"], cap_SB=t["p_SB"])
groups = {"L_inv": pos.inv, "L_scope": pos.sco, "scope_only": pos.sco & ~pos.inv, "inv_only": pos.inv & ~pos.sco, "both": pos.inv & pos.sco, "L_all": pos.inv | pos.sco}
res = {g: stats(pos[mk]) for g, mk in groups.items()}
for g, r in res.items():
    print(g, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
# 군집 부트스트랩 (양성만 재표집; 음성 31.7만은 고정 — 분산 기여 무시 가능)
out = {"point": res}
for key in ["g_owner", "g_bundle", "rgst_no"]:
    rng = np.random.default_rng(777)
    cl = pos[key].values; u, codes = np.unique(cl, return_inverse=True); G = len(u)
    D = {g: [] for g in groups}; DIFF = {"auc_inv_minus_scope": [], "skM0_inv_minus_scope": [], "auc_inv_only_minus_scope_only": [], "skM0_invonly_minus_scopeonly": []}
    for b in range(4000):
        wt = rng.multinomial(G, np.full(G, 1 / G))[codes].astype(float)
        rr = {}
        for g, mk in groups.items():
            ww = wt * mk.values
            if ww.sum() == 0: rr[g] = None; continue
            a = {m: (ww * pos["pl_" + m].values).sum() / ww.sum() for m in M}
            rr[g] = ((a["p_champ"] - a["p_M0"]) / (1 - a["p_M0"]), (a["p_champ"] - a["p_SB"]) / (1 - a["p_SB"]), a["p_champ"])
            D[g].append(rr[g])
        DIFF["auc_inv_minus_scope"].append(rr["L_inv"][2] - rr["L_scope"][2]); DIFF["skM0_inv_minus_scope"].append(rr["L_inv"][0] - rr["L_scope"][0])
        if rr["inv_only"] and rr["scope_only"]:
            DIFF["auc_inv_only_minus_scope_only"].append(rr["inv_only"][2] - rr["scope_only"][2]); DIFF["skM0_invonly_minus_scopeonly"].append(rr["inv_only"][0] - rr["scope_only"][0])
    o = {"G": G}
    for g in groups:
        arr = np.array(D[g]); o[g] = dict(skM0=np.percentile(arr[:, 0], [2.5, 97.5]).round(4).tolist(), skSB=np.percentile(arr[:, 1], [2.5, 97.5]).round(4).tolist(),
                                       auc=np.percentile(arr[:, 2], [2.5, 97.5]).round(4).tolist(), p_skM0_le0=float((arr[:, 0] <= 0).mean()), p_skSB_le0=float((arr[:, 1] <= 0).mean()))
    for k, v in DIFF.items():
        v = np.array(v); o[k] = dict(ci=np.percentile(v, [2.5, 97.5]).round(4).tolist(), p_le0=float((v <= 0).mean()))
    out[key] = o
    print(key, json.dumps(o, ensure_ascii=False))
json.dump(out, open(os.path.join(OUT, "v01_scope.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
