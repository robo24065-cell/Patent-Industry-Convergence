# -*- coding: utf-8 -*-
"""verify_1 v01: L_all_no_prior 독립 재계산 (s01/s02 산출물·labels_rebuilt 미사용, sqlite mode=ro)."""
import os, sqlite3, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT = os.path.dirname(os.path.abspath(__file__))
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
P = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no", "y", "p_champ", "p_M0", "p_SB"])
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
names = pd.read_sql("select rgstCsName, count(*) n from right where rgstCsName like '%무효%예고%' or rgstCsName like '%권리범위%예고%' group by 1", c)
h = pd.read_sql("select rgst_no, rgstCsName, rgstCsDate from right where (rgstCsName like '%무효%예고%' or rgstCsName like '%권리범위%예고%')", c)
h["d"] = pd.to_numeric(h.rgstCsDate.astype(str).str.replace("-", "").str[:8], errors="coerce")
res = {"name_variants": names.to_dict("records"), "null_dates": int(h.d.isna().sum())}
pre = set(h[(h.d < 20230101)].rgst_no.astype(str))
win = h[(h.d >= 20230101) & (h.d <= 20241231)]
P["rgst_no"] = P.rgst_no.astype(str)
P["prior"] = P.rgst_no.isin(pre)
ywin = P.rgst_no.isin(set(win.rgst_no.astype(str)))
res["pop"] = len(P); res["y_sum"] = int(P.y.sum()); res["y_eq_rebuilt"] = bool((ywin.values == (P.y.values == 1)).all())
res["prior_pop"] = int(P.prior.sum()); res["prior_pos"] = int((P.prior & (P.y == 1)).sum())
M = ["p_champ", "p_M0", "p_SB"]
def evalset(mask, y, tag):
    yy = y[mask]; o = {}
    o["n"] = int(mask.sum()); o["pos"] = int(yy.sum())
    k = int(round(0.01 * mask.sum())); o["k"] = k
    for m in M:
        s = P[m].values[mask]
        o["auc_" + m] = float(roc_auc_score(yy, s))
        # 동점 처리: 안정 정렬(원순서), 그리고 동점 경계 여부 확인
        idx = np.argsort(-s, kind="stable")
        o["top1_" + m] = int(yy[idx[:k]].sum()); o["top100_" + m] = int(yy[idx[:100]].sum())
        thr = s[idx[k - 1]]; o["ties_at_k_" + m] = int((s == thr).sum())
    for b in ["p_M0", "p_SB"]:
        o["skill_vs_" + b] = (o["auc_p_champ"] - o["auc_" + b]) / (1 - o["auc_" + b])
    o["cap1"] = {m: o["top1_" + m] / o["pos"] for m in M}
    res[tag] = o
y = (P.y.values == 1); pr = P.prior.values
evalset(np.ones(len(P), bool), y, "L_all")
evalset(~(y & pr), y & ~pr, "no_prior_keep_prior_negs")          # 원 발견 정의
evalset(~pr, y, "no_prior_drop_all_prior")                       # FINAL_REPORT 2.2식(이력 특허 전부 제외, 무효·권리범위만 기준)
# 원 상위 1% 컷(전체 3178)에서 이력 없는 양성 포착
idx = np.argsort(-P.p_champ.values, kind="stable")[:3178]
res["orig_cut_newpos_hits"] = int((y & ~pr)[idx].sum()); res["orig_cut_prior_hits"] = int((y & pr)[idx].sum())
idxs = np.argsort(-P.p_SB.values, kind="stable")[:3178]
res["orig_cut_newpos_hits_SB"] = int((y & ~pr)[idxs].sum())
# skill 분해: 이력 양성 30건의 기여(음성 대비 순위 분위)
neg = P.loc[~y]
def q(m, s):
    v = neg[m].values; return float(((v < s).sum() + 0.5 * (v == s).sum()) / len(v))
pp = P[y & pr]
rc = np.array([q("p_champ", s) for s in pp.p_champ]); rs = np.array([q("p_SB", s) for s in pp.p_SB])
res["prior_pos_mean_rank_champ"] = float(rc.mean()); res["prior_pos_mean_rank_SB"] = float(rs.mean())
res["prior_pos_num_contrib"] = float((rc - rs).sum()); res["prior_pos_den_contrib"] = float((1 - rs).sum())
json.dump(res, open(os.path.join(OUT, "v01_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
