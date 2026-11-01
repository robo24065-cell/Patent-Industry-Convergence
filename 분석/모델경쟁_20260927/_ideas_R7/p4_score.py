# -*- coding: utf-8 -*-
"""IPO 레이더 파일럿 4단계: 채점(원점별 AUC·skill·상위 k), 채택 사슬, 필수성 시험, 군집 부트스트랩.
출력: experiments_ipo.csv, pilot_by_origin_ipo.csv, essential_ipo.csv, p4_summary.json"""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R7 = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(R7, "cache")
P = pd.read_parquet(os.path.join(C, "ipo_panel.parquet")); D = P[P.t0 <= 2021].reset_index(drop=True)
VS = [2017, 2018, 2019, 2020, 2021]
names = [f[6:-8] for f in os.listdir(C) if f.startswith("preds_") and f.endswith(".parquet")]
PR = {n: pd.read_parquet(os.path.join(C, f"preds_{n}.parquet")).set_index("idx").p for n in names}
rows = []
for n in names:
    for v in VS:
        te = D[D.t0 == v]; p = PR[n].reindex(te.index)
        k1 = int(np.ceil(0.01 * len(te))); o = np.argsort(-p.values)
        rows.append({"exp": n, "t0": v, "n": len(te), "pos": int(te.y.sum()), "auc": roc_auc_score(te.y, p),
                     "top1pct_hits": int(te.y.values[o[:k1]].sum()), "k1": k1, "top100_hits": int(te.y.values[o[:100]].sum())})
O = pd.DataFrame(rows)
A = O.pivot(index="t0", columns="exp", values="auc")
def skill(a, b): return (A[a] - A[b]) / (1 - A[b])
O["skill_vs_M0"] = O.apply(lambda r: (r.auc - A.loc[r.t0, "E0"]) / (1 - A.loc[r.t0, "E0"]), axis=1)
O["skill_vs_SB"] = O.apply(lambda r: (r.auc - A.loc[r.t0, "E2"]) / (1 - A.loc[r.t0, "E2"]), axis=1)
O.to_csv(os.path.join(R7, "pilot_by_origin_ipo.csv"), index=False, encoding="utf-8-sig")
S = O.groupby("exp").agg(auc=("auc", "mean"), skill_vs_M0=("skill_vs_M0", "mean"), skill_vs_SB=("skill_vs_SB", "mean"),
                         top1pct_hits=("top1pct_hits", "sum"), top100_hits=("top100_hits", "sum"), pos=("pos", "sum")).round(4)
# 채택 사슬 (E0 → E1 → E2 → C1(+PK) → +PG → +PT)
chain = []; champ = "E0"
def cmp(ch, cand):
    s = skill(cand, ch); return float(s.mean()), int((s > 0).sum())
for cand in ["E1", "E2", "C1"]:
    m, w = cmp(champ, cand); ok = m >= 0.003 and w >= 3
    chain.append({"from": champ, "cand": cand, "mean_skill": round(m, 4), "wins": w, "decision": "ADOPT" if ok else "ROLLBACK"})
    if ok: champ = cand
# +PG, +PT: 챔피언 상태에 따라 사전 계산된 조합 선택
cur = champ
for b in ["PG", "PT"]:
    cand = {("C1", "PG"): "X3", ("C1", "PT"): "X2", ("X3", "PT"): "FULL", ("X2", "PG"): "FULL",
            ("E2", "PG"): "C2", ("E2", "PT"): "C3a", ("C2", "PT"): "C3b"}.get((cur, b))
    if cand is None:
        chain.append({"from": cur, "cand": f"+{b}", "decision": "미실행(사전 계산 조합 없음)"}); continue
    m, w = cmp(cur, cand); ok = m >= 0.003 and w >= 3
    chain.append({"from": cur, "cand": cand, "mean_skill": round(m, 4), "wins": w, "decision": "ADOPT" if ok else "ROLLBACK"})
    if ok: cur = cand
# 군집 부트스트랩: FULL vs 비교 대상 Δskill(비교 대상 기준) 원점 평균
rng = np.random.default_rng(0)
GRP = {}
for v in VS:
    te = D[D.t0 == v]; codes = pd.factorize(te.bizno.astype(str))[0]
    order = np.argsort(codes, kind='stable'); cuts = np.flatnonzero(np.diff(codes[order])) + 1
    GRP[v] = (te, np.split(order, cuts))
def boot(a, b, R=200):
    res = []
    for _ in range(R):
        ds = []
        for v in VS:
            te, cls = GRP[v]
            pick = rng.integers(0, len(cls), len(cls))
            ii = np.concatenate([cls[j] for j in pick])
            y = te.y.values[ii]
            if y.sum() == 0: continue
            pa = PR[a].reindex(te.index).values[ii]; pb = PR[b].reindex(te.index).values[ii]
            aa, ab = roc_auc_score(y, pa), roc_auc_score(y, pb)
            ds.append((aa - ab) / (1 - ab))
        res.append(np.mean(ds))
    return np.percentile(res, [2.5, 97.5]).round(4).tolist()
ess = []
for a, b, lab in [("FULL", "E2", "제공 피처 전부 제거(FULL vs SB)"), ("FULL", "X1", "KSIC 라벨 묶음 제거"), ("FULL", "X2", "5개국 패밀리 제거"),
                  ("FULL", "X3", "제목·IPC 묶음 제거"), ("FULL", "X5", "대체: 제공 KSIC → 공식 IPC-KSIC 연계표"), ("FULL", "X6", "5개국 → KR만"),
                  ("C1", "C1alt", "SB 위 KSIC 묶음: 제공 vs 연계표"), ("C1", "E2", "SB+PK vs SB"), ("X7", "E2", "SB+연계표 산업 vs SB"), ("C2", "E2", "SB+5개국 패밀리 vs SB"), ("C3b", "E2", "SB+5개국+IPC vs SB"),
                  ("L_FULL", "L_SB", "로지스틱 FULL vs SB"), ("L_FULL", "L_X5", "로지스틱 대체 시험")]:
    if a not in PR or b not in PR: continue
    s = (A[a] - A[b]) / (1 - A[b])
    ess.append({"비교": lab, "A": a, "B": b, "AUC_A": round(A[a].mean(), 4), "AUC_B": round(A[b].mean(), 4),
                "skill_A_vs_B": round(float(s.mean()), 4), "A_우세_원점": f"{int((s > 0).sum())}/5", "CI95": boot(a, b),
                "top1pct_A": int(O[O.exp == a].top1pct_hits.sum()), "top1pct_B": int(O[O.exp == b].top1pct_hits.sum()),
                "top100_A": int(O[O.exp == a].top100_hits.sum()), "top100_B": int(O[O.exp == b].top100_hits.sum())})
E = pd.DataFrame(ess); E.to_csv(os.path.join(R7, "essential_ipo.csv"), index=False, encoding="utf-8-sig")
S.to_csv(os.path.join(R7, "experiments_ipo.csv"), encoding="utf-8-sig")
json.dump({"chain": chain, "champion": cur, "by_exp": S.reset_index().to_dict("records")}, open(os.path.join(R7, "p4_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(S.to_string()); print(pd.DataFrame(chain).to_string()); print(E.to_string())
print(A.round(4).to_string())
