# -*- coding: utf-8 -*-
r"""C10-T3 verify_4 v4b: 예측 파일에서 AUC·ΔAUC·CI·보존율을 독립 코드로 다시 계산(작업자 s6 미사용).
- AUC: sklearn roc_auc_score (작업자는 scipy rankdata 공식)
- CI 1: 원점 안 셀 복원추출(원점별 ΔAUC 평균), seed 4242, 2,000회 (작업자 seed 20261005, 1,000회)
- CI 2: 원점 단위 복원추출(9원점 재표집, 원점 평균 ΔAUC) — 원점 간 변동을 반영하는 더 보수적 구간
- 보존율 = (hyb − KR만) / (제공 − KR만), 원점 평균 AUC로, 전체 정밀도와 4자리 반올림 둘 다
읽기만: R7 results/pub_*_preds, c5_R10 t3_results/*_preds, C10-T3 res/t3/*_preds. 쓰기: verify_4/v4b_out.json"""
import os, json, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
R7 = os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "results"); C5 = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10", "t3_results")
F = {"full": os.path.join(R7, "pub_full_preds.parquet"), "drop_ros": os.path.join(R7, "pub_drop_ros_preds.parquet"),
     "kr_rebuild": os.path.join(R7, "pub_kr_rebuild_preds.parquet"),
     "txt_stored": os.path.join(C5, "hyb_txt_preds.parquet"), "free_stored": os.path.join(C5, "hyb_free_preds.parquet"),
     "txt_reg": os.path.join(W, "res", "t3", "hyb_txt_reg_preds.parquet"), "free_reg": os.path.join(W, "res", "t3", "hyb_free_reg_preds.parquet"),
     "txt_pub": os.path.join(W, "res", "t3", "hyb_txt_pub_preds.parquet"), "free_pub": os.path.join(W, "res", "t3", "hyb_free_pub_preds.parquet"),
     "repro_free": os.path.join(W, "res", "t3", "repro_hyb_free_preds.parquet")}
M = {k: pd.read_parquet(v).sort_values(["t0", "cell"]).reset_index(drop=True) for k, v in F.items()}
ORIG = list(range(2010, 2019))
base = M["drop_ros"]
OUT = {"rows": {k: int(len(v)) for k, v in M.items()}}
# 같은 평가 셀·같은 라벨인가
OUT["same_cells_labels"] = {k: bool((v.cell.values == base.cell.values).all() and (v.t0.values == base.t0.values).all() and (v.yb.values == base.yb.values).all()) for k, v in M.items()}
OUT["preds_identical_to_stored"] = {"free_reg_vs_free_stored": float(np.abs(M["free_reg"].p - M["free_stored"].p).max()),
                                    "txt_reg_vs_txt_stored": float(np.abs(M["txt_reg"].p - M["txt_stored"].p).max()),
                                    "repro_vs_free_stored": float(np.abs(M["repro_free"].p - M["free_stored"].p).max()),
                                    "free_reg_vs_free_pub": float(np.abs(M["free_reg"].p - M["free_pub"].p).max())}
idx = {v: np.where(base.t0.values == v)[0] for v in ORIG}
y = base.yb.values
A = {k: np.array([roc_auc_score(y[idx[v]], m.p.values[idx[v]]) for v in ORIG]) for k, m in M.items()}
OUT["auc_mean"] = {k: round(float(a.mean()), 5) for k, a in A.items()}
OUT["auc_by_origin"] = {k: [round(float(x), 4) for x in a] for k, a in A.items()}


def pair_auc(sp, sn):
    # P(양성 점수 > 음성 점수) + 0.5 P(같음), 정렬·searchsorted로 직접 계산(rank 공식·sklearn과 다른 구현)
    s = np.sort(sn); lt = np.searchsorted(s, sp, "left"); le = np.searchsorted(s, sp, "right")
    return float((lt + 0.5 * (le - lt)).sum() / (len(sp) * len(sn)))


def cell_boot(ka, kb, seed=4242, B=2000):
    rng = np.random.default_rng(seed); bm = np.zeros(B)
    pa, pb = M[ka].p.values, M[kb].p.values
    per = np.zeros((len(ORIG), B))
    for i, v in enumerate(ORIG):
        ii = idx[v]; yy, a_, b_ = y[ii], pa[ii], pb[ii]; n = len(ii)
        pos = np.where(yy == 1)[0]; neg = np.where(yy == 0)[0]
        for j in range(B):
            # 층화(사건/비사건 각각 복원추출) — 작업자 비층화와 다른 방식
            sp = rng.choice(pos, len(pos)); sn = rng.choice(neg, len(neg))
            per[i, j] = pair_auc(a_[sp], a_[sn]) - pair_auc(b_[sp], b_[sn])
    bm = per.mean(0)
    return [round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)]


def origin_boot(ka, kb, seed=4242, B=20000):
    rng = np.random.default_rng(seed); d = A[ka] - A[kb]
    s = rng.integers(0, 9, (B, 9)); bm = d[s].mean(1)
    return [round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)]


PAIRS = [("drop_ros", "txt_stored"), ("drop_ros", "free_stored"), ("drop_ros", "txt_reg"), ("drop_ros", "free_reg"),
         ("drop_ros", "txt_pub"), ("drop_ros", "free_pub"), ("drop_ros", "kr_rebuild"),
         ("txt_reg", "kr_rebuild"), ("free_reg", "kr_rebuild"), ("free_reg", "free_stored"), ("txt_reg", "txt_stored")]
OUT["pairs"] = {}
for a_, b_ in PAIRS:
    d = A[a_] - A[b_]
    OUT["pairs"][f"{a_}-{b_}"] = dict(dAUC=round(float(d.mean()), 4), dAUC_raw=float(d.mean()), wins=f"{int((d > 0).sum())}/9",
                                     ci_cell_strat=cell_boot(a_, b_), ci_origin=origin_boot(a_, b_), by_origin=[round(float(x), 4) for x in d])
    print(a_, b_, OUT["pairs"][f"{a_}-{b_}"], flush=True)
den = A["drop_ros"].mean() - A["kr_rebuild"].mean()
r4 = lambda k: round(float(A[k].mean()), 4)
den4 = r4("drop_ros") - r4("kr_rebuild")
OUT["retention"] = {k: dict(full=round(float((A[k].mean() - A["kr_rebuild"].mean()) / den), 4), from4digit=round((r4(k) - r4("kr_rebuild")) / den4, 4))
                    for k in ["txt_stored", "free_stored", "txt_reg", "free_reg", "txt_pub", "free_pub"]}
print(OUT["auc_mean"]); print(OUT["retention"]); print(OUT["same_cells_labels"], OUT["preds_identical_to_stored"])
json.dump(OUT, open(os.path.join(V, "v4b_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
