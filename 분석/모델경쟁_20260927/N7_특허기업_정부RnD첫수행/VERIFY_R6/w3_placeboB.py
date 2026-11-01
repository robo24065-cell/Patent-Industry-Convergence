# -*- coding: utf-8 -*-
r"""위약 B(반복): 학습 레이블(t0 2015–2020)을 섞은 뒤 동결 스펙(챔피언 ens 3시드, M0 xgb 3시드)으로 재학습 → 실제 시험 레이블로 AUC·skill.
--strata t0nps  : 층 = t0 × nps_present (R5 사전 지정 위약 B와 같은 층)
--strata global : 완전 무작위(층 없음)
--strata t0     : 원점 안에서만(구성 효과 제거)
반복 i의 섞기 시드 = 10000 + i (층 방식별 독립). 결과는 cache/placeboB_<strata>_<start>.csv (청크별, 9분 안)."""
import argparse
from w_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--strata", required=True, choices=["t0nps", "global", "t0"])
ap.add_argument("--start", type=int, required=True); ap.add_argument("--n", type=int, required=True); ap.add_argument("--panel", default="orig")
a = ap.parse_args()
D = pd.read_parquet(os.path.join(VC, f"final_panel_{a.panel}.parquet"))
tr = D[D.t0 <= 2020].reset_index(drop=True); te = D[D.t0 == 2022].reset_index(drop=True); y = te.y.values
grp = {"t0nps": tr.t0.astype(str) + "_" + tr.nps_present.astype(str), "global": pd.Series(np.zeros(len(tr))), "t0": tr.t0}[a.strata]
gidx = [ix.values for _, ix in pd.Series(np.arange(len(tr))).groupby(grp.values)]
Xc, X0, Tc, T0 = tr[FE].astype(float), tr[M0F].astype(float), te[FE].astype(float), te[M0F].astype(float)
fn = os.path.join(VC, f"placeboB_{a.strata}_{a.start:04d}.csv")
rows = []
t = time.time()
for i in range(a.start, a.start + a.n):
    rng = np.random.default_rng(10000 + i)
    ys = tr.y.values.copy()
    for ix in gidx:
        ys[ix] = rng.permutation(ys[ix])
    pc = predict(Xc, ys, Tc, model="ens", seeds=SEEDS)
    p0 = predict(X0, ys, T0, model="xgb", seeds=SEEDS)
    ac, a0 = auc_rs(y, pc), auc_rs(y, p0)
    rows.append(dict(rep=i, strata=a.strata, auc_champ=ac, auc_M0=a0, skill=(ac - a0) / (1 - a0)))
    pd.DataFrame(rows).to_csv(fn, index=False)
    if time.time() - t > 480:
        break
log("done", a.strata, a.start, len(rows), round(time.time() - t))
