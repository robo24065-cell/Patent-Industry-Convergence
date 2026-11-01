# -*- coding: utf-8 -*-
"""(1) 결과 A: 5개 개발 원점(2008–2012) 평균 ΔAUC의 법인 군집 부트스트랩 CI(원점 간 같은 법인 가중을 공유).
사용: python a_pool.py P50:CW27 P27:CW27 ... → res/A_pool_<a>_<b>.json
"""
from nc_common import *

RES = os.path.join(OUT, "res")
C = [2008, 2009, 2010, 2011, 2012]
BOOT = 200


def prep(y, s):
    o = np.argsort(s, kind="mergesort"); _, idx = np.unique(s[o], return_index=True)
    return o, idx, (y[o] == 1)


def fast(pr, w):
    o, idx, yp = pr; w = w[o]
    wp = np.add.reduceat(w * yp, idx); wn = np.add.reduceat(w * ~yp, idx)
    cn = np.cumsum(wn) - wn
    return (wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


pairs = [tuple(x.split(":")) for x in sys.argv[1:]]
data = {}
firms = set()
for c in C:
    d = pd.read_parquet(os.path.join(RES, f"A_pred_c{c}.parquet"))
    data[c] = d; firms |= set(d.applicantCd.unique())
firms = np.array(sorted(firms)); fidx = {f: i for i, f in enumerate(firms)}
for a, b in pairs:
    P1, P2, INV, D0 = {}, {}, {}, []
    for c, d in data.items():
        y = d.y.values.astype(np.int8)
        P1[c] = prep(y, d[a].values.astype(np.float64)); P2[c] = prep(y, d[b].values.astype(np.float64))
        INV[c] = d.applicantCd.map(fidx).values
        D0.append(auc_w(y, d[a].values) - auc_w(y, d[b].values))
    rng = np.random.default_rng(7)
    ds = []
    for _ in range(BOOT):
        wc = rng.poisson(1.0, len(firms)).astype(np.float64)
        ds.append(np.mean([fast(P1[c], wc[INV[c]]) - fast(P2[c], wc[INV[c]]) for c in C]))
    r = {"pair": f"{a}-{b}", "per_origin_d": D0, "mean_d": float(np.mean(D0)), "wins": int(sum(x > 0 for x in D0)),
         "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "p_gt0": float(np.mean(np.array(ds) > 0)), "B": BOOT}
    json.dump(r, open(os.path.join(RES, f"A_pool_{a}_{b}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(r)
