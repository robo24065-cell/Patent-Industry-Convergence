# -*- coding: utf-8 -*-
"""b_lib.AucPrep vs c_pool fast() 동치 점검 → res/b_selftest.json"""
import sys; sys.dont_write_bytecode = True
from c10_common import *
from b_lib import AucPrep
import time
d = pd.read_parquet(os.path.join(SRC_RES, 'C_pred_c2012.parquet'))
y = d.y.values.astype(np.int8)
fid = pd.factorize(d.applicantCd)[0]; sid = pd.factorize(d.sc)[0]
out = {}
for arm in ("P50", "FREE50"):
    s = d[arm].values.astype(np.float64)
    pr = prep(y, s); A = AucPrep(y, s, [fid]); A2 = AucPrep(y, s, [fid, sid])
    rng = np.random.default_rng(1); diffs = []; d2 = []
    for _ in range(5):
        wf = rng.poisson(1.0, fid.max() + 1).astype(float); ws = rng.poisson(1.0, sid.max() + 1).astype(float)
        diffs.append(abs(fast(pr, wf[fid]) - A.auc([wf])))
        d2.append(abs(fast(pr, wf[fid] * ws[sid]) - A2.auc([wf, ws])))
    one = abs(auc_w(y, s) - A.auc([np.ones(fid.max() + 1)]))
    t = time.time()
    for _ in range(20): A.auc([wf])
    out[arm] = {"maxdiff_firm": max(diffs), "maxdiff_twoway": max(d2), "diff_unit": one, "sec_per_auc": (time.time() - t) / 20}
json.dump(out, open(os.path.join(RES, "b_selftest.json"), "w"), indent=1); print(out)
