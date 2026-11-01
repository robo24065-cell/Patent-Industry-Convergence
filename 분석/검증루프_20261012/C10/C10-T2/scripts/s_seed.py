# -*- coding: utf-8 -*-
r"""[0][1] seed 변동 풀링: 음성 표집 k, 학습 seed 쌍 p별 5원점 AUC와 법인 Poisson 부트스트랩(B=500, c_pool과 같은 rng 7 가중 흐름 → 앞 200회 = 공식 c_pool 가중).
사용: python -B s_seed.py <k> <pairs: 0,1,2,3,4> → cache/seed_k<k>_p<pairs>.npz (pt[arm,pair,origin], bs[rep,arm,pair,origin])"""
import sys
sys.dont_write_bytecode = True
from c10_common import *
from b_lib import AucPrep

k = int(sys.argv[1]); pairs = [int(x) for x in sys.argv[2].split(",")]
ARM3 = ["P50", "FREE50", "TXT50"]; B = 500


def preds(c, k):
    fs = [f for f in os.listdir(CACHE) if f.startswith(f"pred_c{c}_k{k}_P50-FREE50-TXT50_p") and f.endswith(".parquet")]
    fs = sorted(fs, key=len, reverse=True)  # p012, p34 우선, p0(2008 단독)은 중복
    out = {}
    for f in fs:
        d = pd.read_parquet(os.path.join(CACHE, f))
        for col in d.columns:
            out.setdefault(col, d[col].values)
    return out


rows = {}; firms = set()
for c in C_ORIG:
    a = pd.read_parquet(os.path.join(NC8, "cache", f"A_o{c}.parquet"), columns=["applicantCd", "sc", "first_ky"])
    rows[c] = a; firms |= set(a.applicantCd.unique())
firms = np.array(sorted(firms)); fidx = {f: i for i, f in enumerate(firms)}
PT = np.full((3, len(pairs), 5), np.nan); BS = np.full((B, 3, len(pairs), 5), np.nan)
PR = {}
for j, c in enumerate(C_ORIG):
    a = rows[c]; y = a.first_ky.notna().values.astype(np.int8); inv = a.applicantCd.map(fidx).values
    pp = preds(c, k)
    for i, arm in enumerate(ARM3):
        for q, p in enumerate(pairs):
            s = pp[f"{arm}_p{p}"].astype(np.float64)
            PR[(i, q, j)] = AucPrep(y, s, [inv])
            PT[i, q, j] = auc_w(y, s)
    log("prep", c)
rng = np.random.default_rng(7)
for r in range(B):
    wc = rng.poisson(1.0, len(firms)).astype(np.float64)
    for key, A in PR.items():
        BS[(r,) + key] = A.auc([wc])
    if r % 50 == 0:
        log("rep", r)
np.savez_compressed(os.path.join(CACHE, f"seed_k{k}_p{''.join(map(str, pairs))}.npz"), pt=PT, bs=BS, arms=np.array(ARM3), pairs=np.array(pairs), origins=np.array(C_ORIG))
log("done", k, pairs)
