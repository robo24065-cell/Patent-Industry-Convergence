# -*- coding: utf-8 -*-
r"""반박 검증(seed 견고성) A: 저장 예측에서 15구성 x 5원점 AUC를 sklearn으로 독립 재계산.
- 같은 열이 여러 파일에 있으면 값 일치 확인, 구성 간 예측 복사 여부(해시) 확인
- 작업자 seed_configs.csv와 비교, 앙상블 재계산
출력: verify_0/vs_seed_a.json  (읽기: cache/pred_*, NONCIRC_R8 A_o*; 쓰기: verify_0만)"""
import os, sys, json, time, hashlib, glob
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
NC8 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(W))), "모델경쟁_20260927", "T1_산업대기실", "NONCIRC_R8", "cache")
assert os.path.isdir(NC8), NC8
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
CS = [2008, 2009, 2010, 2011, 2012]; ARM = ["P50", "FREE50", "TXT50"]
out = {"dup_file_check": [], "hash_dupes": [], "auc": {}, "notes": []}
Y = {}
for c in CS:
    a = pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["first_ky"])
    Y[c] = a.first_ky.notna().values.astype(np.int8)
hashes = {}
ens = {c: {arm: np.zeros(len(Y[c])) for arm in ARM} for c in CS}
for k in range(3):
    for c in CS:
        fs = sorted(glob.glob(os.path.join(W, "cache", f"pred_c{c}_k{k}_P50-FREE50-TXT50_p*.parquet")))
        cols = {}
        for f in fs:
            d = pd.read_parquet(f)
            assert len(d) == len(Y[c]), (f, len(d), len(Y[c]))
            for col in d.columns:
                v = d[col].values
                if col in cols:
                    out["dup_file_check"].append({"c": c, "k": k, "col": col, "file": os.path.basename(f),
                                                  "maxabs": float(np.abs(cols[col] - v).max())})
                else:
                    cols[col] = v
        for p in range(5):
            for arm in ARM:
                v = cols[f"{arm}_p{p}"]
                h = hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()
                key = f"c{c}_k{k}_{arm}_p{p}"
                if h in hashes:
                    out["hash_dupes"].append([hashes[h], key])
                hashes[h] = key
                out["auc"][key] = float(roc_auc_score(Y[c], v.astype(np.float64)))
                ens[c][arm] += v.astype(np.float64) / 15
        log("k", k, "c", c)
# 구성별 요약
rows = []
for k in range(3):
    for p in range(5):
        r = {"k": k, "p": p}
        for b in ["FREE50", "TXT50"]:
            po = [out["auc"][f"c{c}_k{k}_P50_p{p}"] - out["auc"][f"c{c}_k{k}_{b}_p{p}"] for c in CS]
            r[f"d_{b}"] = float(np.mean(po)); r[f"wins_{b}"] = int(sum(x > 0 for x in po)); r[f"po_{b}"] = po
        rows.append(r)
df = pd.DataFrame(rows)
wk = pd.read_csv(os.path.join(W, "res", "seed_configs.csv"))
cmp = []
for _, r in df.iterrows():
    for b in ["FREE50", "TXT50"]:
        m = wk[(wk.pair == f"P50-{b}") & (wk.k_neg == r.k) & (wk.train_pair == r.p)]
        cmp.append(abs(float(m.mean_d.iloc[0]) - r[f"d_{b}"]))
out["max_absdiff_vs_worker"] = float(max(cmp))
for b in ["FREE50", "TXT50"]:
    d = df[f"d_{b}"].values
    srt = np.sort(d)
    off = float(df[(df.k == 0) & (df.p == 0)][f"d_{b}"].iloc[0])
    out[f"summary_{b}"] = {"mean": float(d.mean()), "sd_ddof1": float(d.std(ddof=1)), "sd_ddof0": float(d.std()),
                           "min": float(d.min()), "max": float(d.max()), "official": off,
                           "official_rank_from_smallest": int((srt < off).sum() + 1),
                           "cv_sd_over_mean": float(d.std(ddof=1) / d.mean()),
                           "origin_config_pos": int(sum(sum(x > 0 for x in po) for po in df[f"po_{b}"])),
                           "min_origin_delta": float(min(min(po) for po in df[f"po_{b}"]))}
    # 표집 seed 간 / 안 sd
    out[f"summary_{b}"]["sd_between_k_means"] = float(df.groupby("k")[f"d_{b}"].mean().std(ddof=1))
    out[f"summary_{b}"]["mean_within_k_sd"] = float(df.groupby("k")[f"d_{b}"].std(ddof=1).mean())
    eo = [roc_auc_score(Y[c], ens[c]["P50"]) - roc_auc_score(Y[c], ens[c][b]) for c in CS]
    out[f"ensemble_{b}"] = {"mean": float(np.mean(eo)), "per_origin": [float(x) for x in eo]}
out["configs"] = df.drop(columns=["po_FREE50", "po_TXT50"]).to_dict("records")
json.dump(out, open(os.path.join(HERE, "vs_seed_a.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done")
