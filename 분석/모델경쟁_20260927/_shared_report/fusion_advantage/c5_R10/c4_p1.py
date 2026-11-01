# -*- coding: utf-8 -*-
r"""c5_R10 c4: R8 P1(기술의 산업 간 이동) 5개국 증분을 해외 라벨 대체로 다시 평가.
단위·결과(y)는 R8 p1_panel.parquet 그대로(읽기만). 특성 정의는 p1b_build2.py share_feats 그대로 복사, 라벨링만 'hyb_<alt>' = KR 제공 + 해외 대체(c1 alt_assign).
평가는 p1c_eval.py와 같은 XGBoost(300 trees, depth 4, lr 0.05, subsample 0.8, colsample 0.8, hist, random_state 0, n_jobs 2), 학습 t' ≤ t−3, 평가 원점 2010–2018.
재현 점검: FULL(prov__ 전부)을 다시 학습해 R8 p1_by_origin.csv 값과 대조."""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from xgboost import XGBClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(HERE, r"..\..\..")); R8 = os.path.join(MC, "_ideas_R8")
sys.path.insert(0, R8)
from r8_common import auc, topk_hit, boot_ci, skill   # 함수만 사용(r8_common은 import 시 쓰기 없음)
t_ = time.time(); log = lambda *a: print(*a, f"[{time.time() - t_:.0f}s]", flush=True)
PY_MAX, Y0 = 2021, 2000; YRS = list(range(Y0, PY_MAX + 1)); CCS = ["KR", "US", "JP", "CN", "EP"]
con = duckdb.connect(); con.execute("SET threads=2; SET memory_limit='3GB'; SET enable_progress_bar=false")
REC = os.path.join(R8, "cache", "rec.parquet").replace("\\", "/"); ALT = os.path.join(HERE, "alt_assign", "*.parquet").replace("\\", "/")
ALTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["txt", "free", "pkr"]
con.execute(f"""CREATE TABLE r AS SELECT r.cc, r.mg, r.pyear, r.ksic AS prov,
   {', '.join(f"CASE WHEN r.cc='KR' THEN r.ksic ELSE a.k_{x} END AS hyb_{x}" for x in ALTS)}
   FROM '{REC}' r LEFT JOIN read_parquet('{ALT}') a ON a.cc=r.cc AND a.appl_no=r.appl_no
   WHERE r.mg<>'' AND r.pyear BETWEEN {Y0} AND {PY_MAX}""")


def wide(sql_keys, where="TRUE"):
    sel = ", ".join(sql_keys)
    d = con.sql(f"SELECT {sel}, pyear, count(*) n FROM r WHERE {where} GROUP BY ALL").df()
    w = d.pivot_table(index=[k.split(" AS ")[-1] for k in sql_keys], columns="pyear", values="n", aggfunc="sum", fill_value=0)
    w = w.reindex(columns=YRS, fill_value=0).astype(np.int32)
    cs = np.cumsum(w.values, axis=1)
    return w.index, np.concatenate([np.zeros((len(w), 1), np.int64), cs], axis=1)


class WS:
    def __init__(self, idx, cs): self.idx, self.cs = idx, cs

    def get(self, keys, y0, y1):
        pos = self.idx.get_indexer(keys)
        a = max(y0, Y0) - Y0; b = min(y1, PY_MAX) - Y0 + 1
        v = self.cs[:, b] - self.cs[:, a] if b > a else np.zeros(len(self.cs))
        return np.where(pos >= 0, v[np.maximum(pos, 0)], 0).astype(float)


LABS = ["prov"] + [f"hyb_{x}" for x in ALTS]
TOT = WS(*wide(["cc", "mg"]))
CNT = {a: WS(*wide(["cc", "mg", f"{a} AS lab"], f"{a} IS NOT NULL")) for a in LABS}
LAB = {a: WS(*wide(["cc", f"{a} AS lab"], f"{a} IS NOT NULL")) for a in LABS}
log("wide done")
mi = lambda *cols: pd.MultiIndex.from_arrays(list(cols))


def share_feats(u, t, a, lab_kr, lab_f):   # p1b_build2.py 와 같은 코드
    out = {}
    for cc in CCS:
        lab = lab_kr if cc == "KR" else lab_f
        kk = mi([cc] * len(u), u.mg.values, lab); gk = mi([cc] * len(u), u.mg.values)
        sh = {}
        for w, (y0, y1) in {"P": (t - 2, t), "PP": (t - 5, t - 3), "T": (t, t)}.items():
            num = CNT[a].get(kk, y0, y1); den = TOT.get(gk, y0, y1)
            sh[w] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            if w == "P":
                out[f"{cc}_lnn"] = np.log1p(num); out[f"{cc}_lng"] = np.log1p(den)
            if w == "PP":
                out[f"{cc}_lngPP"] = np.log1p(den)
        out[f"{cc}_shP"] = sh["P"]; out[f"{cc}_dsh"] = sh["P"] - sh["PP"]
        if cc == "KR":
            out["KR_shPP"] = sh["PP"]; out["KR_shT"] = sh["T"]; out["KR_dshT"] = sh["T"] - sh["P"]
        lk = mi([cc] * len(u), lab)
        out[f"{cc}_kgrow"] = np.log1p(LAB[a].get(lk, t - 2, t)) - np.log1p(LAB[a].get(lk, t - 5, t - 3))
    o = pd.DataFrame(out, index=u.index)
    o["g_grow"] = o.KR_lng - o.KR_lngPP
    for cc in CCS[1:]:
        o[f"{cc}_gap"] = o[f"{cc}_shP"] - o["KR_shP"]
    ix = CNT[a].idx; krm = ix.get_level_values(0) == "KR"
    v = CNT[a].get(ix[krm], t - 2, t)
    d = pd.DataFrame({"mg": ix[krm].get_level_values(1), "n": v})
    nl = d[d.n > 0].groupby("mg").size()
    o["n_lab_g"] = u.mg.map(nl).fillna(0).values
    return o


P0 = pd.read_parquet(os.path.join(R8, "cache", "p1_panel.parquet"))
P0 = P0[P0.t <= 2018].reset_index(drop=True); assert (P0.y >= 0).all()
U = P0[["t", "mg", "k", "y", "W"] + [c for c in P0.columns if c.startswith("prov__")]].copy()
parts = []
for t, u in U.groupby("t"):
    fs = []
    for a in LABS:
        f = share_feats(u, t, a, u.k.values, u.k.values); f.columns = [f"{'re' if a == 'prov' else a}__{x}" for x in f.columns]; fs.append(f)
    parts.append(pd.concat(fs, axis=1))
F = pd.concat(parts).loc[U.index]
U = pd.concat([U, F], axis=1)
prov_cols = [c for c in U.columns if c.startswith("prov__")]
# 특성 재구성 점검: 'prov' 라벨로 다시 만든 값(re__)이 R8 패널 값과 같은지
chk = {c: float(np.nanmax(np.abs(U[c].values.astype(float) - U["re__" + c[6:]].values.astype(float)))) for c in prov_cols}
log("재구성 최대 차이", max(chk.values()))
SETS = {"FULL_prov5c": prov_cols, "KRONLY_prov": [c for c in prov_cols if c.startswith("prov__KR_")] + ["prov__g_grow", "prov__n_lab_g"]}
for x in ALTS:
    SETS[f"HYB_{x}"] = [f"hyb_{x}__{c[6:]}" for c in prov_cols]


def fit_pred(tr, te, X):
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      n_jobs=2, tree_method="hist", random_state=0, eval_metric="logloss")
    m.fit(tr[X], tr.y); return m.predict_proba(te[X])[:, 1]


rows = []
for t in range(2010, 2019):
    tr = U[U.t <= t - 3]; te = U[U.t == t]; rec = {"t": t, "n": len(te), "pos": int(te.y.sum())}
    for nm, X in SETS.items():
        s = fit_pred(tr, te, X); rec[nm] = auc(te.y, s); rec[nm + "_top1"] = topk_hit(te.y.values, s, 0.01)
    rows.append(rec); log(t, {k: round(v, 4) for k, v in rec.items() if isinstance(v, float) and not k.endswith("top1")})
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "c4_p1_by_origin.csv"), index=False)
ref = pd.read_csv(os.path.join(R8, "p1_by_origin.csv"))
out = {"repro_feature_maxabs": max(chk.values()),
       "repro_FULL_maxabs_auc_vs_R8": float((R.FULL_prov5c.values - ref.FULL_prov5c.values).__abs__().max()),
       "repro_KRONLY_maxabs_auc_vs_R8": float((R.KRONLY_prov.values - ref.KRONLY_prov.values).__abs__().max()),
       "auc": {nm: float(R[nm].mean()) for nm in SETS}, "top1": {nm: float(R[nm + "_top1"].mean()) for nm in SETS}, "compare": {}}
for a_, b_ in [("FULL_prov5c", "KRONLY_prov")] + [("FULL_prov5c", f"HYB_{x}") for x in ALTS] + [(f"HYB_{x}", "KRONLY_prov") for x in ALTS]:
    d = (R[a_] - R[b_]).values; lo, hi = boot_ci(d)
    w = int((d > 0).sum())
    out["compare"][f"{a_}-{b_}"] = {"mean": float(d.mean()), "pos_origins": f"{w}/9", "ci": [lo, hi],
                                     "verdict": "앞 팔 우위" if (d.mean() > 0 and lo > 0 and w >= 6) else ("뒤 팔 우위" if (d.mean() < 0 and hi < 0 and 9 - w >= 6) else "같다")}
full_kr = R.FULL_prov5c.mean() - R.KRONLY_prov.mean()
out["retention_of_5c_increment"] = {x: float((R[f"HYB_{x}"].mean() - R.KRONLY_prov.mean()) / full_kr) for x in ALTS}
json.dump(out, open(os.path.join(HERE, "c4_p1_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)); log("end")
