# -*- coding: utf-8 -*-
r"""C10-T3 s5: c5_R10 c3_t3_run.py 사본(학습·평가 로직 동일). 바꾼 것: 캐시 경로·태그·원점 목록을 인자로 받고,
쓰기는 C10-T3/res/t3/, comp 캐시는 C10-T3/cache/t3_comp. 개발 원점 2010–2018만 허용(최종시험 구간 없음).
  python s5_t3_run.py --cache <dir> --tag <name> [--origins 2010,2011,...]"""
import os, sys, json, time, argparse, copy
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "모델경쟁_20260927"))
T3 = os.path.join(MC, "T3_기술셀부상예측"); R3 = os.path.join(T3, "R3"); R7 = os.path.join(T3, "ESSENTIAL_R7")
ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--origins", default=""); a = ap.parse_args()
for k in ["T3R3_FINAL_UNLOCK", "T3R3_TRAIN_MAX_T0", "T3R3_HIER", "T3R3_DATA_FIRST"]: os.environ.pop(k, None)
os.environ.update(T3R3_VAR="pub", T3R3_CACHE="cache_pub", T3R3_MAXYEAR="2021", T3R3_LABEL_MAX="2021", T3R3_EXT_SUFFIX="", T3R3_EXTB_SUFFIX="")
sys.path.insert(0, R3); sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import r3lib as L
import r3core as C
SPEC = json.load(open(os.path.join(R3, "final_spec_r3.json"), encoding="utf-8"))
CFG0 = SPEC["cfg"]; ORIGINS = SPEC["dev"]["origins"]
assert ORIGINS == list(range(2010, 2019)), ORIGINS
if a.origins:
    ORIGINS = [int(x) for x in a.origins.split(",")]; assert set(ORIGINS) <= set(range(2010, 2019))
ALLF = sorted({f for c in CFG0["components"] for f in c["features"]})
PI = ["pi_prod_g2", "pi_prod_g1", "pi_ship_g2"]; ROS = ["ros_sci", "ros_ppp"]
OUTD = os.path.join(HERE, "res", "t3"); os.makedirs(OUTD, exist_ok=True)
C.COMP_DIR = os.path.join(HERE, "cache", "t3_comp"); os.makedirs(C.COMP_DIR, exist_ok=True)
C.EXT_FILES = [os.path.join(R3, "ext_ksic_t0_pub.parquet")]; C.CELL_EXT_FILE = os.path.join(R3, "ext_cell_t0_pub.parquet")
T0S = list(range(C.FIRST_T0, max(ORIGINS) + 1))


def cfg_drop(cfg, drop):
    c2 = copy.deepcopy(cfg)
    for c in c2["components"]:
        c["features"] = [f for f in c["features"] if f not in set(drop)]
        assert c["features"]
    return c2


def prep_orig(D):
    cf = pd.read_parquet(os.path.join(R3, "extB_cell_pub.parquet"))[["cell", "t0"] + ROS]
    n0 = len(D); D = D.merge(cf, on=["cell", "t0"], how="left"); assert len(D) == n0
    return D


def load_aug(cache, extra_cells):
    df = pd.read_parquet(os.path.join(cache, "panel_ccy.parquet")); assert df.year.max() <= L.Y1
    df["cell"] = df.ksic + "|" + df.mg
    cells = np.array(sorted(set(df.cell.unique()) | set(extra_cells))); cid = {c: i for i, c in enumerate(cells)}
    df["ci"] = df.cell.map(cid).values; T = L.Y1 - L.Y0 + 1
    cols = ["n", "n_app", "n_prio_for", "n_newapp_apps", "n_newapp", "n_newk_apps", "hhi", "n_multi", "n_newpair_apps", "n_newpair", "n_newword_apps", "n_newword"]
    A = {}
    for col in cols:
        arr = np.zeros((len(cells), len(L.CCS), T), dtype=np.float32); sub = df[["ci", "cc", "year", col]].dropna()
        arr[sub.ci.values, sub.cc.map({c: i for i, c in enumerate(L.CCS)}).values, sub.year.values - L.Y0] = sub[col].values; A[col] = arr
    return dict(cells=cells, ksic=np.array([c.split("|")[0] for c in cells]), mg=np.array([c.split("|")[1] for c in cells]), A=A)


def alt_frame(cache, need_D, need_W):
    sv = (L.CACHE, L.LOCAL); L.CACHE = L.LOCAL = cache; L._cache.clear()
    P = load_aug(cache, set(need_D.alt_cell) | set(need_W.alt_cell)); L._cache["P"] = P
    needset = pd.concat([need_D, need_W])[["alt_cell", "t0"]].drop_duplicates()
    F = []
    for tt in T0S:
        f = L.feats(P, tt, 0); keep = set(needset.loc[needset.t0 == tt, "alt_cell"]); F.append(f[f.cell.isin(keep)])
    F = pd.concat(F, ignore_index=True)
    outs = []
    for need in (need_D, need_W):
        k = need[["alt_cell", "t0"]].drop_duplicates().rename(columns={"alt_cell": "cell"})
        X = F.merge(k, on=["cell", "t0"]).reset_index(drop=True)
        X = C.enrich(X); X = C.extras(X)
        outs.append(X)
    L.CACHE, L.LOCAL = sv; L._cache.clear()
    return outs


def swap(T, X, repl):
    X = X[["cell", "t0", "tot_W"] + [c for c in repl if c != "tot_W"]].rename(columns={"cell": "alt_cell", "tot_W": "alt_totW"})
    n0 = len(T); T2 = T.drop(columns=[c for c in repl + ["alt_totW"] if c in T]).merge(X, on=["alt_cell", "t0"], how="left"); assert len(T2) == n0
    T2.index = T.index
    return T2, dict(row_found=float(T2.alt_totW.notna().mean()), alt_totW_pos=float((T2.alt_totW > 0).mean()),
                    alt_vs_target_totW_median_ratio=float(np.nanmedian((T2.alt_totW + 1) / (T.tot_W.values + 1))))


t = time.time(); info = {}; test = a.tag
D = prep_orig(C.get_D(T0S)); C._WIDE[10] = prep_orig(C.get_wide(D, 10))
assert not [f for f in ALLF if f not in D]
P0 = {v: C.baseline_pred(D, v) for v in ORIGINS}
cache = a.cache
assert os.path.exists(os.path.join(cache, "panel_ccy.parquet")), cache
cfg = cfg_drop(CFG0, ROS); repl = [f for f in ALLF if f not in PI + ROS]
W = C._WIDE[10].copy(); D["alt_cell"] = D.cell.values; W["alt_cell"] = W.cell.values
XD, XW = alt_frame(cache, D[["alt_cell", "t0"]], W[["alt_cell", "t0"]])
repl = [f for f in repl if f in XD.columns]
D, covD = swap(D, XD, repl); W, covW = swap(W, XW, repl); C._WIDE[10] = W
info.update(cache=cache, replaced=repl, coverage_D=covD, coverage_W=covW)
res, preds = {}, []
for v in ORIGINS:
    te = D[D.t0 == v]; tr = C.train_rows(D, v, cfg); assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR
    p = C.score(cfg, D, v, "r7"); p0 = P0[v]
    m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
    res[str(v)] = dict(model=m, baseline=b, skill=(m["auc"] - b["auc"]) / (1 - b["auc"]))
    print(f"[{test}] t0={v} AUC {m['auc']:.4f} (기준선 {b['auc']:.4f}) skill {res[str(v)]['skill']:.4f} n={m['n']} evt={m['n_evt']} | {time.time()-t:.0f}s", flush=True)
    preds.append(pd.DataFrame(dict(cell=te.cell.values, mg=te.mg.values, ksic=te.ksic.values, t0=v, y=te.y.values, yb=te.yb.values, p=p, p_base=p0)))
sk = [r["skill"] for r in res.values()]
summ = dict(mean_skill=float(np.mean(sk)), mean_auc=float(np.mean([r["model"]["auc"] for r in res.values()])),
            mean_base_auc=float(np.mean([r["baseline"]["auc"] for r in res.values()])), mean_p10=float(np.mean([r["model"]["p_at10"] for r in res.values()])))
out = dict(test=test, origins=ORIGINS, summary=summ, by_origin=res, info=info, created=time.strftime("%Y-%m-%d %H:%M:%S"), secs=round(time.time() - t))
json.dump(out, open(os.path.join(OUTD, f"{test}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(OUTD, f"{test}_preds.parquet"))
print(json.dumps(dict(summary=summ, coverage_D=covD), ensure_ascii=False))
