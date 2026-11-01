# -*- coding: utf-8 -*-
r"""C15-T3 verify_0 (verify_1 v1_independent.py 의 panel/feat2/run 단계 사본, 경로만 verify_0; 나머지 원점 재학습용): hyb_pkr 대안 A 독립 재현(원점 2개: 2012, 2017). s1–s5와 코드 공유 없음, duckdb 미사용.
 단계(9분 제한 대응, 단계마다 끊음):
  assign : pyarrow 읽기 + pandas groupby/sort/drop_duplicates로 PKR-IPC 대안 A 배정(c1_assign.py 정의를 다시 읽고 직접 구현)
           IPC 표 = KR 문서(출원 ≤2019) 제공 라벨, sg/mg/sc 키별 문서 ≥3, 최빈(동률은 코드 숫자 작은 것). 해외 문서 = REG 연결 KR 문서 제공 라벨 최빈 → 없으면 sg→mg→sc.
           결과를 s1 cache/pkr_alt_*.parquet k_reg와 문서 단위 대조(대조만, 계산에는 안 씀).
  base   : 원본 base_rows를 pyarrow 행 그룹 단위로 읽어 해외 행의 ksic를 배정값으로 바꾸고(없으면 행 제외) ParquetWriter로 저장.
  panel / feat2 : 동결 빌더(build_panel.py, build_feat2.py) 실행 — 동결 코드라 공유가 불가피.
  run    : 새로 쓴 구동 코드. 동결 r3lib·r3core API(get_D, get_wide, feats, enrich, extras, score, baseline_pred)만 호출.
           교체 피처 = 사양의 모든 피처 − KOSIS 3 − RoS 2(RoS는 학습에서도 제외). AUC는 sklearn roc_auc_score.
쓰기: verify_1/ 안에만."""
import os, sys, json, time, threading, ctypes, subprocess, hashlib, copy
from ctypes import wintypes
sys.stdout.reconfigure(encoding="utf-8"); sys.dont_write_bytecode = True
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"; R3 = T3 + "/R3"
V = BASE + "/분석/검증루프_20261012/C15/C15-T3/verify_0"; CA = V + "/cache"; os.makedirs(CA, exist_ok=True)
ALTD = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign"
RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; LINKS = BASE + "/분석/검증루프_20261012/C10/C10-T3/cache/links.parquet"
SRC = R3 + "/cache_pub/base_rows.parquet"; S1 = BASE + "/분석/검증루프_20261012/C15/C15-T3/cache"
t0_ = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0_:.0f}s]", flush=True)


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("pf", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
               ("peak_ws", "ws", "a", "b", "c", "d", "pagefile", "peak_pagefile")]


_k = ctypes.WinDLL("kernel32"); _k.GetCurrentProcess.restype = wintypes.HANDLE
_k.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]


def memgb():
    m = _PMC(); m.cb = ctypes.sizeof(_PMC); _k.K32GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(m), m.cb); return m.ws / 2 ** 30, m.peak_ws / 2 ** 30


def _guard():
    while True:
        if memgb()[0] > 3.8: print("메모리 3.8GB 초과 → 종료", flush=True); os._exit(3)
        time.sleep(1)


threading.Thread(target=_guard, daemon=True).start()
stage = sys.argv[1]
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds, pyarrow.compute as pc


def top_label(df, keys):
    """키별 최빈 k, 동률은 int(k) 최소."""
    df = df.assign(_ki=df["k"].astype(int)).sort_values(keys + ["n", "_ki"], ascending=[True] * len(keys) + [False, True])
    return df.drop_duplicates(keys)[keys + ["k"]]


if stage == "assign":
    kr = pq.read_table(f"{ALTD}/alt_KR.parquet", columns=["appl_no", "appl_year", "ksic_prov"]).to_pandas()
    rcd = ds.dataset(RC)
    krsg = rcd.to_table(columns=["appl_no", "sg"], filter=ds.field("cc") == "KR").to_pandas()
    kr = kr.merge(krsg, on="appl_no", how="left"); del krsg
    assert not kr.appl_no.duplicated().any()
    kr["sg"] = kr["sg"].fillna(""); kr["mg"] = kr["sg"].str.split("/").str[0]; kr["sc"] = kr["sg"].str[:4]
    base = kr[kr.appl_year <= 2019].rename(columns={"ksic_prov": "k"})
    tab = {}
    for lv in ("sg", "mg", "sc"):
        g = base[base[lv] != ""].groupby([lv, "k"]).size().rename("n").reset_index()
        tot = g.groupby(lv)["n"].sum(); g = g[g[lv].isin(tot.index[tot >= 3])]
        tab[lv] = top_label(g, [lv]).set_index(lv)["k"]
    log("IPC 표", {k: len(v) for k, v in tab.items()})
    lk = pq.read_table(LINKS, filters=[("link", "=", "REG")]).to_pandas()[["cc", "appl_no", "kr_appl_no"]].drop_duplicates()
    lk = lk.merge(kr[["appl_no", "ksic_prov"]].rename(columns={"appl_no": "kr_appl_no", "ksic_prov": "k"}), on="kr_appl_no")
    fam = top_label(lk.groupby(["cc", "appl_no", "k"]).size().rename("n").reset_index(), ["cc", "appl_no"]).rename(columns={"k": "k_fam"})
    log("REG 상속 문서", fam.groupby("cc").size().to_dict())
    del kr, base, lk
    CCL = sys.argv[2].split(",") if len(sys.argv) > 2 else ["US", "JP", "EP", "CN"]
    fp = V + "/v1_assign_check.json"; rep = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}
    for cc in CCL:
        f = fam[fam.cc == cc].set_index("appl_no")["k_fam"]; outs = []
        for bt in rcd.to_batches(columns=["appl_no", "sg"], filter=ds.field("cc") == cc, batch_size=1_000_000):
            a = bt.to_pandas(); a["sg"] = a["sg"].fillna("")
            k = a["sg"].map(tab["sg"]); k = k.fillna(a["sg"].str.split("/").str[0].map(tab["mg"])); k = k.fillna(a["sg"].str[:4].map(tab["sc"]))
            a["k"] = a["appl_no"].map(f).fillna(k).values; del k
            outs.append(pa.Table.from_pandas(a[["appl_no", "k"]], preserve_index=False)); del a
        a = pa.concat_tables(outs); del outs
        pq.write_table(a, f"{CA}/v_pkr_{cc}.parquet")
        # 대조: s1 k_reg (alt_assign 문서 집합 기준) — 계산에는 안 씀
        s = pq.read_table(f"{S1}/pkr_alt_{cc}.parquet", columns=["appl_no", "k_reg"])
        m = s.join(a, keys="appl_no", join_type="left outer")
        x = pc.fill_null(m["k_reg"], "∅"); y = pc.fill_null(m["k"], "∅"); eq = pc.equal(x, y)
        rep[cc] = dict(n_s1=s.num_rows, n_v=a.num_rows, n_join=m.num_rows, same=pc.mean(pc.cast(eq, pa.float64())).as_py(), n_diff=int(m.num_rows - pc.sum(pc.cast(eq, pa.int64())).as_py()))
        log(cc, rep[cc], f"mem {memgb()[0]:.2f}GB"); del a, s, m, x, y, eq
    json.dump(rep, open(V + "/v1_assign_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
elif stage == "base":
    out = CA + "/t3_v/base_rows.parquet"; os.makedirs(CA + "/t3_v", exist_ok=True)
    maps = {}
    for cc in ("US", "JP", "EP", "CN"):
        tb0 = pq.read_table(f"{CA}/v_pkr_{cc}.parquet"); maps[cc] = tb0.filter(pc.invert(pc.is_null(tb0["k"])))
    pf = pq.ParquetFile(SRC); sch = pf.schema_arrow; wr = pq.ParquetWriter(out + ".part", sch, compression="zstd"); cnt = {}
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i)
        parts = [tb.filter(pc.equal(tb["cc"], "KR"))]
        for cc, mp in maps.items():
            sub = tb.filter(pc.equal(tb["cc"], cc))
            if sub.num_rows == 0: continue
            j = sub.join(mp.rename_columns(["appl_no", "k_new"]), keys="appl_no", join_type="inner")
            j = j.set_column(j.schema.get_field_index("ksic"), "ksic", j["k_new"]).drop_columns(["k_new"]).select(sch.names).cast(sch)
            parts.append(j)
        for p in parts:
            if p.num_rows:
                c = p["cc"][0].as_py(); cnt[c] = cnt.get(c, 0) + p.num_rows; wr.write_table(p)
    wr.close(); os.replace(out + ".part", out); log("base_rows", cnt, f"peak {memgb()[1]:.2f}GB")
    json.dump(cnt, open(V + "/v1_base_counts.json", "w"), indent=1)
elif stage in ("panel", "feat2"):
    cache = CA + "/t3_v"; env = {**os.environ, "T3_CACHE": cache, "T3_MAXYEAR": "2021", "T3_T0MAX": "2018", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    step = "build_panel.py" if stage == "panel" else "build_feat2.py"
    r = subprocess.run([sys.executable, T3 + "/" + step], cwd=cache, env=env); assert r.returncode == 0; log(step, "ok")
elif stage == "run":
    ORS = [int(x) for x in sys.argv[2].split(",")]; assert set(ORS) <= set(range(2010, 2019))
    for k in ["T3R3_FINAL_UNLOCK", "T3R3_TRAIN_MAX_T0", "T3R3_HIER", "T3R3_DATA_FIRST"]: os.environ.pop(k, None)
    os.environ.update(T3R3_VAR="pub", T3R3_CACHE="cache_pub", T3R3_MAXYEAR="2021", T3R3_LABEL_MAX="2021", T3R3_EXT_SUFFIX="", T3R3_EXTB_SUFFIX="")
    sys.path.insert(0, R3)
    import r3lib as L, r3core as C
    from sklearn.metrics import roc_auc_score
    spec = json.load(open(R3 + "/final_spec_r3.json", encoding="utf-8"))["cfg"]
    KOSIS = {"pi_prod_g2", "pi_prod_g1", "pi_ship_g2"}; RS = {"ros_sci", "ros_ppp"}
    cfg = copy.deepcopy(spec)
    for comp in cfg["components"]: comp["features"] = [f for f in comp["features"] if f not in RS]
    feats_all = sorted({f for comp in spec["components"] for f in comp["features"]})
    C.COMP_DIR = CA + "/comp"; os.makedirs(C.COMP_DIR, exist_ok=True)
    C.EXT_FILES = [R3 + "/ext_ksic_t0_pub.parquet"]; C.CELL_EXT_FILE = R3 + "/ext_cell_t0_pub.parquet"
    t0s = list(range(C.FIRST_T0, 2019))
    ros = pd.read_parquet(R3 + "/extB_cell_pub.parquet")[["cell", "t0", "ros_sci", "ros_ppp"]]
    D = C.get_D(t0s).merge(ros, on=["cell", "t0"], how="left")
    Wd = C.get_wide(D, 10).merge(ros, on=["cell", "t0"], how="left")
    base_p = {v: C.baseline_pred(D, v) for v in ORS}
    # 대체 패널 → 배열 P (셀 = 대체 패널 셀 ∪ 평가 표본 셀)
    cache = CA + "/t3_v"; pan = pd.read_parquet(cache + "/panel_ccy.parquet"); pan["cell"] = pan["ksic"] + "|" + pan["mg"]
    allcells = np.array(sorted(set(pan.cell) | set(D.cell) | set(Wd.cell)))
    ix = pd.Index(allcells); ny = L.Y1 - L.Y0 + 1; ccix = {c: i for i, c in enumerate(L.CCS)}
    cols = ["n", "n_app", "n_prio_for", "n_newapp_apps", "n_newapp", "n_newk_apps", "hhi", "n_multi", "n_newpair_apps", "n_newpair", "n_newword_apps", "n_newword"]
    A = {}
    for col in cols:
        s = pan[["cell", "cc", "year", col]].dropna(); arr = np.zeros((len(allcells), len(L.CCS), ny), dtype=np.float32)
        arr[ix.get_indexer(s.cell), s.cc.map(ccix).values, s.year.values - L.Y0] = s[col].values; A[col] = arr
    P = dict(cells=allcells, ksic=np.array([c.split("|")[0] for c in allcells]), mg=np.array([c.split("|")[1] for c in allcells]), A=A)
    keep_cells = pd.concat([D[["cell", "t0"]], Wd[["cell", "t0"]]]).drop_duplicates()
    old = (L.CACHE, L.LOCAL); L.CACHE = L.LOCAL = cache; L._cache.clear(); L._cache["P"] = P
    fr = []
    for tt in t0s:
        f = L.feats(P, tt, 0); fr.append(f.merge(keep_cells[keep_cells.t0 == tt][["cell"]], on="cell"))
    fr = pd.concat(fr, ignore_index=True)
    XD = C.extras(C.enrich(fr.merge(D[["cell", "t0"]].drop_duplicates(), on=["cell", "t0"]).reset_index(drop=True)))
    XW = C.extras(C.enrich(fr.merge(Wd[["cell", "t0"]].drop_duplicates(), on=["cell", "t0"]).reset_index(drop=True)))
    L.CACHE, L.LOCAL = old; L._cache.clear()
    rep_cols = [f for f in feats_all if f not in KOSIS | RS and f in XD.columns]

    def put(T, X):
        X = X[["cell", "t0", "tot_W"] + [c for c in rep_cols if c != "tot_W"]].rename(columns={"tot_W": "_altW"})
        T2 = T.drop(columns=[c for c in rep_cols if c in T.columns]).merge(X, on=["cell", "t0"], how="left"); assert len(T2) == len(T)
        ratio = float(np.nanmedian((T2["_altW"].values + 1) / (T["tot_W"].values + 1)))
        return T2.drop(columns=["_altW"]), ratio
    D2, ratio = put(D, XD); W2, _ = put(Wd, XW); C._WIDE[10] = W2
    out = {"replaced_n": len(rep_cols), "median_totW_ratio": ratio}
    for v in ORS:
        te = D2[D2.t0 == v]; p = C.score(cfg, D2, v, "r7")
        out[str(v)] = dict(auc=float(roc_auc_score(te.yb.values, p)), base_auc=float(roc_auc_score(te.yb.values, base_p[v])), n=int(len(te)))
        pd.DataFrame(dict(cell=te.cell.values, t0=v, yb=te.yb.values, p=p)).to_parquet(f"{V}/v0_preds_{v}.parquet")
        log(v, out[str(v)], f"mem {memgb()[0]:.2f}GB")
    out["peak_gb"] = memgb()[1]
    json.dump(out, open(V + f"/v0_run_{'_'.join(map(str, ORS))}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
