# -*- coding: utf-8 -*-
r"""C11-T2 verify_0: P1 reg 팔 독립 재현(작업자 s1–s5와 다른 코드 경로, pyarrow/pandas).
 (a) reg txt_assign: 바뀔 수 있는 문서(REG 연결 문서 ∪ 저장 family 문서)만 pandas로 다시 배정, 나머지는 저장값과 같아야 함.
 (b) panel n: reg − r8 패널 차이(n)를 '라벨이 바뀐 문서'만으로 독립 집계(mg 파싱은 파이썬 정규식)해 대조.
 (c) CTX: c_ind_kmom, c_cov, c_n_ind를 독립 식으로 t0 2010·2014·2018에서 계산해 작업자 ctx_txt_reg와 대조.
 (e) ΔAUC·CI: sklearn AUC + 자체 부트스트랩(같은 seed 20261005 → 작업자 값과 같아야; seed 1·2·3 → 판정 안정성).
사용: python v0_reg.py a | b | c | e"""
import os, sys, json, re, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; C10 = BASE + "/분석/검증루프_20261012/C10/C10-T3"
OUT = W + "/verify_0"; step = sys.argv[1]; t = time.time()
log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
R = {}
if step == "a":
    km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)
    multi = sorted(km.code.value_counts().loc[lambda s: s > 1].index)
    tat = pq.read_table(T3 + "/TXT_R9/txt_assign.parquet", columns=["cc", "appl_no", "code", "ksic_txt", "method"])
    mt = pq.read_table(W + "/cache/txt_assign_reg.parquet", columns=["cc", "appl_no", "ksic_txt"]).rename_columns(["cc", "appl_no", "k_mine"])
    J = tat.join(mt, keys=["cc", "appl_no"], join_type="inner"); assert J.num_rows == tat.num_rows == mt.num_rows
    lk = pq.read_table(C10 + "/cache/links.parquet", filters=[("link", "=", "REG")]).to_pandas()
    fammask = pc.equal(J["method"], "family")
    cand_appl = pa.array(sorted(set(lk.appl_no) | set(J.filter(fammask)["appl_no"].to_pylist())))
    foreign_multi = pc.and_(pc.not_equal(J["cc"], "KR"), pc.is_in(J["code"], value_set=pa.array(multi)))
    inc = pc.and_(foreign_multi, pc.is_in(J["appl_no"], value_set=cand_appl))
    diff = pc.not_equal(J["k_mine"], J["ksic_txt"])
    R["a_rows_changed_total"] = int(pc.sum(diff).as_py()); R["a_changed_outside_candidates"] = int(pc.sum(pc.and_(diff, pc.invert(inc))).as_py())
    fx = J.filter(inc).to_pandas()
    kr = J.filter(pc.equal(J["cc"], "KR")).select(["appl_no", "code", "ksic_txt", "method"]).to_pandas()
    del J, tat, mt
    log("후보", len(fx), "바뀐 행", R["a_rows_changed_total"], "후보 밖 변경", R["a_changed_outside_candidates"])
    krk = kr[["appl_no", "code", "ksic_txt"]].rename(columns={"appl_no": "kr_appl_no", "code": "kcode", "ksic_txt": "kk"})
    j = lk.merge(fx[["cc", "appl_no", "code"]], on=["cc", "appl_no"]).merge(krk, on="kr_appl_no")
    j = j[j.kcode == j.code]
    cnt = j.groupby(["cc", "appl_no", "kk"]).size().reset_index(name="n")
    cnt["ki"] = cnt.kk.astype(int); cnt = cnt.sort_values(["cc", "appl_no", "n", "ki"], ascending=[True, True, False, True])
    fam = cnt.groupby(["cc", "appl_no"], sort=False).head(1).set_index(["cc", "appl_no"]).kk
    log("REG 상속 문서", fam.groupby(level=0).size().to_dict())
    rc = pq.read_table(T3 + "/ESSENTIAL_R7/rec_conc.parquet", columns=["cc", "appl_no", "sg"])
    need = sorted(set(fx.loc[fx.method == "family", "appl_no"]) | set(kr.loc[kr.method == "text", "appl_no"]))
    rc = rc.filter(pc.is_in(rc["appl_no"], value_set=pa.array(need))).to_pandas()
    krt = kr[kr.method == "text"].assign(cc="KR").merge(rc[rc.cc == "KR"], on=["cc", "appl_no"])
    krt["mg"] = krt.sg.str.split("/").str[0]; krt["sc"] = krt.sg.str[:4]
    tabs = {}
    for lvl in ("sg", "mg", "sc"):
        g = krt[krt[lvl].fillna("") != ""].groupby(["code", lvl, "ksic_txt"]).size().reset_index(name="n")
        g = g[g.groupby(["code", lvl]).n.transform("sum") >= 3]
        g["ki"] = g.ksic_txt.astype(int); g = g.sort_values(["code", lvl, "n", "ki"], ascending=[True, True, False, True])
        tabs[lvl] = g.groupby(["code", lvl], sort=False).head(1).set_index(["code", lvl]).ksic_txt.to_dict()
    g = krt.groupby(["code", "ksic_txt"]).size().reset_index(name="n"); g["ki"] = g.ksic_txt.astype(int)
    tabs["code"] = g.sort_values(["code", "n", "ki"], ascending=[True, False, True]).groupby("code", sort=False).head(1).set_index("code").ksic_txt.to_dict()
    sf = fx[fx.method == "family"].merge(rc, on=["cc", "appl_no"], how="left")
    def tabv(r):
        sg = r.sg if isinstance(r.sg, str) else ""
        for lvl, key in (("sg", sg), ("mg", sg.split("/")[0]), ("sc", sg[:4])):
            v = tabs[lvl].get((r.code, key))
            if key and v is not None: return v
        return tabs["code"].get(r.code)
    sf["tab"] = sf.apply(tabv, axis=1)
    exp = fx.set_index(["cc", "appl_no"]).ksic_txt.copy()
    tabser = sf.set_index(["cc", "appl_no"]).tab
    exp.loc[tabser.index] = tabser.values
    exp.loc[fam.index] = fam.values
    mine_s = fx.set_index(["cc", "appl_no"]).k_mine
    eq = (exp == mine_s)
    R["a_candidates"] = int(len(eq)); R["a_equal_candidates"] = float(eq.mean()); R["a_mismatch"] = int((~eq).sum())
    R["a_by_cc"] = eq.groupby(level=0).mean().to_dict()
    R["a_changed_vs_stored_by_cc"] = (mine_s != fx.set_index(["cc", "appl_no"]).ksic_txt).groupby(level=0).sum().to_dict()
    log(R)
elif step == "b":
    a8 = pq.read_table(W + "/cache/txt_assign_r8.parquet", columns=["cc", "appl_no", "ksic_txt"]).to_pandas()
    ar = pq.read_table(W + "/cache/txt_assign_reg.parquet", columns=["cc", "appl_no", "ksic_txt"]).to_pandas()
    m = a8.merge(ar, on=["cc", "appl_no"], suffixes=("_8", "_r")); ch = m[m.ksic_txt_8 != m.ksic_txt_r]
    log("바뀐 문서", len(ch))
    br = pq.read_table(T3 + "/R3/cache_pub/base_rows.parquet", columns=["cc", "appl_no", "year", "ipc_raw"],
                       filters=[("appl_no", "in", sorted(set(ch.appl_no)))]).to_pandas()
    br = br.merge(ch, on=["cc", "appl_no"])
    p1 = re.compile(r"^([A-H]\d{2}[A-Z])(\d{1,4})/"); p2 = re.compile(r"^([A-H]\d{2}[A-Z])(\d{3,6})$")
    def mg(s):
        if not isinstance(s, str): return None
        s = s.replace(" ", ""); x = p1.match(s)
        if x: return x.group(1) + " " + str(int(x.group(2)))
        if p2.match(s): return s[:4] + " " + str(int(s[4:len(s) - 2]))
        return None
    br["mg"] = br.ipc_raw.map(mg); br = br.dropna(subset=["mg"]).drop_duplicates(["cc", "appl_no", "mg", "year", "ksic_txt_8", "ksic_txt_r"])
    plus = br.groupby(["ksic_txt_r", "mg", "cc", "year"]).size(); plus.index.names = ["ksic", "mg", "cc", "year"]
    minus = br.groupby(["ksic_txt_8", "mg", "cc", "year"]).size(); minus.index.names = ["ksic", "mg", "cc", "year"]
    dexp = plus.sub(minus, fill_value=0); dexp = dexp[dexp != 0]
    P8 = pd.read_parquet(W + "/cache/txt_r8/panel_ccy.parquet", columns=["ksic", "mg", "cc", "year", "n"]).set_index(["ksic", "mg", "cc", "year"]).n
    PR = pd.read_parquet(W + "/cache/txt_reg/panel_ccy.parquet", columns=["ksic", "mg", "cc", "year", "n"]).set_index(["ksic", "mg", "cc", "year"]).n
    dobs = PR.sub(P8, fill_value=0); dobs = dobs[dobs != 0]
    al = pd.concat([dexp.rename("exp"), dobs.rename("obs")], axis=1).fillna(0)
    R["b_cells_changed_exp"] = int((al.exp != 0).sum()); R["b_cells_changed_obs"] = int((al.obs != 0).sum())
    R["b_max_abs_diff"] = float((al.exp - al.obs).abs().max()); R["b_n_mismatch"] = int(((al.exp - al.obs).abs() > 0).sum())
    R["b_total_moved_docs_rows"] = int(plus.sum())
    log(R)
elif step == "c":
    P = pd.read_parquet(W + "/cache/txt_reg/panel_ccy.parquet", columns=["ksic", "mg", "cc", "year", "n"])
    M = pd.read_parquet(W + "/cache/mg_pub/panel_ccy.parquet", columns=["mg", "cc", "year", "n"])
    C = pd.read_parquet(W + "/cache/ctx_txt_reg.parquet")
    res = {}
    for t0 in (2010, 2014, 2018):
        Wn = P[P.year.between(t0 - 2, t0)]; Pn = P[P.year.between(t0 - 5, t0 - 3)]
        def agg(df, nm):
            a = df.groupby(["ksic", "mg"]).n.sum().rename(nm + "_tot"); k = df[df.cc == "KR"].groupby(["ksic", "mg"]).n.sum().rename(nm + "_kr")
            return pd.concat([a, k], axis=1)
        X = pd.concat([agg(Wn, "W"), agg(Pn, "P")], axis=1).fillna(0)
        X = X[(X.W_tot > 0) | (X.P_tot > 0)].reset_index()
        Ik = X.groupby("ksic")[["W_kr", "P_kr"]].transform("sum")
        X["peer_k"] = np.log((Ik.W_kr - X.W_kr + 1) / (Ik.P_kr - X.P_kr + 1))
        X["w"] = X.W_tot / X.groupby("mg").W_tot.transform("sum")
        kmom = (X.w * X.peer_k).groupby(X.mg).sum(min_count=1)
        nind = (X.w >= 0.05).groupby(X.mg).sum()
        mgt = M[M.year.between(t0 - 2, t0)].groupby("mg").n.sum()
        cov = (X.groupby("mg").W_tot.sum() / np.maximum(mgt.reindex(X.mg.unique()).fillna(0), 1)).clip(0, 1)
        c = C[C.t0 == t0].set_index("mg")
        ok = c.index.intersection(kmom.index)
        dk = (c.loc[ok, "c_ind_kmom"] - kmom.loc[ok]).abs(); dn = (c.loc[ok, "c_n_ind"] - nind.loc[ok]).abs(); dc = (c.loc[ok, "c_cov"] - cov.loc[ok]).abs()
        res[t0] = dict(n_mg=int(len(ok)), n_mg_ctx=int(len(c)), kmom_maxabs=float(np.nanmax(dk.values)), kmom_nan_mismatch=int((c.loc[ok, "c_ind_kmom"].isna() != kmom.loc[ok].isna()).sum()),
                       n_ind_maxabs=float(dn.max()), cov_maxabs=float(np.nanmax(dc.values)))
    R["c"] = res; log(R)
elif step == "e":
    from sklearn.metrics import roc_auc_score
    R8R = T3 + "/NONCIRC_R8/results"
    def load(f): return pd.read_parquet(f)
    prov = load(R8R + "/mg_ctx-prov_preds.parquet"); none = load(R8R + "/mg_ctx-none_preds.parquet")
    txt = {arm: load(f"{W}/res/runs/mg_ctx-txt__{arm}_preds.parquet") for arm in ("r8", "reg", "pub", "pure")}
    ORIG = list(range(2010, 2019))
    def boot(A, B, seed, nb=1000):
        rng = np.random.default_rng(seed); d = []; bs = np.zeros((9, nb))
        for i, v in enumerate(ORIG):
            a = A[A.t0 == v].set_index("cell"); b = B[B.t0 == v].set_index("cell").loc[a.index]
            y = a.yb.values; pa_, pb_ = a.p.values, b.p.values
            d.append(roc_auc_score(y, pa_) - roc_auc_score(y, pb_))
            I = rng.integers(0, len(y), (nb, len(y)))
            for j in range(nb):
                ii = I[j]; yy = y[ii]
                bs[i, j] = np.nan if yy.min() == yy.max() else roc_auc_score(yy, pa_[ii]) - roc_auc_score(yy, pb_[ii])
        bm = np.nanmean(bs, 0); d = np.array(d)
        return dict(d=round(float(d.mean()), 5), lo=round(float(np.nanpercentile(bm, 2.5)), 5), hi=round(float(np.nanpercentile(bm, 97.5)), 5), wins=int((d > 0).sum()))
    out = {}
    for arm in ("reg", "r8", "pure", "pub"):
        out[arm] = {"prov_vs_txt_seed20261005": boot(prov, txt[arm], 20261005)}
        if arm in ("reg", "pure"):
            for s in (1, 2, 3): out[arm][f"prov_vs_txt_seed{s}"] = boot(prov, txt[arm], s)
        out[arm]["txt_auc_mean"] = round(float(np.mean([roc_auc_score(txt[arm][txt[arm].t0 == v].yb, txt[arm][txt[arm].t0 == v].p) for v in ORIG])), 5)
        log(arm, out[arm])
    R["e"] = out
json.dump(R, open(f"{OUT}/v0_{step}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("end")
