# -*- coding: utf-8 -*-
r"""C10-T3 verify_4 v4a: 수정판 T3 캐시(t3_free_reg/base_rows)의 해외 라벨이 '올바른 연결(fam_reg) 상속 → IPC 표' 규칙과 맞는지,
원천에서 독립으로 기대 라벨을 만들어 대조한다(작업자 s1·s2·alt_fix·links.parquet 미사용, pandas/pyarrow만).
같은 코드로 결함 연결(R8 fam_link)을 쓰면 c5 저장 캐시(t3cache_hyb_free)가 재현되는지도 본다(규칙 재현 점검).
나라별(US·JP·EP)로 처리(메모리). CN은 두 연결이 같아 생략. KR 레코드 동일성도 확인.
읽기만: _shared/meta, R7 rec_conc, T1 TXT_R9 kr_txtfree, label_origin_R10 fam_reg, _ideas_R8 fam_link, R3 base_rows, c5 t3cache_hyb_free, C10 t3_free_reg
쓰기: verify_4/v4a_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
ARM = sys.argv[1] if len(sys.argv) > 1 else "free_reg"
SRC = os.path.join(MC, "T3_기술셀부상예측", "R3", "cache_pub", "base_rows.parquet")
STO = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10", "t3cache_hyb_free", "base_rows.parquet")
FIX = os.path.join(W, "cache", f"t3_{ARM}", "base_rows.parquet")
meta = ds.dataset(os.path.join(MC, "_shared", "meta"), format="parquet")
rcd = ds.dataset(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), format="parquet")


def obj(df):
    return df.astype({c: object for c in df.columns if str(df[c].dtype) == "category"})


def rd(path, cc, cols=("appl_no", "ksic")):
    return obj(ds.dataset(path, format="parquet").to_table(columns=list(cols), filter=ds.field("cc") == cc).to_pandas())


# KR 표(출원 ≤2019, rec_conc sg, kr_txtfree k_free): sg→mg→sc 최빈, 최소 3건, 동률은 KSIC 정수 작은 것
K = obj(meta.to_table(columns=["appl_no", "appl_year"], filter=(ds.field("cc") == "KR")).to_pandas())
rk = obj(rcd.to_table(columns=["appl_no", "sg"], filter=ds.field("cc") == "KR").to_pandas())
kf = pq.read_table(os.path.join(MC, "T1_산업대기실", "TXT_R9", "cache", "kr_txtfree.parquet")).to_pandas().dropna()
Tk = K[K.appl_year <= 2019].merge(rk, on="appl_no").merge(kf, on="appl_no")
Tk["mg"] = Tk.sg.str.split("/").str[0]; Tk["sc"] = Tk.sg.str[:4]


def mode_tab(df, key):
    g = df[df[key].notna() & (df[key] != "")].groupby([key, "k_free"]).size().rename("n").reset_index()
    g = g[g.groupby(key).n.transform("sum") >= 3].assign(kk=lambda d: d.k_free.astype(int))
    return g.sort_values([key, "n", "kk"], ascending=[True, False, True]).drop_duplicates(key).set_index(key).k_free


TAB = {lvl: mode_tab(Tk, lvl) for lvl in ("sg", "mg", "sc")}
log("KR 표", {k: len(v) for k, v in TAB.items()})
reg = pq.read_table(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"]).to_pandas()
r8 = pq.read_table(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet")).to_pandas()
kfm = kf.rename(columns={"appl_no": "kr_appl_no"})


def inherit(link, cc, dedup):
    x = link[link.cc == cc]
    if dedup: x = x.drop_duplicates()
    x = x.merge(kfm, on="kr_appl_no")
    g = x.groupby(["appl_no", "k_free"]).size().rename("n").reset_index().assign(kk=lambda d: d.k_free.astype(int))
    return g.sort_values(["appl_no", "n", "kk"], ascending=[True, False, True]).drop_duplicates("appl_no").set_index("appl_no").k_free


OUT = {"arm": ARM}
# KR 레코드 동일성
ks = rd(STO, "KR").sort_values("appl_no").reset_index(drop=True); kx = rd(FIX, "KR").sort_values("appl_no").reset_index(drop=True)
OUT["KR_identical"] = bool(len(ks) == len(kx) and (ks.appl_no.values == kx.appl_no.values).all() and (ks.ksic.values == kx.ksic.values).all())
del ks, kx
log("KR 동일", OUT["KR_identical"])
for cc in ("US", "JP", "EP"):
    P = rd(SRC, cc, ("appl_no",)).drop_duplicates()
    rc = obj(rcd.to_table(columns=["appl_no", "sg"], filter=ds.field("cc") == cc).to_pandas()).drop_duplicates("appl_no")
    P = P.merge(rc, on="appl_no", how="left"); P["mg"] = P.sg.str.split("/").str[0]; P["sc"] = P.sg.str[:4]
    tab = pd.Series(index=P.index, dtype=object)
    for lvl in ("sg", "mg", "sc"):
        tab = tab.fillna(P[lvl].map(TAB[lvl]))
    i_reg = P.appl_no.map(inherit(reg, cc, True)); i_r8 = P.appl_no.map(inherit(r8, cc, False))
    P["exp_reg"] = i_reg.fillna(tab); P["exp_r8"] = i_r8.fillna(tab); P["tab"] = tab
    P["has_reg"] = i_reg.notna(); P["has_r8"] = i_r8.notna()
    S = rd(STO, cc).rename(columns={"ksic": "k_sto"}); X = rd(FIX, cc).rename(columns={"ksic": "k_fix"})
    P = P.merge(S, on="appl_no", how="left").merge(X, on="appl_no", how="left")
    eq = lambda a, b: float(((P[a] == P[b]) | (P[a].isna() & P[b].isna())).mean())
    r = dict(n_pool=int(len(P)), n_sto=int(P.k_sto.notna().sum()), n_fix=int(P.k_fix.notna().sum()),
             share_has_reg=round(float(P.has_reg.mean()), 4), share_has_r8=round(float(P.has_r8.mean()), 4),
             fix_eq_exp_reg=round(eq("k_fix", "exp_reg"), 5), sto_eq_exp_r8=round(eq("k_sto", "exp_r8"), 5),
             fix_eq_exp_reg_among_has_reg=round(float((P.k_fix[P.has_reg] == P.exp_reg[P.has_reg]).mean()), 5),
             changed_fix_vs_sto=round(float((P.k_fix != P.k_sto).mean()), 4),
             fix_eq_tab_among_r8_only=round(float((P.k_fix[P.has_r8 & ~P.has_reg] == P.tab[P.has_r8 & ~P.has_reg]).mean()), 5),
             # 라벨 품질: 제공 KSIC와 일치(상속 대상 문서만)
             )
    prov = rd(SRC, cc, ("appl_no", "ksic")).drop_duplicates("appl_no").rename(columns={"ksic": "prov"})
    P = P.merge(prov, on="appl_no", how="left")
    for nm, msk in (("has_reg", P.has_reg), ("has_r8", P.has_r8)):
        r[f"agree_prov_{nm}"] = dict(fix=round(float((P.k_fix[msk] == P.prov[msk]).mean()), 4), sto=round(float((P.k_sto[msk] == P.prov[msk]).mean()), 4), n=int(msk.sum()))
    mis = P[(P.k_fix != P.exp_reg) & ~(P.k_fix.isna() & P.exp_reg.isna())]
    r["mismatch_examples"] = mis.head(5)[["appl_no", "k_fix", "exp_reg", "k_sto", "has_reg", "has_r8"]].astype(str).values.tolist()
    OUT[cc] = r; log(cc, {k: v for k, v in r.items() if k != "mismatch_examples"})
    del P, S, X, prov, rc
    json.dump(OUT, open(os.path.join(V, f"v4a_out_{ARM}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done")
