# -*- coding: utf-8 -*-
r"""C11-T2 verify_5: 지형도 정의 밖 변형(firstKR, tie_desc, freetab_ge1) 반박 시도. verify_1·s7 코드 미사용.
duckdb는 파일 추출(필터·조인)만, 배정 논리(최빈·표·상속)·RTA·순위는 numpy/pandas로 새로 작성.
쓰기: verify_5/v5_out.json, v5_rows.csv"""
import sys, json, time
sys.dont_write_bytecode = True; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; OUT = W + "/verify_5"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"
RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; KF = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"
TA = T3 + "/TXT_R9/txt_assign.parquet"; LINKS = BASE + "/분석/검증루프_20261012/C10/C10-T3/cache/links.parquet"
CCS = ["KR", "US", "JP", "CN", "EP"]
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
con = duckdb.connect(); con.execute(f"set memory_limit='2500MB'; set threads=2; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]
K50 = sorted(km.ksic); assert len(K50) == 50 and all(len(k) == len(K50[0]) for k in K50)
multi = set(km.code.value_counts().loc[lambda s: s > 1].index)
R = {}

# ---- 창 문서 ----
D = con.sql(f"""select a.cc, a.appl_no, a.ksic_prov, a.code, a.k_txt, a.k_free, r.sg from read_parquet('{ALT}') a
   left join read_parquet('{RC}') r on r.cc=a.cc and r.appl_no=a.appl_no where a.is_inv and a.appl_year>=2017 and a.appl_year<=2019""").df()
D["mg"] = D.sg.str.split("/", n=1).str[0]; D["sc"] = D.sg.str.slice(0, 4)
assert not D.duplicated(["cc", "appl_no"]).any(); R["window"] = D.cc.value_counts().to_dict(); log("window", R["window"])
kfree = pd.read_parquet(KF, columns=["appl_no", "k_free"]).dropna()
kfree = kfree.drop_duplicates("appl_no").set_index("appl_no").k_free
# ---- KR FREE 표 원천 ----
KRf = con.sql(f"""select a.appl_no, r.sg from read_parquet('{ALT}') a left join read_parquet('{RC}') r on r.cc=a.cc and r.appl_no=a.appl_no
   where a.cc='KR' and a.appl_year<=2019""").df()
KRf["k"] = KRf.appl_no.map(kfree); KRf = KRf[KRf.k.notna()].copy()
KRf["mg"] = KRf.sg.str.split("/", n=1).str[0]; KRf["sc"] = KRf.sg.str.slice(0, 4)
# ---- TXT: 저장 txt_assign(창 해외 + KR text) ----
TAw = con.sql(f"""select t.cc, t.appl_no, t.code, t.ksic_txt, t."method" m from read_parquet('{TA}') t
   join (select cc, appl_no from read_parquet('{ALT}') where cc<>'KR' and is_inv and appl_year>=2017 and appl_year<=2019) w on w.cc=t.cc and w.appl_no=t.appl_no""").df()
KRt = con.sql(f"""select t.appl_no, t.code, t.ksic_txt k, r.sg from read_parquet('{TA}') t left join read_parquet('{RC}') r on r.cc=t.cc and r.appl_no=t.appl_no
   where t.cc='KR' and t."method"='text'""").df()
KRt["mg"] = KRt.sg.str.split("/", n=1).str[0]; KRt["sc"] = KRt.sg.str.slice(0, 4)
KRcode = con.sql(f"select appl_no kr_appl_no, code kr_code, ksic_txt kr_k from read_parquet('{TA}') where cc='KR'").df()
L = con.sql(f"""select l.cc, l.appl_no, l.kr_appl_no from read_parquet('{LINKS}') l
   join (select cc, appl_no from read_parquet('{ALT}') where cc<>'KR' and is_inv and appl_year>=2017 and appl_year<=2019) w on w.cc=l.cc and w.appl_no=l.appl_no
   where l.link='REG'""").df().drop_duplicates()
R["reg_pairs_window"] = L.groupby("cc").size().to_dict(); log("links", R["reg_pairs_window"])

def pick_mode(df, keys, desc=False):
    """df: keys + k + n. 최빈, 동률 = 코드 작은(큰) 것. numpy lexsort로 구현."""
    kk = df.k.astype(int).values
    sec = kk if desc else -kk
    keyarr = pd.MultiIndex.from_frame(df[keys]).codes if len(keys) > 1 else [pd.factorize(df[keys[0]])[0]]
    order = np.lexsort((sec, df.n.values, *keyarr[::-1]))
    s = df.iloc[order]
    return s.drop_duplicates(keys, keep="last")  # lexsort 오름차순 -> 같은 키 안 마지막 = n 최대, 동률 시 sec 최대

def count_tab(df, keys, minn, desc):
    g = df.dropna(subset=[keys[-1]]); g = g[g[keys[-1]] != ""]
    g = g.groupby(keys + ["k"], sort=False).size().rename("n").reset_index()
    tot = g.groupby(keys, sort=False).n.transform("sum"); g = g[tot >= minn]
    return pick_mode(g, keys, desc)

def free_labels(rule, desc_inh, desc_tab, fmin):
    F = D[D.cc != "KR"][["cc", "appl_no", "sg", "mg", "sc"]].copy()
    lab = pd.Series(np.nan, index=F.index, dtype=object)
    for lvl in ("sc", "mg", "sg"):   # 낮은 우선순위부터 덮어쓰기
        tb = count_tab(KRf[[lvl, "k"]], [lvl], fmin, desc_tab).set_index(lvl).k
        v = F[lvl].map(tb); lab = v.where(v.notna(), lab)
    LL = L.assign(k=L.kr_appl_no.map(kfree)).dropna(subset=["k"])
    if rule == "first":
        inh = LL.sort_values(["cc", "appl_no", "kr_appl_no"]).drop_duplicates(["cc", "appl_no"])
    else:
        g = LL.groupby(["cc", "appl_no", "k"], sort=False).size().rename("n").reset_index()
        inh = pick_mode(g, ["cc", "appl_no"], desc_inh)
    key = F.cc + "|" + F.appl_no; im = pd.Series(inh.k.values, index=(inh.cc + "|" + inh.appl_no).values)
    v = key.map(im); lab = v.where(v.notna(), lab)
    return pd.Series(lab.values, index=key.values)

def txt_labels(rule, desc_inh, desc_tab, tmin):
    X = TAw.merge(D[D.cc != "KR"][["cc", "appl_no", "sg", "mg", "sc"]], on=["cc", "appl_no"], how="left")
    xm = X[X.code.isin(multi)].copy()
    lab = pd.Series(np.nan, index=xm.index, dtype=object)
    cm = count_tab(KRt[["code", "k"]], ["code"], 1, desc_tab).set_index("code").k
    lab = xm.code.map(cm)
    for lvl in ("sc", "mg", "sg"):
        tb = count_tab(KRt[["code", lvl, "k"]], ["code", lvl], tmin, desc_tab)
        v = xm.merge(tb, on=["code", lvl], how="left").k.values
        v = pd.Series(v, index=xm.index); lab = v.where(v.notna(), lab)
    LL = L.merge(xm[["cc", "appl_no", "code"]], on=["cc", "appl_no"]).merge(KRcode, on="kr_appl_no")
    LL = LL[LL.kr_code == LL.code].rename(columns={"kr_k": "k"})
    if rule == "first":
        inh = LL.sort_values(["cc", "appl_no", "kr_appl_no"]).drop_duplicates(["cc", "appl_no"])
    else:
        g = LL.groupby(["cc", "appl_no", "k"], sort=False).size().rename("n").reset_index()
        inh = pick_mode(g, ["cc", "appl_no"], desc_inh)
    v = xm.merge(inh[["cc", "appl_no", "k"]], on=["cc", "appl_no"], how="left").k.values
    v = pd.Series(v, index=xm.index); lab = v.where(v.notna(), lab)
    X["lab"] = X.ksic_txt; X.loc[xm.index, "lab"] = lab
    pool = pd.Series(X.lab.values, index=(X.cc + "|" + X.appl_no).values)
    return pool, X, xm, inh

def rta(n):
    s = n / n.sum(axis=0); return s.div(s.mean(axis=1), axis=0)

def counts(kr_series, for_lab):
    n = pd.DataFrame(0.0, index=K50, columns=CCS)
    n["KR"] = kr_series.dropna().value_counts().reindex(K50, fill_value=0).astype(float)
    Fd = D[D.cc != "KR"]
    for cc in CCS[1:]:
        m = (Fd.cc == cc).values
        n[cc] = pd.Series(for_lab[m]).dropna().value_counts().reindex(K50, fill_value=0).astype(float).values
    return n

KRw = D[D.cc == "KR"]; Fw = D[D.cc != "KR"]; fkey = (Fw.cc + "|" + Fw.appl_no).values
prov = rta(counts(KRw.ksic_prov, Fw.ksic_prov.values)); lp = prov.idxmax(axis=1); kp = set(lp.index[lp == "KR"])
w = km.assign(w=1.0 / km.groupby("code").ksic.transform("count"))
ncw = pd.DataFrame({cc: D[D.cc == cc][["code"]].merge(w, on="code").groupby("ksic").w.sum().reindex(K50, fill_value=0) for cc in CCS})
cweq = float(spearmanr(rta(ncw).KR, prov.KR).statistic); R["cweq_rho"] = cweq; log("cweq", cweq)
rows = []
def run(name, rule="mode", desc_inh=False, desc_tab=False, fmin=3, tmin=3, check=False):
    fl = free_labels(rule, desc_inh, desc_tab, fmin)
    pool, X, xm, inh = txt_labels(rule, desc_inh, desc_tab, tmin)
    fv = pd.Series(fkey).map(fl).values
    tv = pd.Series(fkey).map(pool); tv = tv.where(tv.notna(), pd.Series(Fw.k_txt.values)).values
    R[f"n_txt_inherit_{name}"] = int(len(inh))
    out = {}
    for scen, kr_free, kr_txt in (("F", KRw.k_free, KRw.k_txt), ("H", KRw.ksic_prov, KRw.ksic_prov)):
        for lab, kr, fo in (("free", kr_free, fv), ("txt", kr_txt, tv)):
            rt = rta(counts(kr, fo)); ld = rt.idxmax(axis=1); krl = set(ld.index[ld == "KR"])
            rho = float(spearmanr(rt.KR, prov.KR).statistic)
            rows.append(dict(arm=name, item=f"{lab}_{scen}", kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=len(krl), kept=len(krl & kp), rho_raw=rho))
            out[f"{lab}_{scen}"] = rho
    v = [cweq, out["txt_F"], out["free_F"]]
    R.setdefault("range_168", {})[name] = dict(min_raw=min(v), max_raw=max(v), once=f"{min(v):.2f}–{max(v):.2f}")
    log(name, {k: round(x, 4) for k, x in out.items()}, R["range_168"][name]["once"])
    return fv, tv

fv0, tv0 = run("REG_base", check=True)
# 기준 점검: s7 결과 파일의 reg 값 및 alt 저장 라벨 대조(비연결 해외 FREE = alt 저장 k_free 이어야: 표 재구축 점검)
alt_free = Fw.k_free.values
linked = pd.Series(fkey).isin(set(L.cc + "|" + L.appl_no)).values
R["free_tab_eq_alt_nonlinked"] = float(np.mean(pd.Series(fv0[~linked]).fillna("NA").values == pd.Series(alt_free[~linked]).fillna("NA").values))
log("FREE 비연결 = alt 저장", R["free_tab_eq_alt_nonlinked"])
tr = pd.read_parquet(W + "/cache/txt_assign_reg.parquet", columns=["cc", "appl_no", "ksic_txt"])
trm = pd.Series(tr.ksic_txt.values, index=(tr.cc + "|" + tr.appl_no).values); del tr
inpool = pd.Series(fkey).isin(trm.index).values
R["txt_pool_eq_c11_reg"] = float(np.mean(pd.Series(fkey[inpool]).map(trm).values == tv0[inpool])); del trm
log("TXT 풀 = C11 txt_assign_reg", R["txt_pool_eq_c11_reg"])
run("REG_firstKR", rule="first")
run("REG_tie_desc_inherit", desc_inh=True)
run("REG_tie_desc_full", desc_inh=True, desc_tab=True)
run("REG_tie_desc_tab_only", desc_tab=True)
run("REG_freetab_ge1", fmin=1)
run("REG_freetab_ge5", fmin=5)
run("REG_txttab_ge1", tmin=1)
run("REG_firstKR_tiefull_ge1", rule="first", desc_inh=True, desc_tab=True, fmin=1)
T = pd.DataFrame(rows); T["rho3"] = T.rho_raw.round(3)
T.to_csv(OUT + "/v5_rows.csv", index=False, encoding="utf-8-sig"); print(T.to_string())
R["rows"] = rows
json.dump(R, open(OUT + "/v5_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R["range_168"], ensure_ascii=False, indent=1)); log("end")
