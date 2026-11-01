# -*- coding: utf-8 -*-
"""verify_5 보조: FREE 표 동률을 균등 분할(중립)했을 때 free_F·free_H ρ. 상속은 REG 최빈(동률 작은 코드, 사전 정의)."""
import sys, json; sys.dont_write_bytecode = True; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, duckdb
from scipy.stats import spearmanr
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"; MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; LINKS = BASE + "/분석/검증루프_20261012/C10/C10-T3/cache/links.parquet"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"; RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; KF = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"
CCS = ["KR", "US", "JP", "CN", "EP"]
con = duckdb.connect(); con.execute(f"set memory_limit='2500MB'; set threads=2; set temp_directory='{W}/cache/_tmp'")
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str); K50 = sorted(km.ksic)
kf = pd.read_parquet(KF).dropna().drop_duplicates("appl_no").set_index("appl_no").k_free
K = con.sql(f"select a.appl_no, r.sg from read_parquet('{ALT}') a left join read_parquet('{RC}') r on r.cc=a.cc and r.appl_no=a.appl_no where a.cc='KR' and a.appl_year<=2019").df()
K["k"] = K.appl_no.map(kf); K = K.dropna(subset=["k", "sg"]); K = K[K.sg != ""]; K["mg"] = K.sg.str.split("/", n=1).str[0]; K["sc"] = K.sg.str[:4]
D = con.sql(f"""select a.cc, a.appl_no, a.ksic_prov, a.k_free, r.sg from read_parquet('{ALT}') a left join read_parquet('{RC}') r on r.cc=a.cc and r.appl_no=a.appl_no
  where a.is_inv and a.appl_year between 2017 and 2019""").df()
D["mg"] = D.sg.str.split("/", n=1).str[0]; D["sc"] = D.sg.str[:4]
Fw = D[D.cc != "KR"].reset_index(drop=True); KRw = D[D.cc == "KR"]
L = con.sql(f"select distinct cc, appl_no, kr_appl_no from read_parquet('{LINKS}') where link='REG'").df(); L["k"] = L.kr_appl_no.map(kf); L = L.dropna(subset=["k"])
g = L.groupby(["cc", "appl_no", "k"]).size().rename("n").reset_index(); g["ki"] = g.k.astype(int)
g = g.sort_values(["cc", "appl_no", "n", "ki"], ascending=[True, True, False, True]).drop_duplicates(["cc", "appl_no"])
inh = pd.Series(g.k.values, index=(g.cc + "|" + g.appl_no).values)
key = Fw.cc + "|" + Fw.appl_no; hinh = key.map(inh)
# 표 수준 결정(sg > mg > sc), 동률 top 집합
lvl_of = pd.Series(None, index=Fw.index, dtype=object); tops = {}
for lvl in ("sc", "mg", "sg"):
    t = K.groupby([lvl, "k"]).size().rename("n").reset_index(); t = t[t.groupby(lvl).n.transform("sum") >= 3]
    t = t[t.n == t.groupby(lvl).n.transform("max")]; t["w"] = 1.0 / t.groupby(lvl).k.transform("size"); tops[lvl] = t
    has = Fw[lvl].isin(set(t[lvl])); lvl_of[has] = lvl
rows = []
def rta(n):
    s = n / n.sum(axis=0); return s.div(s.mean(axis=1), axis=0)
def table_counts(mode):
    n = pd.DataFrame(0.0, index=K50, columns=CCS[1:])
    for cc in CCS[1:]:
        m = (Fw.cc == cc)
        a = hinh[m & hinh.notna()].value_counts()
        tot = a.reindex(K50, fill_value=0).astype(float)
        rest = Fw[m & hinh.isna()]
        for lvl in ("sg", "mg", "sc"):
            r = rest[lvl_of[rest.index] == lvl][[lvl]]
            t = tops[lvl]
            if mode == "frac":
                x = r.merge(t, on=lvl).groupby("k").w.sum()
            else:
                tt = t.assign(ki=t.k.astype(int)).sort_values([lvl, "ki"], ascending=[True, mode == "asc"]).drop_duplicates(lvl)
                x = r.merge(tt, on=lvl).groupby("k").size().astype(float)
            tot = tot.add(x.reindex(K50, fill_value=0), fill_value=0)
        n[cc] = tot.values
    return n
prov_n = pd.DataFrame({cc: (KRw if cc == "KR" else Fw[Fw.cc == cc]).ksic_prov.value_counts().reindex(K50, fill_value=0).astype(float) for cc in CCS})
prov = rta(prov_n)
out = {}
for mode in ("asc", "desc", "frac"):
    nf = table_counts(mode)
    for scen, kr in (("F", KRw.k_free), ("H", KRw.ksic_prov)):
        n = nf.copy(); n.insert(0, "KR", kr.dropna().value_counts().reindex(K50, fill_value=0).astype(float).values)
        rt = rta(n); out[f"{mode}_free_{scen}"] = dict(rho=float(spearmanr(rt.KR, prov.KR).statistic), kr_rta_gt1=int((rt.KR > 1).sum()))
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(W + "/verify_5/v5_frac.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
