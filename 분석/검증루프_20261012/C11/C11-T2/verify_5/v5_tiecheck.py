# -*- coding: utf-8 -*-
"""verify_5 보조: FREE 표 동률 빈도와 동률 규칙 반전 시 해외 창 문서 라벨 변화(다른 구현: groupby max)."""
import sys, json; sys.dont_write_bytecode = True; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, duckdb
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"; MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"; RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; KF = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"
con = duckdb.connect(); con.execute(f"set memory_limit='2500MB'; set threads=2; set temp_directory='{W}/cache/_tmp'")
kf = pd.read_parquet(KF).dropna().drop_duplicates("appl_no").set_index("appl_no").k_free
K = con.sql(f"select a.appl_no, r.sg from read_parquet('{ALT}') a left join read_parquet('{RC}') r on r.cc=a.cc and r.appl_no=a.appl_no where a.cc='KR' and a.appl_year<=2019").df()
K["k"] = K.appl_no.map(kf); K = K.dropna(subset=["k", "sg"]); K = K[K.sg != ""]
K["mg"] = K.sg.str.split("/", n=1).str[0]; K["sc"] = K.sg.str[:4]
Fw = con.sql(f"""select a.cc, a.appl_no, a.k_free, a.m_free, r.sg from read_parquet('{ALT}') a left join read_parquet('{RC}') r on r.cc=a.cc and r.appl_no=a.appl_no
  where a.cc<>'KR' and a.is_inv and a.appl_year between 2017 and 2019""").df()
Fw["mg"] = Fw.sg.str.split("/", n=1).str[0]; Fw["sc"] = Fw.sg.str[:4]
out = {"n_KR_rows": len(K)}
lab = {}
for desc in (False, True):
    v = pd.Series(np.nan, index=Fw.index, dtype=object)
    for lvl in ("sc", "mg", "sg"):
        g = K.groupby([lvl, "k"]).size().rename("n").reset_index()
        g = g[g.groupby(lvl).n.transform("sum") >= 3]
        mx = g.groupby(lvl).n.transform("max"); top = g[g.n == mx].copy(); top["ki"] = top.k.astype(int)
        if not desc: out[f"tie_keys_{lvl}"] = int((top.groupby(lvl).size() > 1).sum()); out[f"keys_{lvl}"] = int(top[lvl].nunique())
        pick = top.loc[(top.groupby(lvl).ki.idxmax() if desc else top.groupby(lvl).ki.idxmin())].set_index(lvl).k
        m = Fw[lvl].map(pick); v = m.where(m.notna(), v)
    lab[desc] = v
nonfam = ~Fw.m_free.fillna("").eq("family")
out["asc_eq_stored_ipc"] = float((lab[False][Fw.m_free.fillna("").str.startswith("ipc_")] == Fw.k_free[Fw.m_free.fillna("").str.startswith("ipc_")]).mean())
chg = (lab[False].fillna("NA") != lab[True].fillna("NA"))
out["table_docs_changed_by_tie_reverse"] = {cc: [int(chg[(Fw.cc == cc)].sum()), int((Fw.cc == cc).sum())] for cc in ["US", "JP", "CN", "EP"]}
out["share_changed_all_foreign"] = float(chg.mean())
# 바뀐 문서가 몰리는 KSIC
out["top_changed_to"] = lab[True][chg].value_counts().head(8).to_dict(); out["top_changed_from"] = lab[False][chg].value_counts().head(8).to_dict()
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(W + "/verify_5/v5_tiecheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
