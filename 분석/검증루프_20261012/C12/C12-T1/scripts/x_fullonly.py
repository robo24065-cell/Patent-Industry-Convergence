import sys, os, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, "scripts")
import n5lib
R = pd.read_parquet("cache/link_records.parquet"); M = pd.read_parquet("cache/link_matches.parquet")
M["apps"] = M.names.fillna("").str.split("|").map(lambda xs: [n5lib.norm(x) for x in xs if x])
st = M[M.strict == 1].drop_duplicates(["rid", "appl_no"])
pair = st.explode("apps").groupby(["comp", "apps"]).size()
g = M.groupby(["rid", "appl_no"]).strict.max().reset_index()
fo = g[g.strict == 0][["rid","appl_no"]].merge(M[M.strict==0].drop_duplicates(["rid","appl_no"]), on=["rid","appl_no"]).merge(R[["rid","PATT_APPLICATION_NUMBER","PUBL_NUMBER","Original_Application_Number","Original_Publication_Number","app_reason"]], on="rid")
fo["agree"] = [any(pair.get((r.comp, a), 0) >= 2 for a in r.apps) for r in fo.itertuples()]
fo["app1"] = fo.apps.str[0]
fo["new_patent"] = ~fo.appl_no.isin(set(st.appl_no))
print(fo.groupby(["kt","app_reason"]).agg(n=("agree","size"), agree=("agree","mean"), new=("new_patent","sum")).to_string())
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(fo[~fo.agree][["kt","key","PATT_APPLICATION_NUMBER","PUBL_NUMBER","Original_Publication_Number","comp","app1","ksic","new_patent"]].head(40).to_string())
print(fo[fo.new_patent & (fo.ksic=="26429")].groupby(["kt","agree"]).size())
