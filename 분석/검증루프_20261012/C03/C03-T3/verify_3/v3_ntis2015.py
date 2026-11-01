import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import pyarrow.parquet as pq
import numpy as np
EXT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T3\verify_3"
pf = pq.ParquetFile(EXT + r"\ntis_project_raw.parquet")
cols = [c for c in pf.schema.names]
want = [c for c in ["pjt_no","year","secret_flag","gov_funds","private_funds","total_funds","performer_type","perf_type","ministry"] if c in cols]
print("cols:", cols)
t = pq.read_table(EXT + r"\ntis_project_raw.parquet", columns=want).to_pandas()
t["year"] = t["year"].astype(int)
y = t[t.year == 2015].copy()
res = {"n_2015": int(len(y)), "n_uniq": int(y.pjt_no.nunique())}
sf = y.secret_flag.astype(bool)
res["n_secret"] = int(sf.sum())
for c in ["gov_funds","total_funds"]+(["private_funds"] if "private_funds" in y else []):
    v = pd_v = y[c].astype(float)
    res[c+"_nonsecret_eok"] = float(v[~sf].sum()/1e8)
    res[c+"_secret_eok"] = float(v[sf].sum()/1e8)
    res[c+"_secret_nonnull"] = int(v[sf].notna().sum())
    res[c+"_secret_nonzero"] = int((v[sf].fillna(0)!=0).sum())
# alternative secret definition by performer text
for pc in ["performer_type","perf_type"]:
    if pc in y:
        res["performer_vals"] = y[pc].astype(str).value_counts().head(15).to_dict()
        alt = y[pc].astype(str).str.contains("보안")
        res["alt_secret_n"] = int(alt.sum()); res["alt_vs_flag_mismatch"] = int((alt != sf).sum())
OFF, DEF = 188747.0, 25441.0
g = res["gov_funds_nonsecret_eok"]; s = res["total_funds_secret_eok"]
res["ratio_nonsecret_gov"] = g/OFF
res["ratio_gov_plus_secret_total"] = (g+s)/OFF
res["ratio_nonsecret_vs_off_minus_def"] = g/(OFF-DEF)
res["ratio_gov_plus_secret_gov"] = (g+res["gov_funds_secret_eok"])/OFF
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(OUT+r"\v3_ntis2015.json","w",encoding="utf8"), ensure_ascii=False, indent=1)
