# -*- coding: utf-8 -*-
# verify_1: NPS 코드 출처(이름 매핑 경로)별 DART 일치. 읽기만. 쓰기: verify_1/v1_src.json
import os, sys, json
sys.dont_write_bytecode = True
import pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.dirname(os.path.abspath(__file__))
S50 = set(['15220','20202','20411','20423','20491','21220','23992','24191','24219','25914','25922','25924','25932','26111','26112','26121','26129','26211','26212','26223','26224','26291','26322','26323','26422','26429','26519','26529','27199','27211','27212','27301','28111','28122','28511','28902','29131','29132','29133','29150','29162','29210','29229','29241','29280','30393','31202','31312','33301','33309'])
U = pd.read_parquet(os.path.join(HERE, r"cache\firm_U.parquet"), columns=["applicantCd", "nps", "dart", "cls_j"])
H = pd.read_parquet(os.path.join(HERE, r"cache\nps_hist_U.parquet"))
h = H[H.snap == "201912"].drop_duplicates("applicantCd")
U = U.merge(h[["applicantCd", "k5", "k5src", "indcd_c"]], on="applicantCd", how="left")
U["d5"] = U.dart.where(U.dart.str.len() == 5)
g = U[U.d5.isin(S50) & U.nps.notna()].copy()
J = {"G1n": len(g), "hist_k5_eq_reg_nps": round(float((g.k5 == g.nps).mean()), 4)}
g["src"] = g.k5src.fillna("NA").replace("", "ksic50def_only")
J["by_src"] = {k: {"n": int(len(x)), "eq5": round(float((x.d5 == x.nps).mean()), 4), "nps_in50": round(float(x.nps.isin(S50).mean()), 4)} for k, x in g.groupby("src")}
J["by_cls"] = {k: {"n": int(len(x)), "eq5": round(float((x.d5 == x.nps).mean()), 4)} for k, x in g.groupby(g.cls_j.fillna("NA"))}
# 불일치 쌍 상위
mis = g[g.d5 != g.nps]
J["top_mismatch_pairs"] = mis.groupby(["d5", "nps"]).size().sort_values(ascending=False).head(15).reset_index().values.tolist()
J["indcd_len"] = g.indcd_c.str.len().value_counts(dropna=False).astype(int).to_dict()
json.dump(J, open(os.path.join(OUT, "v1_src.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(J, ensure_ascii=True, default=str))
