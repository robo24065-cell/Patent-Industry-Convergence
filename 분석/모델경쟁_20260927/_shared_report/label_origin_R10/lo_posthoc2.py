# -*- coding: utf-8 -*-
r"""[사후, PREREG 밖] (d) US 패밀리 문서 중 출원인 칸에 회사 표지가 없는(발명자 이름만 있는) 문서로 P1 재계산.
이런 문서는 출원인 이름으로 한국 등록업종을 찾기 어렵다. 출력 res_posthoc2.json"""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = os.path.join(MC, r"_shared_report\label_origin_R10"); CACHE = os.path.join(OUT, "cache")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
KSIC50 = set("15220 20202 20411 20423 20491 21220 23992 24191 24219 25914 25922 25924 25932 26111 26112 26121 26129 26211 26212 26223 26224 26291 26322 26323 26422 26429 26519 26529 27199 27211 27212 27301 28111 28122 28511 28902 29131 29132 29133 29150 29162 29210 29229 29241 29280 30393 31202 31312 33301 33309".split())
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='2GB'")
fa = con.sql(f"SELECT appl_no, applicant FROM read_parquet('{META}') WHERE cc='US'").df()
fam = pd.read_parquet(os.path.join(CACHE, "fam_reg.parquet")); fam = fam[fam.cc == "US"].merge(fa, on="appl_no", how="left")
corp_pat = r"CO\.|LTD|INC|CORP|COMPANY|LLC|GMBH|\bAG\b|ELECTRONICS|ELECTRIC|SAMSUNG|\bLG\b|HYUNDAI|SK |UNIVERSITY|INSTITUTE|FOUNDATION|KAISHA|LIMITED|L\.P\.|S\.A\.|B\.V\.|PLC|TECHNOLOG|INDUSTR|\bKIA\b|DISPLAY|CHEM"
fam["inventor_only"] = ~fam.applicant.fillna("").str.upper().str.contains(corp_pat, regex=True)
kr = pd.read_parquet(os.path.join(CACHE, "kr_lo.parquet"), columns=["appl_no", "ksic"]); kr["appl_no"] = kr.appl_no.astype(str)
pf = pd.read_parquet(os.path.join(CACHE, "pat_firm.parquet")); reg = pd.read_parquet(os.path.join(CACHE, "firm_reg.parquet"))
X = pf.merge(fam[["kr_appl_no", "fk", "inventor_only"]].rename(columns={"kr_appl_no": "appl_no"}).astype({"appl_no": str}), on="appl_no").merge(kr, on="appl_no")
J = {"us_links": int(len(fam)), "inventor_only_share": round(float(fam.inventor_only.mean()), 4), "rows": []}
rng = np.random.default_rng(3)
for src in ("nps", "dart"):
    R = reg[["applicantCd", src]].dropna(); R = R[R[src].isin(KSIC50)]
    for io in (True, False):
        Y = X[X.inventor_only == io].drop_duplicates(["applicantCd", "appl_no"]).merge(R.rename(columns={src: "r5"}), on="applicantCd")
        g = Y.assign(k=(Y.ksic == Y.r5).astype(float), f=(Y.fk == Y.r5).astype(float)).groupby("applicantCd")[["k", "f"]].mean()
        d = g.k.values - g.f.values; ii = rng.integers(0, len(d), (1000, len(d))); bs = d[ii].mean(1)
        J["rows"].append({"기준": src.upper(), "US 출원인칸=발명자만": io, "기업": len(g), "특허": len(Y), "KR라벨": round(float(g.k.mean()), 4),
                          "US라벨": round(float(g.f.mean()), 4), "KR-US": round(float(d.mean()), 4), "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4),
                          "KR-US 라벨일치": round(float((Y.ksic == Y.fk).mean()), 4)})
json.dump(J, open(os.path.join(OUT, "res_posthoc2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(J, ensure_ascii=False, indent=1))
