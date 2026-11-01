# -*- coding: utf-8 -*-
"""기업 연결 렌즈 파일럿 0단계: 등록특허(제공범위, KSIC 라벨) → 출원인코드 → 법인번호 → DART corp_code 연계표.
출력: cache/pat_firm.parquet (rgst_no, applicantCd, jurir, corp_code, ksic, applDate, rgstDate), cache/link_stats.json
모든 DB는 읽기 전용."""
import os, sqlite3, json
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
os.makedirs(OUT, exist_ok=True)
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
DARTDB = os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite")

con = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
pat = pd.read_sql("select rgst_no, ksic, applDate, rgstDate, applNo from patent", con)
app = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", con)
con.close()
ca = pd.read_parquet(os.path.join(BASE, r"공모전데이터\외부데이터\DART\corp_applicant.parquet"))
ca = ca[["출원인코드", "jurir"]].rename(columns={"출원인코드": "applicantCd"})
ca = ca[ca.jurir.fillna("").str.len() == 13].drop_duplicates("applicantCd")
dc = sqlite3.connect(f"file:{DARTDB}?mode=ro", uri=True)
dart = pd.read_sql("select corp_code, corp_name, jurir_no, induty_code, est_dt, corp_cls from company where status='OK'", dc)
dart = dart[dart.jurir_no.fillna("").str.len() == 13].drop_duplicates("jurir_no")

a = app.merge(ca, on="applicantCd", how="left").merge(dart[["jurir_no", "corp_code"]], left_on="jurir", right_on="jurir_no", how="left")
a = a.merge(pat, on="rgst_no", how="left")
a.drop(columns=["jurir_no"]).to_parquet(os.path.join(OUT, "pat_firm.parquet"), index=False)
dart.to_parquet(os.path.join(OUT, "dart_firms.parquet"), index=False)

st = {
    "registered_patents": int(len(pat)),
    "applicant_rows": int(len(app)),
    "applicant_rows_with_corp_jurir": int(a.jurir.notna().sum()),
    "applicant_rows_with_dart": int(a.corp_code.notna().sum()),
    "patents_with_any_dart_applicant": int(a.loc[a.corp_code.notna(), "rgst_no"].nunique()),
    "unique_jurir": int(a.jurir.nunique()),
    "unique_dart_firms_with_patents": int(a.corp_code.nunique()),
}
f = a[a.corp_code.notna()].groupby("corp_code").rgst_no.nunique().rename("n_pat").reset_index().merge(dart, on="corp_code")
st["dart_firms_by_cls"] = f.corp_cls.value_counts().to_dict()
st["dart_firms_ge5pat_by_cls"] = f[f.n_pat >= 5].corp_cls.value_counts().to_dict()
json.dump(st, open(os.path.join(OUT, "link_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(st, ensure_ascii=False, indent=1))
