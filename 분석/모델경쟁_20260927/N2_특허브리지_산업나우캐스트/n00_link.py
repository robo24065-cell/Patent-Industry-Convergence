# -*- coding: utf-8 -*-
"""N2 연계 0단계 (파일럿 f00_link.py 복사·수정): 제공범위 KR 등록특허(KSIC 라벨) → 출원인코드 → 법인번호(CORP_APPLICANT)
→ DART jurir_no → corp_code. 모든 DB 읽기 전용.
출력: cache/pat_firm.parquet, cache/dart_firms.parquet, ext_link_stats.csv(연계율 단계표)"""
import os, sqlite3, json
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "cache")
os.makedirs(OUT, exist_ok=True)
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
DARTDB = os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite")

con = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
pat = pd.read_sql("select rgst_no, ksic, applDate, rgstDate, applNo, cndrtExptnDate from patent", con)
app = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", con)
con.close()
ca = pd.read_parquet(os.path.join(BASE, r"공모전데이터\외부데이터\DART\corp_applicant.parquet"))
ca = ca[["출원인코드", "jurir"]].rename(columns={"출원인코드": "applicantCd"})
ca = ca[ca.jurir.fillna("").str.len() == 13].drop_duplicates("applicantCd")
dc = sqlite3.connect(f"file:{DARTDB}?mode=ro", uri=True)
dart = pd.read_sql("select corp_code, corp_name, jurir_no, induty_code, est_dt, corp_cls from company where status='OK'", dc)
dc.close()
dart = dart[dart.jurir_no.fillna("").str.len() == 13].drop_duplicates("jurir_no")

a = app.merge(ca, on="applicantCd", how="left").merge(dart[["jurir_no", "corp_code"]], left_on="jurir", right_on="jurir_no", how="left")
a = a.merge(pat, on="rgst_no", how="left")
a.drop(columns=["jurir_no"]).to_parquet(os.path.join(OUT, "pat_firm.parquet"), index=False)
dart.to_parquet(os.path.join(OUT, "dart_firms.parquet"), index=False)

rows = [
    ("1_제공범위 KR 등록특허", len(pat), "건"),
    ("2_출원인 행(특허×출원인)", len(app), "행"),
    ("3_출원인코드→법인번호(CORP_APPLICANT) 연결 행", int(a.jurir.notna().sum()), "행"),
    ("4_법인번호→DART corp_code 연결 행", int(a.corp_code.notna().sum()), "행"),
    ("5_DART 연결 출원인이 1명 이상인 특허", int(a.loc[a.corp_code.notna(), "rgst_no"].nunique()), "건"),
    ("6_연결된 고유 법인번호", int(a.jurir.nunique()), "개"),
    ("7_특허 보유 DART 기업(corp_code)", int(a.corp_code.nunique()), "사"),
]
L = pd.DataFrame(rows, columns=["단계", "값", "단위"])
L["비율_기준대비"] = [1, 1, L.값[2] / L.값[1], L.값[3] / L.값[1], L.값[4] / L.값[0], None, None]
L.to_csv(os.path.join(HERE, "ext_link_stats.csv"), index=False, encoding="utf-8-sig")
f = a[a.corp_code.notna()].groupby("corp_code").rgst_no.nunique().rename("n_pat").reset_index().merge(dart, on="corp_code")
print(L.to_string()); print(f.corp_cls.value_counts().to_dict())
