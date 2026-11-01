# -*- coding: utf-8 -*-
"""IPO 레이더 파일럿 1단계: KIND 상장 사건 → 법인(법인번호·사업자번호) → 출원인코드.
외부 호출 없음(OpenDART API 미사용). 연결 경로:
  KIND 종목코드 → dart_corpcode.csv(corp_code, 현재 DART 상호)
   (a) dart_company.sqlite(ro)에 있으면 그 법인번호·사업자번호
   (b) 없으면 DART 현재 상호 ↔ CORP_APPLICANT(2026-07) 현재 상호 정규화 '유일' 일치 → 법인번호·사업자번호
  법인번호·사업자번호 → CORP_APPLICANT의 모든 출원인코드(같은 법인번호 또는 같은 사업자번호)
출력: cache/ipo_events.parquet (출원인코드 × 상장사건), cache/ipo_link_stats.json
상장사건은 모든 연도를 연결만 한다. 레이블 계산(p2)은 dev에서 상장일 ≤ 2023-12-31만 쓴다."""
import os, re, json, sqlite3, sys
import pandas as pd
sys.dont_write_bytecode = True
R7 = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(R7)
BASE = os.path.dirname(os.path.dirname(ROOT))
EXT = os.path.join(ROOT, "_shared", "ext")
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|재단법인|사단법인|\(주\)|\(유\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())

K = pd.read_csv(os.path.join(EXT, "r7_data_kind_listing.csv"), dtype=str)
K = K[(K["증권구분"] == "주권") & (K["국적"] == "대한민국") & ~K["회사명"].str.contains("스팩|기업인수목적")].copy()
cc = pd.read_csv(os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_corpcode.csv"), dtype=str).dropna(subset=["stock_code"])
cc = cc.drop_duplicates("stock_code")
con = sqlite3.connect(f"file:{os.path.join(BASE, r'공모전데이터/외부데이터/DART/dart_company.sqlite')}?mode=ro", uri=True)
dc = pd.read_sql("select corp_code, jurir_no, bizr_no, corp_cls from company", con)
CA = pd.read_parquet(os.path.join(BASE, r"공모전데이터\외부데이터\DART\corp_applicant.parquet"))
CA["nn"] = CA["출원인명"].map(norm)

K = K.merge(cc[["stock_code", "corp_code", "corp_name"]], left_on="종목코드", right_on="stock_code", how="left")
# 종목코드가 corpcode에 없으면 KIND 상호로 유일 일치
nm_cc = pd.read_csv(os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_corpcode.csv"), dtype=str)
nm_cc["nn"] = nm_cc.corp_name.map(norm); u = nm_cc.groupby("nn").corp_code.nunique()
nm_map = nm_cc[nm_cc.nn.map(u) == 1].set_index("nn")
miss = K.corp_code.isna()
K.loc[miss, "corp_code"] = K.loc[miss, "회사명"].map(norm).map(nm_map.corp_code)
K.loc[miss, "corp_name"] = K.loc[miss, "회사명"].map(norm).map(nm_map.corp_name)
K = K.merge(dc, on="corp_code", how="left")
K["route"] = K.jurir_no.notna().map({True: "dart_company", False: ""})
# (b) 이름 경로: DART 현재 상호와 KIND 상장 당시 상호 둘 다 시도, CORP_APPLICANT 정규화 상호 '유일' 일치(같은 법인번호 하나)
g = CA[CA.jurir.fillna("") != ""].groupby("nn").jurir.nunique()
uni = CA[CA.nn.map(g) == 1].drop_duplicates("nn").set_index("nn")
for col in ["corp_name", "회사명"]:
    m = K.jurir_no.isna()
    key = K.loc[m, col].map(norm)
    hit = key.map(uni.jurir)
    K.loc[m, "jurir_no"] = hit
    K.loc[m, "bizr_no"] = key.map(uni.bizr)
    K.loc[m & K.jurir_no.notna(), "route"] = f"name_{col}"
K["listing_date"] = pd.to_datetime(K["상장일"])
# 법인 → 출원인코드
ev = []
CAj = CA[CA.jurir.fillna("") != ""][["출원인코드", "jurir"]]
CAb = CA[CA.bizr.fillna("") != ""][["출원인코드", "bizr"]]
for key, ca, kc in [("jurir", CAj, "jurir_no"), ("bizr", CAb, "bizr_no")]:
    x = K[K[kc].fillna("") != ""].merge(ca, left_on=kc, right_on=key)
    ev.append(x[["출원인코드", "종목코드", "회사명", "시장", "상장유형", "listing_date", "route"]])
E = pd.concat(ev).drop_duplicates(["출원인코드", "종목코드", "listing_date"]).rename(columns={"출원인코드": "applicantCd"})
E.to_parquet(os.path.join(R7, "cache", "ipo_events.parquet"), index=False)
K.to_parquet(os.path.join(R7, "cache", "kind_linked.parquet"), index=False)
yr = K["상장일"].str[:4]
st = {"KIND 국내 주권 비스팩 상장사건": int(len(K)),
      "법인번호 확보율(전체)": round(float(K.jurir_no.notna().mean()), 3),
      "경로별": K.route.value_counts().to_dict(),
      "연도별 법인번호 확보율(2012–2023)": {y: round(float(g.jurir_no.notna().mean()), 3) for y, g in K[yr.between("2012", "2023")].groupby(yr)},
      "출원인코드 연결 상장사건 수": int(E[["종목코드", "listing_date"]].drop_duplicates().shape[0]),
      "연결 출원인코드 수": int(E.applicantCd.nunique())}
json.dump(st, open(os.path.join(R7, "cache", "ipo_link_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(st, ensure_ascii=False, indent=1))
