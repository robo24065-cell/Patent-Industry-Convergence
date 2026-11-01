# -*- coding: utf-8 -*-
"""보조 후보 2종 저장·연계율 측정 (원본은 스크래치패드에 이미 1회 수신).
1) KRX KIND 상장폐지 현황(kind.krx.co.kr/investwarn/delcompany.do, 로그인 불필요, 2000-01-01~2026-09-27 1회 조회)
2) 중소벤처기업부_벤처기업명단(data.go.kr 15084581, 현재 유효 벤처 스냅샷, 대표자명 컬럼은 저장하지 않음)
출력: firm_krx_delisted.csv, firm_venture_list_202602.parquet, firm_misc_link.csv"""
import os, io, re, sqlite3
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
SP = r"C:\Users\PC\AppData\Local\Temp\claude\C--Users-PC-Downloads------------------------\72c91564-e815-4ac3-bfb0-ac338e5a5b96\scratchpad"
SUF = re.compile(r"주식회사|유한회사|\(주\)|㈜|\(유\)", re.I)
norm = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&]", "", SUF.sub("", str(s or ""))).upper()
out = []

# 1) KIND 상장폐지
t = open(os.path.join(SP, "kind_del.html"), encoding="utf-8", errors="replace").read()
d = pd.read_html(io.StringIO(t))[0].iloc[:, :4]
d.columns = ["no", "corp_name", "delist_date", "reason"]
d["cat"] = d.reason.str.contains("합병|흡수|지주회사|포괄적").map({True: "합병·지주전환", False: "기타"})
d.loc[d.reason.str.contains("신청|자진"), "cat"] = "자진신청"
d.loc[d.reason.str.contains("부도|자본전액|자본잠식|감사의견|의견거절|파산|회생|회사정리|해산|계속성|매출액|영업활동"), "cat"] = "부실"
d.loc[d.reason.str.contains("존속기간|증권투자|SPAC|스팩|상장예비|투자회사"), "cat"] = "펀드·SPAC"
d.to_csv(os.path.join(EXT, "firm_krx_delisted.csv"), index=False, encoding="utf-8-sig")
print(d.shape, d.delist_date.str[:4].value_counts().sort_index().to_dict()); print(d.cat.value_counts().to_dict())
dc = sqlite3.connect(f"file:{os.path.join(BASE, r'공모전데이터/외부데이터/DART/dart_company.sqlite')}?mode=ro", uri=True)
firm = pd.read_sql("select corp_code, corp_name from company where status='OK'", dc)
pf = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\pat_firm.parquet"), columns=["corp_code", "rgst_no"])
pat_dart = firm[firm.corp_code.isin(set(pf.corp_code.dropna()))]
d["n"] = d.corp_name.map(norm)
hit = d[d.n.isin(set(pat_dart.corp_name.map(norm)))]
out += [("KIND 상장폐지 기업(2000–2026.9)", len(d), ""),
        ("  특허 보유 DART 기업(9,562)과 상호 일치", len(hit), f"{len(hit)/len(d):.3f}"),
        ("  그중 부실 사유", int((hit.cat == "부실").sum()), "")]

# 2) 벤처기업명단
v = pd.read_csv(os.path.join(SP, "venture.bin"), encoding="utf-8-sig", dtype=str).drop(columns=["대표자명(익명)"])
v.to_parquet(os.path.join(EXT, "firm_venture_list_202602.parquet"), index=False)
import json
d50 = json.load(open(os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"), encoding="utf-8"))
nn = lambda s: re.sub(r"[\s·,및]", "", str(s))
k50 = {nn(x["codenmKr"]) for x in d50}
v["in50"] = v["업종명(11차)"].map(nn).isin(k50)
ap = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"), columns=["applicantCd", "nname"])
namecnt = ap.nname.value_counts()
v["n"] = v["업체명"].map(norm)
v["uniq_hit"] = v.n.map(namecnt).eq(1)
out += [("벤처기업명단(유효, 2023-06~)", len(v), ""),
        ("  업종명(11차)이 50 KSIC명과 일치", int(v.in50.sum()), f"{v.in50.mean():.3f}"),
        ("  특허 출원인(사업자번호 보유)과 상호 유일 일치", int(v.uniq_hit.sum()), f"{v.uniq_hit.mean():.3f}"),
        ("  50 KSIC 벤처 중 출원인 상호 일치", int((v.in50 & v.uniq_hit).sum()), f"{(v.in50 & v.uniq_hit).sum()/max(v.in50.sum(),1):.3f}")]
O = pd.DataFrame(out, columns=["항목", "값", "비율"]); print(O.to_string())
O.to_csv(os.path.join(EXT, "firm_misc_link.csv"), index=False, encoding="utf-8-sig")
