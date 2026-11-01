# -*- coding: utf-8 -*-
"""R7 데이터 1: KRX KIND 신규상장기업현황(로그인 불필요) 연도×시장별 엑셀(HTML) 다운로드.
출처: https://kind.krx.co.kr/listinvstg/listingcompany.do?method=searchListingTypeMain (EXCEL 버튼과 같은 POST)
보내는 값: 날짜 범위·시장구분·상장유형 코드뿐(제공 데이터 내용 없음). 요청 간 2.5초, 실패 시 지수 백오프, 3회 실패면 중단.
출력: _shared/ext/r7_data_kind_listing.csv (+ 원본 HTML은 _ideas_R7/raw/kind_list_*.xls)
"""
import os, io, time, sys
import requests, pandas as pd

R7 = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(R7), "_shared", "ext")
RAW = os.path.join(R7, "raw")
URL = "https://kind.krx.co.kr/listinvstg/listingcompany.do"
H = {"User-Agent": "Mozilla/5.0", "Referer": URL + "?method=searchListingTypeMain"}
MK = {"1": "유가증권", "2": "코스닥", "6": "코넥스"}

def fetch(y, mk):
    fn = os.path.join(RAW, f"kind_list_{y}_{mk}.xls")
    if os.path.exists(fn) and os.path.getsize(fn) > 500:
        return open(fn, "rb").read(), False
    data = {"method": "searchListingTypeSub", "forward": "listingtype_down", "currentPageSize": "3000",
            "pageIndex": "1", "orderMode": "1", "orderStat": "D", "listTypeArrStr": "01|02|03|04|05|",
            "secuGrpArrStr": "0|ST|FS|MF|SC|RT|IF|DR|", "marketType": mk,
            "fromDate": f"{y}-01-01", "toDate": f"{y}-12-31"}
    for k in range(3):
        try:
            r = requests.post(URL, data=data, headers=H, timeout=60)
            if r.status_code == 200 and len(r.content) > 300:
                open(fn, "wb").write(r.content)
                return r.content, True
            print("status", r.status_code, len(r.content))
        except Exception as e:
            print("err", e)
        time.sleep(5 * 2 ** k)
    sys.exit(f"중단: {y} {mk} 3회 실패")

rows = []
for y in range(1996, 2027):
    for mk in MK:
        if mk == "6" and y < 2013:
            continue
        b, net = fetch(y, mk)
        t = b.decode("cp949", errors="replace")
        try:
            d = pd.read_html(io.StringIO(t))[0]
        except ValueError:
            d = pd.DataFrame()
        if len(d) and d.shape[1] >= 7 and "회사명" in d.columns:
            d = d[d["상장일"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")]
            d["시장"] = MK[mk]
            rows.append(d)
        print(y, MK[mk], len(d), flush=True)
        if net:
            time.sleep(2.5)

D = pd.concat(rows, ignore_index=True)
D["종목코드"] = D["종목코드"].astype(str).str.zfill(6)
D.to_csv(os.path.join(EXT, "r7_data_kind_listing.csv"), index=False, encoding="utf-8-sig")
print(D.shape)
print(D.groupby([D["상장일"].str[:4], "시장"]).size().unstack(fill_value=0).to_string())
print(D["상장유형"].value_counts().to_string())
