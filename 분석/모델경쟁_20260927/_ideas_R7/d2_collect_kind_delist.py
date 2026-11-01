# -*- coding: utf-8 -*-
"""R7 데이터 2: KRX KIND 상장폐지현황 연도×시장별(로그인 불필요). 기존 firm_krx_delisted.csv에는 종목코드·시장이 없어 다시 받는다.
출처: https://kind.krx.co.kr/investwarn/delcompany.do?method=searchDelCompanyMain (EXCEL 버튼과 같은 POST)
보내는 값: 날짜 범위·시장구분뿐. 요청 간 2.5초, 지수 백오프, 3회 실패 시 중단.
출력: _shared/ext/r7_data_kind_delisting.csv"""
import os, io, time, sys
import requests, pandas as pd
R7 = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.join(os.path.dirname(R7), "_shared", "ext"); RAW = os.path.join(R7, "raw")
URL = "https://kind.krx.co.kr/investwarn/delcompany.do"
H = {"User-Agent": "Mozilla/5.0", "Referer": URL + "?method=searchDelCompanyMain"}
MK = {"1": "유가증권", "2": "코스닥", "6": "코넥스"}
def fetch(y, mk):
    fn = os.path.join(RAW, f"kind_del_{y}_{mk}.xls")
    if os.path.exists(fn) and os.path.getsize(fn) > 300: return open(fn, "rb").read(), False
    data = {"method": "searchDelCompanySub", "forward": "delcompany_down", "currentPageSize": "3000", "pageIndex": "1",
            "marketType": mk, "fromDate": f"{y}-01-01", "toDate": f"{y}-12-31", "tabType": "1"}
    for k in range(3):
        try:
            r = requests.post(URL, data=data, headers=H, timeout=60)
            if r.status_code == 200 and len(r.content) > 300:
                open(fn, "wb").write(r.content); return r.content, True
            print("status", r.status_code)
        except Exception as e: print("err", e)
        time.sleep(5 * 2 ** k)
    sys.exit(f"중단 {y} {mk}")
rows = []
for y in range(1996, 2027):
    for mk in MK:
        if mk == "6" and y < 2013: continue
        b, net = fetch(y, mk)
        try: d = pd.read_html(io.StringIO(b.decode("cp949", errors="replace")))[0]
        except ValueError: d = pd.DataFrame()
        if len(d) and "폐지일자" in d.columns:
            d = d[d["폐지일자"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")].copy(); d["시장"] = MK[mk]; rows.append(d)
        print(y, MK[mk], len(d), flush=True)
        if net: time.sleep(2.5)
D = pd.concat(rows, ignore_index=True).drop(columns=["번호"])
D["종목코드"] = D["종목코드"].astype(str).str.zfill(6)
D.to_csv(os.path.join(EXT, "r7_data_kind_delisting.csv"), index=False, encoding="utf-8-sig")
print(D.shape, D["폐지일자"].min(), D["폐지일자"].max())
print(D.groupby([D["폐지일자"].str[:4], "시장"]).size().unstack(fill_value=0).to_string())
