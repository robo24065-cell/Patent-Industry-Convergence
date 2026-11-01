# -*- coding: utf-8 -*-
"""한국 HS6 수출 금액+순중량+수량 (단가 = 품질 사다리 신호) — UN Comtrade public preview (키 불필요).
대상: 분석/fusion/ksic_hs6_map.csv 1,019 HS6 × 2000–2021. 250코드 청크, 요청 간 3초, 429/오류 시 지수 백오프,
403 또는 연속 실패 시 즉시 중단. motCode 0·customsCode C00·partner2 0·partner 0 행만 보존.
저장: _shared/ext/trade_kr_export_hs6_uv_2000_2021.csv (연도 단위로 추가 저장, 재실행 시 완료 연도 건너뜀)"""
import os, json, time, urllib.request, urllib.error
import pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\trade_kr_export_hs6_uv_2000_2021.csv")
mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), encoding="utf-8-sig", dtype=str)
codes = sorted(mp.hs6.str.zfill(6).unique())
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
chunks = [codes[i:i+250] for i in range(0, len(codes), 250)]
def fetch(year, cs):
    u = f"{URL}?reporterCode=410&flowCode=X&partnerCode=0&period={year}&cmdCode={','.join(cs)}"
    wait = 6
    for a in range(6):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8")).get("data", [])
        except urllib.error.HTTPError as e:
            print(f"{year} HTTP {e.code}", flush=True)
            if e.code == 403: raise SystemExit("403 — 즉시 중단")
        except Exception as e:
            print(f"{year} 오류 {str(e)[:60]}", flush=True)
        time.sleep(wait); wait = min(wait*2, 120)
    return None
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT, usecols=["year"]).year.unique())
maxrows = 0
for yr in range(2000, 2022):
    if yr in done: continue
    rows = []
    for cs in chunks:
        d = fetch(yr, cs); time.sleep(3)
        if d is None: raise SystemExit(f"{yr} 실패 — 중단")
        maxrows = max(maxrows, len(d))
        for x in d:
            if x.get("partnerCode") != 0 or x.get("motCode") not in (0, None) or x.get("customsCode") not in ("C00", None) or x.get("partner2Code") not in (0, None):
                continue
            rows.append(dict(year=yr, hs6=str(x["cmdCode"]).zfill(6), hs_rev=x.get("classificationCode"),
                             export_usd=x.get("primaryValue"), net_wgt_kg=x.get("netWgt"),
                             qty=x.get("qty"), qty_unit=x.get("qtyUnitAbbr"), qty_unit_code=x.get("qtyUnitCode")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False, encoding="utf-8")
    print(yr, len(df), "maxrows/call", maxrows, flush=True)
print("완료")
