# -*- coding: utf-8 -*-
"""한국 HS6 수출(전세계) 수집 — UN Comtrade public preview API (키 불필요), PROTOCOL_R2 §5.
대상 HS6 = 분석/fusion/ksic_hs6_map.csv 의 1,019개 (KSIC 50개 연결 품목).
파일럿 규모: 2000–2021 (최종시험 구간 2022+ 는 수집하지 않음 → 결과변수 노출 방지).
요청 간 2.5초, 실패 시 지수 백오프. 연도별 중간 저장, 재실행 시 완료 연도 건너뜀.
저장: _shared/ext/R2global_kr_export_hs6.csv
"""
import os, json, time, urllib.request
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, "분석", "모델경쟁_20260927", "_shared", "ext", "R2global_kr_export_hs6.csv")
mp = pd.read_csv(os.path.join(BASE, "분석", "fusion", "ksic_hs6_map.csv"), encoding="utf-8-sig", dtype=str)
codes = sorted(mp.hs6.str.zfill(6).unique())
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
YEARS = list(range(2000, 2022))
CHUNK = 250


def fetch(year, cs, flow="X", partner=0):
    u = (f"{URL}?reporterCode=410&flowCode={flow}&partnerCode={partner}"
         f"&period={year}&cmdCode={','.join(cs)}")
    wait = 4
    for a in range(7):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8")).get("data", [])
        except Exception as e:
            print(f"  {year} 재시도{a} {str(e)[:80]}", flush=True)
            time.sleep(wait); wait = min(wait * 2, 120)
    return None


done = set()
rows = []
if os.path.exists(OUT):
    old = pd.read_csv(OUT, encoding="utf-8-sig", dtype={"hs6": str})
    rows = old.to_dict("records"); done = set(old.year.unique())
chunks = [codes[i:i + CHUNK] for i in range(0, len(codes), CHUNK)]
for yr in YEARS:
    if yr in done:
        continue
    got, ok = [], True
    for cs in chunks:
        d = fetch(yr, cs)
        time.sleep(2.5)
        if d is None:
            ok = False; break
        for x in d:
            if x.get("partnerCode") not in (0, None):
                continue
            got.append({"year": yr, "flow": "X", "hs6": str(x.get("cmdCode")).zfill(6),
                        "value": x.get("primaryValue")})
    if not ok:
        print(f"{yr} 실패 — 중단", flush=True); break
    rows += got
    df = pd.DataFrame(rows)
    df = df[df.hs6.str.len() == 6].groupby(["year", "flow", "hs6"], as_index=False).value.sum()
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    rows = df.to_dict("records")
    print(f"{yr}: {len(got)}행 저장", flush=True)
print("끝", flush=True)
