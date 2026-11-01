# -*- coding: utf-8 -*-
"""N2: 한국 HS6 수출(전세계) 2022·2023 추가 수집 — UN Comtrade public preview API(키 불필요), PROTOCOL_R2 §5.
_ideas_R2/global/fetch_kr_export.py 와 같은 방식(같은 1,019 HS6, 250코드 청크, partnerCode 0 행 합산)으로 일관성 유지.
N2 개발 원점 t=2022·2023 피처용(결과 구간 ≤2023). 2024 이후는 수집하지 않는다(N2 최종시험 봉인).
요청 간 3초, 실패 시 지수 백오프, HTTP 403/429 누적 시 중단.
저장: _shared/ext/R2N2_kr_export_hs6_2022_2023.csv (컬럼 year, flow, hs6, value)"""
import os, json, time, urllib.request, urllib.error, sys
import pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
YEARS = [int(a) for a in sys.argv[1:]] or [2022, 2023]   # 개발 기본 2022·2023. 2024 는 final_run_r2.py --mode final 에서만
assert max(YEARS) <= 2023 or os.environ.get("N2_FINAL_OK") == "1", "2024+ 수출은 최종시험 실행에서만"
OUT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\R2N2_kr_export_hs6_" + "_".join(map(str, YEARS)) + ".csv")
mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), encoding="utf-8-sig", dtype=str)
codes = sorted(mp.hs6.str.zfill(6).unique())
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
chunks = [codes[i:i + 250] for i in range(0, len(codes), 250)]


def fetch(year, cs):
    u = f"{URL}?reporterCode=410&flowCode=X&partnerCode=0&period={year}&cmdCode={','.join(cs)}"
    wait = 5
    for a in range(5):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8")).get("data", [])
        except urllib.error.HTTPError as e:
            print(f"  {year} HTTP {e.code}", flush=True)
            if e.code == 403:
                return None
            time.sleep(wait); wait *= 2
        except Exception as e:
            print(f"  {year} err {type(e).__name__}", flush=True)
            time.sleep(wait); wait *= 2
    return None


rows = []
for yr in YEARS:
    for cs in chunks:
        d = fetch(yr, cs)
        time.sleep(3)
        if d is None:
            print("STOP", yr); sys.exit(1)
        for x in d:
            if x.get("partnerCode") not in (0, None):
                continue
            rows.append({"year": yr, "flow": "X", "hs6": str(x.get("cmdCode")).zfill(6), "value": x.get("primaryValue")})
    print(yr, "rows so far", len(rows), flush=True)
df = pd.DataFrame(rows)
df = df[df.hs6.str.len() == 6].groupby(["year", "flow", "hs6"], as_index=False).value.sum()
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("saved", len(df), df.groupby("year").size().to_dict())
