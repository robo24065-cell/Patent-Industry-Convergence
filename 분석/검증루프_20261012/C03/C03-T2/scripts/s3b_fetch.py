# -*- coding: utf-8 -*-
"""s3b: UN Comtrade 공개 미리보기 API(로그인·키 없음) 보충 수집. 요청 간 4초, 403/429 반복 시 중단. 2022년 이후 요청 금지.
 (1) TOTAL: 5개 보고국 × 2004–2021 (D1 총량 대조, A3 분모)
 (2) 구버전 코드: H4(2015·2016) 101개, H3(2008·2009·2010) 123개 (A1)
 (3) 재조회 표본: 원 청크 0(100코드) × 2015·2016·2017·2021 (D5 사후 수정)
원 응답 JSON을 download/comtrade/ 에 그대로 저장(파싱은 별도 스크립트)."""
import os, sys, json, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
OUTD = os.path.join(DL, "comtrade"); os.makedirs(OUTD, exist_ok=True)
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REPS_C = "410,156,392,842,276"
LOG = os.path.join(DL, "comtrade_fetch_log.txt")
need = json.load(open(os.path.join(CACHE, "s3a_fetch_need.json")))
codes = sorted(load_map().hs6.unique())
jobs = []
for y in range(2004, 2022):
    jobs.append((f"TOTAL_{y}", y, ["TOTAL"]))
for ver, yrs in [("H4", [2015, 2016]), ("H3", [2008, 2009, 2010])]:
    cs = need[ver]; ch = [cs[i:i + 80] for i in range(0, len(cs), 80)]
    for y in yrs:
        for i, c in enumerate(ch):
            jobs.append((f"OLD{ver}_{y}_{i}", y, c))
for y in [2015, 2016, 2017, 2021]:
    jobs.append((f"REFETCH_{y}_c0", y, codes[:100]))
assert max(j[1] for j in jobs) <= 2021


def fetch(y, cs):
    u = f"{URL}?reporterCode={REPS_C}&flowCode=X&partnerCode=0&period={y}&cmdCode={','.join(cs)}"
    w = 10; n429 = 0
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 403: raise SystemExit("403 중단")
            if e.code == 429:
                n429 += 1
                if n429 >= 2: raise SystemExit("429 반복 중단")
            print("  HTTP", e.code, flush=True)
        except Exception as e:
            print("  err", str(e)[:80], flush=True)
        time.sleep(w); w *= 2
    return None


with open(LOG, "a", encoding="utf-8") as lg:
    for key, y, cs in jobs:
        fp = os.path.join(OUTD, key + ".json")
        if os.path.exists(fp): continue
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
        j = fetch(y, cs); time.sleep(4.0)
        if j is None:
            lg.write(f"{t0} FAIL {key}\n"); print("FAIL", key, flush=True); continue
        json.dump(j, open(fp, "w", encoding="utf-8"))
        n = len(j.get("data", []))
        lg.write(f"{t0} OK {key} year={y} n_cmd={len(cs)} rows={n} error={j.get('error','')!r}\n")
        print(key, n, j.get("error", ""), flush=True)
print("done")
