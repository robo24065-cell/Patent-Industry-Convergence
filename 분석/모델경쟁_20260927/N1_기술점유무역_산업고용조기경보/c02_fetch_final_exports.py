# -*- coding: utf-8 -*-
"""🔒 최종시험 때 1회만 실행: 한국(410) HS6 대세계 수출 2019–2024 (UN Comtrade 공개 미리보기 API, 키 불필요, 요청 간 ≥2.5초, 지수 백오프).
2019–2021 도 같은 호출로 다시 받아 2022–24 와 같은 보고 기준(HS 개정판)으로 품목 바구니를 맞춘다(final_run_r2.py run_final_D).
출력: _shared/ext/N1final_kr_export_hs6_2019_2024.csv (+ SOURCES.md 에 출처·수집일 기록). 개발 단계에서는 실행 금지."""
import os, sys, json, time, urllib.request, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from n1lib import BASE, B
if not os.path.exists(os.path.join(HERE, "final_result.json.lock_ok")) and "--final" not in sys.argv:
    sys.exit("최종시험 단계에서 --final 인자로만 실행한다.")
OUT = os.path.join(B, "_shared", "ext", "N1final_kr_export_hs6_2019_2024.csv")
mp = pd.read_csv(os.path.join(BASE, "분석", "fusion", "ksic_hs6_map.csv"), dtype=str); codes = sorted(mp.hs6.str.zfill(6).unique())
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
rows = []
for y in range(2019, 2025):
    for i in range(0, len(codes), 250):
        cs = codes[i:i+250]; u = f"{URL}?reporterCode=410&flowCode=X&partnerCode=0&period={y}&cmdCode={','.join(cs)}"; w = 10
        for a in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=180) as r:
                    j = json.loads(r.read().decode("utf-8"))
                for x in j.get("data", []):
                    if x.get("motCode", 0) != 0 or x.get("customsCode") != "C00" or x.get("partner2Code", 0) != 0: continue
                    rows.append((y, "X", str(x["cmdCode"]).zfill(6), x["primaryValue"], x.get("classificationCode")))
                break
            except Exception as e:
                print("재시도", y, i, a, str(e)[:80]); time.sleep(w); w = min(w * 2, 240)
        time.sleep(2.5)
    print(y, len(rows), flush=True)
pd.DataFrame(rows, columns=["year", "flow", "hs6", "value", "hs_rev"]).to_csv(OUT, index=False)
with open(os.path.join(B, "_shared", "ext", "SOURCES.md"), "a", encoding="utf-8") as f:
    f.write(f"\n## N1final_kr_export_hs6_2019_2024.csv  (트랙: N1, 수집 {time.strftime('%Y-%m-%d')})\n- 출처: UN Comtrade public preview API, reporter=410, flow=X, partner=0, HS6 {len(codes)}개(ksic_hs6_map) × 2019–2024. 최종시험 1회 수집.\n")
