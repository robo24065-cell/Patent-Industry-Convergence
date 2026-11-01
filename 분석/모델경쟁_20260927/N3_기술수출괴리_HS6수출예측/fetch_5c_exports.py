# -*- coding: utf-8 -*-
"""N3 보충 수집 — UN Comtrade 공개 미리보기 API(키 불필요). R2ind 파일에 없는 연도/청크만(2015 잔여, 2018, 2021).
reporter KR 410·CN 156·JP 392·US 842·DE 276, flow X, partner 0(세계). 대상 HS6 = ksic_hs6_map 1,019개(100코드 청크, R2ind와 동일 분할).
🔒 2022 이후 연도는 요청하지 않는다(최종시험 결과 구간). 요청 간 ≥3초. HTTP 403(Quota) 또는 429 반복 시 즉시 중단.
출력: _shared/ext/R2N3_comtrade_exports_5c_hs6.csv (+ _done.txt)"""
import os, json, time, sys, urllib.request, urllib.error, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, "분석", "모델경쟁_20260927", "_shared", "ext")
OUT = os.path.join(EXT, "R2N3_comtrade_exports_5c_hs6.csv")
DONE = OUT.replace(".csv", "_done.txt")
R2IND_DONE = os.path.join(EXT, "R2ind_comtrade_exports_5c_hs6_2004_2021_done_chunks.txt")
mp = pd.read_csv(os.path.join(BASE, "분석", "fusion", "ksic_hs6_map.csv"), dtype=str)
codes = sorted(mp.hs6.str.zfill(6).unique())
chunks = [codes[i:i + 100] for i in range(0, len(codes), 100)]
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REP = "410,156,392,842,276"
YEARS = [int(a) for a in sys.argv[1:]] or [2015, 2018, 2021]
assert max(YEARS) <= 2021, "최종시험 결과 구간 수집 금지"
COLS = ["year", "reporter", "hs6", "export_usd", "hs_rev"]
ind_done = set(open(R2IND_DONE).read().split()) if os.path.exists(R2IND_DONE) else set()
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
rows = list(pd.read_csv(OUT, dtype={"hs6": str})[COLS].itertuples(index=False, name=None)) if os.path.exists(OUT) else []


class Stop(Exception):
    pass


def fetch(y, cs):
    u = f"{URL}?reporterCode={REP}&flowCode=X&partnerCode=0&period={y}&cmdCode={','.join(cs)}"
    w = 10
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=180) as r:
                j = json.loads(r.read().decode("utf-8"))
            if j.get("error"):
                raise RuntimeError(j["error"])
            return j.get("data", [])
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} {y}: {str(e)[:80]}", flush=True)
            if e.code == 403:
                raise Stop("403 Quota — 즉시 중단")
            if e.code == 429 and a >= 2:
                raise Stop("429 반복 — 중단")
        except Exception as e:
            print(f"  재시도 {y} {a}: {str(e)[:80]}", flush=True)
        time.sleep(w); w *= 2
    return None


try:
    for y in YEARS:
        for ci, cs in enumerate(chunks):
            key = f"{y}_{ci}"
            if key in done or key in ind_done:
                continue
            d = fetch(y, cs); time.sleep(3.0)
            if d is None:
                print("실패", key, flush=True); continue
            for x in d:
                if x.get("motCode", 0) != 0 or x.get("customsCode") != "C00" or x.get("partner2Code", 0) != 0:
                    continue
                rows.append((y, x["reporterCode"], str(x["cmdCode"]).zfill(6), x["primaryValue"], x.get("classificationCode")))
            done.add(key); open(DONE, "a").write(f" {key}")
            pd.DataFrame(rows, columns=COLS).to_csv(OUT, index=False)
            print(key, len(d), len(rows), flush=True)
except Stop as e:
    print("중단:", e, flush=True)
print("완료 청크", sorted(done))
