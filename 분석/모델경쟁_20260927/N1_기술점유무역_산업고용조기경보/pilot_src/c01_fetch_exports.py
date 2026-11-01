# -*- coding: utf-8 -*-
"""UN Comtrade 공개 미리보기 API(키 불필요) — 5개 특허국 대응 수출국(KR 410, CN 156, JP 392, US 842, DE 276)의
HS6 대세계 수출액. 대상 HS6 = ksic_hs6_map(제공 50 KSIC ↔ HS6 1,019개). 연도 2004–2021만(최종시험 결과 구간 2022+ 미수집).
요청 간 ≥2초, 실패 시 지수 백오프. 1회 응답 500행 상한 → 5개국×100코드 청크."""
import os, json, time, urllib.request, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, "분석", "모델경쟁_20260927", "_shared", "ext"); os.makedirs(EXT, exist_ok=True)
OUT = os.path.join(EXT, "R2ind_comtrade_exports_5c_hs6_2004_2021.csv")
mp = pd.read_csv(os.path.join(BASE, "분석", "fusion", "ksic_hs6_map.csv"), dtype=str)
codes = sorted(mp.hs6.str.zfill(6).unique())
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REP = "410,156,392,842,276"
chunks = [codes[i:i+100] for i in range(0, len(codes), 100)]
def fetch(y, cs):
    u = f"{URL}?reporterCode={REP}&flowCode=X&partnerCode=0&period={y}&cmdCode={','.join(cs)}"
    w = 15
    for a in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=180) as r:
                j = json.loads(r.read().decode("utf-8"))
            if j.get("error"): raise RuntimeError(j["error"])
            return j.get("data", [])
        except Exception as e:
            print(f"  재시도 {y} {a}: {str(e)[:80]}", flush=True); time.sleep(w); w = min(w*2, 240)
    return None
COLS = ["year", "reporter", "hs6", "export_usd", "hs_rev", "net_wgt_kg", "qty", "qty_unit"]
DONE = OUT.replace(".csv", "_done_chunks.txt")   # (연도,청크) 완료 기록 → 재실행 시 건너뜀
old = pd.read_csv(OUT, dtype={"hs6": str}) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
if "net_wgt_kg" not in old.columns: old = pd.DataFrame(columns=COLS)
rows = list(old[COLS].itertuples(index=False, name=None))
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
if not os.path.exists(DONE):
    done = {f"{y}_{ci}" for y in sorted(set(old.year)) for ci in range(len(chunks))}
    open(DONE, "w").write(" ".join(sorted(done)))
# 3년 간격 연도 우선(비중첩 3년 창), 이후 나머지
YEARS = [2006, 2009, 2012, 2015, 2018, 2021, 2007, 2010, 2013, 2016, 2019, 2005, 2008, 2011, 2014, 2017, 2020, 2004]
fails = []
for y in YEARS:
    for ci, cs in enumerate(chunks):
        if f"{y}_{ci}" in done: continue
        d = fetch(y, cs); time.sleep(8.0)
        if d is None: fails.append((y, ci)); continue
        if len(d) >= 500: print(f"  경고: {y} chunk{ci} 500행 상한 도달", flush=True)
        for x in d:
            if x.get("motCode", 0) != 0 or x.get("customsCode") != "C00" or x.get("partner2Code", 0) != 0: continue
            rows.append((y, x["reporterCode"], str(x["cmdCode"]).zfill(6), x["primaryValue"], x.get("classificationCode"), x.get("netWgt"), x.get("qty"), x.get("qtyUnitAbbr")))
        done.add(f"{y}_{ci}"); open(DONE, "a").write(f" {y}_{ci}")
        pd.DataFrame(rows, columns=COLS).to_csv(OUT, index=False)
    print(y, len(rows), flush=True)
print("실패", fails)
