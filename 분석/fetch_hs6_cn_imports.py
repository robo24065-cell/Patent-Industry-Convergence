# -*- coding: utf-8 -*-
"""
HS6 수준 대(對)중국 수입액 수집 — UN Comtrade public preview API (키 불필요)
reporter=410(한국), flow=M(수입), partner=156(중국) / partner=0(전세계)

목적: KSIC 세세분류(5자리) 해상도의 '대중 수입침투' 시계열 확보.
      IO코드-KSIC-HS 대조표(한국은행 2020 실측표)로 KSIC5 ↔ HS6 를 연결한다.
      → 1,019개 HS6 중 83%가 단일 KSIC5 에만 대응하므로 중분류 내(within-mid) 변동이 살아난다.

제약: preview 엔드포인트는 (a) 1회 1개 연도만, (b) URL 길이 ~2.2KB 한계 → 코드 300개씩 청크.
"""
import os, json, time, urllib.request
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, "분석", "fusion")
SCR = (r"C:\Users\PC\AppData\Local\Temp\claude"
       r"\C--Users-PC-Downloads------------------------"
       r"\72c91564-e815-4ac3-bfb0-ac338e5a5b96\scratchpad")

# ── KSIC5 ↔ HS6 대조표 구축 ─────────────────────────────────────
conc = os.path.join(BASE, "공모전데이터", "외부데이터", "산업연관표",
                    "IO코드와 KSIC 및 HS코드 대조표.xlsx")
k = pd.read_excel(conc, sheet_name="IO코드와 KSIC 비교표", header=1, dtype=str)
h = pd.read_excel(conc, sheet_name="IO코드와 HS코드 비교표", header=1, dtype=str)
k.columns = ["io", "nm", "ksic"]
h.columns = ["io", "nm", "hs"]
k["io"] = k.io.ffill().str.strip()
h["io"] = h.io.ffill().str.strip()
k = k.dropna(subset=["ksic"]).copy()
h = h.dropna(subset=["hs"]).copy()
k["k5"] = k.ksic.str.strip().str[:5]          # 제조업은 KSIC8자리 → 앞 5자리가 세세분류
h["hs6"] = h.hs.str.strip().str.zfill(6)

tgt = pd.read_csv(os.path.join(OUT, "industry_country_year.csv"),
                  encoding="utf-8-sig", dtype={"ksic": str}).ksic.unique()
kk = k[k.k5.isin(tgt)][["io", "k5"]].drop_duplicates()
mp = kk.merge(h[["io", "hs6"]].drop_duplicates(), on="io").drop_duplicates()
mp.to_csv(os.path.join(OUT, "ksic_hs6_map.csv"), index=False, encoding="utf-8-sig")
codes = sorted(mp.hs6.unique())
print(f"KSIC5 {mp.k5.nunique()}개 ↔ HS6 {len(codes)}개 (쌍 {len(mp)})", flush=True)

# ── Comtrade 수집 ───────────────────────────────────────────────
URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
YEARS = list(range(2000, 2025))
CHUNK = 250


def fetch(year, partner, cs):
    u = (f"{URL}?reporterCode=410&flowCode=M&partnerCode={partner}"
         f"&period={year}&cmdCode={','.join(cs)}")
    wait = 3
    for a in range(7):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8")).get("data", [])
        except Exception as e:
            msg = str(e)
            if "429" in msg or "50" in msg[:20]:
                time.sleep(wait); wait = min(wait * 2, 60)
            else:
                print(f"  {year}/{partner} 실패: {e}", flush=True)
                time.sleep(wait)
    return None


rows = []
chunks = [codes[i:i + CHUNK] for i in range(0, len(codes), CHUNK)]
for yr in YEARS:
    for partner, lab in [(156, "CN"), (0, "WLD")]:
        got = 0
        for ci, cs in enumerate(chunks):
            d = fetch(yr, partner, cs)
            if d is None:
                print(f"  포기 {yr} {lab} chunk{ci}", flush=True)
                continue
            for x in d:
                rows.append({"year": yr, "partner": lab,
                             "hs6": str(x.get("cmdCode")).zfill(6),
                             "value": x.get("primaryValue")})
            got += len(d)
            time.sleep(2.0)
        print(f"{yr} {lab}: {got}행", flush=True)
    # 중간 저장
    pd.DataFrame(rows).to_csv(os.path.join(SCR, "comtrade_hs6_raw.csv"),
                              index=False, encoding="utf-8-sig")

raw = pd.DataFrame(rows)
raw = raw[raw.hs6.str.len() == 6]
raw = raw.groupby(["year", "partner", "hs6"], as_index=False).value.sum()
raw.to_csv(os.path.join(OUT, "hs6_import_yearly.csv"), index=False, encoding="utf-8-sig")
print(f"저장 완료: {len(raw):,}행, {raw.year.min()}~{raw.year.max()}", flush=True)
