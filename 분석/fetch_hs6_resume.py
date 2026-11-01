# -*- coding: utf-8 -*-
"""
HS6 대(對)중국 수입액 수집 '이어받기' — 2012~2024 (이전 실행은 2011에서 중단됨)

이전 산출물: scratchpad/comtrade_hs6_raw.csv (2000~2011, CN/WLD)
이 스크립트는 아직 없는 연도만 추가 수집해 같은 파일에 누적 저장한다.

UN Comtrade public preview API (키 불필요)
  reporter=410(한국), flow=M(수입), partner=156(중국) / partner=0(전세계)
  제약: 1회 1개 연도, URL 길이 한계 → 코드 250개씩 청크
"""
import os, json, time, urllib.request
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, "분석", "fusion")
SCR = (r"C:\Users\PC\AppData\Local\Temp\claude"
       r"\C--Users-PC-Downloads------------------------"
       r"\72c91564-e815-4ac3-bfb0-ac338e5a5b96\scratchpad")
RAW = os.path.join(SCR, "comtrade_hs6_raw.csv")
log = lambda m: print(m, flush=True)

mp = pd.read_csv(os.path.join(OUT, "ksic_hs6_map.csv"), encoding="utf-8-sig", dtype=str)
codes = sorted(mp.hs6.unique())
log(f"대상 HS6 {len(codes)}개 (KSIC5 {mp.k5.nunique()}개)")

# ── 기존 수집분 로드 → 이미 확보한 (연도,파트너) 조합은 건너뛴다 ──
if os.path.exists(RAW):
    prev = pd.read_csv(RAW, encoding="utf-8-sig", dtype={"hs6": str})
    done = set(map(tuple, prev[["year", "partner"]].drop_duplicates().values))
    rows = prev.to_dict("records")
    log(f"기존 {len(prev):,}행 로드, 완료조합 {len(done)}개 "
        f"({prev.year.min()}~{prev.year.max()})")
else:
    done, rows = set(), []

URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
YEARS = list(range(2000, 2025))
CHUNK = 250
chunks = [codes[i:i + CHUNK] for i in range(0, len(codes), CHUNK)]


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
                log(f"  {year}/{partner} 재시도{a}: {e}")
                time.sleep(wait)
    return None


def save():
    pd.DataFrame(rows).to_csv(RAW, index=False, encoding="utf-8-sig")


for yr in YEARS:
    for partner, lab in [(156, "CN"), (0, "WLD")]:
        if (yr, lab) in done:
            continue
        got, fail = 0, 0
        for ci, cs in enumerate(chunks):
            d = fetch(yr, partner, cs)
            if d is None:
                log(f"  포기 {yr} {lab} chunk{ci}"); fail += 1; continue
            for x in d:
                rows.append({"year": yr, "partner": lab,
                             "hs6": str(x.get("cmdCode")).zfill(6),
                             "value": x.get("primaryValue")})
            got += len(d)
            time.sleep(1.5)
        log(f"{yr} {lab}: {got}행 (실패청크 {fail})")
        save()

log("수집 종료 — 최종 정리")
raw = pd.DataFrame(rows)
raw["hs6"] = raw.hs6.astype(str).str.zfill(6)
raw = raw[raw.hs6.str.len() == 6]
raw["value"] = pd.to_numeric(raw.value, errors="coerce")
raw = raw.dropna(subset=["value"])
raw = raw.groupby(["year", "partner", "hs6"], as_index=False).value.sum()
raw.to_csv(os.path.join(OUT, "hs6_import_yearly.csv"), index=False, encoding="utf-8-sig")
log(f"저장 hs6_import_yearly.csv : {len(raw):,}행, {raw.year.min()}~{raw.year.max()}, "
    f"HS6 {raw.hs6.nunique()}개")
