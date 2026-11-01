# -*- coding: utf-8 -*-
"""OpenDART fnlttMultiAcnt 수집 (PROTOCOL_R2 §5 준수).
대상: 제공범위 KR 등록특허 출원인 중 DART 연결 법인 전체(약 9.5천사), 100사씩 묶어 호출.
사업연도 보고서: 2015, 2018, 2021, 2023 (각 보고서 = 당기·전기·전전기 → 2013–2023 커버).
2024 이후 보고서는 받지 않는다(최종시험 결과 구간 보호).
단일 스레드, 요청 간 0.7초, 000/013 외 상태·연결 오류 시 즉시 중단. 키는 파일에서 읽고 출력하지 않는다.
출력: _shared/ext/R2firm_dart_fnltt_raw.jsonl (응답 행 그대로), 진행 로그는 이 폴더 f02.log"""
import os, re, json, time, urllib.request, urllib.error, sys
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
os.makedirs(EXT, exist_ok=True)
RAW = os.path.join(EXT, "R2firm_dart_fnltt_raw.jsonl")
DONE = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\f02_done.txt")
txt = open(os.path.join(BASE, "API키_기록.txt"), encoding="utf-8").read()
KEY = re.search(r"키:\s*([0-9a-f]{40})", txt.split("[금융감독원 OpenDART]")[1]).group(1)
YEARS = [2015, 2018, 2021, 2023]
GAP = 0.7

pf = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\pat_firm.parquet"))
codes = sorted(pf.corp_code.dropna().unique())
chunks = [codes[i:i + 100] for i in range(0, len(codes), 100)]
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
print(f"firms {len(codes)} chunks {len(chunks)} years {YEARS} done {len(done)}", flush=True)

n = 0
for y in YEARS:
    for ci, ch in enumerate(chunks):
        tag = f"{y}_{ci}"
        if tag in done:
            continue
        u = ("https://opendart.fss.or.kr/api/fnlttMultiAcnt.json?crtfc_key=" + KEY +
             "&corp_code=" + ",".join(ch) + f"&bsns_year={y}&reprt_code=11011")
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                j = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"STOP {tag} network error {type(e).__name__}", flush=True)
            sys.exit(1)
        st = j.get("status")
        if st not in ("000", "013"):
            print(f"STOP {tag} status {st} {j.get('message')}", flush=True)
            sys.exit(1)
        lst = j.get("list", []) if st == "000" else []
        with open(RAW, "a", encoding="utf-8") as f:
            for x in lst:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        with open(DONE, "a") as f:
            f.write(tag + "\n")
        n += 1
        if n % 20 == 0:
            print(f"{tag} rows {len(lst)} calls {n}", flush=True)
        time.sleep(GAP)
print("DONE calls", n, flush=True)
