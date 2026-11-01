# -*- coding: utf-8 -*-
"""N2: OpenDART fnlttMultiAcnt 추가 수집 — 파일럿이 받지 않은 사업연도 보고서(2014·2016·2017·2019·2020·2022).
목적: 회계연도 t 값을 't년 사업보고서(t+1년 3월 공시)'의 당기·전기로 바꿔 실시간(나우캐스트 시점) 정보집합을 재현.
PROTOCOL_R2 §5: 단일 스레드, 요청 간 ≥0.7초, 000/013 외 상태·연결 오류 시 즉시 중단. 키는 API키_기록.txt에서 읽고 출력하지 않는다.
2024 이후 보고서는 받지 않는다(최종시험 봉인 — 최종시험 실행 시점에만 별도 수집).
사용: python n01_dart_collect.py 2016   (연도 1개씩, 한 실행 9분 이내)
출력: _shared/ext/R2N2_dart_fnltt_{year}.jsonl, 완료 청크 기록 cache/n01_done.txt"""
import os, re, json, time, urllib.request, sys
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
DONE = os.path.join(HERE, "cache", "n01_done.txt")
txt = open(os.path.join(BASE, "API키_기록.txt"), encoding="utf-8").read()
KEY = re.search(r"키:\s*([0-9a-f]{40})", txt.split("[금융감독원 OpenDART]")[1]).group(1)
GAP = 0.7
y = int(sys.argv[1])
flag = sys.argv[2] if len(sys.argv) > 2 else ""
ok = y in (2014, 2016, 2017, 2019, 2020, 2022) or (y == 2024 and flag == "--final") or (y == 2025 and flag == "--future")
assert ok, "허용 연도만 (개발: ≤2023 파일럿 미수집 연도 / 2024는 --final, 2025는 --future 에서만 — final_run_r2.py 가 호출)"
RAW = os.path.join(EXT, f"R2N2_dart_fnltt_{y}.jsonl")

pf = pd.read_parquet(os.path.join(HERE, "cache", "pat_firm.parquet"), columns=["corp_code"])
codes = sorted(pf.corp_code.dropna().unique())
chunks = [codes[i:i + 100] for i in range(0, len(codes), 100)]
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
print(f"year {y} firms {len(codes)} chunks {len(chunks)} done_total {len(done)}", flush=True)
n = 0; t0 = time.time()
for ci, ch in enumerate(chunks):
    tag = f"{y}_{ci}"
    if tag in done:
        continue
    if time.time() - t0 > 480:
        print("TIME budget reached; rerun to resume", flush=True); sys.exit(0)
    u = ("https://opendart.fss.or.kr/api/fnlttMultiAcnt.json?crtfc_key=" + KEY +
         "&corp_code=" + ",".join(ch) + f"&bsns_year={y}&reprt_code=11011")
    try:
        with urllib.request.urlopen(u, timeout=60) as r:
            j = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"STOP {tag} network error {type(e).__name__}", flush=True); sys.exit(1)
    st = j.get("status")
    if st not in ("000", "013"):
        print(f"STOP {tag} status {st} {j.get('message')}", flush=True); sys.exit(1)
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
print("DONE year", y, "calls", n, flush=True)
