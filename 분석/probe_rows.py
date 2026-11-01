# -*- coding: utf-8 -*-
"""numOfRows 상한 및 RC:007 의미 확인"""
import json, os, sys, io, time
import urllib.request, urllib.error
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
sys.path.insert(0, H)
from collect_registry import DATAGO_KEYS
URL = "http://apis.data.go.kr/1430000/PttRgstRtInfoInqSvc/getBusinessRightList"
KEY = DATAGO_KEYS[-1]
CD = "119980045698"          # SK하이닉스 — 앞서 rc=000 / total=27,378 확인됨
log = lambda m: print(m, flush=True)


def call(rows, page=1, cd=CD):
    u = (f"{URL}?serviceKey={KEY}&pageNo={page}&numOfRows={rows}"
         f"&type=json&searchType=2&searchVal={cd}")
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "p"}), timeout=30)
        b = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return -1, f"{type(e).__name__}"
    return 200, b


log("[numOfRows 상한 탐색]")
for n in (10, 50, 100, 200, 300, 500, 999, 1000):
    st, b = call(n)
    try:
        j = json.loads(b)
        rc, tc = j.get("resultCode"), j.get("totalCount")
        it = (j.get("items") or {}).get("rightList") or []
        if isinstance(it, dict):
            it = [it]
        log(f"  numOfRows={n:5,} → rc={rc} total={tc} 받은행={len(it)}"
            + (f"  msg={j.get('resultMsg')}" if rc != "000" else ""))
    except Exception:
        log(f"  numOfRows={n:5,} → 파싱실패 {b[:120]}")
    time.sleep(0.4)

log("\n[페이지 이동 확인 — 상한 안에서]")
for p in (1, 2, 3):
    st, b = call(100, page=p)
    try:
        j = json.loads(b)
        it = (j.get("items") or {}).get("rightList") or []
        if isinstance(it, dict):
            it = [it]
        first = str(it[0].get("rgstNo")) if it else "-"
        log(f"  pageNo={p} → 받은행 {len(it)} 첫 등록번호 {first}")
    except Exception:
        log(f"  pageNo={p} → 실패")
    time.sleep(0.4)

log("\n[응답 필드]")
st, b = call(3)
try:
    j = json.loads(b)
    it = (j.get("items") or {}).get("rightList") or []
    if isinstance(it, dict):
        it = [it]
    if it:
        log(f"  키: {list(it[0])}")
        log(f"  샘플: {json.dumps(it[0], ensure_ascii=False)[:300]}")
except Exception as e:
    log(f"  {e}")
