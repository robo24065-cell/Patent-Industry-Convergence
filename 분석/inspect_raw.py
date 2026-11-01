# -*- coding: utf-8 -*-
"""
원본 응답 구조 확인 — 감사에서 두 가지가 걸렸다
  (1) datago_pay 소스 209,609건(38.8%)에 등록일·청구항이 0% 유효
      → 응답에 정말 없는 것인가, 아니면 우리 파서가 못 뽑은 것인가?
        후자라면 재수집 없이 raw_gz 에서 복구할 수 있다.
  (2) 감사의 '등록연도 < 출원연도 37,008건'은 출원번호 앞 4자리를 연도로 본 내 가정 때문일 수 있다
      → 출원번호 실제 형식을 확인한다
"""
import sqlite3, gzip, json, io, sys
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = BASE + r"\수집파이프라인\registry.sqlite"
log = lambda m: print(m, flush=True)
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

log("[0] 출원번호 실제 형식")
for rn, an, ty in con.execute(
        "SELECT rgst_no, appl_no, CASE WHEN rgst_no LIKE '2%' THEN '실용' ELSE '특허' END "
        "FROM registry WHERE status='OK' ORDER BY RANDOM() LIMIT 8"):
    log(f"  {ty} 등록={rn} 출원={an} (길이 {len(str(an))})")

log("\n[1] 소스별 원본 응답 구조")
for src in ("datago", "datago_pay"):
    r = con.execute("SELECT rgst_no, raw_gz FROM registry WHERE status='OK' AND source=? "
                    "AND raw_gz IS NOT NULL LIMIT 1", (src,)).fetchone()
    if not r:
        log(f"  {src}: 원본 없음"); continue
    rn, gz = r
    txt = gzip.decompress(gz).decode("utf-8", errors="replace")
    log(f"\n  ── {src} (등록번호 {rn}) ──")
    log(f"  원문 길이 {len(txt):,}자")
    try:
        j = json.loads(txt)

        def walk(o, path="", depth=0):
            if depth > 4:
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    p = f"{path}.{k}" if path else k
                    if isinstance(v, (dict, list)):
                        n = len(v)
                        log(f"    {p}  <{type(v).__name__} n={n}>")
                        walk(v, p, depth + 1)
                    else:
                        s = str(v)
                        log(f"    {p} = {s[:60]}")
            elif isinstance(o, list) and o:
                walk(o[0], path + "[0]", depth + 1)

        walk(j)
    except Exception as e:
        log(f"  JSON 파싱 실패: {e}")
        log(f"  원문 앞 400자: {txt[:400]}")

log("\n[2] datago_pay 응답에 등록일/청구항이 실제로 있는가 — 20건 스캔")
rows = con.execute("SELECT rgst_no, raw_gz FROM registry WHERE status='OK' AND source='datago_pay' "
                   "AND raw_gz IS NOT NULL ORDER BY RANDOM() LIMIT 20").fetchall()
keys_seen, has_regdate, has_claim = {}, 0, 0
for rn, gz in rows:
    txt = gzip.decompress(gz).decode("utf-8", errors="replace")
    low = txt.lower()
    if any(t in low for t in ("regidate", "regdate", "등록일")):
        has_regdate += 1
    if any(t in low for t in ("claim", "청구항")):
        has_claim += 1
    try:
        j = json.loads(txt)

        def collect(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys_seen[k] = keys_seen.get(k, 0) + 1
                    collect(v)
            elif isinstance(o, list):
                for x in o[:2]:
                    collect(x)
        collect(j)
    except Exception:
        pass
log(f"  등록일 흔적 {has_regdate}/20 | 청구항 흔적 {has_claim}/20")
log(f"  등장 키 상위 20: {sorted(keys_seen.items(), key=lambda x: -x[1])[:20]}")

log("\n[3] datago(정상) 응답의 키 — 대조군")
rows = con.execute("SELECT raw_gz FROM registry WHERE status='OK' AND source='datago' "
                   "AND raw_gz IS NOT NULL ORDER BY RANDOM() LIMIT 20").fetchall()
ks = {}
for (gz,) in rows:
    try:
        j = json.loads(gzip.decompress(gz).decode("utf-8", errors="replace"))

        def c2(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    ks[k] = ks.get(k, 0) + 1
                    c2(v)
            elif isinstance(o, list):
                for x in o[:2]:
                    c2(x)
        c2(j)
    except Exception:
        pass
log(f"  등장 키 상위 20: {sorted(ks.items(), key=lambda x: -x[1])[:20]}")
con.close()
