# -*- coding: utf-8 -*-
"""datago_pay(폴백) 응답 구조 확인 — 등록일·청구항이 정말 없는가"""
import sqlite3, gzip, json
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
con = sqlite3.connect(f"file:{BASE}\\수집파이프라인\\registry.sqlite?mode=ro", uri=True)
log = lambda m: print(m, flush=True)

rn, gz = con.execute("SELECT rgst_no, raw_gz FROM registry WHERE status='OK' "
                     "AND source='datago_pay' AND raw_gz IS NOT NULL LIMIT 1").fetchone()
txt = gzip.decompress(gz).decode("utf-8", errors="replace")
log(f"[datago_pay 원본] 등록번호 {rn} | 길이 {len(txt):,}자")
log(txt[:1500])

log("\n[키 등장 빈도 — pay 40건 스캔]")
ks = {}
for (g2,) in con.execute("SELECT raw_gz FROM registry WHERE status='OK' AND source='datago_pay' "
                         "AND raw_gz IS NOT NULL ORDER BY RANDOM() LIMIT 40"):
    try:
        j = json.loads(gzip.decompress(g2).decode("utf-8", errors="replace"))
    except Exception:
        continue

    def c(o):
        if isinstance(o, dict):
            for k, v in o.items():
                ks[k] = ks.get(k, 0) + 1
                c(v)
        elif isinstance(o, list):
            for x in o[:2]:
                c(x)
    c(j)
log("  " + str(sorted(ks.items(), key=lambda x: -x[1])))

log("\n[datago(정상) 최상위 items 키 — 우리가 안 쓰고 버린 필드]")
rn2, g3 = con.execute("SELECT rgst_no, raw_gz FROM registry WHERE status='OK' AND source='datago' "
                      "AND raw_gz IS NOT NULL LIMIT 1").fetchone()
j = json.loads(gzip.decompress(g3).decode("utf-8", errors="replace"))
it = j.get("items", {})
log(f"  items 키 {len(it)}개: {list(it)}")
log("\n  [리스트형 필드의 원소 수 = 우리가 못 쓰고 있는 정보]")
for k, v in it.items():
    if isinstance(v, list):
        log(f"    {k:14s} n={len(v)}  키: {list(v[0]) if v and isinstance(v[0], dict) else '-'}")

log("\n[전용실시권·권리이전 보유율 — datago 500건 표본]")
n = lic = own2 = pay3 = 0
for (g4,) in con.execute("SELECT raw_gz FROM registry WHERE status='OK' AND source='datago' "
                         "AND raw_gz IS NOT NULL ORDER BY RANDOM() LIMIT 500"):
    try:
        it = json.loads(gzip.decompress(g4).decode("utf-8", errors="replace")).get("items", {})
    except Exception:
        continue
    n += 1
    ls = it.get("elsLcnsList") or []
    if any((x or {}).get("elsLcnsPsnNm") for x in ls):
        lic += 1
    ow = it.get("owner") or []
    if len(ow) > 1:
        own2 += 1
    if len(it.get("pay") or []) >= 3:
        pay3 += 1
log(f"  표본 {n}건 | 전용실시권 설정 {lic} ({lic/max(n,1)*100:.1f}%) | "
    f"권리자 2인 이상(이전 이력) {own2} ({own2/max(n,1)*100:.1f}%) | 납부이력 3건+ {pay3} ({pay3/max(n,1)*100:.1f}%)")
con.close()
