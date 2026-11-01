# -*- coding: utf-8 -*-
"""내보낸 CSV 무결성 검사 — 행수 일치 + 열 정렬"""
import csv, os, sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
csv.field_size_limit(10**8)
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
D = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\내보내기\등록원부_20260831"
log = lambda m: print(m, flush=True)

def dbcount(db, t):
    c = sqlite3.connect(f"file:{os.path.join(H,db)}?mode=ro", uri=True)
    try: return c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    finally: c.close()

def scan(path):
    raw = sum(1 for _ in open(path, "rb"))
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f); hdr = next(r); n = len(hdr)
        rows = bad = 0
        for row in r:
            rows += 1
            if len(row) != n: bad += 1
    return raw, rows, n, bad

log("[CSV 무결성 — DB 행수 대비]")
log(f"  {'파일':24s} {'DB':>10s} {'CSV':>10s} {'열':>4s} {'열불일치':>8s} {'물리-논리':>9s}")
tot_bad = 0
targets = [("registry_full.sqlite", t, os.path.join(D, f"{t}.csv")) for t in
           ("patent","applicant","inventor","owner","pay","right","applAgent",
            "rgstAgent","cofpr","elsLcnsList","noelsLcnsList","utilityModelOwner")]
targets += [("business_rights.sqlite", t, os.path.join(D, "사업자권리목록", f"{t}.csv"))
            for t in ("applicant_query","right_list")]
for db, t, p in targets:
    if not os.path.exists(p): log(f"  {t:24s} 파일 없음"); continue
    dbn = dbcount(db, t)
    raw, rows, ncol, bad = scan(p)
    ok = "✓" if (dbn == rows and bad == 0) else "←"
    tot_bad += bad
    log(f"  {ok} {t:22s} {dbn:10,} {rows:10,} {ncol:4d} {bad:8,} {raw-1-rows:9,}")

log("\n[통합표]")
p = os.path.join(D, "통합표_특허별.csv")
raw, rows, ncol, bad = scan(p)
log(f"  행 {rows:,} / 열 {ncol} / 열불일치 {bad} / 물리-논리 차이 {raw-1-rows}")
log(f"  patent DB {dbcount('registry_full.sqlite','patent'):,} → {'✓ 일치' if rows==dbcount('registry_full.sqlite','patent') else '← 불일치'}")
log(f"\n총 열불일치 {tot_bad}건 → {'✓ 전부 정상' if tot_bad==0 else '← 확인필요'}")
