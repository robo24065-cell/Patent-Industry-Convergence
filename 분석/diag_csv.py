# -*- coding: utf-8 -*-
"""
통합 CSV 무결성 진단
────────────────────────────────────────────────────────────────
증상: 중간에 빈칸이 쭈르륵 있거나 열이 안 맞는 부분이 있다.

의심 원인 3가지를 각각 검정한다.
  [A] 필드 안에 개행문자(\n, \r)가 들어 있어 한 행이 여러 줄로 쪼개짐
      → 엑셀·일부 도구가 이를 새 행으로 오인 → 열이 밀림
  [B] 아직 재수집 안 된 레코드(datago_pay)는 스칼라 필드가 전부 NULL
      → 진짜로 빈칸이 맞음 (버그 아님)
  [C] 따옴표·구분자 이스케이프 문제
"""
import csv, os, sqlite3, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
CSVP = os.path.join(BASE, "공모전데이터", "내보내기", "등록원부_20260815", "통합표_특허별.csv")
log = lambda m: print(m, flush=True)

log("[A] 원본 데이터에 개행문자가 들어있는가")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
found = {}
for t in tabs:
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
    txt = [c for c in cols if c not in ("rgst_no", "seq")]
    if not txt:
        continue
    conds = " OR ".join(
        f"\"{c}\" LIKE '%' || char(10) || '%' OR \"{c}\" LIKE '%' || char(13) || '%'" for c in txt)
    try:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}" WHERE {conds}').fetchone()[0]
    except Exception:
        n = -1
    if n:
        found[t] = n
        log(f"  {t:20s} 개행 포함 {n:,}행")
if not found:
    log("  개행 포함 레코드 없음")

log("\n  [어느 컬럼인가 — 상위]")
for t in list(found)[:4]:
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")') if r[1] not in ("rgst_no", "seq")]
    for c in cols:
        try:
            n = con.execute(
                f'SELECT COUNT(*) FROM "{t}" WHERE "{c}" LIKE \'%\' || char(10) || \'%\' '
                f'OR "{c}" LIKE \'%\' || char(13) || \'%\'').fetchone()[0]
        except Exception:
            continue
        if n:
            s = con.execute(
                f'SELECT "{c}" FROM "{t}" WHERE "{c}" LIKE \'%\' || char(10) || \'%\' '
                f'OR "{c}" LIKE \'%\' || char(13) || \'%\' LIMIT 1').fetchone()[0]
            log(f"    {t}.{c}: {n:,}건  예: {repr(str(s)[:90])}")

log("\n[B] 스칼라가 전부 비어있는 레코드 = 아직 재수집 안 된 건")
tot = con.execute("SELECT COUNT(*) FROM patent").fetchone()[0]
empty = con.execute("SELECT COUNT(*) FROM patent WHERE rgstNo IS NULL AND applNo IS NULL "
                    "AND claimCount IS NULL AND title IS NULL").fetchone()[0]
log(f"  patent {tot:,}행 중 스칼라 전부 NULL = {empty:,}행 ({empty/tot*100:.1f}%)")
log("  → 이건 CSV 버그가 아니라 '아직 연차료만 받은 건'이 그대로 비어 나온 것")

log("\n[C] 따옴표 포함 필드")
q = con.execute("SELECT COUNT(*) FROM patent WHERE title LIKE '%\"%'").fetchone()[0]
log(f"  title 에 큰따옴표 포함 {q:,}건")
con.close()

log("\n[D] 실제 CSV 파일 검사")
if not os.path.exists(CSVP):
    log(f"  파일 없음: {CSVP}")
    sys.exit()
size = os.path.getsize(CSVP) / 1024 ** 2
raw_lines = 0
with open(CSVP, "rb") as f:
    for _ in f:
        raw_lines += 1
log(f"  크기 {size:.0f} MB | 물리적 줄 수 {raw_lines:,}")

csv.field_size_limit(10 ** 7)
with open(CSVP, encoding="utf-8-sig", newline="") as f:
    r = csv.reader(f)
    hdr = next(r)
    ncol = len(hdr)
    rows = bad = 0
    badex = []
    for row in r:
        rows += 1
        if len(row) != ncol:
            bad += 1
            if len(badex) < 5:
                badex.append((rows, len(row), row[:3]))
log(f"  헤더 열 수 {ncol}")
log(f"  csv 파서가 읽은 논리적 행 수 {rows:,}")
log(f"  물리 줄 − 논리 행 = {raw_lines-1-rows:,}  ← 필드 안 개행 때문에 생긴 차이")
log(f"  열 수 불일치 행 {bad:,}")
for i, n, ex in badex:
    log(f"    행 {i}: {n}열 (기대 {ncol}) {ex}")
if bad == 0:
    log("  ✓ csv 규격 자체는 정상 — 모든 행의 열 수가 일치한다")
    log("    (엑셀에서 밀려 보인다면 필드 안 개행을 엑셀이 새 행으로 그린 탓)")
