# -*- coding: utf-8 -*-
"""getBusinessRightList 수집 대상 규모 산정"""
import os, sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
log = lambda m: print(m, flush=True)

f = sqlite3.connect(f"file:{os.path.join(H,'registry_full.sqlite')}?mode=ro", uri=True)
rows = f.execute("""SELECT applicantCd, applicantName, COUNT(*) c FROM applicant
                    WHERE applicantCd IS NOT NULL AND applicantCd<>''
                    GROUP BY 1 ORDER BY c DESC""").fetchall()
f.close()
tot = sum(r[2] for r in rows)
log(f"[출원인번호] 고유 {len(rows):,}개 / 특허-출원인 연결 {tot:,}건")
for k in (100, 500, 1000, 3000, 10000, 30000, 100000):
    if k > len(rows):
        break
    s = sum(r[2] for r in rows[:k])
    log(f"  상위 {k:7,}개가 전체의 {s/tot*100:5.1f}% 보유 ({s:,}건)")
log("  상위 5: " + ", ".join(f"{r[1]}({r[2]:,})" for r in rows[:5]))

c = sqlite3.connect(f"file:{os.path.join(H,'registry.sqlite')}?mode=ro", uri=True)
tp = c.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
d = c.execute("SELECT COUNT(*) FROM registry WHERE status='OK' AND source='datago'").fetchone()[0]
c.close()
log(f"\n[등록원부] 전체필드 {d:,}/{tp:,} ({d/tp*100:.1f}%) | 잔여 {tp-d:,}")

dd = sqlite3.connect(os.path.join(H, "..", "공모전데이터", "외부데이터", "DART", "dart_company.sqlite"))
nb = dd.execute("SELECT COUNT(*) FROM company WHERE bizr_no<>''").fetchone()[0]
dd.close()
log(f"[DART] 사업자번호 보유 {nb:,}개사")

log("\n[콜 수 추정] numOfRows=1000 기준, totalCount/1000 페이지 필요")
log("  대부분 출원인은 특허가 적어 1콜이면 끝난다.")
log("  대형(LG전자 10.3만건)은 103콜. 상위 소수가 콜을 많이 먹는다.")
log(f"  일일 한도 10키 x 10,000 = 100,000콜")
