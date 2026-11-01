# -*- coding: utf-8 -*-
"""사업자 권리목록 수집 오류 진단"""
import json, os, sqlite3, sys, io, time
import urllib.request, urllib.error
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
sys.path.insert(0, H)
from collect_registry import DATAGO_KEYS
log = lambda m: print(m, flush=True)

c = sqlite3.connect(f"file:{os.path.join(H,'business_rights.sqlite')}?mode=ro", uri=True)
log("[상태 분포]")
for st, n in c.execute("SELECT status, COUNT(*) FROM applicant_query GROUP BY 1 ORDER BY 2 DESC"):
    log(f"  {st:14s} {n:6,}")

log("\n[오류 건의 특성 — 융합데이터 내 보유량별]")
for st, in c.execute("SELECT DISTINCT status FROM applicant_query WHERE status<>'OK'"):
    r = c.execute("SELECT COUNT(*), AVG(n_in_fusion), MIN(n_in_fusion), MAX(n_in_fusion) "
                  "FROM applicant_query WHERE status=?", (st,)).fetchone()
    log(f"  {st}: {r[0]:,}건 | 융합 내 보유 평균 {r[1]:.0f} (범위 {r[2]}~{r[3]})")

log("\n[성공 건]")
r = c.execute("SELECT COUNT(*), AVG(n_in_fusion), AVG(total_count) FROM applicant_query "
              "WHERE status='OK'").fetchone()
log(f"  {r[0]:,}건 | 융합 평균 {r[1]:.0f} → 실제 평균 {r[2]:.0f}")

log("\n[오류 샘플 재현 — 실제 응답 확인]")
bad = c.execute("SELECT applicant_cd, applicant_name, n_in_fusion, status FROM applicant_query "
                "WHERE status<>'OK' ORDER BY n_in_fusion DESC LIMIT 5").fetchall()
c.close()
K = DATAGO_KEYS[-1]
U = "http://apis.data.go.kr/1430000/PttRgstRtInfoInqSvc/getBusinessRightList"
for cd, nm, nf, st in bad:
    u = f"{U}?serviceKey={K}&pageNo=1&numOfRows=10&type=json&searchType=2&searchVal={cd}"
    try:
        b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "x"}),
                                   timeout=25).read().decode("utf-8", "replace")
        log(f"  {str(nm)[:16]:18s} cd={cd} (융합 {nf:,}) 저장상태={st}")
        log(f"    → {b[:180]}")
    except urllib.error.HTTPError as e:
        log(f"  {str(nm)[:16]:18s} cd={cd} HTTP {e.code}")
    except Exception as e:
        log(f"  {str(nm)[:16]:18s} {type(e).__name__}")
    time.sleep(0.4)
