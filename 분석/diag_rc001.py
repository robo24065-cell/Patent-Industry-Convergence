# -*- coding: utf-8 -*-
"""RC:001 의 정체 — 진짜 오류인가, 데이터 없음인가"""
import json, os, sqlite3, sys, io, time
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
sys.path.insert(0, H)
from collect_registry import DATAGO_KEYS
log = lambda m: print(m, flush=True)

c = sqlite3.connect(f"file:{os.path.join(H,'business_rights.sqlite')}?mode=ro", uri=True)
bad = c.execute("SELECT applicant_cd, applicant_name, n_in_fusion FROM applicant_query "
                "WHERE status='RC:001' ORDER BY n_in_fusion DESC LIMIT 6").fetchall()
good = c.execute("SELECT applicant_cd, applicant_name, n_in_fusion FROM applicant_query "
                 "WHERE status='OK' AND n_in_fusion<=3 LIMIT 3").fetchall()
c.close()

K = DATAGO_KEYS[-1]
U = "http://apis.data.go.kr/1430000/PttRgstRtInfoInqSvc/getBusinessRightList"


def probe(cd):
    u = f"{U}?serviceKey={K}&pageNo=1&numOfRows=10&type=json&searchType=2&searchVal={cd}"
    try:
        b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "x"}),
                                   timeout=25).read().decode("utf-8", "replace")
        j = json.loads(b)
        return j.get("resultCode"), j.get("resultMsg"), j.get("totalCount")
    except Exception as e:
        return "EXC", type(e).__name__, None


log("[RC:001 로 기록된 출원인 — 다시 조회]")
for cd, nm, nf in bad:
    rc, msg, tc = probe(cd)
    log(f"  {str(nm)[:22]:24s} cd={cd} (융합 {nf:2d}건) → rc={rc} msg={msg} total={tc}")
    time.sleep(0.4)

log("\n[대조: 성공한 소규모 출원인]")
for cd, nm, nf in good:
    rc, msg, tc = probe(cd)
    log(f"  {str(nm)[:22]:24s} cd={cd} (융합 {nf:2d}건) → rc={rc} msg={msg} total={tc}")
    time.sleep(0.4)

log("\n[출원인번호 앞자리로 본 유형]")
log("  특허청 출원인번호 체계: 1=국내법인 2=국내단체 4=국내개인 5=외국법인 …")
c = sqlite3.connect(f"file:{os.path.join(H,'business_rights.sqlite')}?mode=ro", uri=True)
for st in ("OK", "RC:001"):
    rows = c.execute("SELECT substr(applicant_cd,1,1) p, COUNT(*) FROM applicant_query "
                     "WHERE status=? GROUP BY 1 ORDER BY 2 DESC", (st,)).fetchall()
    log(f"  {st:8s}: " + ", ".join(f"{p}…{n:,}" for p, n in rows))
c.close()
