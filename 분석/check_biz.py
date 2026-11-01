# -*- coding: utf-8 -*-
"""사업자 권리목록 수집 결과 확인"""
import os, sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
p = os.path.join(H, "business_rights.sqlite")
if not os.path.exists(p):
    print("DB 없음"); sys.exit()
c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
nq = c.execute("SELECT COUNT(*) FROM applicant_query").fetchone()[0]
nr = c.execute("SELECT COUNT(*) FROM right_list").fetchone()[0]
print(f"조회 {nq:,} | 권리행 {nr:,}")
print("\n[융합데이터 대비 실제 보유]")
for nm, nf, tc, fe, pg, tr, st in c.execute(
        "SELECT applicant_name,n_in_fusion,total_count,fetched,pages,truncated,status "
        "FROM applicant_query ORDER BY total_count DESC LIMIT 10"):
    r = f"{tc/nf:.1f}배" if nf and tc else "-"
    print(f"  {str(nm)[:18]:20s} 융합 {nf:6,} → 실제 {tc or 0:7,} ({r:>6s})  "
          f"받음 {fe:5,} p{pg} {'잘림' if tr else ''} {st}")
print("\n[상태]", dict(c.execute("SELECT status,COUNT(*) FROM applicant_query GROUP BY 1").fetchall()))
print("\n[샘플 권리행]")
for x in c.execute("SELECT rgst_no,title,claim_count,business_no,corp_rgstno "
                   "FROM right_list LIMIT 3"):
    print("  ", x)
ok = c.execute("SELECT SUM(total_count), SUM(n_in_fusion) FROM applicant_query "
               "WHERE status='OK'").fetchone()
if ok and ok[0]:
    print(f"\n[합계] 융합데이터 {ok[1]:,}건 → 실제 보유 {ok[0]:,}건 "
          f"({ok[0]/max(ok[1],1):.1f}배)")
c.close()
