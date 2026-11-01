# -*- coding: utf-8 -*-
import os, sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
log = lambda m: print(m, flush=True)
def con(db): return sqlite3.connect(f"file:{os.path.join(H,db)}?mode=ro", uri=True)

log("[3] 사업자 권리목록 (business_rights.sqlite)")
b = con("business_rights.sqlite")
f = con("registry_full.sqlite")
tgt = f.execute("SELECT COUNT(DISTINCT applicantCd) FROM applicant WHERE applicantCd IS NOT NULL AND applicantCd<>''").fetchone()[0]
f.close()
log(f"  전체 출원인 대상      {tgt:,}")
for s,n in b.execute("SELECT status,COUNT(*) FROM applicant_query GROUP BY 1 ORDER BY 2 DESC"):
    lab = {"OK":"성공(법인)","RC:001":"해당없음(개인)"}.get(s,s)
    log(f"    {lab:18s} {n:8,}  ({n/tgt*100:5.1f}%)")
q = b.execute("SELECT COUNT(*) FROM applicant_query").fetchone()[0]
log(f"  처리 완료             {q:,} / {tgt:,}  ({q/tgt*100:.1f}%)")
log(f"  ★ 미처리              {tgt-q:,}")
tr = b.execute("SELECT COUNT(*) FROM applicant_query WHERE truncated=1").fetchone()[0]
log(f"  ★ 잘림(이어받기 필요)  {tr:,}")
if tr:
    for nm,tc,fe,pg in b.execute("SELECT applicant_name,total_count,fetched,pages FROM applicant_query WHERE truncated=1 ORDER BY total_count DESC LIMIT 5"):
        log(f"      {str(nm)[:20]:22s} 보유 {tc:7,} / 받음 {fe:7,} (p{pg})")
nr = b.execute("SELECT COUNT(*) FROM right_list").fetchone()[0]
log(f"\n  수집 권리행           {nr:,}")
s = b.execute("SELECT SUM(total_count), SUM(fetched), SUM(n_in_fusion) FROM applicant_query WHERE status='OK'").fetchone()
log(f"  API 보고 보유 합계     {s[0]:,}")
log(f"  실제 받은 합계         {s[1]:,}   {'✓ 일치' if s[0]==s[1] else f'차이 {s[0]-s[1]:,} (잘림분)'}")
log(f"  제공범위 내 보유       {s[2]:,}  → 범위 밖 포함 실제 {s[0]/max(s[2],1):.1f}배")
d = b.execute("SELECT COUNT(*) FROM (SELECT applicant_cd,rgst_no FROM right_list GROUP BY 1,2 HAVING COUNT(*)>1)").fetchone()[0]
log(f"  ★ (출원인,등록번호) 중복 {d:,} {'✓' if d==0 else '←확인'}")
b.close()
