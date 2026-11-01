# -*- coding: utf-8 -*-
import os, sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
log = lambda m: print(m, flush=True)
def C(db): return sqlite3.connect(f"file:{os.path.join(H,db)}?mode=ro", uri=True)

log("[3] 사업자 권리목록  business_rights.sqlite")
b = C("business_rights.sqlite"); f = C("registry_full.sqlite")
tgt = f.execute("SELECT COUNT(DISTINCT applicantCd) FROM applicant WHERE applicantCd IS NOT NULL AND applicantCd<>''").fetchone()[0]
f.close()
q = b.execute("SELECT COUNT(*) FROM applicant_query").fetchone()[0]
st = dict(b.execute("SELECT status,COUNT(*) FROM applicant_query GROUP BY 1").fetchall())
tr = b.execute("SELECT COUNT(*) FROM applicant_query WHERE truncated=1").fetchone()[0]
s  = b.execute("SELECT SUM(total_count),SUM(fetched) FROM applicant_query WHERE status='OK'").fetchone()
nr = b.execute("SELECT COUNT(*) FROM right_list").fetchone()[0]
uq = b.execute("SELECT COUNT(*) FROM (SELECT DISTINCT applicant_cd,rgst_no FROM right_list)").fetchone()[0]
log(f"  대상 출원인 {tgt:,} / 처리 {q:,} ({q/tgt*100:.1f}%) | 미처리 {tgt-q:,}")
log(f"  상태 {st}")
log(f"  ★ 잘림(미완료) {tr:,}")
log(f"  API 보유합계 {s[0]:,} / 받음 {s[1]:,} → 미수집 {s[0]-s[1]:,}")
log(f"  권리행 {nr:,} (고유 출원인×등록번호 {uq:,}, 차이 {nr-uq:,}=실시권 다중행)")
log(f"  판정: {'✓ 완전' if (q==tgt and tr==0 and s[0]==s[1]) else '← 확인필요'}")

log("\n  [최대 기업 완결 확인]")
for nm,tc,fe,pg,t in b.execute("SELECT applicant_name,total_count,fetched,pages,truncated FROM applicant_query ORDER BY total_count DESC LIMIT 6"):
    mark = "✓" if (t==0 and fe>=tc) else "←"
    log(f"    {mark} {str(nm)[:18]:20s} 보유 {tc:7,} 받음 {fe:7,} p{pg}")

log("\n  [권리행 필드 결측]")
for col in ("rgst_no","rgst_date","appl_no","title","claim_count","corp_rgstno"):
    v = b.execute(f'SELECT COUNT(*) FROM right_list WHERE "{col}" IS NOT NULL AND "{col}"<>""').fetchone()[0]
    log(f"    {col:14s} {v:9,} ({v/nr*100:5.1f}%)")

log("\n[4] 제공범위 밖 규모")
r = C("registry.sqlite"); ours = {x[0] for x in r.execute("SELECT rgst_no FROM pending")}; r.close()
tot_r = out = 0
for (rn,) in b.execute("SELECT DISTINCT rgst_no FROM right_list"):
    tot_r += 1
    if str(rn).strip() not in ours: out += 1
log(f"  고유 등록번호 {tot_r:,} | 제공범위(50 KSIC) 밖 {out:,} ({out/tot_r*100:.1f}%)")
b.close()
