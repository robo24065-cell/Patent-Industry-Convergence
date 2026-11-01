# -*- coding: utf-8 -*-
import os, sqlite3, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
log = lambda m: print(m, flush=True)
b = sqlite3.connect(f"file:{os.path.join(H,'business_rights.sqlite')}?mode=ro", uri=True)

log("[A] (출원인,등록번호) 중복 3,906건의 정체")
rows = b.execute("""SELECT applicant_cd, rgst_no, COUNT(*) n FROM right_list
                    GROUP BY 1,2 HAVING n>1 ORDER BY n DESC LIMIT 3""").fetchall()
for cd, rn, n in rows:
    log(f"\n  출원인 {cd} / 등록번호 {rn} — {n}행")
    for (raw,) in b.execute("SELECT raw FROM right_list WHERE applicant_cd=? AND rgst_no=?", (cd, rn)):
        j = json.loads(raw)
        log(f"    rn={j.get('rn')} pertDiv={j.get('pertDiv')} rkNo={j.get('rkNo')} rgstStatus={j.get('rgstStatus')}")
log("\n  [중복행의 pertDiv 조합]")
for pd, n in b.execute("""SELECT pd, COUNT(*) FROM (
    SELECT applicant_cd, rgst_no, json_extract(raw,'$.pertDiv') pd FROM right_list
    WHERE (applicant_cd,rgst_no) IN (SELECT applicant_cd,rgst_no FROM right_list GROUP BY 1,2 HAVING COUNT(*)>1)
  ) GROUP BY 1 ORDER BY 2 DESC"""):
    log(f"    {pd}: {n:,}")
log("\n  고유 (출원인,등록번호) 쌍: "
    f"{b.execute('SELECT COUNT(*) FROM (SELECT DISTINCT applicant_cd,rgst_no FROM right_list)').fetchone()[0]:,}")

log("\n[B] 잘림 6건 — 이어받기가 왜 안 됐나")
for nm, tc, fe, pg, tr, st in b.execute(
    "SELECT applicant_name,total_count,fetched,pages,truncated,status FROM applicant_query WHERE truncated=1"):
    log(f"    {str(nm)[:20]:22s} 보유 {tc:7,} 받음 {fe:7,} 마지막p {pg} → 남은 {(tc-fe)//100+1:,}페이지")
log("\n  ※ max-pages 60 x 여러 세션 누적으로 p263~323 까지 진행됨. 이어받기는 작동 중.")
log("     남은 6개는 다음 실행에서 계속 이어받는다.")
b.close()
