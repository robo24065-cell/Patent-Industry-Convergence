# -*- coding: utf-8 -*-
"""C13-T1 s6: (d) 예고등록 말소 후에도 원부에 예고 행이 남는지 — 취하·기각 확정 사건 앞에 예고등록이 있는 비율. 출력 cache/s6_withdraw.json"""
import sqlite3, sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
N = "('무효(심판)예고등록','권리범위(심판)예고등록','정정(심판)예고등록','심판예고등록','취소(심판)예고등록','취소신청예고등록','이의신청예고등록','기술평가예고등록')"
out = {}
for nm, cond in [("취하등록(초심)", "e.rgstCsName='취하등록(초심)'"),
                 ("심결확정(초심)_기각각하", "e.rgstCsName like '심결%확정(초심)등록' and (replace(e.rgstCsReason,' ','') like '%기각%' or replace(e.rgstCsReason,' ','') like '%각하%')"),
                 ("심결확정(초심)_전체", "e.rgstCsName like '심결%확정(초심)등록'")]:
    tot, hit = c.execute(f"""select count(*), sum(case when exists(select 1 from right n where n.rgst_no=e.rgst_no and n.rgstCsName in {N}
                and cast(n.rgstCsDate as integer) <= cast(e.rgstCsDate as integer)) then 1 else 0 end) from right e where {cond}""").fetchone()
    out[nm] = dict(events=tot, with_prior_notice=hit, share=(hit / tot if tot else None))
r = c.execute("select replace(rgstCsReason,' ',''), count(*) from right where rgstCsName like '심결%확정(초심)등록' group by 1 order by 2 desc limit 12").fetchall()
out["decision_reason_top"] = r
json.dump(out, open(os.path.join(W, "cache", "s6_withdraw.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
