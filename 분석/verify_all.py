# -*- coding: utf-8 -*-
"""전체 수집 데이터 정합성 검증"""
import os, sqlite3, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
log = lambda m: print(m, flush=True)
def q(db, sql, p=()):
    c = sqlite3.connect(f"file:{os.path.join(H,db)}?mode=ro", uri=True)
    try: return c.execute(sql, p).fetchall()
    finally: c.close()

log("="*70); log("[1] 등록원부 (registry.sqlite)")
tot = q("registry.sqlite","SELECT COUNT(*) FROM pending")[0][0]
log(f"  대상(pending)        {tot:,}")
for s,n in q("registry.sqlite","SELECT source,COUNT(*) FROM registry WHERE status='OK' GROUP BY 1 ORDER BY 2 DESC"):
    log(f"    {s:14s} {n:9,}  ({n/tot*100:5.1f}%)")
ok = q("registry.sqlite","SELECT COUNT(*) FROM registry WHERE status='OK'")[0][0]
bad = q("registry.sqlite","SELECT COUNT(*) FROM registry WHERE status<>'OK'")[0][0]
raw = q("registry.sqlite","SELECT COUNT(*) FROM registry WHERE raw_gz IS NOT NULL")[0][0]
log(f"  수집 성공            {ok:,} / {tot:,}  ({ok/tot*100:.2f}%)")
log(f"  실패 잔존            {bad:,}")
log(f"  원본 보관(raw_gz)    {raw:,}  ({raw/ok*100:.1f}%)")
miss = q("registry.sqlite","SELECT COUNT(*) FROM pending p LEFT JOIN registry r ON p.rgst_no=r.rgst_no WHERE r.rgst_no IS NULL")[0][0]
log(f"  ★ 미수집             {miss:,}  {'✓ 없음' if miss==0 else '← 확인 필요'}")
dup = q("registry.sqlite","SELECT COUNT(*) FROM (SELECT rgst_no FROM registry GROUP BY 1 HAVING COUNT(*)>1)")[0][0]
log(f"  ★ 등록번호 중복       {dup:,}  {'✓ 없음' if dup==0 else '← 확인 필요'}")
