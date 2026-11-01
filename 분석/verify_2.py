# -*- coding: utf-8 -*-
import os, sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
log = lambda m: print(m, flush=True)
def con(db): return sqlite3.connect(f"file:{os.path.join(H,db)}?mode=ro", uri=True)

log("[2] 완전추출 (registry_full.sqlite)")
c = con("registry_full.sqlite")
tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
for t in tabs:
    n = c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    mark = "  ← 0이어야 정상" if t=="unknown_keys" else ""
    log(f"    {t:20s} {n:10,}{mark}")
np = c.execute("SELECT COUNT(*) FROM patent").fetchone()[0]
log(f"\n  patent {np:,}건 — 등록원부 656,360 대비 {'✓ 일치' if np==656360 else f'차이 {656360-np:,}'}")
log("\n  [주요 필드 결측률]")
for col in ("rgstNo","applNo","applDate","rgstDate","claimCount","cndrtExptnDate","ipcCd","title"):
    v = c.execute(f'SELECT COUNT(*) FROM patent WHERE "{col}" IS NOT NULL AND "{col}"<>""').fetchone()[0]
    st = "✓" if v/np > 0.99 else "!"
    log(f"    {st} {col:16s} {v:9,} ({v/np*100:6.2f}%)")
log("\n  [고아 레코드 — patent 에 없는 rgst_no 를 참조하는 자식 행]")
for t in ("applicant","inventor","owner","pay","right"):
    if t not in tabs: continue
    o = c.execute(f'SELECT COUNT(*) FROM "{t}" WHERE rgst_no NOT IN (SELECT rgst_no FROM patent)').fetchone()[0]
    log(f"    {t:12s} 고아 {o:,} {'✓' if o==0 else '←확인'}")
c.close()
