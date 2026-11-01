# -*- coding: utf-8 -*-
"""전체 수집 자산 최종 검증"""
import os, sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
log = lambda m: print(m, flush=True)
def C(db): return sqlite3.connect(f"file:{os.path.join(H,db)}?mode=ro", uri=True)

log("="*72)
log("[1] 등록원부  registry.sqlite")
c = C("registry.sqlite")
tot = c.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
ok  = c.execute("SELECT COUNT(*) FROM registry WHERE status='OK'").fetchone()[0]
raw = c.execute("SELECT COUNT(*) FROM registry WHERE raw_gz IS NOT NULL").fetchone()[0]
src = dict(c.execute("SELECT source,COUNT(*) FROM registry WHERE status='OK' GROUP BY 1").fetchall())
miss= c.execute("SELECT COUNT(*) FROM pending p LEFT JOIN registry r ON p.rgst_no=r.rgst_no WHERE r.rgst_no IS NULL").fetchone()[0]
dup = c.execute("SELECT COUNT(*) FROM (SELECT rgst_no FROM registry GROUP BY 1 HAVING COUNT(*)>1)").fetchone()[0]
bad = c.execute("SELECT COUNT(*) FROM registry WHERE status<>'OK'").fetchone()[0]
c.close()
log(f"  대상 {tot:,} / 수집 {ok:,} ({ok/tot*100:.2f}%)")
log(f"  소스 {src}")
log(f"  원본보관 {raw:,} ({raw/ok*100:.1f}%) | 실패 {bad} | 미수집 {miss} | 중복 {dup}")
log(f"  판정: {'✓ 완전' if (ok==tot and miss==0 and dup==0 and bad==0 and raw==ok) else '← 확인필요'}")

log("\n[2] 완전추출  registry_full.sqlite")
f = C("registry_full.sqlite")
tabs = [r[0] for r in f.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
np_ = f.execute("SELECT COUNT(*) FROM patent").fetchone()[0]
uk  = f.execute("SELECT COUNT(*) FROM unknown_keys").fetchone()[0]
tot_rows = 0
for t in tabs:
    n = f.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]; tot_rows += n
    log(f"    {t:20s} {n:10,}")
log(f"  patent {np_:,} vs 등록원부 {tot:,} → {'✓ 일치' if np_==tot else '← 불일치'}")
log(f"  unknown_keys {uk} → {'✓ 스키마 밖 필드 없음' if uk==0 else '← 확인필요'}")
log(f"  총 {tot_rows:,}행")
miss_cols = []
for col in ("rgstNo","applNo","applDate","rgstDate","claimCount","cndrtExptnDate","ipcCd","title","applTpcd"):
    v = f.execute(f'SELECT COUNT(*) FROM patent WHERE "{col}" IS NOT NULL AND "{col}"<>""').fetchone()[0]
    if v/np_ < 0.999: miss_cols.append((col, v/np_))
log(f"  주요필드 결측: {'✓ 없음(9개 전부 99.9%+)' if not miss_cols else miss_cols}")
orph = sum(f.execute(f'SELECT COUNT(*) FROM "{t}" WHERE rgst_no NOT IN (SELECT rgst_no FROM patent)').fetchone()[0]
           for t in ("applicant","inventor","owner","pay","right"))
log(f"  고아 레코드 {orph} → {'✓' if orph==0 else '← 확인필요'}")
f.close()
