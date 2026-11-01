# -*- coding: utf-8 -*-
"""
무엇을 못 받았는가 — 정확히 특정한다
────────────────────────────────────────────────────────────────
수집 시 오퍼레이션이 두 종류였다.
  getPatentRegisterHistory / getUtilityModelHistory  → 28개 필드 전부 (주 오퍼레이션)
  getPayHistory                                      → payList 4필드만 (폴백)

폴백은 주 오퍼레이션의 일일 한도(키당 9,500)가 소진됐을 때 남은 한도를 쓰려고 넣었다.
연차료는 얻었지만 나머지를 잃었다.

단, 잃은 것 중 상당수는 **제공 데이터(융합데이터)에 이미 있다.**
따라서 '진짜 손실'은 융합데이터에도 없는 필드로 좁혀서 봐야 한다.
"""
import sqlite3, os, io, sys
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
log = lambda m: print(m, flush=True)

con = sqlite3.connect(f"file:{BASE}\\수집파이프라인\\registry.sqlite?mode=ro", uri=True)
log("[1] 소스별 현황")
tot = 0
for s, n in con.execute("SELECT source, COUNT(*) FROM registry WHERE status='OK' GROUP BY 1 ORDER BY 2 DESC"):
    log(f"  {s:12s} {n:8,}")
    tot += n
log(f"  {'합계':12s} {tot:8,}")

rich = sqlite3.connect(f"file:{BASE}\\수집파이프라인\\registry_rich.sqlite?mode=ro", uri=True)
nrich = rich.execute("SELECT COUNT(*) FROM patent").fetchone()[0]
log(f"\n  28필드 복구 완료 {nrich:,} / {tot:,} ({nrich/tot*100:.1f}%)")
log(f"  ★ 미복구 {tot-nrich:,}건 ({(tot-nrich)/tot*100:.1f}%) — 이들이 '못 받은' 대상")

# 미복구 목록
lost = pd.read_sql("SELECT rgst_no, ksic, appl_no, last_annl FROM registry "
                   "WHERE status='OK' AND source<>'datago'", con)
log(f"\n[2] 미복구 {len(lost):,}건이 현재 갖고 있는 것")
log("  · 최종 납부 연차(last_annl)  ✓")
log("  · 납부 이력(시점·금액)        ✓  ← payList 에 있으나 우리 DB 엔 최종연차만 저장됨")
log("  · 그 외 27개 필드            ✗")

# 융합데이터로 메울 수 있는가
kr = pd.read_parquet(os.path.join(BASE, "공모전데이터", "작업데이터", "kr_registered.parquet"),
                     columns=["출원번호", "출원일", "등록번호", "등록일자", "발명명칭",
                              "메인IPC", "출원인명", "패밀리출원번호"])
kr["등록번호"] = kr["등록번호"].astype(str).str.strip()
lost["rgst_no"] = lost.rgst_no.astype(str).str.strip()
m = lost.merge(kr, left_on="rgst_no", right_on="등록번호", how="left")
hit = m["출원일"].notna()
log(f"\n[3] 융합데이터로 메울 수 있는가 — 등록번호 매칭 {hit.sum():,}/{len(m):,} ({hit.mean()*100:.1f}%)")
log("  융합데이터에서 회수 가능:")
for c in ("출원일", "등록일자", "발명명칭", "메인IPC", "출원인명", "패밀리출원번호"):
    v = m[c].notna() & (m[c].astype(str).str.strip() != "")
    log(f"    {c:12s} {v.sum():7,} ({v.mean()*100:5.1f}%)")

log("\n[4] ★ 어디서도 회수 불가 — 진짜 손실")
log("  다음은 등록원부 주 오퍼레이션에만 있고 융합데이터에 없다:")
for f, why in (("claimCount 청구항 수", "특허 규모·가치의 표준 대리변수"),
               ("applicantCd 출원인 코드", "기업 매칭 정확도 (이름 매칭은 위양성 62%)"),
               ("applicantNatl 출원인 국적", "내·외국인 판별"),
               ("owner 권리자 이력", "★ 권리 이전 여부 — 특허 거래·기술이전 흔적"),
               ("inventor 발명자", "발명자 수·국적"),
               ("cndrtExptnDate 존속만료예정일", "조기포기 판정 (등록일+20년으로 근사는 가능)"),
               ("right 등록사건 이력", "설정등록·이전·소멸 사건")):
    log(f"    · {f:26s} — {why}")

log("\n[5] 그래서 실제로 얼마나 영향받는가")
ow = rich.execute("SELECT COUNT(*) FROM patent WHERE owner_changed=1").fetchone()[0]
log(f"  권리이전 18.2%({ow:,}건)는 복구된 {nrich:,}건 기준이다.")
log(f"  전체 {tot:,}건 중 {nrich/tot*100:.0f}% 만 관측되므로,")
log(f"  '권리이전' 분석은 표본이 {nrich:,}건으로 제한된다 (전수 아님).")
log(f"  청구항·출원인코드·발명자도 마찬가지.")

log("\n[6] 되찾으려면")
log(f"  미복구 {tot-nrich:,}건을 주 오퍼레이션으로 재수집 = {tot-nrich:,} 콜")
log(f"  10개 키 × 9,500 = 95,000 콜/일 → 약 {(tot-nrich)/95000:.1f}일")
log("  ※ 다만 주제가 청구항·권리이전을 쓰지 않으면 재수집할 이유가 없다.")
con.close(); rich.close()
