# -*- coding: utf-8 -*-
"""사업자 수집 속도 병목 진단"""
import os, sqlite3, sys, io, time, json
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
sys.path.insert(0, H)
from collect_registry import DATAGO_KEYS
log = lambda m: print(m, flush=True)

c = sqlite3.connect(f"file:{os.path.join(H,'business_rights.sqlite')}?mode=ro", uri=True)
r = c.execute("SELECT COUNT(*), SUM(pages), SUM(fetched), SUM(total_count), SUM(truncated) "
              "FROM applicant_query WHERE status='OK'").fetchone()
n, pages, fetched, total, trunc = r
log(f"[완료 {n:,}개 출원인]")
log(f"  받은 페이지 {pages:,} = API 콜 수")
log(f"  받은 행 {fetched:,} | 실제 보유 합계 {total:,}")
log(f"  평균 {pages/n:.1f}페이지/출원인  → 1개 처리에 콜 {pages/n:.1f}회")
log(f"  잘림 {trunc:,}개 (max-pages 상한 도달)")

log("\n[병목 계산]")
log(f"  현재 설정: 10워커 × (출원인간 0.7초 + 페이지간 0.2초)")
avg_pg = pages / n
per_item = avg_pg * (0.3 + 0.2) + 0.7      # 네트워크 0.3s 가정
log(f"  출원인 1개 예상 소요 = {avg_pg:.0f}p × (통신 0.3 + 대기 0.2) + 0.7 = {per_item:.1f}초")
log(f"  10워커 이론 처리율 = {10/per_item:.2f} 건/초  (관측 0.9)")
log("  → 워커는 정상. 병목은 **출원인당 페이지 수**다.")

log("\n[남은 작업량 추정]")
f = sqlite3.connect(f"file:{os.path.join(H,'registry_full.sqlite')}?mode=ro", uri=True)
rows = f.execute("""SELECT applicantCd, COUNT(*) c FROM applicant
                    WHERE applicantCd IS NOT NULL AND applicantCd<>''
                    GROUP BY 1 ORDER BY c DESC""").fetchall()
f.close()
done = {x[0] for x in c.execute("SELECT applicant_cd FROM applicant_query WHERE status='OK'")}
left = [x for x in rows if x[0] not in done]
log(f"  남은 출원인 {len(left):,}개")
# 완료분에서 (융합내 건수 → 실제 페이지수) 관계 추정
pairs = c.execute("SELECT n_in_fusion, pages FROM applicant_query WHERE status='OK' "
                  "AND n_in_fusion>0").fetchall()
c.close()
if pairs:
    import statistics as st
    ratio = st.median(p / max(nf, 1) for nf, p in pairs)
    log(f"  완료분 중앙값: 융합 내 1건당 {ratio:.2f}페이지")
    est = sum(max(1, round(cnt * ratio)) for _, cnt in left)
    log(f"  남은 예상 콜 {est:,}회")
    log(f"  일일 한도 10키 × 9,500 = 95,000콜 → 약 {est/95000:.1f}일")

log("\n[속도 실측 — 0.7초 간격이 정말 필요한가]")
K = DATAGO_KEYS[0]
U = "http://apis.data.go.kr/1430000/PttRgstRtInfoInqSvc/getBusinessRightList"
for gap in (0.0, 0.1, 0.2):
    ok = err = 0
    t0 = time.time()
    for i in range(12):
        u = f"{U}?serviceKey={K}&pageNo={i+1}&numOfRows=100&type=json&searchType=2&searchVal=119980045698"
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "x"}),
                                       timeout=25).read().decode("utf-8", "replace")
            ok += 1 if '"resultCode":"000"' in b else 0
            err += 0 if '"resultCode":"000"' in b else 1
        except Exception:
            err += 1
        time.sleep(gap)
    log(f"  간격 {gap}s: 성공 {ok}/12 실패 {err} | {12/(time.time()-t0):.1f} 콜/초")
