# -*- coding: utf-8 -*-
"""
격차의 정체 규명 — '융합데이터가 놓쳤다'는 표현이 맞는가
────────────────────────────────────────────────────────────────
관측: 삼성전자 융합데이터 36,490 vs getBusinessRightList 158,236 (4.3배)

이 격차가 무엇인지 후보가 여럿이다. 라벨을 붙이기 전에 확인한다.
  [H1] KSIC 50개 코드 밖의 특허      → '절단' 이 맞다
  [H2] 특허가 아닌 다른 권리(디자인·상표·실용신안)  → 애초에 다른 종류
  [H3] 5개국 중 한국만 뽑아서 생긴 차이 → 사용자 지적. 다만 이 API 도 한국 등록원부다
  [H4] 등록/소멸 상태 차이            → 우리는 등록특허만 봤다
  [H5] 권리자로서 보유(양수) vs 출원인  → pertDiv 필드가 '권리자란' 이었다
"""
import json, os, sqlite3, sys, io
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
B = os.path.dirname(H)
log = lambda m: print(m, flush=True)

bz = sqlite3.connect(f"file:{os.path.join(H,'business_rights.sqlite')}?mode=ro", uri=True)
rows = bz.execute("SELECT applicant_cd, rgst_no, raw FROM right_list").fetchall()
log(f"[표본] 시험 수집한 권리 {len(rows):,}행 (상위 20개사, 각 300건)")

# ── 권리 종류 판별: 등록번호 앞자리
kind = {"1": "특허", "2": "실용신안", "3": "디자인", "4": "상표"}
cnt = Counter(kind.get(str(r[1])[:1], f"기타({str(r[1])[:1]})") for r in rows)
log(f"\n[H2] 권리 종류 — 등록번호 앞자리")
for k, v in cnt.most_common():
    log(f"  {k:10s} {v:6,} ({v/len(rows)*100:5.1f}%)")

# ── raw 안의 필드로 추가 확인
sample = json.loads(rows[0][2])
log(f"\n  raw 필드: {list(sample)}")
pd = Counter(json.loads(r[2]).get("pertDiv") for r in rows)
log(f"  pertDiv 분포: {dict(pd)}")
ic = Counter(json.loads(r[2]).get("indvCorpTpcd") for r in rows)
log(f"  indvCorpTpcd 분포: {dict(ic)}")

# ── 우리 융합데이터(=KR 등록특허 65.6만) 안에 있는가
c = sqlite3.connect(f"file:{os.path.join(H,'registry.sqlite')}?mode=ro", uri=True)
ours = {x[0] for x in c.execute("SELECT rgst_no FROM pending")}
c.close()
got = [str(r[1]).strip() for r in rows]
inside = sum(1 for g in got if g in ours)
log(f"\n[H1] 융합데이터(KR 등록특허 50개 KSIC) 안에 있는가")
log(f"  {inside:,}/{len(got):,} ({inside/len(got)*100:.1f}%) 가 안쪽")
log(f"  {len(got)-inside:,} 건이 **밖**")

# ── 밖의 것들은 무엇인가: 특허만 골라서 다시
pat_out = [g for g in got if g.startswith("1") and g not in ours]
log(f"\n  그중 '특허(1…)'이면서 밖인 것: {len(pat_out):,}건")
if pat_out:
    log(f"  예시 등록번호: {pat_out[:5]}")

# ── 융합데이터 원본에서 해당 출원인의 KR 건수와 5개국 전체 건수 비교
log(f"\n[H3] 5개국 중 한국만 뽑아서 생긴 차이인가")
log("  getBusinessRightList 는 **한국 등록원부** API 다.")
log("  → 반환값도 한국 권리뿐이므로, 이 격차는 국가 필터 때문이 아니다.")
log("  다만 융합데이터 쪽 비교 기준을 확인한다:")
try:
    import pandas as pd
    kr = pd.read_parquet(os.path.join(B, "공모전데이터", "작업데이터", "kr_registered.parquet"),
                         columns=["출원인명", "등록번호"])
    log(f"  kr_registered.parquet = KR 등록특허 {len(kr):,}건 (50개 KSIC)")
    log("  → 우리가 쓴 '융합데이터 내 보유'는 이 안에서 센 값이다.")
except Exception as e:
    log(f"  (parquet 확인 실패: {e})")

log(f"\n[결론 재료]")
log(f"  · 권리 종류 구성: {dict(cnt)}")
log(f"  · 융합데이터 밖 비율: {(len(got)-inside)/len(got)*100:.1f}%")
bz.close()
