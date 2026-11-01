# -*- coding: utf-8 -*-
# verify_0 독립 절차 2: 표본 행마다 사전에 정한 키워드로 원고 본문(1–446행)을 기계 검색해 공개 여부를 다시 판정한다.
# 규칙: (트랙 키워드 AND 요소 키워드)가 같은 줄에 있으면 '공개 후보', 요소 키워드만 있고 트랙 키워드가 없으면 '부분 후보', 없으면 '미공개'.
import os, sys, json
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(H, r"..\..\..\보고서_20261012\manuscript_v3.md")
L = open(MS, encoding="utf-8").read().splitlines()[:446]
K = {  # 요소ID: (트랙 키워드들, 요소 키워드들)
 "T1-03": (["T1", "대기실", "위험쌍"], ["사후", "이후 등록", "조건부"]),
 "T1-05": (["T1", "대기실"], ["조회 실패", "RC:001", "휴면"]),
 "T1-06": (["T1", "대기실", "진입"], ["2025-08까지", "등록번호", "−0.002"]),
 "T1-10": (["T1", "물가"], ["현행 빈티지", "빈티지"]),
 "T3-09": (["T3", "기술셀"], ["출원인명", "출원인 이름", "출원인 피처", "공동출원"]),
 "T4-06": (["T4", "포기"], ["생산지수", "출하", "재고지수"]),
 "T4-09": (["T4"], ["원점 뒤 공개분", "시간여행", "기준일 이전"]),
 "N4-04": (["N4", "폐업"], ["지연", "소급", "1회 스냅샷", "스냅샷"]),
 "N4-07": (["N4", "폐업"], ["NTIS", "과제"]),
 "N4-10": (["N4", "폐업"], ["KSIC 라벨", "현행 라벨", "라벨 빈티지"]),
 "N5-09": (["N5", "표준특허"], ["반영", "선언률"]),
 "N6-02": (["N6", "첫 특허"], ["연결", "구조적", "출원인코드"]),
 "N6-05": (["N6", "첫 특허"], ["스냅샷", "NTIS"]),
 "N6-06": (["N6", "첫 특허"], ["국민연금", "NPS"]),
 "N6-07": (["N6"], ["제목 사전", "특허 제목"]),
 "N7-03": (["N7", "정부 R&D에 처음"], ["보안과제", "관측되지"]),
 "N8-01": (["N8", "분쟁"], ["소멸", "존속", "살아 있"]),
 "N8-03": (["N8", "분쟁"], ["예고등록", "무효심판"]),
}
out = {}
for k, (tk, ek) in K.items():
    both = [i + 1 for i, s in enumerate(L) if any(t in s for t in tk) and any(e in s for e in ek)]
    only = [i + 1 for i, s in enumerate(L) if any(e in s for e in ek)]
    out[k] = dict(both=both[:8], only_n=len(only), verdict=("공개 후보" if both else ("부분 후보" if only else "미공개")))
json.dump(out, open(os.path.join(H, "verify_0", "v0_ms_grep.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sys.stdout.reconfigure(encoding="utf-8")
for k, v in out.items(): print(k, v)
