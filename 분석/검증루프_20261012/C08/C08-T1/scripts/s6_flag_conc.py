# -*- coding: utf-8 -*-
# concentration_by_cell.csv에 자료 품질 표시 열 추가(s2 산출 + s4 점검 결과). 같은 파일을 덮어쓴다(s2 재실행 시 이 스크립트도 다시 실행).
import os, sys, pandas as pd
sys.dont_write_bytecode = True
WD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f = os.path.join(WD, "concentration_by_cell.csv")
C = pd.read_csv(f, dtype={"ksic": str})
C = C.drop(columns=[c for c in ["quality_flag"] if c in C.columns])
def flag(r):
    if r.cc == "US" and r.period in ("E0507", "W0809", "P1"):
        return "US 출원인 필드=발명자(2012 AIA 이전, 법인형 이름 0.2%) → 법인 집중도 아님"
    if r.cc == "CN" and r.period in ("E0507", "W0809", "P1"):
        return "CN 출원인 결측 다수(2005 84%, 2008 54%) → 결측 제외 집중도"
    if r.cc == "KR":
        return "KR 출원인 1명만 수록(공동출원 5.3%에서 선택 규칙 불일정)"
    return ""
C["quality_flag"] = C.apply(flag, axis=1)
C.to_csv(f, index=False, encoding="utf-8-sig")
print(C.quality_flag.value_counts())
