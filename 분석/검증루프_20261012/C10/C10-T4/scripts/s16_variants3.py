# C10-T4 s16: V9 = V7 + 제공 데이터 행 '(KISTA)' 삭제 + Chen 참고문헌 학회명 약칭(줄 상쇄 시험)
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
src = open(os.path.join(T, "ms_V7.md"), encoding="utf-8").read()
E0 = "지식재산처·한국특허전략개발원(KISTA). 공유 금지"
E1 = "지식재산처·한국특허전략개발원. 공유 금지"
C0 = "*Proceedings of the 22nd ACM SIGKDD*, 785–794."
C1 = "*KDD 2016*, 785–794."
for s in (E0, C0):
    assert src.count(s) == 1, s
open(os.path.join(T, "ms_V9.md"), "w", encoding="utf-8").write(src.replace(E0, E1).replace(C0, C1))
open(os.path.join(T, "ms_V10.md"), "w", encoding="utf-8").write(src.replace(E0, E1))
print("ok")
