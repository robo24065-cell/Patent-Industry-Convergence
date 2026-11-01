# C10-T4 s17: V11 = V9 + 제공 데이터 행 국가 나열을 '5개국'으로(줄 상쇄 시험)
import os
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
s = open(os.path.join(T, "ms_V9.md"), encoding="utf-8").read()
a = "특허-산업 융합데이터(KSIC 50개, KR·US·JP·CN·EP), 지식재산처"
assert s.count(a) == 1
open(os.path.join(T, "ms_V11.md"), "w", encoding="utf-8").write(s.replace(a, "특허-산업 융합데이터(KSIC 50개, 5개국), 지식재산처"))
print("ok")
