# -*- coding: utf-8 -*-
# p2_within_group.py 복사본을 만들어 출력 경로·변형 인자만 바꾼다(원본은 읽기만).
import os
HS = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(HS, "p2_within_group_orig_copy.py"), encoding="utf-8").read()
head = '''# C07-T1 복사본: _ideas_R8\\p2_within_group.py (원본 읽기만). 바뀐 점: (1) 출력은 C07-T1\\cache\\p2_<VAR>_*, (2) VAR 인자로 원점·코드 집합만 바꿈.
#   B0 원판 / V1 결과 창이 2016을 넘지 않는 원점만(2009·2011·2012·2013) / V2a KSIC 9→10 비-1:1 코드 제외
import sys, os
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
VAR = sys.argv[1] if len(sys.argv) > 1 else "B0"
sys.path.insert(0, r"C:\\Users\\PC\\Downloads\\특허-산업 융합데이터 활용 아이디어 공모전\\분석\\모델경쟁_20260927\\_ideas_R8")
from r8_common import *
_H = r"C:\\Users\\PC\\Downloads\\특허-산업 융합데이터 활용 아이디어 공모전\\분석\\검증루프_20261012\\C07\\C07-T1"
def pout(n):
    return os.path.join(_H, "cache", f"p2_{VAR}_" + n)
if VAR == "V2a":
    import pandas as _pd
    _m = _pd.read_csv(os.path.join(_H, "ksic9_10_map_check.csv"), dtype={"ksic10": str})
    _non = set(_m[~_m.type_9to10.isin(["1:1", "1:1(코드 변경)"])].ksic10)
    KSIC50 = [k for k in KSIC50 if k not in _non]
    _gc = _pd.Series([K2G[k] for k in KSIC50]).value_counts()
    MULTI_K = sorted([k for k in KSIC50 if _gc[K2G[k]] > 1])
    print("V2a units", len(MULTI_K))
'''
old = "import sys\nsys.dont_write_bytecode = True\nfrom r8_common import *\n"
assert old in s
s = s.replace(old, head)
o2 = "ORIG = [2009, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]"
assert o2 in s
s = s.replace(o2, o2 + "\nif VAR == 'V1':\n    ORIG = [2009, 2011, 2012, 2013]")
for a, b in [("p('cache', 'rec.parquet')", "os.path.join(CACHE, 'rec.parquet')"), ("p('cache', 'kr_assign.parquet')", "os.path.join(CACHE, 'kr_assign.parquet')"),
             ('D.to_csv(p("p2_panel.csv")', 'D.to_csv(pout("panel.csv")'), ('PER.to_csv(p("p2_by_origin.csv")', 'PER.to_csv(pout("by_origin.csv")'),
             ('open(p("p2_results.json"), "w"', 'open(pout("results.json"), "w"')]:
    assert a in s, a
    s = s.replace(a, b)
assert 'p("' not in s and "p('" not in s
open(os.path.join(HS, "s4_p2_sens.py"), "w", encoding="utf-8").write(s)
print("ok")
