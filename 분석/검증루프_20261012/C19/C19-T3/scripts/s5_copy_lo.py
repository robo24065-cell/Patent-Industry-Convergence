# C19-T3 s5: label_origin_R10 계열(lo_fam, lo_tests, lo_posthoc2, R11 fp_prep, fp_tests) 사본 — 경로만 바꿈
#  쓰기: C19_OUT(= cache/lo_<V>), fam_reg는 C19_OUT/cache/fam_reg.parquet, R11 산출은 C19_OUT/fp/
#  읽기: 메타 = C19_META, 그 밖 캐시(kr_lo·pat_firm·firm_reg·alt_assign·rec_conc 등)는 원래 위치(읽기 전용)
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
LO = MC + r'\_shared_report\label_origin_R10'; FP = LO + r'\foreign_public_R11'
D = os.path.join(W, 'scripts', 'lo'); os.makedirs(D, exist_ok=True)
HDR = "import os as _os\n_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']\n_os.makedirs(_os.path.join(_C19_OUT, 'cache'), exist_ok=True); _os.makedirs(_os.path.join(_C19_OUT, 'fp', 'cache', '_tmp'), exist_ok=True)\n"
FAMNEW = 'os.path.join(_C19_OUT, "cache", "fam_reg.parquet")'
jobs = {
 'lo_fam.py': (LO, [
   ('OUT = os.path.join(MC, r"_shared_report\\label_origin_R10")', 'OUT = _C19_OUT'),
   ('META = os.path.join(MC, r"_shared\\meta\\*.parquet").replace("\\\\", "/")', 'META = _C19_META')]),
 'lo_tests.py': (LO, [
   ('OUT = os.path.join(MC, r"_shared_report\\label_origin_R10"); CACHE = os.path.join(OUT, "cache")',
    'OUT = _C19_OUT; CACHE = os.path.join(MC, r"_shared_report\\label_origin_R10", "cache")'),
   ('os.path.join(CACHE, "fam_reg.parquet")', FAMNEW)]),
 'lo_posthoc2.py': (LO, [
   ('OUT = os.path.join(MC, r"_shared_report\\label_origin_R10"); CACHE = os.path.join(OUT, "cache")',
    'OUT = _C19_OUT; CACHE = os.path.join(MC, r"_shared_report\\label_origin_R10", "cache")'),
   ('META = os.path.join(MC, r"_shared\\meta\\*.parquet").replace("\\\\", "/")', 'META = _C19_META'),
   ('os.path.join(CACHE, "fam_reg.parquet")', FAMNEW)]),
 'fp_prep.py': (FP, [
   ('HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)',
    'HERE = os.path.join(_C19_OUT, "fp"); CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)'),
   ('MC = os.path.abspath(os.path.join(HERE, r"..\\..\\.."))', 'MC = os.path.abspath(os.path.join(r"%s", r"..\\..\\.."))' % FP),
   ('FAM = f("_shared_report", "label_origin_R10", "cache", "fam_reg.parquet")', 'FAM = ' + FAMNEW + '.replace("\\\\", "/")')]),
 'fp_tests.py': (FP, [
   ('HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")', 'HERE = os.path.join(_C19_OUT, "fp"); CACHE = os.path.join(HERE, "cache")'),
   ('MC = os.path.abspath(os.path.join(HERE, r"..\\..\\.."))', 'MC = os.path.abspath(os.path.join(r"%s", r"..\\..\\.."))' % FP)]),
}
for fn, (src, reps) in jobs.items():
    s = open(os.path.join(src, fn), encoding='utf-8').read()
    for a, b in reps:
        assert s.count(a) >= 1, (fn, a, s.count(a))
        s = s.replace(a, b)
    i = s.find('\nimport ')
    s = s[:i + 1] + HDR + s[i + 1:]
    s = '# [C19-T3 사본] 원본: ' + os.path.join(src, fn) + ' — 경로만 바꿈(쓰기→C19_OUT, META→C19_META, fam_reg→C19_OUT/cache)\n' + s
    # 남은 원래 쓰기 경로 점검: OUT/HERE 외 절대 경로 쓰기가 없는지
    open(os.path.join(D, fn), 'w', encoding='utf-8').write(s)
    out('copied', fn)
