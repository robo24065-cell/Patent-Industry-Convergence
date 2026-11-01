# C18-T2 s3b: 원고 3.1 생성 스크립트(map5 m1·m1b·m2·m3c·m4, split34 s1·s2)를 scripts/map/ 에 복사하면서 경로만 바꾼다.
#  - 쓰기 위치(OUT·HERE) → 환경변수 C18_OUT(이 폴더 cache/map_<V>/)
#  - 메타 경로 → 환경변수 C18_META(P0 = 원본, V1·V1E = cache/meta_<V>.parquet)
#  - 그 밖의 입력(MC·BASE·EXT)은 원래 위치(읽기 전용)
# 사용: python s3b_map_copy.py  (복사만)
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
FA = MC + r'\_shared_report\fusion_advantage'
D = os.path.join(W, 'scripts', 'map'); os.makedirs(D, exist_ok=True)
ORIG_MAP5 = FA + r'\map5'; ORIG_S34 = FA + r'\split34'
HDR = "import os as _os\n_C18_OUT = _os.environ['C18_OUT']; _C18_META = _os.environ['C18_META']\n"
jobs = {
    'm1_build_counts.py': (ORIG_MAP5, [
        ('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = _C18_OUT'),
        ('META = os.path.join(OUT, r"..\\..\\..\\_shared\\meta\\*.parquet").replace("\\\\", "/")', 'META = _C18_META')]),
    'm1b_field_counts.py': (ORIG_MAP5, [
        ('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = _C18_OUT'),
        ('EXT = os.path.join(OUT, r"..\\..\\..\\_shared\\ext")', 'EXT = os.path.join(r"%s", r"..\\..\\..\\_shared\\ext")' % ORIG_MAP5),
        ('META = os.path.join(OUT, r"..\\..\\..\\_shared\\meta\\*.parquet").replace("\\\\", "/")', 'META = _C18_META')]),
    'm2_trend_map.py': (ORIG_MAP5, [
        ('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = _C18_OUT'),
        ('MC = os.path.abspath(os.path.join(OUT, r"..\\..\\.."))', 'MC = os.path.abspath(os.path.join(r"%s", r"..\\..\\.."))' % ORIG_MAP5)]),
    'm3c_contemporaneous.py': (ORIG_MAP5, [
        ('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = _C18_OUT')]),
    'm4_industry_quadrant.py': (ORIG_MAP5, [
        ('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = _C18_OUT'),
        ('MC = os.path.abspath(os.path.join(OUT, r"..\\..\\.."))', 'MC = os.path.abspath(os.path.join(r"%s", r"..\\..\\.."))' % ORIG_MAP5)]),
    'm5_dashboard.py': (ORIG_MAP5, [
        ('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = _C18_OUT'),
        ('MC = os.path.abspath(os.path.join(OUT, r"..\..\.."))', 'MC = os.path.abspath(os.path.join(r"%s", r"..\..\.."))' % ORIG_MAP5)]),
    's1_panel.py': (ORIG_S34, [
        ('HERE = os.path.dirname(os.path.abspath(__file__))', 'HERE = _C18_OUT'),
        ('MC = os.path.abspath(os.path.join(HERE, r"..\\..\\.."))', 'MC = os.path.abspath(os.path.join(r"%s", r"..\\..\\.."))' % ORIG_S34),
        ('META = os.path.join(MC, r"_shared\\meta\\*.parquet").replace("\\\\", "/")', 'META = _C18_META')]),
    's2_decomp.py': (ORIG_S34, [
        ('HERE = os.path.dirname(os.path.abspath(__file__))', 'HERE = _C18_OUT'),
        ('MC = os.path.abspath(os.path.join(HERE, r"..\\..\\.."))', 'MC = os.path.abspath(os.path.join(r"%s", r"..\\..\\.."))' % ORIG_S34)]),
}
for fn, (src, reps) in jobs.items():
    s = open(os.path.join(src, fn), encoding='utf-8').read()
    for a, b in reps:
        assert s.count(a) == 1, (fn, a, s.count(a))
        s = s.replace(a, b)
    # 헤더는 첫 import 앞에(독스트링 뒤) 넣는다
    i = s.find('\nimport ')
    s = s[:i + 1] + HDR + s[i + 1:]
    s = '# [C18-T2 사본] 원본: ' + os.path.join(src, fn) + ' — 경로만 바꿈(OUT/HERE→C18_OUT, META→C18_META)\n' + s
    open(os.path.join(D, fn), 'w', encoding='utf-8').write(s)
    out('copied', fn)
