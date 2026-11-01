# C19-T3 s4: 원고 3.1 계열 생성 스크립트(map5 m1·m1b·m2·m3·m3c·m4·m5·m7, split34 s1–s5) 전부를 scripts/map/ 에 복사하면서 경로만 바꾼다.
#  - 쓰기 위치(OUT·HERE) → 환경변수 C19_OUT
#  - 메타 경로 → 환경변수 C19_META (P0 = 원본, V1E = C18-T2 cache/meta_V1E.parquet; s0에서 검증)
#  - MC·BASE·EXT 등 그 밖의 입력은 원래 위치(읽기 전용)
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
FA = MC + r'\_shared_report\fusion_advantage'
D = os.path.join(W, 'scripts', 'map'); os.makedirs(D, exist_ok=True)
HDR = "import os as _os\n_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']\n"
JOBS = [('map5', s) for s in ['m1_build_counts', 'm1b_field_counts', 'm2_trend_map', 'm3_leadlag', 'm3c_contemporaneous', 'm4_industry_quadrant', 'm5_dashboard', 'm7_supplement']] + \
       [('split34', s) for s in ['s1_panel', 's2_decomp', 's3_forecast', 's4_figs_robust', 's5_cases']]
log = []
for sub, fn in JOBS:
    src = os.path.join(FA, sub, fn + '.py'); od = os.path.join(FA, sub)
    s = open(src, encoding='utf-8').read(); n0 = len(s)
    reps = 0
    # 1) MC/EXT/META 상대 경로(OUT·HERE 기준)를 원래 폴더 기준으로
    for var in ('OUT', 'HERE'):
        pat = f'os.path.join({var}, r"..\\..\\..'
        if pat in s:
            k = s.count(pat); s = s.replace(pat, f'os.path.join(r"{od}", r"..\\..\\..'); reps += k
    # 2) META 정의 줄 → _C19_META
    lines = s.split('\n'); nm = 0
    for i, L in enumerate(lines):
        if re.match(r'^META\s*=', L):
            lines[i] = 'META = _C19_META'; nm += 1
    s = '\n'.join(lines)
    # 3) OUT/HERE = dirname(__file__) → _C19_OUT
    k = len(re.findall(r'^(OUT|HERE) = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)', s, flags=re.M))
    s = re.sub(r'^(OUT|HERE) = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)', r'\1 = _C19_OUT', s, flags=re.M)
    assert k == 1, (fn, 'OUT/HERE', k)
    i = s.find('\nimport ')
    s = s[:i + 1] + HDR + s[i + 1:]
    s = '# [C19-T3 사본] 원본: ' + src + ' — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)\n' + s
    open(os.path.join(D, fn + '.py'), 'w', encoding='utf-8').write(s)
    log.append((fn, 'rel_reps', reps, 'meta_lines', nm))
    out(fn, reps, nm)
jdump(log, 's4_copy_map.json')
