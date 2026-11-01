# C18-T2 s2: 공식 9트랙·원고 3.1 생성 코드에서 해외(US·EP) 메타 행 사용처 grep → exposure.csv, cache/s2_grep_hits.csv
# 분류(사용 유형·노출 판단)는 grep 결과를 읽고 사람이 적은 표(아래 ROWS)이며, 근거 행은 스크립트가 다시 찾아 붙인다.
import os, sys, re, glob, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
PAT = {
    'meta_read': r"_shared[/\\]+meta|META\b|'\{M\}'|read_parquet\('\{meta\}'\)",
    'us_rows': r"cc\s*=\s*'US'|cc\s*==\s*['\"]US['\"]|in \('US'|\"US\"|'US'",
    'ep_rows': r"cc\s*=\s*'EP'|in \('US','EP'|\"EP\"|'EP'",
    'rgst': r"rgst_no|rgst_date|least\(nullif\(pub_date",
    'family': r"family|fam_",
    'count': r"count\(distinct appl_no\)|count\(\*\)|groupby\(.*\)\.size\(\)|nunique",
}
TRACK_FILES = {
    'T1': ['T1_산업대기실/R3/final_run_r3.py', 'T1_산업대기실/final_run.py', 'T1_산업대기실/R2/r2ext.py', 'T1_산업대기실/R3/r3ext2.py', 'T1_산업대기실/R3/build_ipcx.py'],
    'T3': ['T3_기술셀부상예측/R3/build_base_pub.py', 'T3_기술셀부상예측/build_panel.py', 'T3_기술셀부상예측/build_feat2.py', 'T3_기술셀부상예측/R3/r3lib.py',
           'T3_기술셀부상예측/R3/build_ros_pubrule_r3d.py', '_shared/ext/r3ros_02_build.py'],
    'T4': ['T4_특허포기예측/R2/final_run_r2.py', 'T4_특허포기예측/R2/r2lib.py', 'T4_특허포기예측/s2/extract_family_dated.py', 'T4_특허포기예측/extract_family.py', 'T4_특허포기예측/R2/build_meta_cells.py'],
    'N4': ['N4_특허행태_기업폐업조기경보/n4lib.py', 'N4_특허행태_기업폐업조기경보/n4feat.py'],
    'N5': ['N5_표준특허선언레이더/n5lib.py'],
    'N6': ['N6_정부RnD기업_첫특허전환/n6_lib.py', 'N6_정부RnD기업_첫특허전환/n6_features.py'],
    'N7': ['N7_특허기업_정부RnD첫수행/n7lib.py', 'N7_특허기업_정부RnD첫수행/n7feat.py'],
    'N8': ['N8_특허분쟁표적레이더/n8lib.py'],
    'N9': ['N9_정부RnD과제_특허성과예측/n9_lib.py', 'N9_정부RnD과제_특허성과예측/n9_build.py'],
    '원고3.1': ['_shared_report/fusion_advantage/map5/m1_build_counts.py', '_shared_report/fusion_advantage/map5/m2_trend_map.py',
              '_shared_report/fusion_advantage/map5/m3c_contemporaneous.py', '_shared_report/fusion_advantage/map5/m4_industry_quadrant.py',
              '_shared_report/fusion_advantage/map5/m7_supplement.py'],
    '표2·표3': ['_shared_report/label_origin_R10/lo_fam.py', '_shared_report/fusion_advantage/c5_R10/c1_assign.py'],
}
hits = []
for tr, fs in TRACK_FILES.items():
    for f in fs:
        p = os.path.join(MC, f)
        if not os.path.exists(p):
            hits.append((tr, f, 0, 'MISSING', '')); continue
        for i, line in enumerate(open(p, encoding='utf-8', errors='replace'), 1):
            for k, rg in PAT.items():
                if re.search(rg, line):
                    hits.append((tr, f, i, k, line.strip()[:220]))
with open(os.path.join(C, 's2_grep_hits.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.writer(fh); w.writerow(['track', 'file', 'line', 'pattern', 'text']); w.writerows(hits)
out('grep hits', len(hits))
for tr in TRACK_FILES:
    sub = [h for h in hits if h[0] == tr]
    out(tr, {k: sum(1 for h in sub if h[3] == k) for k in list(PAT) + ['MISSING']})
