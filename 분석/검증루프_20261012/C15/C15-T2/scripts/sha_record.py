# -*- coding: utf-8 -*-
"""읽기 전용으로 import·참조한 트랙 코드·입력의 sha256·수정 시각 기록, 사본 대조 → copy_sha256.csv (인자 tag: before/after)"""
import sys, os, hashlib, time, csv, pathlib
HERE = pathlib.Path(__file__).resolve().parent.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석")
MC = B/'모델경쟁_20260927'; T1 = MC/'T1_산업대기실'; T3 = MC/'T3_기술셀부상예측'; LP = B/'검증루프_20261012'
files = [T1/'R3'/x for x in ('final_run_r3.py', 'r3.py', 'r3ext.py', 'r3ext2.py', 'r3ext3.py', 'final_spec_r3.json', 'final_result_r3.json', 'dev_result_r3.json')] + \
        [T1/'R2'/'r2.py', T1/'R2'/'r2ext.py', T1/'final_run.py', T1/'lib.py', T1/'s2lib.py', T1/'VERIFY_R5'/'sens_t1.py', T1/'VERIFY_R5'/'ci_sb_t1.py', T1/'VERIFY_R5'/'ci_sb_t1.json',
         T1/'R3'/'runs'/'FROZEN_final'/'o2019.parquet', T1/'R3'/'cache_final'/'o2019'/'baseline_scores.parquet', T1/'R3'/'cache_final'/'o2019'/'prep_tr.parquet', T1/'R3'/'cache_final'/'o2019'/'prep_te.parquet'] + \
        [T3/'R3'/x for x in ('final_run_r3.py', 'r3core.py', 'r3lib.py', 'final_spec_r3.json', 'final_result_r3.json')] + [T3/'VERIFY_R5'/'sens_t3.py', T3/'VERIFY_R5'/'ci_sb_t3.py', T3/'VERIFY_R5'/'ci_sb_t3.json']
pairs = [(LP/'C11'/'C11-T3'/'scripts'/'t3run'/x, HERE/'scripts'/'t3run'/x) for x in ('c09core.py', 'c09lib.py', 'run_ros.py')] + [(LP/'C10'/'C10-T1'/'scripts'/'guard.py', None)]
sh = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
tag = sys.argv[1] if len(sys.argv) > 1 else 'before'
out = HERE/'copy_sha256.csv'; new = not out.exists()
with open(out, 'a', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    if new: w.writerow(['tag', 'kind', 'path', 'sha256', 'mtime', 'copy_path', 'copy_sha256', 'equal'])
    for p in files:
        w.writerow([tag, 'track_readonly', str(p), sh(p), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p))), '', '', ''])
    for a, c in pairs:
        w.writerow([tag, 'copy', str(a), sh(a), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(a))), str(c or ''), sh(c) if c else '', (sh(a) == sh(c)) if c else ''])
print('ok', tag)
