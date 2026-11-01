# -*- coding: utf-8 -*-
"""입력 sha256 기록. 사용: python s0_sha.py before|after"""
import os, sys, glob
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
J = os.path.join
A = J(AN, '검증루프_20261012')
fs = [J(T4, f) for f in ['final_spec.json', 'final_model.py', 'final_run.py', 'eval_final.py', 'extract_family.py', 'feat2.py', 'holdout_result.json', 'HOLDOUT_REPORT.md',
                         J('s2', 'extract_family_dated.py'), J('cache', 's2_fam_tokens.parquet'), J('cache', 's2_fam_pid.parquet'), J('cache', 'patent.parquet'),
                         J('R2', 'final_spec_r2.json'), J('R2', 'final_result.json'), J('R2', 'final_run_r2.py'), J('R2', 'r2lib.py'), J('R2', 'r2boot.py'),
                         J('VERIFY_R5', 'cache', 'strong_full.npy'), J('VERIFY_R5', 'v3_analysis.py')]]
fs += sorted(glob.glob(J(T4, 'preds', '*'))) + sorted(glob.glob(J(F22, '*.*'))) + sorted(glob.glob(J(F22, 'pred', '*.npy')))
fs += sorted(glob.glob(J(MIRROR, '*.*'))) + sorted(glob.glob(J(MIRROR, 'R2', '*.*')))
fs += [J(A, 'C13', 'C13-T2', 'scripts', 't4_job.py'), J(A, 'C13', 'C13-T2', 'scripts', 't4_score.py'), J(A, 'C15', 'C15-T1', 'scripts', 's3a_t4prep.py'),
       J(A, 'C15', 'C15-T1', 'scripts', 's3b_boot.py'), J(A, 'C18', 'C18-T2', 'scripts', 's3f_t4_famn.py'), J(A, 'C18', 'C18-T2', 'cache', 's3f_t4_famn.json'),
       J(A, 'C18', 'C18-T2', 'cache', 's1e_family_token_examples.csv'), J(A, 'C18', 'C18-T2', 'exposure.csv'),
       J(AN, '보고서_20261012', 'manuscript_v3.md'), J(AN, '보고서_20261012', 'facts', 'facts.csv'), J(AN, '보고서_20261012', 'facts', 'facts_supplement.csv')]
fs += sorted(glob.glob(J(MC, '_shared', 'meta', '*.parquet')))
with open(J(C, f'input_sha256_{sys.argv[1]}.txt'), 'w', encoding='utf-8') as f:
    for p in fs:
        if os.path.isfile(p):
            f.write(f'{sha256(p)}  {os.path.getsize(p)}  {os.path.relpath(p, BASE)}\n')
        else:
            f.write(f'MISSING  0  {os.path.relpath(p, BASE)}\n')
out('files', len(fs))
