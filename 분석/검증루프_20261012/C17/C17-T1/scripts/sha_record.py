# -*- coding: utf-8 -*-
"""입력 파일 sha256·mtime 기록(before/after). python sha_record.py before|after → copy_sha256.csv(열 추가)"""
import sys, pathlib, hashlib, os; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
tag = sys.argv[1]
files = [T1/'lib.py', T1/'prep1.py', T1/'prep2.py', T1/'prep_app.py', T1/'final_run.py', T1/'s2lib.py', T1/'DATA_NOTES.md', R3/'final_run_r3.py', R3/'r3.py', R3/'r3ext.py', R3/'r3ext2.py', R3/'r3ext3.py', T1/'R2'/'r2.py', T1/'R2'/'r2ext.py',
         R3/'final_result_r3.json', RD/'o2019.parquet', FC/'prep_te.parquet', FC/'baseline_scores.parquet', FC/'feat_c2019.parquet', FC/'base_app_c2019.parquet',
         CACHE/'ins_firm.parquet', CACHE/'ins_applicant.parquet', CACHE/'ins_pat.parquet', CACHE/'oos.parquet', CACHE/'app_pub.parquet', CACHE/'firms.parquet', CACHE/'firm_meta.parquet', CACHE/'rl_pat.parquet',
         KR, KP, CA, RGF, BRS, PIPE/'collect_biz.py',
         B/'분석/보고서_20261012/manuscript_v3.md', B/'분석/보고서_20261012/facts/facts.csv', B/'분석/보고서_20261012/facts/facts_supplement.csv',
         C16C/'labels.parquet', C16C/'p0'/'sbb_all.npy', C16C/'build'/'pop.pkl', C16C/'build'/'s1F'/'scores.parquet', C16C/'p0'/'s1_te.parquet',
         C16C/'models'/'xgb_seed0.json', C16C/'models'/'xgb_seed1.json', C16C/'models'/'logit.pkl', C16C/'models'/'firm_seed0.json', C16C/'models'/'sbb.json',
         C16C/'models'/'base_xgb_y.json', C16C/'models'/'base_xgb_y_u.json'] + [C16C/'models'/f'base_logit_{a}_{b}.pkl' for a in ('stack', 'single') for b in ('y', 'y_u')]


def h(p):
    s = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b''): s.update(ch)
    return s.hexdigest()


rows = [dict(file=str(p), **{f'sha256_{tag}': h(p), f'mtime_{tag}': os.path.getmtime(p)}) for p in files]
D = pd.DataFrame(rows); fo = HERE/'copy_sha256.csv'
if fo.exists() and tag != 'before':
    O = pd.read_csv(fo); D = O.merge(D, on='file', how='outer')
    D['same'] = (D.sha256_before == D[f'sha256_{tag}']) & (D.mtime_before == D[f'mtime_{tag}'])
    print('all same:', bool(D.same.all()), int(D.same.sum()), '/', len(D))
D.to_csv(fo, index=False, encoding='utf-8-sig'); print(len(D), 'files', G.rss_peak())
