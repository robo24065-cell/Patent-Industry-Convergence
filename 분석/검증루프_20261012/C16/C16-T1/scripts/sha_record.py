# -*- coding: utf-8 -*-
"""읽기 전용으로 참조한 트랙·원천·원고 파일의 sha256·수정 시각 기록 → copy_sha256.csv (인자 tag: before/after)"""
import sys, os, hashlib, time, csv, pathlib
HERE = pathlib.Path(__file__).resolve().parent.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
A = B/'분석'; MC = A/'모델경쟁_20260927'; T1 = MC/'T1_산업대기실'; LP = A/'검증루프_20261012'
FC = T1/'R3'/'cache_final'/'o2019'
files = [T1/'R3'/x for x in ('FINAL_REPORT_R3.md', 'final_run_r3.py', 'r3.py', 'r3ext.py', 'r3ext2.py', 'r3ext3.py', 'final_spec_r3.json', 'final_result_r3.json', 'audit_r3.py', 'verify_final_r3.py')] + \
        [T1/'HOLDOUT_REPORT.md', T1/'DATA_NOTES.md', T1/'R2'/'R2_NOTES.md', T1/'R2'/'r2.py', T1/'R2'/'r2ext.py', T1/'final_run.py', T1/'lib.py', T1/'common.py', T1/'feats.py', T1/'s2lib.py', T1/'prep_app.py',
         T1/'VERIFY_R5'/'sens_t1_scores.parquet', T1/'R3'/'runs'/'FROZEN_final'/'o2019.parquet', T1/'R3'/'runs'/'FROZEN_final'/'o2019_px0.npy', T1/'R3'/'runs'/'FROZEN_final'/'o2019_px1.npy', T1/'R3'/'runs'/'FROZEN_final'/'o2019_pl.npy'] + \
        [FC/x for x in ('baseline_scores.parquet', 'prep_tr.parquet', 'prep_te.parquet', 'feat_c2019.parquet', 'base_app_c2019.parquet', 'P_base.npy', 'P_os51.npy', 'PO_os51.npy', 'P_ins_cf.npy', 'ep_firm.parquet', 'r3boot_R200_0.npy')] + \
        [T1/'cache'/x for x in ('ins_firm.parquet', 'oos.parquet', 'app_pub.parquet', 'firm_meta.parquet', 'firms.parquet', 'ins_applicant.parquet', 'ins_pat.parquet')] + \
        [B/'공모전데이터'/'작업데이터'/'kr_registered.parquet', B/'공모전데이터'/'작업데이터'/'kr_patents.parquet', B/'공모전데이터'/'외부데이터'/'DART'/'corp_applicant.parquet',
         B/'수집파이프라인'/'registry_full.sqlite', B/'수집파이프라인'/'business_rights.sqlite',
         A/'보고서_20261012'/'manuscript_v3.md', A/'보고서_20261012'/'facts'/'facts.csv',
         LP/'C15'/'C15-T2'/'cache'/'t1'/'A'/'o2019'/'trpre.parquet', LP/'C15'/'C15-T2'/'cache'/'t1'/'A'/'o2019'/'te_ext.parquet', LP/'C15'/'C15-T2'/'cache'/'t1'/'A'/'o2019'/'ftab.pkl',
         LP/'C15'/'C15-T2'/'cache'/'t1'/'final_labels.parquet', LP/'C15'/'C15-T2'/'cache'/'t1'/'btr_labels.npz']


def sh(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''): h.update(blk)
    return h.hexdigest()


tag = sys.argv[1] if len(sys.argv) > 1 else 'before'
out = HERE/'copy_sha256.csv'; new = not out.exists()
with open(out, 'a', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    if new: w.writerow(['tag', 'kind', 'path', 'sha256', 'mtime', 'size'])
    for p in files:
        if not p.exists(): w.writerow([tag, 'missing', str(p), '', '', '']); continue
        w.writerow([tag, 'readonly_input', str(p), sh(p), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p))), os.path.getsize(p)])
print('ok', tag, len(files))
