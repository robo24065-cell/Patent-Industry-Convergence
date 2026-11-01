# -*- coding: utf-8 -*-
"""읽기 전용 입력 파일의 sha256·크기·수정시각을 before/after로 기록(copy_sha256.csv). 사용: python sha_record.py before|after"""
import sys, pathlib, hashlib, os, time, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
HERE = pathlib.Path(__file__).resolve().parent.parent
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'
FILES = [B/'수집파이프라인/registry_full.sqlite', B/'수집파이프라인/business_rights.sqlite',
         B/'공모전데이터/작업데이터/kr_patents.parquet', B/'공모전데이터/외부데이터/DART/corp_applicant.parquet',
         T1/'cache/app_pub.parquet', T1/'cache/firms.parquet', T1/'cache/ins_applicant.parquet', T1/'cache/rl_pat.parquet', T1/'cache/ins_firm.parquet',
         T1/'R3/cache_final/o2019/prep_te.parquet', T1/'R3/cache_final/o2019/baseline_scores.parquet', T1/'R3/runs/FROZEN_final/o2019.parquet',
         B/'분석/검증루프_20261012/C16/C16-T1/cache/alias_docs.parquet', B/'분석/검증루프_20261012/C16/C16-T1/cache/labels.parquet',
         B/'분석/검증루프_20261012/C16/C16-T1/alias_audit.csv', B/'분석/검증루프_20261012/C03/C03-T1/cache/eval_o2019.parquet',
         B/'분석/보고서_20261012/manuscript_v3.md', B/'분석/보고서_20261012/facts/facts.csv', B/'분석/보고서_20261012/facts/facts_supplement.csv',
         B/'분석/검증루프_20261012/ERRATA.txt', B/'분석/검증루프_20261012/FINDINGS.txt']
tag = sys.argv[1]
out = HERE/'copy_sha256.csv'
rows = []
if out.exists():
    rows = list(csv.DictReader(open(out, encoding='utf-8-sig')))
for f in FILES:
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for ch in iter(lambda: fh.read(1 << 22), b''): h.update(ch)
    st = os.stat(f)
    rows.append(dict(tag=tag, path=str(f), size=st.st_size, mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)), sha256=h.hexdigest()))
w = csv.DictWriter(open(out, 'w', encoding='utf-8-sig', newline=''), fieldnames=['tag', 'path', 'size', 'mtime', 'sha256'])
w.writeheader(); w.writerows(rows)
print('ok', tag, len(FILES))
