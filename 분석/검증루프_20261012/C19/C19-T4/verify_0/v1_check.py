# -*- coding: utf-8 -*-
# 반박 검증: v20 상태 델타 22행 재확인
import csv, collections, sys, os, hashlib
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012'
R = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012'
r = list(csv.DictReader(open(os.path.join(B, 'C18', 'C18-T3', 'errata_status_v18.csv'), encoding='utf-8-sig')))
print(len(r), collections.Counter(x['상태'] for x in r))
print(collections.Counter((x['상태'], x['분류']) for x in r))
ids = ['E-C17T3-01','E-C16T2-06','E-C17T3-04','E-C16T2-04','E-C17T1-01b','E-C16T3-S','E-C13T2-02','E-C09T2-04','E-C13T1-01','E-C03T3-11','E-C03T3-10','E-C16T4-02','E-C02T1-07','E-C15T1-03','E-C17T3-B2','E-C16T4-03','E-C02-T3-12','E-C07T1-08','E-C14T2-05','E-C15T1-02','E-C15T4-A1','E-C01-X2','E-C11T2-03','E-C12T1-04','E-C10T3-05','E-C14T2-03','E-C10T3-07']
d = {x['ID']: x for x in r}
for i in ids:
    x = d.get(i)
    print(i, x and (x['상태'], x['분류'], x['범위'], x['v19줄'], x['대체ID'], x['근거'][:260]))
print('부분:', [x['ID'] for x in r if x['상태'] == '부분'])
for fn in ['manuscript_v3.md', r'drafts\manuscript_for_v19.md', r'drafts\manuscript_for_v20.md']:
    p = os.path.join(R, fn)
    if os.path.exists(p):
        b = open(p, 'rb').read(); print(fn, hashlib.sha1(b).hexdigest()[:8], b.count(b'\n'))
