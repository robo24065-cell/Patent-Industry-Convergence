# -*- coding: utf-8 -*-
"""제공 융합데이터 50개 TSV(29.5GB) → 5개국 메타 parquet (요약·대표청구항 제외). 파일별 part 로 저장."""
import pandas as pd, glob, os, sys, io, time
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(B, r"분석\모델경쟁_20260927\_shared\meta")
COLS = ['출원번호', '출원일', '출원연도', '등록번호', '등록일자', '공개일자', '발명의 명칭', '국가코드',
        '출원인명', '메인 ipc', 'IPC', '메인 cpc', '패밀리 출원번호', '우선권 주장 국가코드',
        '우선권 주장 출원일자', '세세분류 코드']
REN = {'발명의 명칭': 'title', '메인 ipc': 'main_ipc', '메인 cpc': 'main_cpc', '패밀리 출원번호': 'family',
       '우선권 주장 국가코드': 'prio_cc', '우선권 주장 출원일자': 'prio_date', '세세분류 코드': 'ksic',
       '출원번호': 'appl_no', '출원일': 'appl_date', '출원연도': 'appl_year', '등록번호': 'rgst_no',
       '등록일자': 'rgst_date', '공개일자': 'pub_date', '국가코드': 'cc', '출원인명': 'applicant', 'IPC': 'ipc_all'}


def one(f):
    t = time.time()
    out = os.path.join(OUT, os.path.basename(f).replace('.csv', '.parquet'))
    if os.path.exists(out):
        return f, -1, 0
    parts = []
    for ch in pd.read_csv(f, sep='\t', encoding='utf-8-sig', usecols=COLS, dtype=str, chunksize=400_000):
        ch = ch.rename(columns=REN)
        for c in ch.columns:
            ch[c] = ch[c].str.strip().replace('', None)
        parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    d['appl_year'] = pd.to_numeric(d.appl_year, errors='coerce').astype('Int16')
    for c in ('cc', 'ksic', 'prio_cc'):
        d[c] = d[c].astype('category')
    d.to_parquet(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    return f, len(d), time.time() - t


if __name__ == '__main__':
    fs = sorted(glob.glob(os.path.join(B, r'공모전데이터\융합데이터_*\*.csv')), key=os.path.getsize, reverse=True)
    print(f'files {len(fs)}', flush=True)
    n = 0
    with Pool(5) as p:
        for f, r, s in p.imap_unordered(one, fs):
            n += max(r, 0)
            print(f'{os.path.basename(f)} rows={r:,} {s:.0f}s total={n:,}', flush=True)
    print('DONE', flush=True)
