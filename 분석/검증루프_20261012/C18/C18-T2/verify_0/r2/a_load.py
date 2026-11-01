# C18-T2 verify_0 r2 (재시도, 반박 목적): 원천 meta parquet에서 pyarrow/pandas로 독립 추출 (duckdb 미사용)
# 출력: r2/us_ep.pkl (US·EP 행 일부 열), r2/kr.pkl (KR 등록번호), r2/tok.pkl (해외 family KR 등록 토큰)
import glob, os, re, sys, pickle, time
import pyarrow.parquet as pq, pyarrow.compute as pc, pandas as pd
t0 = time.time()
META = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta'
OUT = os.path.dirname(os.path.abspath(__file__))
fs = sorted(glob.glob(os.path.join(META, '*.parquet')))
ue, kr, tk = [], [], []
pat = re.compile(r'^KR(10|20)[0-9]{7}$')
for f in fs:
    t = pq.read_table(f, columns=['appl_no', 'appl_date', 'cc', 'ksic', 'rgst_no', 'family', 'title', 'pub_date'])
    cc = pc.cast(t['cc'], 'string')
    t = t.drop(['cc']).append_column('cc', cc)
    ks = pc.cast(t['ksic'], 'string')
    t = t.drop(['ksic']).append_column('ksic', ks)
    d = t.to_pandas()
    m = d.cc.isin(['US', 'EP'])
    ue.append(d.loc[m, ['cc', 'appl_no', 'appl_date', 'ksic', 'title', 'pub_date']])
    k = d[(d.cc == 'KR') & d.rgst_no.notna()]
    k = k[k.rgst_no.str.len() >= 9]
    kr.append(pd.DataFrame({'kr_appl_no': k.appl_no.values, 'r9': k.rgst_no.str[:9].values, 'kk': k.ksic.values}))
    fo = d[(d.cc != 'KR') & d.family.notna() & d.family.str.contains('KR', regex=False)]
    ex = fo[['cc', 'appl_no', 'ksic', 'family']].assign(t=fo.family.str.split('|')).explode('t')
    ex = ex[ex.t.notna() & ex.t.str.match(r'^KR(10|20)[0-9]{7}$')]
    tk.append(pd.DataFrame({'cc': ex.cc.values, 'appl_no': ex.appl_no.values, 'fk': ex.ksic.values, 'r9': ex.t.str[2:].values}))
    del d, t
ue = pd.concat(ue, ignore_index=True); kr = pd.concat(kr, ignore_index=True); tk = pd.concat(tk, ignore_index=True)
ue.to_pickle(os.path.join(OUT, 'us_ep.pkl')); kr.to_pickle(os.path.join(OUT, 'kr.pkl')); tk.to_pickle(os.path.join(OUT, 'tok.pkl'))
print(len(ue), len(kr), len(tk), ue.cc.value_counts().to_dict(), round(time.time() - t0), flush=True)
