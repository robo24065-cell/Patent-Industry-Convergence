# -*- coding: utf-8 -*-
"""vC 보조: (iii)에서 새로 SAB가 된 4셀의 최대 기여 출원인 키 구성(공동출원 상대 유형) 확인. 개인 이름은 해시."""
import sys, re, html, sqlite3, hashlib, duckdb, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
BR = (ROOT + r"\분석\모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet").replace(chr(92),'/')
META = (ROOT + r"\분석\모델경쟁_20260927\_shared\meta\*.parquet").replace(chr(92),'/')
RF = ROOT + r"\수집파이프라인\registry_full.sqlite"
cells = ['27199|A61N 7', '28111|H02K 23', '29210|A01D 33', '29280|A61G 7']
con = duckdb.connect()
br = con.execute(f"select ksic, appl_no, year, ipc_raw, app1 from read_parquet('{BR}') where cc='KR' and year between 2019 and 2024").df()
def mg(s):
    s2 = (s or '').replace(' ', ''); m = re.match(r'^([A-H]\d{2}[A-Z])(\d{1,4})/', s2)
    return f"{m.group(1)} {int(m.group(2))}" if m else None
br['cell'] = br['ksic'] + '|' + br['ipc_raw'].map(mg)
d = br[br.cell.isin(cells)].drop_duplicates(['cell', 'appl_no'])
meta = con.execute(f"select appl_no, any_value(nullif(trim(rgst_no),'')) rgst_no from read_parquet('{META}') where cc='KR' group by appl_no").df()
d = d.merge(meta, on='appl_no', how='left')
sq = sqlite3.connect(f"file:{RF}?mode=ro", uri=True)
ap = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", sq)
def lab(n):
    n = n or ''
    if any(k in n for k in ['대학', '학교', '연구원', '재단', '공사', '주식회사', '(주)', '㈜', '협동조합', 'Co', 'Inc']): return n
    return 'P#' + hashlib.sha1(n.encode()).hexdigest()[:6]
for c in cells:
    x = d[d.cell == c].copy(); x['w'] = np.where(x.year <= 2021, 'past', 'fut')
    j = x.merge(ap, on='rgst_no', how='left')
    j['key'] = j['applicantCd'].fillna('N:' + j['app1'].map(lab))
    g = j.groupby('key').apply(lambda t: pd.Series({'fut': (t.w == 'fut').sum(), 'past': (t.w == 'past').sum(), 'name': lab(t.applicantName.dropna().iloc[0]) if t.applicantName.notna().any() else lab(t.app1.iloc[0])}))
    g['delta'] = g.fut - g.past
    print('==', c, 'docs', len(x), 'reg', x.rgst_no.notna().sum(), 'fut', (x.w == 'fut').sum(), 'past', (x.w == 'past').sum())
    print(g.sort_values('delta', ascending=False).head(4).to_string())
    print(' provided app1 top:', x.groupby('w').app1.agg(lambda s: s.map(lab).value_counts().head(3).to_dict()).to_dict())
