# DART 기업개황(설립일) 결합: 권리자명 → (CORP_APPLICANT 법인번호) → DART est_dt. 상장구분·상태는 스냅샷이라 제외.
import sys, sqlite3, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import norm
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\외부데이터\DART"
ca = pd.read_parquet(B + r"\corp_applicant.parquet", columns=['출원인명', 'jurir'])
ca['nm'] = ca['출원인명'].map(norm)
con = sqlite3.connect('file:' + B + r'\dart_company.sqlite?mode=ro', uri=True)
dc = pd.read_sql("select jurir_no jurir, est_dt, induty_code from company where jurir_no is not null and jurir_no<>''", con)
m = ca.merge(dc, on='jurir').dropna(subset=['est_dt'])
m['est'] = pd.to_datetime(m.est_dt, format='%Y%m%d', errors='coerce')
o = m.dropna(subset=['est']).groupby('nm').agg(est=('est', 'min'), induty=('induty_code', 'first')).reset_index()
o.to_parquet('cache/owner_dart.parquet', index=False)
oa = pd.read_parquet('cache/owner_attr.parquet')
va = pd.read_parquet('cache/panel_L2014.parquet', columns=['cur', 'cur_type'])
print('DART matched owner names', len(o), '| L2014 patents with matched owner', round(va.cur.isin(o.nm).mean(), 3),
      '| among cur_type=1', round(va[va.cur_type == 1].cur.isin(o.nm).mean(), 3))
