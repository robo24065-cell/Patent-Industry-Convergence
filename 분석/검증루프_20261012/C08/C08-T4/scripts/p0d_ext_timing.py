# C08-T4 P0-d (PREREG 전 시점 점검): 불납 소멸등록 날짜가 '처리일'인지 '소급 소멸일'인지. 캐시(< 2025-01-01)만.
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import *
pat = pd.read_parquet(os.path.join(C, 'pat.parquet')).set_index('rgst_no')
rgt = pd.read_parquet(os.path.join(C, 'rgt.parquet'))
pay = pd.read_parquet(os.path.join(C, 'pay.parquet'))
pay['lastAnnl'] = pd.to_numeric(pay.lastAnnl, errors='coerce')
cov = pay.groupby('rgst_no').lastAnnl.max()
ext = rgt[(rgt.nm == '소멸등록') & (rgt.rs == '등록료불납')][['rgst_no', 'd']].copy()
ext['d'] = ok8(ext.d); ext = ext.dropna().drop_duplicates('rgst_no', keep='last')
ext['rd'] = ext.rgst_no.map(pat.rgstDate); ext['cov'] = ext.rgst_no.map(cov)
ext = ext.dropna()
ext = ext[ext.rd.str[4:] != '0229']
end = pd.to_datetime((ext.rd.str[:4].astype(int) + ext['cov'].astype(int)).astype(str) + ext.rd.str[4:], format='%Y%m%d', errors='coerce')
ext['gap'] = (pd.to_datetime(ext.d, format='%Y%m%d') - end).dt.days
e = ext[(ext.d >= '20150101') & (ext.d < '20230101')]
out = {'n': int(len(e)), 'gap_q': e.gap.quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99]).to_dict(),
       'share_gap_le_0': float((e.gap <= 0).mean()), 'share_gap_le_1': float((e.gap <= 1).mean()), 'share_gap_183_300': float(((e.gap >= 183) & (e.gap <= 300)).mean())}
# 회복등록 날짜와 소멸등록 날짜 순서
rec = rgt[rgt.nm == '회복등록'][['rgst_no', 'd']].copy(); rec['d'] = ok8(rec.d)
x = ext.merge(rec.rename(columns={'d': 'dr'}), on='rgst_no')
x['dd'] = (pd.to_datetime(x.dr, format='%Y%m%d') - pd.to_datetime(x.d, format='%Y%m%d')).dt.days
out['rec_minus_ext_q'] = x.dd.quantile([.05, .25, .5, .75, .95]).to_dict(); out['rec_pairs'] = int(len(x))
# 회복 특허의 마지막 소멸 앞뒤: 같은 특허의 모든 소멸등록 행
allx = rgt[(rgt.nm == '소멸등록')][['rgst_no', 'd', 'seq']]
r1 = rec.drop_duplicates('rgst_no')
m = allx.merge(r1.rename(columns={'d': 'dr'}), on='rgst_no')
m['before'] = m.d < m.dr
out['rec_patents'] = int(r1.rgst_no.nunique()); out['rec_with_ext_before'] = int(m[m.before].rgst_no.nunique()); out['rec_with_ext_after_only'] = int(m.groupby('rgst_no').before.max().eq(False).sum())
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
json.dump(out, open(os.path.join(C, 'p0d_ext_timing.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
