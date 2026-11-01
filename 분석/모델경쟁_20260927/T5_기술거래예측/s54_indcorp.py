# [v33 피처] 개인 발명가의 법인화 신호 → cache/ind_L{Y}.parquet
#  개인 권리자(현재 권리자 유형 4)의 이름이 최근 5년 출원된 '국내법인 출원' 특허의 발명자로 등장하는가(= 본인이 법인을 세웠거나 합류 → 보유 특허를 법인에 양도할 가능성).
#  시점: 법인 특허는 L 이전에 등록된 것만(발명자 정보가 L 시점에 확정·공개, 미래 등록 여부로 조건화하지 않음).
#  또 그 법인 특허의 출원인(법인)명을 기록해, 같은 KSIC 인지 여부(동일 기술 법인화)를 함께 본다.
import sys, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
t0 = time.time()
pat = pd.read_parquet('cache/pat.parquet', columns=['rgst_no', 'ksic', 'applDate', 'rgstDate'])
for c in ['applDate', 'rgstDate']: pat[c] = pd.to_datetime(pat[c], format='%Y%m%d', errors='coerce')
app = pd.read_parquet('cache/app.parquet', columns=['rgst_no', 'seq', 'applicantCd'])
a1 = app.sort_values(['rgst_no', 'seq']).groupby('rgst_no').applicantCd.first().str[:1]
corp = pat[pat.rgst_no.map(a1) == '1']
inv = pd.read_parquet('cache/inv.parquet', columns=['rgst_no', 'inv_names'])
x = corp.merge(inv, on='rgst_no')
x = x.assign(inv=x.inv_names.fillna('').str.split('|')).explode('inv')
x['inv'] = x.inv.fillna('').str.replace(' ', '')
x = x[x.inv.str.fullmatch(r'[가-힣]{2,4}')][['inv', 'ksic', 'applDate', 'rgstDate']]
print('corp patent-inventor rows', len(x), round(time.time() - t0, 1), flush=True)
for Y in range(2003, LMAX + 1):
    L = pd.Timestamp(Y, 1, 1)
    s = pd.read_parquet(f'cache/panel_L{Y}.parquet', columns=['rgst_no', 'cur', 'cur_type', 'ksic'])
    xx = x[(x.rgstDate < L) & (x.applDate >= pd.Timestamp(Y - 5, 1, 1))]
    s['ind_corp_inv5'] = s.cur.map(xx.groupby('inv').size()).fillna(0)
    ki = pd.MultiIndex.from_arrays([s.cur, s.ksic])
    s['ind_corp_inv5_ksic'] = ki.map(xx.groupby(['inv', 'ksic']).size()).fillna(0).values
    s.loc[s.cur_type != 4, ['ind_corp_inv5', 'ind_corp_inv5_ksic']] = np.nan
    s[['rgst_no', 'ind_corp_inv5', 'ind_corp_inv5_ksic']].assign(L=Y).to_parquet(f'cache/ind_L{Y}.parquet', index=False)
    k = s.cur_type == 4
    print(Y, 'individual rows', int(k.sum()), 'with corp-inventor match', round((s.ind_corp_inv5[k] > 0).mean(), 3), flush=True)
