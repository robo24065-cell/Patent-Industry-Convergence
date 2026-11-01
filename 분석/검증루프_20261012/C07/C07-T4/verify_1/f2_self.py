# 같은 산업 판정이 특허 자신(p)의 포트폴리오 등재 때문인지, 그리고 hc를 독립 재계산
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
C='cache'
docs=pd.read_parquet(C+'/docs.parquet',columns=['appl_no','ksic']).drop_duplicates('appl_no').set_index('appl_no').ksic
PI=pd.read_parquet(C+'/PI.parquet'); PI['ksic']=PI.appl_no.map(docs); PI=PI.dropna(subset=['ksic'])
print('PI date sample', PI.date.head(3).tolist(), PI.date.str.len().value_counts().to_dict())
rows=[]
for Y in range(2015,2023):
    b=pd.read_parquet(f'{C}/base_{Y}.parquet'); l=pd.read_parquet(f'{C}/lab_prov_{Y}.parquet')
    b['y']=l.y.values; b['bne']=l.buyer_nonempty.values
    rows.append(b[(b.has_ev==1)&(~b.buyer_type.isin([4,6]))&(l.buyer_nonempty.values==1)])
nn=pd.concat(rows,ignore_index=True)
nn['t0s']=nn.t0.astype(str)+'0101'
names=set(nn.buyer)
P=PI[PI.name.isin(names)]
m=nn[['rgst_no','applNo','t0s','buyer','ksic','y']].reset_index().merge(P[['name','appl_no','ksic','date','path']],left_on=['buyer','ksic'],right_on=['name','ksic'],how='left')
m=m[m.date.notna()&(m.date<m.t0s)]
hc=m.groupby('index').size()
hc_noself=m[m.appl_no!=m.applNo].groupby('index').size()
nn['hc']=nn.index.isin(hc.index); nn['hc_ns']=nn.index.isin(hc_noself.index)
print('recomputed cross (not hc):', (~nn.hc).sum(), 'vs y', nn.y.sum(), 'agree', ((~nn.hc).astype(int)==nn.y).mean())
print('cross if exclude self:', (~nn.hc_ns).sum(), 'share', (~nn.hc_ns).mean())
s=m[m.appl_no==m.applNo]; print('self rows by path', s.path.value_counts().to_dict(), 'patents', s['index'].nunique())
# same-industry rows supported only by path C (holding), or only by A/B
pa=m.groupby('index').path.agg(lambda x: ''.join(sorted(set(x))))
print(pa.value_counts().head(10))
# number of portfolio docs supporting same-industry
cnt=m[m.appl_no!=m.applNo].groupby('index').appl_no.nunique()
print('support docs (excl self) quantiles', cnt.describe().to_dict())
print('support ==1 share among same', (cnt==1).mean(), ' <=3', (cnt<=3).mean())
nn.to_parquet('verify_1/nn_rows.parquet')
