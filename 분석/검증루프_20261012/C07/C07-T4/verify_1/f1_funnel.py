import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
C='cache'
docs=pd.read_parquet(C+'/docs.parquet',columns=['appl_no','ksic']).drop_duplicates('appl_no').set_index('appl_no').ksic
rows=[]
for Y in range(2015,2023):
    b=pd.read_parquet(f'{C}/base_{Y}.parquet'); l=pd.read_parquet(f'{C}/lab_prov_{Y}.parquet')
    assert (b.rgst_no.values==l.rgst_no.values).all()
    b['kd']=b.applNo.map(docs); b['y']=l.y.values; b['bne']=l.buyer_nonempty.values; b['lk']=l.ksic.values
    rows.append(b[b.has_ev==1])
e=pd.concat(rows)
print('accepted',len(e))
ni=e[~e.buyer_type.isin([4,6])]
print('buyer_type NaN among ev', e.buyer_type.isna().sum())
print('nonindiv',len(ni))
nn=ni[ni.bne==1]; print('nonempty',len(nn),'pos',nn.y.sum(), nn.y.mean())
z=nn[nn.y==0]
print('y0 with ksic NA (lab ksic)', z.lk.isna().sum(), ' base ksic NA', z.ksic.isna().sum())
print('base ksic vs docs ksic mismatch among nn', (nn.ksic.astype(str)!=nn.lk.astype(str)).sum())
print('same-industry share excluding NA', (len(z)-z.lk.isna().sum())/(len(nn)-nn.lk.isna().sum()))
# per year
g=nn.groupby('t0').agg(n=('y','size'),pos=('y','sum'))
print(g)
# bundles: same-industry share by bundle
nn=nn.assign(bulk=nn.bundle>=10)
print(nn.groupby('bulk').y.agg(['size','mean']))
# dedupe by (buyer, date) event-level
ev=nn.groupby(['buyer','ev_date']).y.agg(['size','max','mean'])
print('event-level (buyer,date) clusters',len(ev),'share with any cross',ev['max'].mean(),'mean share',ev['mean'].mean())
# unique buyers
print('buyers', nn.buyer.nunique(), 'cross buyers', nn[nn.y==1].buyer.nunique())
# buyer-type composition of nn and cross rate by type
print(nn.groupby('buyer_type').y.agg(['size','mean']))
