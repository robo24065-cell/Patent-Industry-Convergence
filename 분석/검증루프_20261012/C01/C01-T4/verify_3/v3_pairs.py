# verify_3 v3: 창별 (법인 매입자 상호 x KSIC) 쌍 상한, 일괄 이전 비중 — 읽기전용. 잠금 창은 집계 수만.
import sqlite3, pandas as pd, re
p=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
out=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C01\C01-T4\verify_3\v3_pairs.txt'
con=sqlite3.connect('file:'+p+'?mode=ro',uri=True)
A="('양도','일부양도','지분양도','신탁계약에 의한 양도')"
d=pd.read_sql(f"select o.rgst_no, o.ownerRgstCsDate d, o.ownerName nm, o.ownerNatl nat, p.ksic from owner o join patent p on p.rgst_no=o.rgst_no where o.ownerRgstCsName in {A} and o.ownerRgstCsDate>='2013'",con)
CORP=re.compile(r'주식회사|\(주\)|㈜')
d['corp']=d.nm.fillna('').map(lambda s: bool(CORP.search(s)))
d['nn']=d.nm.fillna('').str.replace(r'주식회사|\(주\)|㈜|\s','',regex=True)
# 매입자 기존 KSIC 보유(창 시작 전 owner 또는 applicant 명) 근사: 창 시작 전 같은 상호가 그 KSIC 특허의 권리자였던 적이 있으면 제외
own_all=pd.read_sql("select o.ownerName nm, o.ownerRgstCsDate d, p.ksic from owner o join patent p on p.rgst_no=o.rgst_no where o.ownerRgstCsDate is not null",con)
own_all['nn']=own_all.nm.fillna('').str.replace(r'주식회사|\(주\)|㈜|\s','',regex=True)
L=[]
for t0 in range(2012,2025):
    s=f"{t0+1}0101"; e=f"{t0+2}0731"
    w=d[(d.d>=s)&(d.d<=e)&d.corp]
    g=w.groupby(['nn','d']).size(); bulk=w.set_index(['nn','d']).index.map(g)>=10
    pairs=w[['nn','ksic']].drop_duplicates()
    prior=own_all[own_all.d<s][['nn','ksic']].drop_duplicates()
    m=pairs.merge(prior,how='left',indicator=True)
    newp=int((m._merge=='left_only').sum())
    L.append(f"t0={t0}: 법인매입 행 {len(w)}, 일괄(같은날 같은매입자>=10) 비중 {bulk.mean():.3f}, 상호xKSIC 쌍 {len(pairs)}, 그중 창 전 같은상호가 그 KSIC 권리자 이력 없음 {newp}")
open(out,'w',encoding='utf-8').write('\n'.join(L))
