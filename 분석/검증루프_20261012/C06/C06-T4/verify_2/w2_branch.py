import sys, re, json; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
B=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
EXT=B+r'\분석\모델경쟁_20260927\_shared\ext'; W=B+r'\분석\검증루프_20261012\C06\C06-T4'
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
TC={str(x) for x in range(81,89)}
s0='202312'
d=pd.read_parquet(EXT+rf'\firm_nps_wkpl_{s0}.parquet',columns=['name','biz6','stat','n_emp','indnm'])
d=d[(d.stat.astype(str)=='1')&d.biz6.astype(str).str[3:5].isin(TC)].copy(); d['nn']=d.name.map(norm); d['b6']=d.biz6.astype(str)
U=pd.read_parquet(W+r'\cache\B_units.parquet'); U=U[U.s0==s0]
ub=pd.DataFrame({'b6':U.key.str.split('|',n=1).str[0],'nn_u':U.key.str.split('|',n=1).str[1],'E0':U.E0.values})
ub=ub[ub.nn_u.str.len()>=2]
m=d.merge(ub,on='b6'); br=m[(m.nn!=m.nn_u)&np.array([a.startswith(b) for a,b in zip(m.nn,m.nn_u)])].copy()
tag=lambda s: '일용/현장' if re.search(r'일용|현장|공사|PJT|신축',s) else ('지점/공장/지사/사업소' if re.search(r'지점|공장|지사|사업소|사업장|센터|연구소|영업소|출장소|본부',s) else '기타')
br['tag']=br.name.astype(str).map(tag)
print('rows',len(br),'firms',br.nn_u.nunique(),'emp',int(br.n_emp.sum()),'unitE0',int(ub.E0.sum()))
print(br.groupby('tag').agg(rows=('name','size'),firms=('nn_u','nunique'),emp=('n_emp','sum')))
print(br.sample(25,random_state=1)[['name','nn_u','n_emp','tag']].to_string())
# 기업별 지점형 고용 비중 분포
f=br.groupby('nn_u').n_emp.sum(); e=ub.groupby('nn_u').E0.sum()
sh=(f/(f+e.reindex(f.index))).dropna()
print('firm-level branch share among firms having branch rows: median',round(sh.median(),3),'p90',round(sh.quantile(.9),3),'n',len(sh))
print('firms where branch emp > key emp:',int((sh>0.5).sum()))
# 대기업 단일 행 고용
for k in ['삼성전자','현대자동차','엘지전자','에스케이하이닉스','기아','엘지디스플레이','삼성에스디아이','엘지화학']:
    x=ub[ub.nn_u==k]; print(k, x.E0.tolist())
# 상위 20 단위: 키 고용 vs 같은 biz6 지점형
top=ub.sort_values('E0',ascending=False).head(20)
top['branch_emp']=top.nn_u.map(f).fillna(0).astype(int)
print(top.to_string())
