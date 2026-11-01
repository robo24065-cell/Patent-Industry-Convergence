# r3kip_ 원시 XML → 엣지 parquet + 인용 시점 부여
import os, re, glob, pandas as pd, numpy as np, duckdb, sqlite3
D=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(D,'r3kip_raw')
B=os.path.dirname(os.path.dirname(D)); ROOT=os.path.dirname(os.path.dirname(B))
def items(body,tag):
    out=[]
    for blk in re.findall(rf'<{tag}>(.*?)</{tag}>',body,re.S):
        out.append({k:v.strip() for k,v in re.findall(r'<(\w+)>(.*?)</\1>',blk,re.S)})
    return out
S=pd.read_parquet(os.path.join(D,'r3kip_sample.parquet'))
ci=[];cn=[];per=[]
for f in glob.glob(os.path.join(RAW,'*.xml')):
    kind,a=os.path.basename(f)[:-4].split('_'); b=open(f,encoding='utf-8').read()
    it=items(b,'citingInfo' if kind=='citing' else 'citationInfoV3')
    per.append((kind,a,len(it)))
    for r in it: r['q']=a; (ci if kind=='citing' else cn).append(r)
per=pd.DataFrame(per,columns=['kind','applNo','n_rows'])
ci=pd.DataFrame(ci); cn=pd.DataFrame(cn)
print('호출별 행수 분포', per.groupby('kind').n_rows.describe().round(2).to_dict())
ci=ci.rename(columns={'StandardCitationApplicationNumber':'cited_appl','ApplicationNumber':'citing_appl','CitationLiteratureTypeCode':'ctype','CitationLiteratureTypeCodeName':'ctype_name','StandardStatusCode':'std_status'})
print('citing 열',ci.columns.tolist()); print(ci.groupby(['ctype','ctype_name']).size())
assert (ci.cited_appl==ci.q).all()
cn=cn.rename(columns={'ApplicationNumber':'citing_appl','StandardCitationLiteraturenumber':'cited_std_no','StandardCitationLiteratureCountryCode':'cited_cc',
  'StandardCitationIdentificationCode':'cited_kind','StandardCitationLiteraturePublicationDate':'cited_pub_date','CitationLiteratureTypeCode':'ctype','CitationLiteratureTypeCodeName':'ctype_name',
  'OriginalcitationLiteraturenumber':'orig_no','StandardStatusCode':'std_status'})
print('citation 열',cn.columns.tolist()); print(cn.groupby(['ctype','ctype_name']).size()); print(cn.cited_cc.value_counts().head(10))
# 인용한 출원(citing_appl)의 시점: 제공 메타(KR) + 등록원부
apps=pd.Series(pd.concat([ci.citing_appl,S.applNo]).unique(),name='a')
con=duckdb.connect(); con.register('apps',apps.to_frame())
m=con.sql(f"""select appl_no a, min(appl_date) appl_date, min(pub_date) pub_date, min(rgst_date) rgst_date, min(ksic) ksic_first,
  string_agg(distinct ksic,'|') ksics, min(main_ipc) main_ipc from read_parquet('{B}/_shared/meta/*.parquet') m join apps on m.appl_no=apps.a where cc='KR' group by 1""").df()
r=sqlite3.connect(f'file:{ROOT}/수집파이프라인/registry_full.sqlite?mode=ro',uri=True)
reg=pd.read_sql('select applNo a, rgstDate r_rgst, opnDate r_opn, pubDate r_pub, applDate r_appl from patent',r)
reg=reg[reg.a.isin(apps)]
br=sqlite3.connect(f'file:{ROOT}/수집파이프라인/business_rights.sqlite?mode=ro',uri=True)
brl=pd.read_sql("select appl_no a, min(appl_date) b_appl, min(rgst_date) b_rgst from right_list where substr(rgst_no,1,2) in ('10','20') group by appl_no",br)
brl=brl[brl.a.isin(apps)]
A=apps.to_frame().merge(m,on='a',how='left').merge(reg.drop_duplicates('a'),on='a',how='left').merge(brl,on='a',how='left')
A['r_appl']=A.r_appl.fillna(A.b_appl); A['r_rgst']=A.r_rgst.fillna(A.b_rgst)
A['src']=np.where(A.appl_date.notna(),'meta',np.where(A.b_appl.notna()|A.r_appl.notna(),'rights','number_only'))
def dt(s): return pd.to_datetime(s.replace('',None),format='%Y%m%d',errors='coerce')
A['appl_dt']=dt(A.appl_date.fillna(A.r_appl))
A['yr_from_no']=A.a.str[2:6].astype(int)
A['pub_dt']=dt(A.pub_date.fillna(A.r_opn))
A['rgst_dt']=dt(A.rgst_date.fillna(A.r_rgst))
A['in_meta']=A.appl_date.notna()
# 시점 규칙: 보수(cons)=인용한 출원의 등록일, 미등록·미상이면 출원연도+4년 1월 1일 / 완화(lib)=인용한 출원의 공개일, 없으면 출원일+18개월
fallback=pd.to_datetime((A.yr_from_no+4).astype(str)+'-01-01')
A['vis_cons']=A.rgst_dt.fillna(fallback)
A['vis_lib']=A.pub_dt.fillna((A.appl_dt+pd.DateOffset(months=18)) if True else None).fillna(pd.to_datetime((A.yr_from_no+2).astype(str)+'-01-01'))
ci=ci.merge(A[['a','in_meta','src','ksics','main_ipc','appl_dt','pub_dt','rgst_dt','vis_cons','vis_lib','yr_from_no']].rename(columns={'a':'citing_appl'}),on='citing_appl',how='left')
ci=ci.merge(S[['applNo','set','ksic']].rename(columns={'applNo':'cited_appl','ksic':'cited_ksic'}),on='cited_appl',how='left')
ci['self_ksic']=[ (isinstance(k,str) and c in k.split('|')) for k,c in zip(ci.ksics,ci.cited_ksic)]
ci=ci.drop(columns=['q'])
ci.to_parquet(os.path.join(D,'r3kip_citing_edges.parquet'),index=False)
cn['cited_pub_dt']=dt(cn.cited_pub_date)
cn=cn.merge(S[['applNo','ksic']].rename(columns={'applNo':'citing_appl','ksic':'citing_ksic'}),on='citing_appl',how='left').drop(columns=['q'])
cn.to_parquet(os.path.join(D,'r3kip_citation_edges.parquet'),index=False)
per.to_parquet(os.path.join(D,'r3kip_calls_per_query.parquet'),index=False)
u=ci.drop_duplicates('citing_appl')
print('피인용 엣지',len(ci),'고유 (피인용,인용) 쌍',ci[['cited_appl','citing_appl']].drop_duplicates().shape[0],'고유 인용 출원',len(u))
print('인용 출원 메타(50KSIC) 연결률',u.in_meta.mean().round(4),'등록일 확보',u.rgst_dt.notna().mean().round(4),'공개일 확보',u.pub_dt.notna().mean().round(4))
print('시점 출처',u.src.value_counts(normalize=True).round(4).to_dict()); print('인용 출원 접두',u.citing_appl.str[:2].value_counts().to_dict())
print('같은 KSIC 인용 비율',ci.self_ksic.mean().round(4))
