import pandas as pd, numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
SP=r'C:/Users/PC/AppData/Local/Temp/claude/C--Users-PC-Downloads------------------------/72c91564-e815-4ac3-bfb0-ac338e5a5b96/scratchpad'
t0=time.time()
kr=pd.read_parquet('공모전데이터/작업데이터/kr_registered.parquet',columns=['등록번호','발명명칭','세세분류코드','출원연도'])
kr['ay']=kr.출원연도.astype(int); kr['k']=kr.세세분류코드.astype(str)
rl=pd.read_parquet(SP+'/rl.parquet',columns=['applicant_cd','rgst_no','appl_date','title'])
rl=rl[rl.rgst_no.str[:2]=='10']
rl['ay']=pd.to_numeric(rl.appl_date.str[:4],errors='coerce'); rl=rl[rl.ay.between(1985,2025)]
rl=rl.merge(kr[['등록번호','k']],left_on='rgst_no',right_on='등록번호',how='left')
rl['ins']=rl.k.notna()
K=sorted(kr.k.unique()); kidx={k:i for i,k in enumerate(K)}
# classifier trained on in-scope <=2009
tr=kr[kr.ay<=2009]
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=5,max_features=400000,sublinear_tf=True,dtype=np.float32)
X=vec.fit_transform(tr.发明명칭 if False else tr.발명명칭.fillna(''))
clf=SGDClassifier(loss='log_loss',alpha=2e-6,max_iter=12,tol=None,random_state=0).fit(X,tr.k)
assert list(clf.classes_)==K
out=rl[~rl.ins & rl.title.notna()].copy()
P=clf.predict_proba(vec.transform(out.title)).astype(np.float32)
print('scored out-of-scope',len(out),time.time()-t0)
def build(cut, h):
    past=rl[rl.ay<=cut]
    firms=past.groupby('applicant_cd').size(); firms=firms[firms>=5].index
    past=past[past.applicant_cd.isin(firms)]
    fi={f:i for i,f in enumerate(firms)}; nF=len(firms)
    # in-scope counts matrix
    ins=past[past.ins]; M=np.zeros((nF,50),np.float32)
    np.add.at(M,(ins.applicant_cd.map(fi).values,ins.k.map(kidx).values),1)
    # proximity from co-presence across all firms with past patents
    allp=rl[(rl.ay<=cut)&rl.ins]; A=pd.crosstab(allp.applicant_cd,allp.k).reindex(columns=K,fill_value=0).values>0
    A=A.astype(np.float32); co=A.T@A; d=np.diag(co); prox=co/np.maximum.outer(d,d); np.fill_diagonal(prox,0)
    B=(M>0).astype(np.float32); dens=(B@prox)/np.maximum(prox.sum(0),1e-9)
    # text exposure
    msk=(out.ay<=cut)&out.applicant_cd.isin(firms)
    o=out[msk]; Po=P[msk.values]; rows=o.applicant_cd.map(fi).values
    T=np.zeros((nF,50),np.float32); np.add.at(T,rows,Po)
    H=np.zeros((nF,50),np.float32); hi=(Po.max(1)>0.5); np.add.at(H,(rows[hi],Po[hi].argmax(1)),1)
    nout=np.bincount(rows,minlength=nF).astype(np.float32)
    Tm=T/np.maximum(nout,1)[:,None]
    # outcome
    fut=rl[(rl.ay>cut)&(rl.ay<=cut+h)&rl.ins&rl.applicant_cd.isin(firms)]
    Y=np.zeros((nF,50),np.int8); Y[fut.applicant_cd.map(fi).values,fut.k.map(kidx).values]=1
    ntot=np.bincount(past.applicant_cd.map(fi).values,minlength=nF).astype(np.float32)
    nin=M.sum(1)
    risk=(M==0)
    f,k=np.nonzero(risk)
    df=pd.DataFrame({'lsize':np.log1p(ntot[f]),'lin':np.log1p(nin[f]),'inshare':nin[f]/ntot[f],'dens':dens[f,k],
        'kpop':np.log1p(A.sum(0))[k],'txt_mean':Tm[f,k],'txt_hi':np.log1p(H[f,k]),'lout':np.log1p(nout[f]),'y':Y[f,k],'k':k,'newc':(nin[f]==0)})
    return df
d1=build(2009,5); d2=build(2014,5)
print('train pairs',len(d1),d1.y.sum(),' test pairs',len(d2),d2.y.sum(),' test newcomer pairs',d2.newc.sum(),d2[d2.newc].y.sum())
base=['lsize','lin','inshare','dens','kpop','lout']; full=base+['txt_mean','txt_hi']
res={}
for name,cols in [('base',base),('base+text',full),('text_only',['txt_mean','txt_hi','lsize','kpop'])]:
    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0).fit(d1[cols],d1.y)
    s=m.predict_proba(d2[cols])[:,1]; res[name]=s
    nc=d2.newc.values
    print(name,'AUC',round(roc_auc_score(d2.y,s),4),'PR-AUC',round(average_precision_score(d2.y,s),4),
      '| newcomers AUC',round(roc_auc_score(d2.y[nc],s[nc]),4),'PR',round(average_precision_score(d2.y[nc],s[nc]),4))
# permutation placebo: shuffle text features within k across firms in test
rng=np.random.RandomState(1); aucs=[]
m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0).fit(d1[full],d1.y)
for r in range(5):
    d3=d2.copy()
    for c in ['txt_mean','txt_hi']: d3[c]=d3.groupby('k')[c].transform(lambda x: x.sample(frac=1,random_state=rng.randint(1e9)).values)
    aucs.append(average_precision_score(d3.y,m.predict_proba(d3[full])[:,1]))
print('placebo PR-AUC (text shuffled within industry):',np.round(aucs,4))
# precision@top: for each industry top-20 predicted entrants
d2['s_base']=res['base']; d2['s_full']=res['base+text']
for s in ['s_base','s_full']:
    top=d2.sort_values(s,ascending=False).groupby('k').head(20)
    print(s,'precision@20 per industry (mean)',top.y.mean().round(4),' hits',top.y.sum())
print('time',time.time()-t0)
