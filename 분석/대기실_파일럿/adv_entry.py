import pandas as pd, numpy as np, time, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize
sys.stdout.reconfigure(encoding='utf-8')
SP=r'C:/Users/PC/AppData/Local/Temp/claude/C--Users-PC-Downloads------------------------/72c91564-e815-4ac3-bfb0-ac338e5a5b96/scratchpad'
B=r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/'
t0=time.time()
kr=pd.read_parquet(B+'공모전데이터/작업데이터/kr_registered.parquet',columns=['등록번호','발명명칭','세세분류코드','출원연도'])
kr['ay']=kr.출원연도.astype(int); kr['k']=kr.세세분류코드.astype(str)
rl0=pd.read_parquet(SP+'/rl.parquet',columns=['applicant_cd','rgst_no','appl_date','title'])
rl0=rl0[rl0.rgst_no.str[:2]=='10']
rl0['ay']=pd.to_numeric(rl0.appl_date.str[:4],errors='coerce'); rl0=rl0[rl0.ay.between(1985,2025)]
rl0=rl0.merge(kr[['등록번호','k']],left_on='rgst_no',right_on='등록번호',how='left')
rl0['ins']=rl0.k.notna()
K=sorted(kr.k.unique()); kidx={k:i for i,k in enumerate(K)}
tr=kr[kr.ay<=2009]
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=5,max_features=400000,sublinear_tf=True,dtype=np.float32)
X=vec.fit_transform(tr.발명명칭.fillna(''))
clf=SGDClassifier(loss='log_loss',alpha=2e-6,max_iter=12,tol=None,random_state=0).fit(X,tr.k)
rl0['title']=rl0.title.fillna('')
Xall=vec.transform(rl0.title); Xall=normalize(Xall)
P0=np.zeros((len(rl0),50),np.float32)
o=(~rl0.ins).values
P0[o]=clf.predict_proba(Xall[o]).astype(np.float32)
rl0['row']=np.arange(len(rl0))
print('scored',o.sum(),time.time()-t0,flush=True)

def build(rl, cut, h, nearthr=None):
    past=rl[rl.ay<=cut]
    firms=past.groupby('applicant_cd').size(); firms=firms[firms>=5].index
    past=past[past.applicant_cd.isin(firms)]
    fi={f:i for i,f in enumerate(firms)}; nF=len(firms)
    ins=past[past.ins]; M=np.zeros((nF,50),np.float32)
    np.add.at(M,(ins.applicant_cd.map(fi).values,ins.k.map(kidx).values),1)
    allp=rl[(rl.ay<=cut)&rl.ins]; A=pd.crosstab(allp.applicant_cd,allp.k).reindex(columns=K,fill_value=0).values>0
    A=A.astype(np.float32); co=A.T@A; d=np.diag(co); prox=co/np.maximum.outer(d,d); np.fill_diagonal(prox,0)
    Bm=(M>0).astype(np.float32); dens=(Bm@prox)/np.maximum(prox.sum(0),1e-9)
    o=past[~past.ins]; Po=P0[o.row.values]; rows=o.applicant_cd.map(fi).values
    T=np.zeros((nF,50),np.float32); np.add.at(T,rows,Po)
    H=np.zeros((nF,50),np.float32); hi=(Po.max(1)>0.5); np.add.at(H,(rows[hi],Po[hi].argmax(1)),1)
    nout=np.bincount(rows,minlength=nF).astype(np.float32)
    Tm=T/np.maximum(nout,1)[:,None]
    fut=rl[(rl.ay>cut)&(rl.ay<=cut+h)&rl.ins&rl.applicant_cd.isin(firms)].copy()
    fut['f']=fut.applicant_cd.map(fi).values; fut['kk']=fut.k.map(kidx).values
    fut=fut[M[fut.f.values,fut.kk.values]==0]
    Y=np.zeros((nF,50),np.int8); Y[fut.f.values,fut.kk.values]=1
    # near-duplicate: entry patent vs firm's own prior out-of-scope titles
    ND=np.zeros((nF,50),np.int8); simmax={}
    if nearthr is not None:
        for f,g in fut.groupby('f'):
            po=o[o.applicant_cd.map(fi).values==f] if False else None
        ofi=o.assign(f=rows)
        og=ofi.groupby('f').row.apply(np.array)
        for (f,kk),g in fut.groupby(['f','kk']):
            first=g[g.ay==g.ay.min()]
            if f not in og.index: s=0.0
            else:
                s=float((Xall[first.row.values]@Xall[og[f]].T).max())
            simmax[(f,kk)]=s
            if s>=nearthr: ND[f,kk]=1
    ntot=np.bincount(past.applicant_cd.map(fi).values,minlength=nF).astype(np.float32)
    nin=M.sum(1)
    f,k=np.nonzero(M==0)
    df=pd.DataFrame({'lsize':np.log1p(ntot[f]),'lin':np.log1p(nin[f]),'inshare':nin[f]/ntot[f],'dens':dens[f,k],
        'kpop':np.log1p(A.sum(0))[k],'txt_mean':Tm[f,k],'txt_hi':np.log1p(H[f,k]),'lout':np.log1p(nout[f]),'y':Y[f,k],'k':k,'newc':(nin[f]==0),'nd':ND[f,k]})
    df['sim']=[simmax.get((a,b),np.nan) for a,b in zip(f,k)] if nearthr is not None else np.nan
    return df
base=['lsize','lin','inshare','dens','kpop','lout']; full=base+['txt_mean','txt_hi']
def evalset(d1,d2,tag,mask=None):
    m1=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0).fit(d1[base],d1.y)
    m2=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0).fit(d1[full],d1.y)
    sb=m1.predict_proba(d2[base])[:,1]; sf=m2.predict_proba(d2[full])[:,1]
    out=[]
    for nm,mk in [('all',np.ones(len(d2),bool)),('newc',d2.newc.values)]+([] if mask is None else [(k,v) for k,v in mask.items()]):
        y=d2.y.values[mk]
        if y.sum()==0: continue
        out.append((tag,nm,mk.sum(),y.sum(),round(roc_auc_score(y,sb[mk]),3),round(roc_auc_score(y,sf[mk]),3),round(average_precision_score(y,sb[mk]),4),round(average_precision_score(y,sf[mk]),4)))
    for r in out: print(r,flush=True)
    return sb,sf

print('--- V0 original (incl pre-2000)')
d1=build(rl0,2009,5); d2=build(rl0,2014,5,nearthr=0.8)
print('pairs',len(d2),'pos',d2.y.sum(),'newc pairs',d2.newc.sum(),'newc pos',d2[d2.newc].y.sum())
print('entry first-patent max cos sim to own prior out titles: quantiles',d2[d2.y==1].sim.quantile([.1,.25,.5,.75,.9]).round(3).to_dict())
print('share entries sim>=0.8',(d2[d2.y==1].sim>=0.8).mean().round(3),' >=0.9',(d2[d2.y==1].sim>=0.9).mean().round(3),' ==1',(d2[d2.y==1].sim>=0.999).mean().round(3))
sb,sf=evalset(d1,d2,'V0')
# drop near-dup entries (treat as not-evaluable)
keep=~((d2.y==1)&(d2.sim>=0.8))
d2k=d2[keep.values]
evalset(d1,d2k,'V0-dropND0.8')
keep=~((d2.y==1)&(d2.sim>=0.6))
evalset(d1,d2[keep.values],'V0-dropND0.6')

print('--- V1 drop pre-2000 patents (in-scope data starts 2000)')
rl1=rl0[rl0.ay>=2000]
d1=build(rl1,2009,5); d2=build(rl1,2014,5,nearthr=0.8)
print('pairs',len(d2),'pos',d2.y.sum(),'newc pairs',d2.newc.sum(),'newc pos',d2[d2.newc].y.sum())
evalset(d1,d2,'V1')
keep=~((d2.y==1)&(d2.sim>=0.8)); evalset(d1,d2[keep.values],'V1-dropND0.8')
keep=~((d2.y==1)&(d2.sim>=0.6)); evalset(d1,d2[keep.values],'V1-dropND0.6')
d2.to_parquet(SP+'/adv_entry_d2.parquet')
print('time',time.time()-t0)
