import pandas as pd, numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
SP=r'C:/Users/PC/AppData/Local/Temp/claude/C--Users-PC-Downloads------------------------/72c91564-e815-4ac3-bfb0-ac338e5a5b96/scratchpad'
t0=time.time()
kr=pd.read_parquet('공모전데이터/작업데이터/kr_registered.parquet',columns=['등록번호','발명명칭','세세분류코드','출원연도'])
kr['ay']=kr.출원연도.astype(int)
u=pd.read_parquet(SP+'/rl_u.parquet',columns=['rgst_no','title','kind','insc','ay'])
out=u[(u.kind=='10')&(~u.insc)&(u.ay>=2000)].copy()
out['y']='OTHER'
ins=kr.rename(columns={'발명명칭':'title','세세분류코드':'y'})[['title','y','ay']]
ins['y']=ins.y.astype(str)
rng=np.random.RandomState(0)
out=out.sample(400000,random_state=0)[['title','y','ay']]
df=pd.concat([ins,out]).dropna(subset=['title'])
tr=df[df.ay<=2015]; te=df[df.ay>=2017]
print('train',len(tr),'test',len(te))
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=5,max_features=600000,sublinear_tf=True,dtype=np.float32)
Xtr=vec.fit_transform(tr.title); Xte=vec.transform(te.title)
print('vec',time.time()-t0)
# Model 1: 50-class on in-scope only
m50=SGDClassifier(loss='log_loss',alpha=2e-6,max_iter=15,tol=None,random_state=0)
itr=(tr.y!='OTHER').values; ite=(te.y!='OTHER').values
m50.fit(Xtr[itr],tr.y[itr])
p=m50.predict(Xte[ite]); print('50-class title-only temporal holdout acc',accuracy_score(te.y[ite],p),'macroF1',f1_score(te.y[ite],p,average='macro'))
P=m50.predict_proba(Xte); msp=P.max(1)
print('open-set AUROC via max-softmax (50-class model, in vs out):',roc_auc_score(ite,msp))
# Model 2: 51-class with real negatives
m51=SGDClassifier(loss='log_loss',alpha=2e-6,max_iter=15,tol=None,random_state=0)
m51.fit(Xtr,tr.y)
P2=m51.predict_proba(Xte); cls=list(m51.classes_); oi=cls.index('OTHER')
print('open-set AUROC 51-class (1-P(other)):',roc_auc_score(ite,1-P2[:,oi]))
p2=m51.predict(Xte); 
print('51-class acc overall',accuracy_score(te.y,p2),' in-scope rows acc',accuracy_score(te.y[ite],p2[ite]),' OTHER recall',(p2[~ite]=='OTHER').mean())
# how many out-of-scope test patents get confident 50-industry signal
po=P2[~ite]; ind=np.delete(po,oi,axis=1)
print('out-of-scope test: share with max industry prob>0.5:',(ind.max(1)>0.5).mean(),' >0.8:',(ind.max(1)>0.8).mean())
print('time',time.time()-t0)
