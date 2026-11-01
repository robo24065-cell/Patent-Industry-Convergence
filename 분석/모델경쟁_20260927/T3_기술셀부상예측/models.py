import numpy as np, pandas as pd, sys, os
import t3lib as L, harness as H
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
BASE_AUC=0.6519
MOM=['kr_mom','kr_mom_l1','kr_mom_l2','kr_g1','kr_acc','kr_lvl','kr_last','kr_slope5','kr_zero_yrs']
FOR=[f'{c}_{s}' for c in ['US','CN','JP','EP'] for s in ['mom','mom_l1','mom_l2','lvl','g1','slope5']]+['fx_mom','fx_mom_l1','tot_mom','tot_lvl','kr_share','kr_share_chg','gap_fx_kr']
STR=['kr_newapp_sh','kr_newk_sh','kr_prio_for_sh','kr_multi_sh','kr_newpair_sh','kr_newword_sh','all_newapp_sh','all_newpair_sh','kr_napp','kr_newapp_n','kr_hhi','all_newapp_n','kr_forin_mom','kr_dom_mom']
GRP=['ksic_kr_mom','mg_kr_mom','mg_tot_mom','mg_fx_mom','ksic_tot_mom']
def logit(cols,C=1.0):
    def fp(tr,te):
        m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=C,max_iter=3000))
        m.fit(tr[cols].values,tr.yb.values); return m.predict_proba(te[cols].values)[:,1]
    return fp
def xgbc(cols,params=None,target='yb',rounds=300):
    p=dict(max_depth=3,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,nthread=2,verbosity=0)
    if target=='yb': p.update(objective='binary:logistic',eval_metric='auc')
    elif target=='y': p.update(objective='reg:squarederror')
    if params: p.update(params)
    def fp(tr,te):
        yt=tr[target].values
        d=xgb.DMatrix(tr[cols].values,label=yt); m=xgb.train(p,d,rounds)
        return m.predict(xgb.DMatrix(te[cols].values))
    return fp
from scipy.stats import nbinom, rankdata
ALL=MOM+FOR+STR+GRP
def tau_of(tr):
    return float(np.median([np.quantile(g.y.values,0.9) for _,g in tr.groupby('t0')]))
def fit_alpha(F,mu):
    best=None
    for a in np.exp(np.linspace(np.log(0.01),np.log(5),60)):
        r=1/a; p=r/(r+mu); ll=nbinom.logpmf(F,r,p).sum()
        if best is None or ll>best[0]: best=(ll,a)
    return best[1]
def nb_tail(W,mu,alpha,tau):
    need=np.ceil((W+1)*np.exp(tau)-1-1e-9)   # F >= need
    r=1/alpha; p=r/(r+mu)
    return nbinom.sf(need-1,r,p)
def nbfp(cols,params=None,rounds=400,return_mu=False):
    pr=dict(objective='count:poisson',max_depth=3,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,nthread=2,verbosity=0,max_delta_step=0.7)
    if params: pr.update(params)
    def fp(tr,te):
        d=xgb.DMatrix(tr[cols].values,label=tr.kr_fut.values); m=xgb.train(pr,d,rounds)
        mu_tr=m.predict(d); a=fit_alpha(tr.kr_fut.values,mu_tr)
        mu=m.predict(xgb.DMatrix(te[cols].values)); tau=tau_of(tr)
        print('  alpha %.3f tau %.3f'%(a,tau))
        s=nb_tail(te.kr_W.values,mu,a,tau)
        return (s,mu) if return_mu else s
    return fp
def rankavg(*ps):
    return np.mean([rankdata(p)/len(p) for p in ps],axis=0)

def enrich(D):
    """feat2 결합 + 장기 KR 이력·상대 점유율 피처."""
    P=L.load(); kr=P['A']['n'][:,0,:]; tot=P['A']['n'].sum(1)
    f2=pd.read_parquet(L.os.path.join(L.CACHE,'feat2.parquet')).drop(columns=['sub'])
    D=D.merge(f2,on=['ksic','mg','t0'],how='left')
    idx=np.searchsorted(P['cells'],D.cell.values)
    long=np.full(len(D),np.nan); dd=np.full(len(D),np.nan); tl=np.full(len(D),np.nan)
    for t0,g in D.groupby('t0'):
        a=max(L.Y0,t0-8); ii=idx[g.index]
        hist=kr[ii, a-L.Y0:t0-3-L.Y0+1]
        long[g.index]=np.log1p(hist.mean(1))
        dd[g.index]=np.log1p(kr[ii, a-L.Y0:t0-L.Y0+1].max(1))-np.log1p(kr[ii,t0-2-L.Y0:t0-L.Y0+1].mean(1))
        tl[g.index]=np.log1p(tot[ii, a-L.Y0:t0-3-L.Y0+1].mean(1))
    D['kr_long']=long; D['kr_drawdown']=dd; D['kr_vs_long']=D.kr_lvl-np.log(3)-D.kr_long
    D['tot_vs_long']=D.tot_lvl-np.log(3)-tl
    D['kr_share_rel_ksic']=D.kr_share-D.groupby(['t0','ksic']).kr_share.transform('median')
    D['kr_share_rel_sub']=D.kr_share-D.sub_krsh
    D['kr_share_rel_mg']=D.kr_share-D.groupby(['t0','mg']).kr_share.transform('median')
    return D
NEW=['app_mom','app_size','big_sh','top1_sh','nb_kmom','nb_tmom','nb_w','nb_deg','sub_kmom','sub_fmom','sub_krsh',
     'kr_long','kr_drawdown','kr_vs_long','tot_vs_long','kr_share_rel_ksic','kr_share_rel_sub','kr_share_rel_mg']
ALL2=ALL+NEW
XP=dict(max_depth=2,eta=0.05,min_child_weight=5)
from scipy.stats import norm
def rank_norm(D, cols):
    """t0 단면 내 순위→정규분위 변환(결측 유지). 같은 t0의 피처만 사용하므로 누수 없음."""
    D=D.copy()
    for c in cols:
        r=D.groupby('t0')[c].rank(pct=True)
        n=D.groupby('t0')[c].transform('count')
        D[c+'_rn']=norm.ppf((r*n-0.5)/n)
    return D
PRUNE=[c for c in MOM+FOR+NEW if not (c.startswith('JP') or c.startswith('CN') or c in ['app_mom','app_size','big_sh','top1_sh','sub_kmom','sub_fmom','sub_krsh'])]
def multi_thr(fp_maker, tops=(0.05,0.10,0.20,0.30)):
    """여러 상위 비율 레이블로 각각 학습 후 순위평균 (레이블 잡음 평활)."""
    def fp(tr,te):
        ps=[]
        for q in tops:
            t2=tr.copy(); t2['yb']=0
            for t,g in t2.groupby('t0'):
                t2.loc[g.index,'yb']=L.breakout(g.y.values,top=q,tie=(g.kr_fut-g.kr_W).values)
            ps.append(fp_maker(t2,te))
        return rankavg(*ps)
    return fp
def logit_w(cols,half=4):
    def fp(tr,te):
        w=0.5**((tr.t0.max()-tr.t0)/half)
        m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=3000))
        m.fit(tr[cols].values,tr.yb.values,logisticregression__sample_weight=w.values); return m.predict_proba(te[cols].values)[:,1]
    return fp
def te_enc(tr,te,key,prior=20):
    g=tr.groupby(key).yb.agg(['sum','count']); gm=tr.yb.mean()
    enc=(g['sum']+prior*gm)/(g['count']+prior)
    return tr[key].map(enc).fillna(gm).values, te[key].map(enc).fillna(gm).values
def nbfp_het(cols,rounds=400):
    """음이항 꼬리확률, 분산(alpha)을 KR 과거규모 구간별로 따로 추정."""
    pr=dict(objective='count:poisson',max_depth=3,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,nthread=2,verbosity=0,max_delta_step=0.7)
    bins=[-1,0,5,20,100,1e12]
    def fp(tr,te):
        d=xgb.DMatrix(tr[cols].values,label=tr.kr_fut.values); m=xgb.train(pr,d,rounds)
        mu_tr=m.predict(d); mu=m.predict(xgb.DMatrix(te[cols].values)); tau=tau_of(tr)
        btr=np.digitize(tr.kr_W.values,bins); bte=np.digitize(te.kr_W.values,bins); s=np.zeros(len(te))
        for b in np.unique(bte):
            a=fit_alpha(tr.kr_fut.values[btr==b],mu_tr[btr==b]) if (btr==b).sum()>50 else fit_alpha(tr.kr_fut.values,mu_tr)
            s[bte==b]=nb_tail(te.kr_W.values[bte==b],mu[bte==b],a,tau)
        return s
    return fp
def oof_origin(tr, fitpred, gap=2):
    """학습셋 내부 OOF: 원점 o 예측은 |o'-o|>gap 원점만으로 학습(결과창 중첩 배제)."""
    out=np.full(len(tr),np.nan)
    for o in sorted(tr.t0.unique()):
        m=(tr.t0==o).values; fitm=(np.abs(tr.t0-o)>gap).values
        if fitm.sum()<500: continue
        out[m]=fitpred(tr[fitm],tr[m])
    return out
def best_fp(extra_nb=True):
    """현재 최선(v22) 파이프라인."""
    def fp(tr,te):
        tr=tr.copy(); te=te.copy()
        tr['te_ksic']=oof_origin(tr, lambda a,b: te_enc(a,b,'ksic')[1]); te['te_ksic']=te_enc(tr,te,'ksic')[1]
        ps=[logit(PRUNE+['te_ksic'])(tr,te), xgbc(ALL2+['te_ksic'],XP,rounds=300)(tr,te)]
        if extra_nb: ps.append(nbfp(ALL2)(tr,te))
        return rankavg(*ps)
    return fp
def relabel(D, mask=None, ycol='y', tiecol=None):
    D=D if mask is None else D[mask]
    D=D.copy(); D['yb']=0
    for t,g in D.groupby('t0'):
        tie=g[tiecol].values if tiecol else (g.kr_fut-g.kr_W).values
        D.loc[g.index,'yb']=L.breakout(g[ycol].values,tie=tie)
    return D
def add_lead(D):
    """국가별 선행 시차 분석 결과 반영: 시차3 성장(US/JP/EP), 4개국 동일가중 평균 성장, 성장국 수(폭)."""
    D=D.reset_index(drop=True).copy()
    P_=L.load(); n=P_['A']['n']; idx=np.searchsorted(P_['cells'],D.cell.values)
    for j,c in [(1,'US'),(3,'JP'),(4,'EP'),(2,'CN')]:
        out=np.full(len(D),np.nan)
        for t,g in D.groupby('t0'):
            tt=t-3
            if tt-5<L.Y0: continue
            x=n[:,j,:]; out[g.index]=np.log((L.wsum(x,tt,-2,0)+1)/(L.wsum(x,tt-3,-2,0)+1))[idx[g.index]]
        D[f'{c}_mom_l3']=out
    M=D[['US_mom','CN_mom','JP_mom','EP_mom']]
    D['fx_eq']=M.mean(1); D['fx_breadth']=(M>0).sum(1)
    D['fx_eq_l1']=D[['US_mom_l1','CN_mom_l1','JP_mom_l1','EP_mom_l1']].mean(1)
    D['fx_eq_l2']=D[['US_mom_l2','CN_mom_l2','JP_mom_l2','EP_mom_l2']].mean(1)
    D['fx_eq_l3']=D[['US_mom_l3','JP_mom_l3','EP_mom_l3']].mean(1)
    D['gap_eq_kr']=D.fx_eq-D.kr_mom
    return D
LEAD=['US_mom_l3','JP_mom_l3','EP_mom_l3','CN_mom_l3','fx_eq','fx_breadth','fx_eq_l1','fx_eq_l2','fx_eq_l3','gap_eq_kr']
def best_fp2(lcols,xcols,nbcols=None):
    def fp(tr,te):
        tr=tr.copy(); te=te.copy()
        tr['te_ksic']=oof_origin(tr, lambda a,b: te_enc(a,b,'ksic')[1]); te['te_ksic']=te_enc(tr,te,'ksic')[1]
        ps=[logit(lcols+['te_ksic'])(tr,te), xgbc(xcols+['te_ksic'],XP,rounds=300)(tr,te)]
        if nbcols: ps.append(nbfp(nbcols)(tr,te))
        return rankavg(*ps)
    return fp
