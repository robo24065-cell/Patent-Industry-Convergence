"""2단계(감사·개선) 공용: 중립 동률깨기 레이블, 추가 원점(선택 미사용) 평가."""
import numpy as np, pandas as pd, hashlib, os
import t3lib as L, harness as H
from models import *
EXTRA_T0=[2010,2012,2014]   # 1단계 선택에 쓰지 않은 원점(결과창 ≤2017) — 선택 편향 점검용
def hkey(cells):
    return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8],16) for c in cells],dtype=np.int64)
def breakout_neutral(y, tie, key, top=0.10):
    """상위 10%: y → 절대증가 → md5(cell) (셀 정렬순서=KSIC 오름차순 의존 제거)"""
    k=max(1,int(round(top*len(y)))); o=np.lexsort((key,-tie,-y)); yb=np.zeros(len(y),dtype=int); yb[o[:k]]=1; return yb
def add_labels_neutral(df,min_total=H.MIN_TOTAL):
    d=df[df.tot_W>=min_total].copy(); d['yb']=0
    for t,g in d.groupby('t0'):
        d.loc[g.index,'yb']=breakout_neutral(g.y.values,(g.kr_fut-g.kr_W).values,hkey(g.cell.values))
    return d
def data(neutral=True,t0s=range(2005,2016),min_total=H.MIN_TOTAL):
    df=pd.concat([L.get_feats(t) for t in t0s],ignore_index=True)
    d=add_labels_neutral(df,min_total) if neutral else L.add_labels(df,min_total)
    return enrich(d.reset_index(drop=True)).reset_index(drop=True)
def base_fp(tr,te):
    m=LogisticRegression(C=1e6,max_iter=1000).fit(tr[['kr_mom']].values,tr.yb.values); return m.predict_proba(te[['kr_mom']].values)[:,1]
def run(fp,D,val=L.VAL_T0):
    return H.rolling(fp,D,val=val)
def both(fp,D,name=''):
    R,P=run(fp,D); Rx,_=run(fp,D,EXTRA_T0)
    s=H.summarize(R); sx=H.summarize(Rx)
    print(f'{name}: 검증 {s["auc"]:.4f} ({s["auc_by_t0"]}) | 추가원점 {sx["auc"]:.4f} ({sx["auc_by_t0"]})',flush=True)
    return R,P,Rx

# ---- [2단계] 국가 간 우선권 흐름 피처 (feat3_panel, t0 이하 연도만) ----
FLOW=['out_kr_lvl','out_kr_mom','out_kr_ratio','out_kr_ratio_chg','out_us_kr_mom','in_US_mom','in_JP_mom','in_EU_mom','in_CN_mom',
      'in_US_sh','in_JP_sh','in_EU_sh','in_CN_sh']
_f3={}
def _flow_arrays():
    if 'A' in _f3: return _f3['A']
    P=L.load(); f=pd.read_parquet(os.path.join(L.CACHE,'feat3_panel.parquet'))
    assert f.year.max()<=L.Y1
    f['cell']=f.ksic+'|'+f.mg; f=f[f.cell.isin(set(P['cells']))]
    ci=np.searchsorted(P['cells'],f.cell.values); T=L.Y1-L.Y0+1; ccs={c:i for i,c in enumerate(L.CCS)}
    A={}
    for col in ['n_pKR','n_pUS','n_pJP','n_pCN','n_pEU']:
        arr=np.zeros((len(P['cells']),5,T),dtype=np.float32); arr[ci,f.cc.map(ccs).values,f.year.values-L.Y0]=f[col].values; A[col]=arr
    _f3['A']=A; return A
def add_flow(D):
    D=D.reset_index(drop=True).copy(); A=_flow_arrays(); P=L.load(); idx=np.searchsorted(P['cells'],D.cell.values)
    out=A['n_pKR'][:,1:,:].sum(1); usout=A['n_pKR'][:,1,:]; kr=P['A']['n'][:,0,:]
    for c in FLOW: D[c]=np.nan
    for t,g in D.groupby('t0'):
        ii=idx[g.index]; W=lambda x: L.wsum(x,t,-2,0)[ii]
        D.loc[g.index,'out_kr_lvl']=np.log1p(W(out)); D.loc[g.index,'out_kr_mom']=L._g(out,t,0)[ii]
        r=np.log((W(out)+1)/(W(kr)+1)); D.loc[g.index,'out_kr_ratio']=r
        if t-5>=L.Y0: D.loc[g.index,'out_kr_ratio_chg']=r-np.log((L.wsum(out,t,-5,-3)[ii]+1)/(L.wsum(kr,t,-5,-3)[ii]+1))
        D.loc[g.index,'out_us_kr_mom']=L._g(usout,t,0)[ii]
        for c,col in [('US','n_pUS'),('JP','n_pJP'),('EU','n_pEU'),('CN','n_pCN')]:
            x=A[col][:,0,:]; D.loc[g.index,f'in_{c}_mom']=L._g(x,t,0)[ii]; D.loc[g.index,f'in_{c}_sh']=W(x)/np.maximum(W(kr),1)
    return D
def ens(lcols,xcols,nbcols,te=False,w=(1,1,1)):
    def fp(tr,te_):
        ps=[logit(lcols)(tr,te_), xgbc(xcols,XP,rounds=300)(tr,te_)]
        if nbcols: ps.append(nbfp(nbcols)(tr,te_))
        R=[rankdata(p)/len(p) for p in ps]; return np.average(R,axis=0,weights=w[:len(R)])
    return fp

# ---- [2단계] 산업 간 확산 피처: 같은 메인그룹이 몇 개 KSIC에서 KR 출원되는가 (t0 이하) ----
DIFF=['mg_nksic','mg_nksic_chg','mg_kr_lvl_all','cell_sh_in_mg','cell_sh_in_mg_chg']
def add_diff(D):
    D=D.reset_index(drop=True).copy(); P=L.load(); kr=P['A']['n'][:,0,:]; mg=P['mg']; idx=np.searchsorted(P['cells'],D.cell.values)
    for c in DIFF: D[c]=np.nan
    for t,g in D.groupby('t0'):
        w=L.wsum(kr,t,-2,0); wp=L.wsum(kr,t,-5,-3) if t-5>=L.Y0 else None
        s=pd.DataFrame({'mg':mg,'w':w,'pos':(w>0).astype(int)})
        if wp is not None: s['wp']=wp; s['posp']=(wp>0).astype(int)
        G=s.groupby('mg').sum(); ii=idx[g.index]; mgi=mg[ii]
        D.loc[g.index,'mg_nksic']=G.pos.reindex(mgi).values
        D.loc[g.index,'mg_kr_lvl_all']=np.log1p(G.w.reindex(mgi).values)
        sh=(w[ii]+0.5)/(G.w.reindex(mgi).values+1); D.loc[g.index,'cell_sh_in_mg']=np.log(sh)
        if wp is not None:
            D.loc[g.index,'mg_nksic_chg']=G.pos.reindex(mgi).values-G.posp.reindex(mgi).values
            D.loc[g.index,'cell_sh_in_mg_chg']=np.log(sh)-np.log((wp[ii]+0.5)/(G.wp.reindex(mgi).values+1))
    return D
def soft_xgb(cols,s=0.25,rounds=300):
    """부드러운 레이블: sigmoid((y - t0별 상위10% 경계)/s) — 경계 근처 계수잡음 완화"""
    def fp(tr,te):
        q=tr.groupby('t0').y.transform(lambda v: np.quantile(v,0.9)); lab=1/(1+np.exp(-(tr.y-q)/s))
        p=dict(max_depth=2,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,nthread=2,verbosity=0,objective='binary:logistic')
        m=xgb.train(p,xgb.DMatrix(tr[cols].values,label=lab.values),rounds); return m.predict(xgb.DMatrix(te[cols].values))
    return fp
def tr_rank(tr,p):
    return pd.Series(p,index=tr.index).groupby(tr.t0.values).rank(pct=True).values
def stack(comps, meta_extra=('kr_lvl',)):
    """성분 점수의 원점-OOF(±2원점 제외)로 로지스틱 메타학습 (성분 순위 + 규모 상호작용)"""
    def fp(tr,te):
        O=np.column_stack([oof_origin(tr,c) for c in comps])
        O=np.column_stack([tr_rank(tr,O[:,j]) for j in range(O.shape[1])])
        T=np.column_stack([rankdata(c(tr,te))/len(te) for c in comps])
        ok=~np.isnan(O).any(1)
        if ok.sum()<1000:
            print('  meta: OOF 부족 → 동일 순위평균'); return T.mean(1)
        def X(R,d): 
            ex=[d[e].values for e in meta_extra]; return np.column_stack([R]+[R*e[:,None] for e in ex]+ex)
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000)).fit(X(O[ok],tr[ok]),tr.yb.values[ok])
        print('  meta n',ok.sum(),'coef',np.round(m[-1].coef_.ravel(),2))
        return m.predict_proba(X(T,te))[:,1]
    return fp
def segmented(fp_maker, thr=5):
    """규모별 분리모형: KR 과거 3년 ≤thr / >thr 따로 학습, 확률 그대로 합침"""
    def fp(tr,te):
        out=np.zeros(len(te))
        for m_tr,m_te in [(tr.kr_W<=thr, te.kr_W<=thr),(tr.kr_W>thr, te.kr_W>thr)]:
            if m_te.sum()==0: continue
            out[m_te.values]=fp_maker(tr[m_tr.values],te[m_te.values])
        return out
    return fp
