"""T3 3라운드 공용 라이브러리 (2라운드 r2lib 사본 + 변형 스위치).
변형 T3R3_VAR:
  app = 출원연도 기준(2라운드와 동일). 캐시 R2/cache_dev(읽기 전용), 연도 2000~2019, 레이블 결과창 ≤2019 (F=2020).
  pub = 공개연도 기준(3라운드 최종시험 변형). 캐시 R3/cache_pub, 공개연도 2000~2021, 레이블 결과창 ≤2021 (🔒 F'=2022).
     셀 집계의 year = 공개연도(least(pub_date, rgst_date)) → t0 피처는 t0년 말까지 공개된 문서만, 레이블은 KR 공개연도 t0+1..t0+3 성장.
"""
import numpy as np, pandas as pd, duckdb, os
HERE=os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

VAR=os.environ.get('T3R3_VAR','pub')
if VAR=='app':
    CACHE=os.path.join(HERE,'..','R2','cache_dev'); Y0,Y1=2000,2019; LABEL_MAX_YEAR=2019; F_YEAR=2020
    LOCAL=os.path.join(HERE,'cache_app_local')
else:
    CACHE=os.path.join(HERE, os.environ.get('T3R3_CACHE','cache_pub')); Y0=2000
    Y1=int(os.environ.get('T3R3_MAXYEAR','2021')); LABEL_MAX_YEAR=int(os.environ.get('T3R3_LABEL_MAX','2021')); F_YEAR=2022
    LOCAL=CACHE
    if (Y1>=F_YEAR or LABEL_MAX_YEAR>=F_YEAR) and os.environ.get('T3R3_FINAL_UNLOCK')!='1':
        raise RuntimeError('🔒 공개연도 2022 이후 자료는 최종시험 전용: T3R3_FINAL_UNLOCK=1 필요')
os.makedirs(LOCAL,exist_ok=True)
CCS=['KR','US','CN','JP','EP']
DEV_T0=list(range(2010,2017)) if VAR=='app' else list(range(2010,2019))

_cache={}
def load():
    if 'P' in _cache: return _cache['P']
    df=pd.read_parquet(os.path.join(CACHE,'panel_ccy.parquet'))
    assert df.year.max()<=Y1
    df['cell']=df.ksic+'|'+df.mg
    cells=np.array(sorted(df.cell.unique()))
    cid={c:i for i,c in enumerate(cells)}
    df['ci']=df.cell.map(cid).values
    T=Y1-Y0+1
    cols=['n','n_app','n_prio_for','n_newapp_apps','n_newapp','n_newk_apps','hhi','n_multi','n_newpair_apps','n_newpair','n_newword_apps','n_newword']
    A={}
    for col in cols:
        arr=np.zeros((len(cells),len(CCS),T),dtype=np.float32)
        sub=df[['ci','cc','year',col]].dropna()
        arr[sub.ci.values, sub.cc.map({c:i for i,c in enumerate(CCS)}).values, sub.year.values-Y0]=sub[col].values
        A[col]=arr
    ks=np.array([c.split('|')[0] for c in cells]); mg=np.array([c.split('|')[1] for c in cells])
    P=dict(cells=cells,ksic=ks,mg=mg,A=A)
    _cache['P']=P; return P

def wsum(arr, t0, a, b):
    """arr[..., years t0+a .. t0+b] 합 (a<=b, 연도 인덱스). 범위 밖 연도는 오류."""
    y0=t0+a-Y0; y1=t0+b-Y0
    assert y0>=0 and y1<=Y1-Y0, (t0,a,b)
    return arr[..., y0:y1+1].sum(-1)

def labels(P, t0):
    assert t0+3<=LABEL_MAX_YEAR, '🔒 홀드아웃/미관측 레이블 접근 금지'
    kr=P['A']['n'][:,0,:]
    fut=wsum(kr,t0,1,3); past=wsum(kr,t0,-2,0)
    return np.log((fut+1)/(past+1))

def eligible(P,t0,min_total):
    tot=wsum(P['A']['n'].sum(1),t0,-2,0)
    return tot>=min_total

def breakout(y, top=0.10, tie=None):
    """상위 top 비율 = breakout. 동률은 tie(절대 증가량 등)로 깨고, 그래도 남으면 정렬 순서(안정 정렬)로 k개만."""
    k=max(1,int(round(top*len(y))))
    if tie is None: tie=np.zeros_like(y)
    order=np.lexsort((-tie,-y))
    yb=np.zeros(len(y),dtype=int); yb[order[:k]]=1
    return yb

def evaluate(y, p, yb=None):
    """y: 연속 성장률(적격 셀, 한 t0), p: 점수. yb: breakout(없으면 y로 계산)"""
    if yb is None: yb=breakout(y)
    k=max(1,int(round(0.10*len(y))))
    top=np.argsort(-p)[:k]
    return dict(auc=roc_auc_score(yb,p), prauc=average_precision_score(yb,p), base_rate=yb.mean(),
                p_at10=yb[top].mean(), rho=spearmanr(y,p).statistic, n=len(y), n_evt=int(yb.sum()))

# ---------------- 피처 ----------------
def _g(x, t0, lag=0, w=3):
    """log((최근 w년 합+1)/(그 이전 w년 합+1)), 최근창 = t0-lag-w+1..t0-lag. 범위 밖이면 NaN."""
    a=t0-lag-2*w+1
    if a<Y0: return np.full(x.shape[0],np.nan,dtype=np.float32)
    return np.log((wsum(x,t0-lag,-w+1,0)+1)/(wsum(x,t0-lag-w,-w+1,0)+1))

def _slope(x, t0, w=5):
    a=max(Y0,t0-w+1); yrs=np.arange(a,t0+1)
    Z=np.log1p(x[:, a-Y0:t0-Y0+1]); tt=yrs-yrs.mean()
    return (Z*tt).sum(1)/(tt**2).sum()

def _grp_mom(keys, x, t0, lag=0):
    """그룹(ksic 또는 mg) 합계 모멘텀을 셀에 붙임 (자기 셀 포함)."""
    df=pd.DataFrame(x); df['k']=keys
    G=df.groupby('k').sum()
    gm=_g(G.values, t0, lag)
    return pd.Series(gm,index=G.index).reindex(keys).values

def feats(P, t0, min_total=10):
    A=P['A']; n=A['n']; kr=n[:,0,:]; tot=n.sum(1); fx=n[:,1:,:].sum(1)
    W=lambda arr: wsum(arr,t0,-2,0)
    F={}
    F['kr_mom']=_g(kr,t0,0)
    F['kr_mom_l1']=_g(kr,t0,1); F['kr_mom_l2']=_g(kr,t0,2)
    F['kr_mom5']=_g(kr,t0,0,w=2) if t0-3>=Y0 else np.nan
    F['kr_g1']=np.log((kr[:,t0-Y0]+1)/(kr[:,t0-1-Y0]+1))
    F['kr_acc']=np.log1p(kr[:,t0-Y0])-2*np.log1p(kr[:,t0-1-Y0])+np.log1p(kr[:,t0-2-Y0])
    F['kr_lvl']=np.log1p(W(kr)); F['kr_last']=np.log1p(kr[:,t0-Y0])
    F['kr_slope5']=_slope(kr,t0,5)
    F['kr_zero_yrs']=(kr[:,max(0,t0-4-Y0):t0-Y0+1]==0).sum(1)   # (2라운드: t0<2004 인덱스 음수 방지, t0≥2004 값 동일)
    F['tot_lvl']=np.log1p(W(tot)); F['fx_mom']=_g(fx,t0,0); F['fx_mom_l1']=_g(fx,t0,1)
    F['tot_mom']=_g(tot,t0,0)
    for j,c in enumerate(CCS[1:],start=1):
        x=n[:,j,:]
        F[f'{c}_mom']=_g(x,t0,0); F[f'{c}_mom_l1']=_g(x,t0,1); F[f'{c}_mom_l2']=_g(x,t0,2)
        F[f'{c}_lvl']=np.log1p(W(x)); F[f'{c}_g1']=np.log((x[:,t0-Y0]+1)/(x[:,t0-1-Y0]+1))
        F[f'{c}_slope5']=_slope(x,t0,5)
    F['kr_share']=np.log((W(kr)+1)/(W(tot)+1))
    F['kr_share_chg']=F['kr_share']-np.log((wsum(kr,t0,-5,-3)+1)/(wsum(tot,t0,-5,-3)+1)) if t0-5>=Y0 else np.nan
    F['gap_fx_kr']=F['fx_mom']-F['kr_mom']
    # 구조 (KR / 전체)
    def share(col, j=0, allcc=False):
        num=A[col].sum(1) if allcc else A[col][:,j,:]
        den=tot if allcc else n[:,j,:]
        return W(num)/np.maximum(W(den),1)
    F['kr_newapp_sh']=share('n_newapp_apps'); F['kr_newk_sh']=share('n_newk_apps')
    F['kr_prio_for_sh']=share('n_prio_for'); F['kr_multi_sh']=share('n_multi')
    F['kr_newpair_sh']=share('n_newpair_apps'); F['kr_newword_sh']=share('n_newword_apps')
    F['all_newapp_sh']=share('n_newapp_apps',allcc=True); F['all_newpair_sh']=share('n_newpair_apps',allcc=True)
    F['kr_napp']=np.log1p(W(A['n_app'][:,0,:])); F['kr_newapp_n']=np.log1p(W(A['n_newapp'][:,0,:]))
    F['kr_hhi']=np.where(W(kr)>0, (A['hhi'][:,0,t0-2-Y0:t0-Y0+1]*kr[:,t0-2-Y0:t0-Y0+1]).sum(1)/np.maximum(W(kr),1), np.nan)
    F['all_newapp_n']=np.log1p(W(A['n_newapp'].sum(1)))
    # 국외 출원인의 한국 유입(우선권 해외) 모멘텀
    pf=A['n_prio_for'][:,0,:]; dom=kr-pf
    F['kr_forin_mom']=_g(pf,t0,0); F['kr_dom_mom']=_g(dom,t0,0)
    # 그룹 모멘텀
    F['ksic_kr_mom']=_grp_mom(P['ksic'],kr,t0); F['mg_kr_mom']=_grp_mom(P['mg'],kr,t0)
    F['mg_tot_mom']=_grp_mom(P['mg'],tot,t0); F['mg_fx_mom']=_grp_mom(P['mg'],fx,t0)
    F['ksic_tot_mom']=_grp_mom(P['ksic'],tot,t0)
    df=pd.DataFrame({k:(np.full(len(kr),v) if np.isscalar(v) else v) for k,v in F.items()})
    df.insert(0,'cell',P['cells']); df.insert(1,'ksic',P['ksic']); df.insert(2,'mg',P['mg']); df.insert(3,'t0',t0)
    df['tot_W']=W(tot); df['kr_W']=W(kr)
    df=df[df.tot_W>=min_total].reset_index(drop=True)
    if t0+3<=LABEL_MAX_YEAR:
        idx=np.searchsorted(P['cells'],df.cell.values)
        fut=wsum(kr,t0,1,3)[idx]
        df['kr_fut']=fut; df['y']=np.log((fut+1)/(df.kr_W+1))
    return df

def get_feats(t0, min_total=10):
    fn=os.path.join(CACHE,f'feat_{t0}.parquet')
    if os.path.exists(fn): return pd.read_parquet(fn)
    fn=os.path.join(LOCAL,f'feat_{t0}.parquet')
    if os.path.exists(fn): return pd.read_parquet(fn)
    df=feats(load(),t0,min_total); df.to_parquet(fn); return df

def add_labels(df, min_total):
    """적격 필터 + t0별 breakout(상위10%, 동률은 절대증가량으로)"""
    d=df[df.tot_W>=min_total].copy()
    d['yb']=0
    for t,g in d.groupby('t0'):
        d.loc[g.index,'yb']=breakout(g.y.values, tie=(g.kr_fut-g.kr_W).values)
    return d
