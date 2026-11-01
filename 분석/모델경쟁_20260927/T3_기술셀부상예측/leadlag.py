# 국가별 선행 시차: 기준모형(KR 모멘텀·수준·장기이력) + 국가 c의 k년 전 3년성장(1개 피처) 추가 시 (1) 표준화 계수 (2) 검증 AUC 증분
from models import *
import statsmodels.api as sm
D=enrich(H.data()); DA=relabel(D, D.kr_W>=5)
P_=L.load(); n=P_['A']['n']
def lagg(Dx, j, k, w=3):
    out=np.full(len(Dx),np.nan); idx=np.searchsorted(P_['cells'],Dx.cell.values)
    for t,g in Dx.groupby('t0'):
        tt=t-k
        if tt-2*w+1<L.Y0: continue
        x=n[:,j,:]; out[g.index]=np.log((L.wsum(x,tt,-w+1,0)+1)/(L.wsum(x,tt-w,-w+1,0)+1))[idx[g.index]]
    return out
BASEF=['kr_mom','kr_mom_l1','kr_lvl','kr_last','kr_long','kr_drawdown','kr_zero_yrs','tot_lvl']
rows=[]
for name,Dx in [('주타깃',D),('보조A(KR≥5)',DA)]:
    Dx=Dx.reset_index(drop=True).copy()
    R0,_=H.rolling(logit(BASEF),Dx); a0=H.summarize(R0)['auc']
    for j,c in enumerate(L.CCS[1:],start=1):
        for k in range(0,4):
            f=f'{c}_g_l{k}'; Dx[f]=lagg(Dx,j,k)
            R,_=H.rolling(logit(BASEF+[f]),Dx); a=H.summarize(R)['auc']
            d=Dx[Dx.t0>=2005+k].dropna(subset=[f]); X=d[BASEF+[f]].fillna(d[BASEF+[f]].median()); X=(X-X.mean())/X.std()
            m=sm.Logit(d.yb.values,sm.add_constant(X.values)).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.mg)[0]})
            rows.append(dict(target=name,cc=c,lag=k,coef=m.params[-1],z=m.tvalues[-1],auc_base=a0,auc_add=a,delta_auc=a-a0,n=len(d)))
            print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('leadlag_summary.csv',index=False,encoding='utf-8-sig')
