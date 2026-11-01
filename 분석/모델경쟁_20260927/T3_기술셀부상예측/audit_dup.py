# 감사: 같은 메인그룹의 서로 다른 KSIC 셀이 결과(과거·미래 KR 건수)를 그대로 공유하는 '복제 표본' 정도
import numpy as np, pandas as pd, harness as H, t3lib as L
from sklearn.metrics import roc_auc_score
D=H.data(); V=D[D.t0.isin(L.VAL_T0)].copy()
P=L.load(); n=P['A']['n']; idx=np.searchsorted(P['cells'],V.cell.values)
# 5개국 전체 시계열(2000~t0+3)이 완전히 같은 셀 = 사실상 같은 출원 집합
for t,g in V.groupby('t0'):
    ii=idx[g.index.values - 0] if False else np.searchsorted(P['cells'],g.cell.values)
    sig=[hash(n[i,:,:t+3-L.Y0+1].tobytes()) for i in ii]
    s=pd.Series(sig); dup=s.duplicated(keep=False)
    g2=g.assign(sig=sig)
    kr_sig=g2.groupby(['mg','kr_W','kr_fut']).cell.transform('size')
    print(t,'셀',len(g),'메인그룹',g.mg.nunique(),'전체시계열 동일 셀 비중 %.3f'%dup.mean(),'같은 mg 내 KR과거·미래 동일 비중 %.3f'%(kr_sig>1).mean(),
          '사건 중 %.3f'%(kr_sig[g.yb==1]>1).mean())
# v22 예측으로: 셀 가중 AUC vs 메인그룹 1/m 가중 AUC vs 중복 제거(전체시계열 동일 셀 1개만) AUC
Pr=pd.read_parquet('preds/v22.parquet'); P0=pd.read_parquet('preds/v0.parquet')
for name,PP in [('v22',Pr),('v0',P0)]:
    PP=PP.merge(V[['cell','t0','mg','kr_W','kr_fut','tot_W']],on=['cell','t0'])
    out=[]
    for t,g in PP.groupby('t0'):
        w=1/g.groupby('mg').cell.transform('size')
        dd=g.drop_duplicates(['mg','kr_W','kr_fut','tot_W'])
        out.append((roc_auc_score(g.yb,g.p),roc_auc_score(g.yb,g.p,sample_weight=w),roc_auc_score(dd.yb,dd.p),len(dd)))
    o=np.array(out); print(name,'셀AUC %.4f  mg가중AUC %.4f  중복제거AUC %.4f (n %s)'%(o[:,0].mean(),o[:,1].mean(),o[:,2].mean(),o[:,3]))
