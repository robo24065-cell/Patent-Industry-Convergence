# 감사: breakout 경계 동률(같은 y·같은 절대증가)이 셀 정렬순서(KSIC 오름차순)로 깨지는지 점검
import numpy as np, pandas as pd, harness as H, t3lib as L
D=H.data()
for t,g in D.groupby('t0'):
    k=int(round(0.1*len(g))); inc=(g.kr_fut-g.kr_W).values; y=g.y.values
    o=np.lexsort((-inc,-y)); ky,ki=y[o[k-1]],inc[o[k-1]]
    tie=(np.isclose(y,ky))&(inc==ki); pos=g.yb.values==1
    ks=g.ksic.values
    print(t,'n',len(g),'k',k,'경계값 y=%.3f inc=%d'%(ky,ki),'경계동률셀',tie.sum(),'그중 양성',(tie&pos).sum(),
          '양성 KSIC 중앙',np.median(ks[tie&pos].astype(int)) if (tie&pos).any() else None,'음성 KSIC 중앙',np.median(ks[tie&~pos].astype(int)) if (tie&~pos).any() else None,
          'past/fut 예', g.kr_W.values[tie][:1], g.kr_fut.values[tie][:1])
# KSIC별 양성률이 정렬순서에 좌우되는 정도: KSIC 순위와 양성률 상관
from scipy.stats import spearmanr
r=D.groupby('ksic').yb.mean(); print('KSIC코드 순위 vs 양성률 Spearman', spearmanr(r.index.astype(int), r.values))
