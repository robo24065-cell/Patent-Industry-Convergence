import os, sys, json, numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
V = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T3\verify_1"
R = pd.read_csv(os.path.join(V,'v1_by_origin.csv')); D = R[R.origin<=2018]; F = R[R.origin==2021].iloc[0]
out={}
def pi(col, x0=2021, xs=None):
    d = D if xs is None else D[D.origin.isin(xs)]
    x, y = d.origin.values.astype(float), d[col].values; n=len(x)
    b = np.polyfit(x,y,1); res = y-np.polyval(b,x); s=np.sqrt((res**2).sum()/(n-2))
    se = s*np.sqrt(1+1/n+(x0-x.mean())**2/((x-x.mean())**2).sum()); tq=stats.t.ppf(0.975,n-2)
    f=np.polyval(b,x0); return dict(slope=b[0],fit=f,resid_sd=s,pi_lo=f-tq*se,pi_hi=f+tq*se,final=float(F[col]),z=(F[col]-f)/se, dev_min=y.min(), dev_max=y.max(),
        outside_range=bool(not (y.min()<=F[col]<=y.max())), outside_pi=bool(not (f-tq*se<=F[col]<=f+tq*se)))
for c in ['sab_rate','pos_krW_le1','pos_krW0','pos_D_le3','sab_rate_D5','sab_rate_krW_ge2','sab_rate_krW_le1','elig_krW_le1','sab100_rate']:
    out[c]=pi(c); print(c, {k:(round(v,4) if isinstance(v,float) else v) for k,v in out[c].items()})
out['sab_rate_fit2012_18']=pi('sab_rate',xs=range(2012,2019)); print('2012-18',out['sab_rate_fit2012_18'])
# 구성 표준화: 2018 층별 SAB 비율 × 최종 층 구성
r18=R[R.origin==2018].iloc[0]
out['composition_std']=dict(final_mix_dev2018_rates=F.pos_krW_le1*r18.sab_rate_krW_le1+(1-F.pos_krW_le1)*r18.sab_rate_krW_ge2,
    final_mix_devmean_rates=F.pos_krW_le1*D.sab_rate_krW_le1.mean()+(1-F.pos_krW_le1)*D.sab_rate_krW_ge2.mean(),
    dev2018_mix_final_rates=r18.pos_krW_le1*F.sab_rate_krW_le1+(1-r18.pos_krW_le1)*F.sab_rate_krW_ge2)
print(out['composition_std'])
json.dump(out,open(os.path.join(V,'v2_trend.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
