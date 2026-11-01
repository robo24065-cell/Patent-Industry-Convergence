# 독립 재계산: 정렬규칙과 무관한 동점 범위(min/max) 포함
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
MC="C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/"
OUT="C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C01/C01-T2/verify_0/indep_results.json"
def tie_range(y,p,k):
    y=np.asarray(y).astype(int); p=np.asarray(p,dtype=float)
    ps=np.sort(p)[::-1]; thr=ps[k-1]
    above=p>thr; tie=p==thr
    na=above.sum(); need=k-na
    ha=y[above].sum(); ht=y[tie].sum(); nt=tie.sum()
    lo=ha+max(0,need-(nt-ht)); hi=ha+min(need,ht)
    o_q=np.argsort(-p)[:k]; o_s=np.argsort(-p,kind='stable')[:k]; o_m=np.argsort(-p,kind='mergesort')[:k]
    return dict(k=int(k),min=int(lo),max=int(hi),n_tie=int(nt),tie_pos=int(ht),need_from_tie=int(need),
                quick=int(y[o_q].sum()),stable=int(y[o_s].sum()),merge=int(y[o_m].sum()))
R={}
spec=[('N9','N9_정부RnD과제_특허성과예측/final_preds_r6.parquet','y',['p_champ','p_M0_orig','p_pilot_L2','p_M0_L2','p_strong_L2','p_strong_orig','p_pilot_orig']),
('N6','N6_정부RnD기업_첫특허전환/final_preds_r5.parquet','y',['p_champ','p_m0']),
('N7','N7_특허기업_정부RnD첫수행/final_test_pred_r5.parquet','y',['p_champ','p_M0','p_SB']),
('N4','N4_특허행태_기업폐업조기경보/final_test_pred_r4.parquet','y',['p_champ','p_M0']),
('T3','T3_기술셀부상예측/R3/preds/FINAL_R3_t0_2021.parquet','yb',['p','p_base'])]
for name,f,yc,cols in spec:
    d=pd.read_parquet(MC+f); y=d[yc].values.astype(int); n=len(y)
    r=dict(n=n,pos=int(y.sum()),dup_rows=int(d.duplicated().sum()),na_y=int(d[yc].isna().sum()))
    for c in cols:
        p=d[c].values
        r[c]=dict(auc=float(roc_auc_score(y,p)),na=int(np.isnan(p).sum()),
                  top100=tie_range(y,p,100),top1=tie_range(y,p,max(1,int(round(0.01*n)))),top10=tie_range(y,p,max(1,int(round(0.10*n)))))
    R[name]=r
json.dump(R,open(OUT,'w',encoding='utf-8'),ensure_ascii=False,indent=1)
for t,r in R.items():
    print('==',t,r['n'],r['pos'],'dup',r['dup_rows'])
    for c,v in r.items():
        if isinstance(v,dict): print('  ',c,'auc=%.4f'%v['auc'],'na',v['na'],'| t100',v['top100'],'| t1',v['top1'],'| t10',v['top10'])
