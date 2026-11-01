# 동결 예측 재채점만 (재학습 없음). 원천 parquet 읽기 전용.
import json, numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
MC = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/"
OUT = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C01/C01-T2/recompute_results.json"
def skill(a,b): return (a-b)/(1-b)
def topk(y,p,frac=None,k=None):
    y=np.asarray(y); o=np.argsort(-np.asarray(p),kind="stable")
    if k is None: k=max(1,int(round(frac*len(y))))
    return dict(k=int(k), hits=int(y[o[:k]].sum()), prec=float(y[o[:k]].mean()), capture=float(y[o[:k]].sum()/y.sum()))
R={}
def track(name, f, ycol, pm, bases):
    d=pd.read_parquet(MC+f); y=d[ycol].values.astype(int)
    r=dict(file=f, n=int(len(y)), pos=int(y.sum()), base_rate=float(y.mean()))
    r['auc_'+pm]=float(roc_auc_score(y,d[pm].values))
    for fr in (0.01,0.05,0.10): r[f'top{int(fr*100)}_{pm}']=topk(y,d[pm].values,fr)
    r[f'hit100_{pm}']=topk(y,d[pm].values,k=100)['hits']
    for b in bases:
        ab=float(roc_auc_score(y,d[b].values)); r['auc_'+b]=ab; r[f'skill_vs_{b}']=float(skill(r['auc_'+pm],ab))
        for fr in (0.01,0.05,0.10): r[f'top{int(fr*100)}_{b}']=topk(y,d[b].values,fr)
        r[f'hit100_{b}']=topk(y,d[b].values,k=100)['hits']
    R[name]=r
track('N9','N9_정부RnD과제_특허성과예측/final_preds_r6.parquet','y','p_champ',['p_M0_L2','p_M0_orig','p_strong_L2','p_strong_orig','p_pilot_L2','p_pilot_orig'])
track('N6','N6_정부RnD기업_첫특허전환/final_preds_r5.parquet','y','p_champ',['p_m0'])
track('N7','N7_특허기업_정부RnD첫수행/final_test_pred_r5.parquet','y','p_champ',['p_M0','p_SB'])
track('N8','N8_특허분쟁표적레이더/final_preds_r6.parquet','y','p_champ',['p_M0','p_SB','p_logit_champ'])
track('N4','N4_특허행태_기업폐업조기경보/final_test_pred_r4.parquet','y','p_champ',['p_M0'])
track('T3','T3_기술셀부상예측/R3/preds/FINAL_R3_t0_2021.parquet','yb','p',['p_base'])
# N9 subgroup by org (org column)
d=pd.read_parquet(MC+'N9_정부RnD과제_특허성과예측/final_preds_r6.parquet')
R['N9']['org_values']=d['org'].value_counts().head(10).to_dict()
# N8 calibration: sum of predicted prob vs actual
d8=pd.read_parquet(MC+'N8_특허분쟁표적레이더/final_preds_r6.parquet')
R['N8']['sum_p_champ']=float(d8.p_champ.sum()); R['N8']['calib_ratio']=float(d8.p_champ.sum()/d8.y.sum())
# T3 top100 and top-10% ties in baseline
d3=pd.read_parquet(MC+'T3_기술셀부상예측/R3/preds/FINAL_R3_t0_2021.parquet')
R['T3']['cols']=list(d3.columns); R['T3']['t0']=d3.t0.unique().tolist()
k=int(round(0.1*len(d3)));
o=np.argsort(-d3.p.values,kind='stable')[:k]
R['T3']['top10_zeroKR_share']=float((d3.kr_W.values[o]==0).mean()) if 'kr_W' in d3 else None
R['T3']['top10_k']=k
m5=d3.kr_W.values>=5
R['T3']['krge5']=dict(n=int(m5.sum()), auc=float(roc_auc_score(d3.yb[m5],d3.p[m5])), auc_base=float(roc_auc_score(d3.yb[m5],d3.p_base[m5])))
for kk in ['quicksort','stable']:
    ob=np.argsort(-d3.p_base.values,kind=kk)[:k]; R['T3'][f'top10_base_{kk}']=float(d3.yb.values[ob].mean())
json.dump(R,open(OUT,'w',encoding='utf-8'),ensure_ascii=False,indent=1,default=str)
for t,r in R.items():
    print('==',t); [print(' ',a,'=',b) for a,b in r.items()]
