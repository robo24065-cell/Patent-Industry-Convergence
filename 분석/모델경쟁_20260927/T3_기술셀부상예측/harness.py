"""롤링 원점 검증: 검증 t0=v 에서 학습은 t0<=v-3 (레이블이 v 이전에 완전 관측된 것만)."""
import numpy as np, pandas as pd, os, datetime, json
import t3lib as L
MIN_TOTAL=50
TRAIN_START=2005
HERE=L.HERE
def data(min_total=MIN_TOTAL):
    df=pd.concat([L.get_feats(t) for t in range(TRAIN_START,2016)],ignore_index=True)
    return L.add_labels(df,min_total)

def rolling(fit_predict, D=None, min_total=MIN_TOTAL, val=L.VAL_T0):
    if D is None: D=data(min_total)
    res=[]; preds=[]
    for v in val:
        tr=D[D.t0<=v-3]; te=D[D.t0==v]
        assert tr.t0.max()+3<=v-0
        p=np.asarray(fit_predict(tr,te),dtype=float)
        m=L.evaluate(te.y.values,p,te.yb.values); m['t0']=v; res.append(m)
        preds.append(pd.DataFrame({'cell':te.cell.values,'t0':v,'p':p,'y':te.y.values,'yb':te.yb.values}))
    R=pd.DataFrame(res); P=pd.concat(preds,ignore_index=True)
    return R,P

def summarize(R):
    return dict(auc=R.auc.mean(), prauc=R.prauc.mean(), p_at10=R.p_at10.mean(), rho=R.rho.mean(),
                auc_by_t0='/'.join(f'{a:.3f}' for a in R.auc))

LB=os.path.join(HERE,'leaderboard.csv')
def record(version, change, hypothesis, R, P, baseline_auc, kept, note=''):
    s=summarize(R)
    skill=(s['auc']-baseline_auc)/(1-baseline_auc)
    row=dict(version=version,date=datetime.date.today().isoformat(),change=change,hypothesis=hypothesis,
             metric='mean breakout AUC (t0=2011/13/15)',value=round(s['auc'],4),baseline_value=round(baseline_auc,4),
             skill=round(skill,4),kept=kept,
             note=f"AUC by t0 {s['auc_by_t0']}; PR-AUC {s['prauc']:.3f}; P@10% {s['p_at10']:.3f}; rho {s['rho']:.3f}. "+note)
    df=pd.read_csv(LB) if os.path.exists(LB) else pd.DataFrame()
    if len(df) and version in df.version.values: df=df[df.version!=version]
    df=pd.concat([df,pd.DataFrame([row])],ignore_index=True); df.to_csv(LB,index=False,encoding='utf-8-sig')
    P.to_parquet(os.path.join(HERE,'preds',f'{version}.parquet'))
    print(version, json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items()},ensure_ascii=False), 'skill',round(skill,4))
    print(R.round(4).to_string())
    return s['auc']
