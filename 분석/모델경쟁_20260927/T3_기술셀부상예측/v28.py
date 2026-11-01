from models import *
D=enrich(H.data())
def fp28(tr,te):
    tr=tr.copy(); te=te.copy()
    tr['te_ksic']=oof_origin(tr, lambda a,b: te_enc(a,b,'ksic')[1]); te['te_ksic']=te_enc(tr,te,'ksic')[1]
    xs=[xgbc(ALL2+['te_ksic'],dict(XP,seed=s),rounds=300)(tr,te) for s in range(5)]
    nbs=[nbfp(ALL2,dict(seed=s))(tr,te) for s in range(3)]
    return rankavg(logit(PRUNE+['te_ksic'])(tr,te), np.mean(xs,0), np.mean(nbs,0))
R,P=H.rolling(fp28,D); H.record('v28','v22 + XGB 5시드·NB 3시드 배깅(subsample 난수 평균)','트리 성분 분산 감소',R,P,BASE_AUC,kept='?')
