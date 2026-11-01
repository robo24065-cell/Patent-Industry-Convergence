# v29 감사 재측정: 중립 동률깨기 레이블로 기준선·v22 재측정 + 추가 원점(2010/2012/2014)에서 1단계 버전들의 선택 편향 점검
from s2lib import *
import json
res={}
for neutral in [False,True]:
    D=data(neutral)
    for name,fp in [('v0',base_fp),('v13',logit(PRUNE)),('v16',lambda tr,te: rankavg(logit(PRUNE)(tr,te), xgbc(ALL2,XP,rounds=300)(tr,te))),
                    ('v17',lambda tr,te: rankavg(logit(PRUNE)(tr,te), xgbc(ALL2,XP,rounds=300)(tr,te), nbfp(ALL2)(tr,te))),('v22',best_fp())]:
        R,P,Rx=both(fp,D,f'{"중립" if neutral else "원래"} {name}')
        res[(neutral,name)]=(R,P,Rx)
import pickle; pickle.dump({k:(v[0],v[2]) for k,v in res.items()},open('cache/s2_v29.pkl','wb'))
b=H.summarize(res[(True,'v0')][0])['auc']; bx=H.summarize(res[(True,'v0')][2])['auc']
R,P,Rx=res[(True,'v22')]; sx=H.summarize(Rx)
H.record('v29','[2단계 감사] 레이블 경계 동률의 잔여 깨기를 셀 정렬순서(KSIC 오름차순)→md5(셀) 중립키로 교체, v22 재측정. 기준선도 같은 레이블로 재계산',
 '경계 동률 1~15셀이 KSIC 코드순으로 양성 배정되면 KSIC 타깃인코딩이 인위 신호를 학습할 수 있음',R,P,b,kept='Y',
 note=f'중립 기준선 {b:.4f}. 추가원점(2010/12/14, 1단계 선택 미사용) v22 {sx["auc"]:.4f} vs 기준선 {bx:.4f}')
