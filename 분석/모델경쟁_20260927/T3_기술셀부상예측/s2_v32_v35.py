from s2lib import *
import sys
D=add_diff(add_flow(data(True))); b=0.6527
which=sys.argv[1:]
def rec(v,change,hyp,fp,kept='?'):
    R,P,Rx=both(fp,D,v); sx=H.summarize(Rx)
    H.record(v,change,hyp,R,P,b,kept=kept,note=f'추가원점(2010/12/14) {sx["auc"]:.4f} ({sx["auc_by_t0"]})')
if 'v32' in which:
    rec('v32','[2단계] v30 + 산업 간 확산 5피처(메인그룹이 KR 출원된 KSIC 수·변화, 메인그룹 전체 KR 규모, 셀의 메인그룹 내 KR 비중·변화) 3성분 모두',
        '기술이 여러 산업으로 퍼지는 중(확산 폭 증가)이거나 해당 산업 비중이 낮은 셀은 따라잡기 부상 가능',ens(PRUNE+DIFF,ALL2+DIFF,ALL2+DIFF))
if 'v33' in which:
    rec('v33','[2단계] v30의 XGB 분류 성분을 부드러운 레이블 XGB(sigmoid((y−원점별 q90)/0.25))로 교체',
        '경계 부근 0/1 레이블의 계수잡음을 연속 가중으로 완화',lambda tr,te: rankavg(logit(PRUNE)(tr,te), soft_xgb(ALL2)(tr,te), nbfp(ALL2)(tr,te)))
if 'v34' in which:
    rec('v34','[2단계] v30 3성분을 동일 순위평균 대신 원점-OOF 로지스틱 메타학습(성분 순위 + 성분×KR규모 상호작용)으로 결합',
        '규모 구간마다 어느 성분이 강한지 다름(음이항은 소규모, 트리는 대규모)',
        stack([logit(PRUNE), xgbc(ALL2,XP,rounds=300), nbfp(ALL2)]))
if 'v35' in which:
    rec('v35','[2단계] v30의 로지스틱·XGB 성분을 KR 과거 3년 ≤5 / >5 규모별 분리모형으로 교체(음이항 성분은 그대로)',
        '0→3 류 소규모 부상과 실질 부상의 결정요인이 다름',
        lambda tr,te: rankavg(segmented(logit(PRUNE))(tr,te), segmented(xgbc(ALL2,XP,rounds=300))(tr,te), nbfp(ALL2)(tr,te)))
