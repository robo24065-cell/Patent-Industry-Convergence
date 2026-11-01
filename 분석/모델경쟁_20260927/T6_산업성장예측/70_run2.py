# 2단계 버전 실행기(final_run 라이브러리 사용). python 70_run2.py v38 [v39 ...]
import sys, os, json, time, copy
import numpy as np, pandas as pd
import final_run as fr
from t6lib import log_version, boot_ci, by_year

SECT2 = ['15', '20', '21', '23', '24', '25', '26', '27', '28', '29', '30', '31', '33']
COMPACT = ['is_q43', 'ip_q43', 'inv_ship3', 'is_q42', 'cost_ratio1', 'va_ratio1', 'asset_int1', 'lsize1', 'peer3_1', 'gestab1', 'gemp1', 'ldev3', 'g1', 'wage_pe1']
AR = ['g1', 'g2', 'cm_all1', 'gmean_exp']
KINT = AR + ['gva1', 'gemp1', 'gestab1', 'gasset1', 'peer3_1', 'peer2_1', 'lsize1', 'va_ratio1', 'cost_ratio1', 'wage_pe1', 'asset_int1', 'ldev3', 'gsd_exp']
IDX = ['ip_g3', 'is_g3', 'ip_q43', 'is_q43', 'inv_ship3', 'ip_g2', 'is_q42', 'mfg_ip_g', 'mfg_is_q4', 'bsi_sales', 'imp_share', 'imp_share_d']
EXT_CU = ['cu_q4x', 'cu_gx']
EXT_EX = ['exp_q4x', 'dom_q4x', 'expsh_gx']
EXT_FIN = ['fin_sales', 'fin_inv_gap', 'fin_tang']
PATC7 = ['cn_share_d3_rel_L2', 'cn_vs_kr3_rel_L2', 'pg3_kr_rel_L2', 'us_fp_share_L2', 'pg3_tot_rel_L2', 'kr_lapse_rate_rel', 'kr_transfer_rate_rel']
XGB = dict(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
           reg_lambda=10.0, objective='reg:absoluteerror', nthread=2, verbosity=0, base_score=0.0)
BASE = dict(features=COMPACT, sectors=SECT2, min_rows=500, drop_years=[2010, 2011], min_year=2009, ycap=1.0,
            seeds=[0, 1, 2, 3, 4], xgb=XGB, extra=True)


def S(**kw):
    s = copy.deepcopy(BASE)
    x = kw.pop('xgb', None)
    if x: s['xgb'].update(x)
    s.update(kw)
    return s


def legacy_rolling(F, years, spec):
    """1단계 방식 재현: 학습행은 학습행 안에서, 예측행은 50개(y 관측) 안에서 순위화."""
    out = []
    import xgboost as xg
    for T in years:
        tr = fr._train_rows(F, T, spec)
        te = F[(F.year == T) & (F.in50 == 1) & F.y.notna()]
        if len(tr) < spec['min_rows']:
            p = np.zeros(len(te))
        else:
            y = fr._target(tr, spec)
            Xtr = tr[spec['features']].astype(float).groupby(tr.year.values).rank(pct=True)
            Xte = te[spec['features']].astype(float).groupby(te.year.values).rank(pct=True)
            p = np.mean([xg.XGBRegressor(**spec['xgb'], random_state=s).fit(Xtr, y).predict(Xte) for s in spec['seeds']], axis=0)
        o = te[['ksic', 'year', 'y']].copy(); o['pred'] = p; out.append(o)
    return pd.concat(out, ignore_index=True)


def add_rel(F):
    F = F.copy()
    ok = ~F.year.isin([2010])
    med = F[ok].groupby('year').y.median()
    F['rel'] = np.where(ok, F.y - F.year.map(med), np.nan)
    F = F.sort_values(['ksic', 'year'])
    g = F.groupby('ksic').rel
    F['rel_med_hist'] = g.transform(lambda s: s.shift(1).expanding(min_periods=2).median())
    F['rel_med3'] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=2).median())
    return F.drop(columns=['rel']).sort_index()


def rank50_rolling(F, years, spec):
    s2 = dict(spec); s2['rank50'] = True
    return fr.rolling(F, years, s2)


V = {
    'v31r': dict(spec=S(), legacy=True, change='[감사 재현] v31을 final_run 패널·코드로 재현(1단계 순위 방식 그대로)', hyp='재현성 확인', log=False),
    'v38': dict(spec=S(), change='[감사 수정] v31 + 순위 기준집단 통일: 학습·예측 행 모두 그해 13개 업종 5자리 전체(y 관측 무관)에서 백분위 순위',
                hyp='1단계 코드는 예측행을 50개(그중 y 관측 행) 안에서 순위화 → 학습 분포와 불일치(평균 |순위차| 0.02–0.10), y 관측 여부가 피처에 스며듦. 수정 후 수치가 정직한 값'),
    'v39': dict(spec=S(features=KINT + IDX), change='[감사 수정] v21 전체 피처(25개)를 순위 기준집단 수정 후 재평가',
                hyp='v31 피처 축소의 낙관편향 없는 대조군'),
    'v40': dict(spec=S(features=COMPACT + EXT_CU + EXT_EX), change='v38 + 새 외부자료: 가동률지수(3자리) 4분기·연간 변화, 내수·수출 출하지수 4분기 모멘텀·수출/내수 비율 변화',
                hyp='설비 가동률 상승과 수출 모멘텀이 다음 해 출하를 선행'),
    'v41': dict(spec=S(features=COMPACT + EXT_FIN), change='v38 + ECOS 기업경영분석(3/2자리) T-1 매출액증가율·재고-매출 증가율 격차·유형자산증가율',
                hyp='기업 재무의 재고 누적과 설비투자가 업종 출하 사이클을 선행'),
    'v42': dict(spec=S(features=COMPACT, sectors=None, min_rows=800), change='v38 설정으로 488개 전 산업 학습·순위(업종 제한 해제)',
                hyp='순위 기준 수정 후에도 유사업종 제한이 유효한지 재확인'),
    'v43': dict(spec=S(), patent=PATC7, change='[절제: 특허 있음] v38 + 특허 7개(중국 비중 3년 변화·중국 대 한국 모멘텀·KR 3년 상대성장·US 해외우선권·5개국 3년 상대성장·KR 소멸률·이전율 상대)',
                hyp='순위 기준 수정 후 특허 신호 재검정'),
}

F41 = COMPACT + EXT_FIN
MONO = {'is_q43': 1, 'ip_q43': 1, 'inv_ship3': -1, 'is_q42': 1, 'lsize1': -1, 'peer3_1': 1, 'gestab1': 1, 'gemp1': 1, 'ldev3': -1, 'g1': -1, 'fin_sales': 1}
def mono(feats):
    return '(' + ','.join(str(MONO.get(f, 0)) for f in feats) + ')'
V.update({
    'v44': dict(spec=S(features=F41, xgb=dict(max_depth=1, n_estimators=600)), change='v41 + 깊이 1(가법 모형), 트리 600',
                hyp='v41까지 상호작용 이득이 불분명 → 가법 구조로 분산을 줄이면 소표본에서 더 안정'),
    'v45': dict(spec=S(features=F41, xgb=dict(monotone_constraints=mono(F41))), change='v41 + 경제적 사전 부호 단조 제약(모멘텀·동료·사업체/종사자 증가 +, 재고/출하·규모·평균회귀(g1, ldev3) −, 나머지 자유)',
                hyp='작은 표본에서 반대 부호로 잡음에 맞는 분할을 막으면 일반화 개선'),
    'v46': dict(spec=S(features=F41, seeds=list(range(10))), change='v41 + 시드 10개 배깅', hyp='예측 분산 추가 감소'),
    'v47': dict(spec=S(features=F41, xgb=dict(min_child_weight=60, reg_lambda=20)), change='v41 + 잎 최소가중 60·L2 20(강한 정규화)',
                hyp='극단값 지배 표본에서 잎 크기를 키우면 중앙값 추정이 안정'),
})

F45 = F41
V.update({
    'v48': dict(spec=S(features=F41 + ['rel_med_hist', 'rel_med3'], xgb=dict(monotone_constraints=mono(F41 + ['rel_med_hist', 'rel_med3']))), addrel=True,
                change='v45 + 산업 고유 상대성장 이력: (y − 그해 488개 중앙값)의 T-1까지 확장 중앙값·최근 3년 중앙값(2010 보간행 제외)',
                hyp='2차전지·의약품처럼 구조적으로 업종 평균을 웃도는 산업의 지속성을 강건 통계로 포착(v20 평균 드리프트는 극단값에 오염)'),
    'v49': dict(spec=S(features=F41, xgb=dict(monotone_constraints=mono(F41))), rank50=True,
                change='[진단] v45에서 예측행 순위만 그해 50개 산업 전체(y 관측 무관) 안에서 매김(1단계 방식에서 y 필터만 제거)',
                hyp='1단계 v31이 v38보다 좋았던 이유가 50개 내 순위의 정보인지 우연인지 판별'),
    'v50': dict(spec=None, blend=['v38', 'v41', 'v45'], change='v38·v41·v45 예측 단순평균(피처 집합·제약이 다른 3개 모형)',
                hyp='시드 안정성 진단상 세 모형 차이는 잡음 수준 → 평균으로 분산 감소'),
})

V.update({
    'v51': dict(spec=S(rank50=True), change='[감사 수정 대안] v31 피처(14개) + 50개 산업 행은 50개 안에서 순위(y 필터 제거, 학습행에도 동일 규칙)',
                hyp='v49 진단: 50개 내 순위가 시드 4세트 모두 +0.007 skill → 대상 집단 내 상대 위치가 정보. 1단계 v31의 누수성 요소(y 관측 필터)만 제거한 정직 버전'),
    'v52': dict(spec=S(features=F41, rank50=True, seeds=list(range(10)), xgb=dict(monotone_constraints=mono(F41))),
                change='v49(=v41 피처 17개 + 단조 제약 + 50개 내 순위) 시드 10개 배깅', hyp='v49의 시드 분산을 줄여 안정화'),
    'v53': dict(spec=S(features=F41, rank50=True, seeds=list(range(10)), xgb=dict(monotone_constraints=mono(F41 + PATC7))), patent=PATC7,
                change='[절제: 특허 있음] v52 + 특허 7개(단조 제약 없음)', hyp='최종 후보 모형에서 특허 신호 재검정'),
})


def run(name):
    t = time.time()
    v = V[name]
    spec = v['spec']
    if v.get('blend'):
        ps = [pd.read_parquet(f'preds/{k}.parquet').sort_values(['ksic', 'year']).reset_index(drop=True) for k in v['blend']]
        pred = ps[0][['ksic', 'year', 'y']].copy(); pred['pred'] = np.mean([q.pred.values for q in ps], axis=0)
        spec = dict(features=[])
    F = fr.build_panel(2019, 2019, extra=True)
    if v.get('addrel'):
        F = add_rel(F)
    feats = spec['features']
    if v.get('patent'):
        P = fr.build_patent_features(2019)
        F = F.merge(P, on=['ksic', 'year'], how='left')
        feats = feats + v['patent']
    if v.get('blend'):
        pass
    elif v.get('legacy'):
        pred = legacy_rolling(F, fr.VAL_YEARS, spec)
    elif v.get('rank50'):
        pred = rank50_rolling(F, fr.VAL_YEARS, spec)
    else:
        pred = fr.rolling(F, fr.VAL_YEARS, spec, feats=feats)
    pred = pred[['ksic', 'year', 'y', 'pred']]
    b = pd.read_parquet('preds/baseline.parquet')
    bm = float(np.abs(b.y).mean())
    assert len(pred.merge(b[['ksic', 'year']], on=['ksic', 'year'])) == len(b) == len(pred)
    lo, hi = boot_ci(pred, b, B=2000)
    if v.get('log', True):
        log_version(name, v['change'], v['hyp'], pred, bm, v.get('kept', '?'), note=f'skill95CI(산업부트)=[{lo:.3f},{hi:.3f}]; 2단계')
    else:
        pred.to_parquet(f'preds/{name}.parquet', index=False)
        mae = float((pred.y - pred.pred).abs().mean())
        print(f'[{name}] MAE={mae:.5f} skill={1-mae/bm:.4f}')
    print('  연도별 MAE', by_year(pred).round(4).to_dict(), f'CI[{lo:.3f},{hi:.3f}] {time.time()-t:.0f}s')
    return pred


if __name__ == '__main__':
    for n in sys.argv[1:]:
        run(n)
