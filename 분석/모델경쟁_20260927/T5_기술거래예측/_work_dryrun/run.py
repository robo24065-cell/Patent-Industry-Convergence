# 버전 실행기: python run.py v3  (검증 L=2014,2015 / 학습 L<=2011)
import sys, json, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from common import *
from evalkit import *

BASE_NUM = ['age', 'lclaims', 'rgst_year']; BASE_CAT = ['orig_type', 'ksic']
OWN_STATUS = ['cur_is_orig', 'l_own_port', 'cur_n']; OWN_STATUS_CAT = ['cur_type']
OWN_HIST = ['prior_MAIN', 'prior_CORP', 'yrs_since_change', 'own_out3', 'own_out3_days', 'own_in3', 'own_out_all', 'own_corp3',
            'own_lapse3', 'own_lapse1', 'own_lapse_rate', 'own_out_rate', 'ksic_rate3']
PAY = ['n_pay', 'paid_thru', 'prepaid', 'days_since_pay', 'amt_per_yr', 'multi']
STATIC = ['n_inv', 'n_inv_for', 'n_app', 'n_prio', 'for_prio', 'inv_is_app', 'exam_yrs', 'has_agent', 'pct', 'lic_any', 'lic_active',
          'lic_excl', 'n_trial', 'age_appl']
EXTRA = ['own_out1', 'own_out6m', 'own_in1', 'own_lapse_trend', 'own_app3', 'own_app_prev', 'own_app_mom', 'own_years',
         'own_ksic_share', 'own_ipc_share', 'own_n_ksic', 'ipc_rate3', 'ksic_rate1']
LOGCOLS = ['own_out3', 'own_out3_days', 'own_in3', 'own_out_all', 'own_corp3', 'own_lapse3', 'own_lapse1', 'n_trial', 'n_inv', 'n_inv_for',
           'own_out1', 'own_out6m', 'own_in1', 'own_app3', 'own_app_prev', 'own_n_ksic', 'amt_per_yr', 'days_since_pay']

XGB_DEF = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20)
V = {
    'v1': dict(model='lr', num=BASE_NUM + OWN_STATUS, cat=BASE_CAT + OWN_STATUS_CAT,
               change='기준선 + 현재 권리자 상태(현재 권리자 유형·원출원인 유지 여부·보유 포트폴리오 규모·공유자 수)',
               hyp='거래 주체는 원출원인이 아니라 L 시점의 권리자. 소규모 보유자·개인이 양도를 많이 한다'),
    'v2': dict(model='lr', num=BASE_NUM + OWN_STATUS + OWN_HIST, cat=BASE_CAT + OWN_STATUS_CAT,
               change='v1 + 권리자/특허의 L 이전 거래·불납 이력(양도 out/in, 합병, 불납 수·비율, KSIC 최근 거래율)',
               hyp='포트폴리오 정리 중인 권리자(최근 양도·불납 증가)는 계속 판다'),
    'v3': dict(model='lr', num=BASE_NUM + OWN_STATUS + OWN_HIST + PAY + STATIC, cat=BASE_CAT + OWN_STATUS_CAT,
               change='v2 + 납부행동 + 특허 정적 속성(발명자·우선권·PCT·실시권·심판·대리인)',
               hyp='선납·감면액(규모 신호)·실시권·해외우선권이 거래가능성을 가른다'),
    'v4': dict(model='xgb', num=BASE_NUM + OWN_STATUS + OWN_HIST + PAY + STATIC + ['ksic_c', 'cur_type', 'orig_type'], cat=[],
               change='모형족 변경: v3 피처 그대로 XGBoost(깊이6, 400트리)', hyp='비선형·상호작용(유형×규모×이력)이 크다'),
    'v5': dict(model='xgb', num=BASE_NUM + OWN_STATUS + OWN_HIST + PAY + STATIC + EXTRA + ['ksic_c', 'cur_type', 'orig_type'], cat=[],
               change='v4 + 권리자 단기동학(최근 6개월·1년 양도)·출원 모멘텀·포트폴리오 핵심성·IPC4 거래율',
               hyp='대형 번들 매각은 직전 6~12개월의 매각 개시·출원 중단·비핵심 특허에서 예고된다'),
}

FULL = BASE_NUM + OWN_STATUS + OWN_HIST + PAY + STATIC + EXTRA + ['ksic_c', 'cur_type', 'orig_type']
V['v6'] = dict(model='xgb', num=FULL + ['text_score'], cat=[], text='cache/text_2006_2011.parquet',
               change='v5 + 제목 문자 n-gram TF-IDF 로지스틱 점수(학습구간 권리자군집 5-fold OOF)',
               hyp='기술 내용(예: 반도체·디스플레이·센서 vs 가전)이 시장성=거래가능성을 반영')

def bundle_w(tr, va, alpha=0.5):
    """학습 양성의 번들 크기(같은 L·양도인·같은 날) 기준 가중 1/size^alpha. 음성은 1."""
    tr = tr.copy(); pos = tr.ytr == 1
    bk = tr[pos].groupby(['L', 'cur', 'y_date']).rgst_no.transform('size')
    tr['w'] = 1.0; tr.loc[pos, 'w'] = 1.0 / bk.values ** alpha
    tr.loc[pos, 'w'] *= pos.sum() / tr.loc[pos, 'w'].sum()   # 양성 총가중 보존
    return tr, va
V['v7'] = dict(model='xgb', num=FULL, cat=[], wcol='w', prep=bundle_w,
               change='v5 + 학습 양성 번들 가중 1/sqrt(같은날·같은양도인 건수), 양성 총가중 보존',
               hyp='수백 건 번들이 학습을 지배해 특정 권리자 특이성을 외운다 → 번들 가중을 낮추면 일반화 개선')

def horizon_aug(tr, text=None, years=(2012, 2013), cutoff=pd.Timestamp('2014-01-01')):
    """절단 수평 보강: L=2012(2년창)·2013(1년창)의 레이블은 2014-01-01 전에 관측 완료 → 규칙상 학습 허용.
    y = [L, 2014-01-01) 안 MAIN 양도. 수평 h 를 피처로 넣고 예측 시 h=3."""
    ad = load(list(years), text=text)
    ad['ytr'] = ((ad.y_MAIN == 1) & (ad.y_date < cutoff)).astype(int)
    ad['h'] = cutoff.year - ad.L
    for c in [c for c in ad.columns if c.startswith('y_') and c != 'y_date']: ad[c] = -1   # 3년 레이블 사용 차단
    tr = tr.copy(); tr['h'] = 3
    return pd.concat([tr, ad], ignore_index=True)
V['v8'] = dict(model='xgb', num=FULL + ['h'], cat=[], aug=horizon_aug,
               change='v5 + 절단수평 보강 학습(L2012 2년창·L2013 1년창 레이블 추가, 수평 h 피처, 예측 h=3)',
               hyp='가장 최근(2012~13) 권리자 매각 동학을 학습에 넣으면 2014~15 번들 매각 예측이 는다')

UNIT = ['u_ipc_out3', 'u_ipc_out1', 'u_ksic_out3', 'u_grp_out3', 'u_ipc_lapse3', 'u_ipc_alive', 'u_ipc_out_share', 'rel_age', 'rel_claims']
V['v9'] = dict(model='xgb', num=FULL + UNIT, cat=[],
               change='v5 + 사업부 단위 매각 신호(같은 권리자가 최근 3년/1년 같은 IPC4·메인그룹·KSIC 특허를 판 건수, 같은 IPC4 불납, 포트폴리오 내 상대연령·상대청구항)',
               hyp='대기업 번들 매각은 사업부(기술) 단위로 단계적으로 진행 → 같은 기술을 이미 팔기 시작한 권리자의 나머지 특허가 다음 매각 대상')

XGB_T = dict(n_estimators=600, max_depth=8, learning_rate=0.03, colsample_bytree=0.5, min_child_weight=5, reg_lambda=10.0)
V['v10'] = dict(model='xgb', num=FULL, cat=[], xgb=XGB_T,
                change='v5 피처 + 내부 시간분할(학습 L2006-08→내부검증 L2011) 튜닝 하이퍼파라미터: 깊이8·min_child_weight5·colsample0.5·lambda10·lr0.03·600트리',
                hyp='min_child_weight 20은 희소 양성(권리자별 소수 사건) 분기를 막아 과소적합 → 내부분할 AUC 0.626→0.684')
V['v11'] = dict(model='xgb', num=FULL + UNIT, cat=[], xgb=XGB_T,
                change='v10 + 사업부 단위 매각 신호(UNIT) 재투입',
                hyp='v9 하락은 과적합 설정 탓이었는지 확인: 튜닝된 설정에서는 사업부 신호가 기여할 것')

OWNSALE = ['own_out3', 'own_out3_days', 'own_out_all', 'own_out_rate', 'own_out1', 'own_out6m']
F12 = [f for f in FULL if f not in OWNSALE]
V['v12'] = dict(model='xgb', num=F12, cat=[],
                change='v5 - 권리자 매각 건수 피처 6개(own_out3·out3_days·out_all·out_rate·out1·out6m) 제거 (그룹 ablation 결과 근거)',
                hyp='권리자 과거 매각 건수는 학습기(2006-11) 일회성 구조조정 번들을 외우는 통로 → 제거하면 일반화 개선')

TECH = ['k_app3', 'k_app_prev', 'k_mom', 'k_ipc_app3', 'k_ipc_app_prev', 'k_ipc_share', 'tech_exit', 'k_dormant']
V['v13'] = dict(model='xgb', num=F12 + TECH, cat=[],
                change='v12 + 기술철수·휴면 신호(KR 전체 출원 108만건 중 L 이전 공개분: 권리자 최근3년/이전3년 출원, 이 특허 IPC4에서의 최근 출원 유무, 마지막 출원 후 경과연수)',
                hyp='권리자가 해당 기술에서 출원을 멈췄거나(철수) 전체 출원이 끊긴(휴면) 경우 보유 특허를 매각한다 — ablation에서 출원모멘텀이 최대 기여')


def prep(d):
    d = d.copy()
    d['lclaims'] = np.log1p(d.claims.fillna(d.claims.median()))
    d['l_own_port'] = np.log1p(d.own_port)
    d['ksic_c'] = d.ksic.astype(int); d['pct'] = (d.applTpcd == '국제출원').astype(int)
    return d


def load(years, extra=True, text=None, unit=True, tech=True):
    d = load_panel(years)
    if tech:
        d = d.merge(pd.concat([pd.read_parquet(f'cache/tech_L{y}.parquet') for y in years]), on=['rgst_no', 'L'], how='left')
    if unit:
        d = d.merge(pd.concat([pd.read_parquet(f'cache/unit_L{y}.parquet') for y in years]), on=['rgst_no', 'L'], how='left')
    if extra:
        d = d.merge(pd.concat([pd.read_parquet(f'cache/extra_L{y}.parquet') for y in years]), on=['rgst_no', 'L'], how='left')
    if text is not None:
        d = d.merge(pd.read_parquet(text), on=['rgst_no', 'L'], how='left')
    return prep(d)


def fit_predict(cfg, tr, va, y='ytr'):
    num, cat = cfg['num'], cfg['cat']
    w = tr[cfg['wcol']].values if cfg.get('wcol') else None
    if cfg['model'] == 'lr':
        X = lambda d: d[num + cat].assign(**{c: np.log1p(d[c].clip(lower=0)) for c in LOGCOLS if c in num})
        ct = ColumnTransformer([('n', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), num),
                                ('c', OneHotEncoder(handle_unknown='ignore'), cat)])
        m = make_pipeline(ct, LogisticRegression(max_iter=3000, C=cfg.get('C', 1.0)))
        m.fit(X(tr), tr[y], logisticregression__sample_weight=w)
        return m.predict_proba(X(va))[:, 1], m
    else:
        prm = dict(XGB_DEF); prm.update(cfg.get('xgb', {}))
        m = xgb.XGBClassifier(**prm, nthread=2, tree_method='hist', eval_metric='auc', random_state=cfg.get('seed', 0))
        m.fit(tr[num], tr[y], sample_weight=w)
        return m.predict_proba(va[num])[:, 1], m


def main(v):
    cfg = V[v]; t = time.time()
    trY = cfg.get('train_years', range(2006, TRAIN_MAX_L + 1))
    text = cfg.get('text')
    tr = load(trY, text=text); va = load(VAL_L, text=text)
    tr['ytr'] = tr.y_MAIN
    if cfg.get('aug'):
        tr = cfg['aug'](tr, text)
    if cfg.get('prep'): tr, va = cfg['prep'](tr, va)
    if 'h' in cfg['num']: va = va.assign(h=3)
    p, m = fit_predict(cfg, tr, va)
    mt = metrics(va.y_MAIN, p); base = json.load(open('cache/v0_metrics.json'))['y_MAIN']
    ci = cluster_boot(va.y_MAIN.values, p, va.cur.values, B=200)
    print(v, 'n_tr', len(tr), 'n_va', len(va), {k: round(x, 4) for k, x in mt.items()}, 'skill', round(skill(mt['auc'], base['auc']), 4),
          'CI', ci[0].round(4), ci[1].round(4), round(time.time() - t, 1), flush=True)
    for L in VAL_L:
        k = (va.L == L).values; print('  L', L, {kk: round(x, 4) for kk, x in metrics(va.y_MAIN[k], p[k]).items()})
    pd.DataFrame({'rgst_no': va.rgst_no, 'L': va.L, 'p': p}).to_parquet(f'cache/pred_{v}.parquet', index=False)
    if cfg['model'] == 'xgb':
        imp = pd.Series(m.get_booster().get_score(importance_type='gain')).sort_values(ascending=False)
        print('  gain top:', imp.head(15).round(1).to_dict())
    rec = dict(auc=mt['auc'], prauc=mt['prauc'], p1=mt['p1'], ci_auc=list(ci[0]), ci_pr=list(ci[1]))
    json.dump(rec, open(f'cache/metrics_{v}.json', 'w'))
    return mt, ci, base


if __name__ == '__main__':
    v = sys.argv[1]; kept = sys.argv[2] if len(sys.argv) > 2 else '?'
    mt, ci, base = main(v)
    note = f"PR-AUC {mt['prauc']:.4f} (기준선 {base['prauc']:.4f}), p@1% {mt['p1']:.4f}; AUC 95%CI(양도인 군집) {ci[0][0]:.3f}-{ci[0][1]:.3f}"
    log(v, V[v]['change'], V[v]['hyp'], mt['auc'], base['auc'], kept, note)
