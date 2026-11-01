"""T3 2라운드 챔피언/도전자 실험 목록. 각 실험 = 현재 챔피언 설정에 변경 1개.
사용: python exps.py E01 E02 ...   (이미 기록된 실험은 --force 없으면 건너뜀)
"""
import sys, copy, json
import r2core as C
import pandas as pd

KOSIS = ['ks_ship_g', 'ks_emp_g', 'ks_va_g', 'ks_estab_g', 'ks_asset_g', 'ks_pat_int', 'ks_pat_per_emp', 'ks_va_ratio']
TRADE_IMP = ['tr_imp_g', 'tr_imp_cn_g', 'tr_cn_sh', 'tr_cn_sh_chg', 'tr_imp_lvl']
TRADE_EXP = ['tr_exp_g', 'tr_exp_lvl', 'tr_net_exp', 'tr_net_exp_chg']
IO_K = ['io_up_kmom', 'io_dn_kmom', 'io_up_fmom', 'io_dn_fmom']
IO_C = ['io_up_cell_km', 'io_dn_cell_km', 'io_up_cell_fm', 'io_dn_cell_fm', 'io_up_cell_klvl', 'io_dn_cell_klvl']
DM = [c + '_dm' for c in C.MOMS]
AGE = ['cell_age', 'mg_age', 'kr_age']
PI = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2', 'pi_inv_ship', 'pi_util', 'pi_util_chg']
LAG = ['fx_mom_l3', 'fx_mom_l4', 'US_mom_l3', 'CN_mom_l3', 'JP_mom_l3', 'EP_mom_l3']

def comp(cfg, t):
    return [c for c in cfg['components'] if c['type'] == t][0]
def set_param(t, **kw):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c = comp(c2, t)
        for k, v in kw.items():
            if k in ('params',): c['params'].update(v)
            else: c[k] = v
        return c2
    return f
def set_all(**kw):
    def f(cfg):
        c2 = copy.deepcopy(cfg)
        for c in c2['components']: c.update(kw)
        return c2
    return f
def set_top(**kw):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c2.update(kw); return c2
    return f
def feats(add=(), drop=(), comps=('logit', 'xgbc', 'nb')):
    return lambda cfg: C.with_feats(cfg, add, drop, comps)
def add_comp(c_new, w=1):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c2['components'].append(c_new); c2['weights'] = list(c2.get('weights', [1] * (len(c2['components']) - 1))) + [w]; return c2
    return f
def replace_raw_with_dm(comps=('logit',)):
    def f(cfg):
        c2 = copy.deepcopy(cfg)
        for c in c2['components']:
            if c['type'] in comps: c['features'] = [x + '_dm' if x in C.MOMS else x for x in c['features']]
        return c2
    return f
def chain(*fs):
    def f(cfg):
        for g in fs: cfg = g(cfg)
        return cfg
    return f

XREG = dict(type='xgbreg', features=None, params=dict(objective='reg:squarederror', max_depth=2, eta=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, nthread=2, verbosity=0), rounds=300)
def add_xreg(cfg):
    c = copy.deepcopy(XREG); c['features'] = list(comp(cfg, 'xgbc')['features']); return add_comp(c)(cfg)

def to_spline(cfg):
    c2 = copy.deepcopy(cfg); c = comp(c2, 'logit'); c['type'] = 'logit_spline'
    c['spline'] = ['kr_lvl', 'kr_mom', 'US_mom', 'EP_mom', 'kr_drawdown', 'tot_lvl']; return c2
def comp(cfg, t):
    return [c for c in cfg['components'] if c['type'] == t][0]

EXPS = {
 # ---- 외부 데이터 (KSIC 연결) ----
 'E01': ('KOSIS 광업제조업조사 8피처(출하·종사자·부가가치·사업체·유형자산 성장, 특허/출하·특허/종사자 집약도, 부가가치율; 조사연도 ≤t0-1) → 트리 2성분', feats(KOSIS, comps=('xgbc', 'nb')), '2010 결측·2008 시작 → 학습 원점 2005–2011은 결측'),
 'E02': ('무역: KSIC↔HS6 한국 수입 5피처(수입 성장·대중 수입 성장·중국 비중·비중 변화·수입 규모; ≤t0) → 3성분', feats(TRADE_IMP), 'UN Comtrade HS6(2000–), ksic_hs6_map 48/50 KSIC'),
 'E03': ('산업연관표(2010 접속 통합소분류 투입계수) 전후방 KSIC의 KR·해외 특허 모멘텀 가중평균 4피처 → 3성분', feats(IO_K), ''),
 'E04': ('산업연관표 셀 수준: 같은 IPC 메인그룹의 전후방 KSIC 셀 KR·해외 모멘텀·규모 6피처 → 3성분', feats(IO_C), '기술이 공급망을 따라 이동하는가'),
 'E18': ('KOSIS 광공업생산지수·가동률(소분류 3자리, 월별→연평균, ≤t0) 6피처: 생산 2년·1년 성장, 출하 2년 성장, 재고/출하, 가동률 수준·2년 변화 → 3성분', feats(PI), '2005.01~ 계열이라 KOSIS 조사보다 학습 원점 결측이 적음'),
 'E19': ('무역: KSIC↔HS6 한국 수출 4피처(수출 성장·규모, 무역특화지수와 변화; ≤t0) → 3성분', feats(TRADE_EXP), '_shared/ext/R2global_kr_export_hs6.csv (다른 트랙 수집, 2000–2021)'),
 'E20': ('E03 변형: 산업연관 전후방 KSIC 특허 모멘텀 4피처를 트리 2성분에만', feats(IO_K, comps=('xgbc', 'nb')), ''),
 'E21': ('근소 미달 묶음: 산업연관 전후방 모멘텀(E03) + 신설코드 경과연수(E07) + 로지스틱 가중 2(E15)', chain(feats(IO_K), feats(AGE), lambda cfg: set_top(weights=[2] + list(cfg.get('weights', [1, 1, 1]))[1:])(cfg)), '개별로 +0.001~0.0026이던 변경의 조합'),
 'E22': ('4번째 성분: 랜덤포레스트(300그루, 잎 20, 피처 30%) XGB 피처로 추가', lambda cfg: add_comp(dict(type='rf', features=list(comp(cfg, 'xgbc')['features']), n=300, leaf=20, mf=0.3))(cfg), '모형 계열 다양화'),
 'E23': ('XGB 분류 깊이 2→3', set_param('xgbc', params=dict(max_depth=3)), ''),
 'E24': ('XGB 분류 min_child_weight 5→20, colsample 0.8→0.5', set_param('xgbc', params=dict(min_child_weight=20, colsample_bytree=0.5)), ''),
 'E25': ('음이항 성분 XGB-Poisson 깊이 3→2, 라운드 400→600', chain(set_param('nb', params=dict(max_depth=2)), set_param('nb', rounds=600)), ''),
 'E26': ('E18 변형: 생산지수 6피처를 로지스틱에서 빼고 트리 2성분에만', feats(drop=PI, comps=('logit',)), ''),
 'E27': ('산업연관×생산지수: 전방(수요)·후방(공급) KSIC의 생산 2년 성장 가중평균 2피처 → 3성분', feats(['io_up_pi_g2', 'io_dn_pi_g2']), '수요 견인 가설'),
 'E28': ('생산지수 제조업 전체 대비 상대성장 + 수출/내수 출하 상대성장 2피처 → 3성분', feats(['pi_prod_g2_rel', 'pi_exp_dom_g']), ''),
 'E29': ('근소 미달 묶음 재시험(새 챔피언 위): 신설코드 경과연수(E07) + 로지스틱 가중 2(E15)', chain(feats(AGE), lambda cfg: set_top(weights=[2] + list(cfg.get('weights', [1, 1, 1]))[1:])(cfg)), ''),
 'E30': ('생산지수 피처 정리: 가동률 수준·변화 2피처 제거', feats(drop=['pi_util', 'pi_util_chg']), ''),
 'E31': ('생산지수 피처 정리: 재고/출하 비율 제거', feats(drop=['pi_inv_ship']), ''),
 'E32': ('생산지수 피처 정리: 생산 1년 성장 제거', feats(drop=['pi_prod_g1']), ''),
 'E33': ('계층 결합: KSIC 부상률 릿지 예측(KSIC 특허 모멘텀+생산지수+산업연관+무역 16피처, 원점 ≤t-3 학습) 1피처 → 3성분', feats(['ksic_hier']), '산업 수준 신호를 셀 모형에 주입'),
 'E34': ('산업 성장 최선값: KOSIS 5자리 출하 성장(있으면)·없으면 생산지수 3자리 출하 성장 1피처 → 3성분', feats(['ind_ship_best']), 'KOSIS 5자리의 산업 내 변동을 결측 없이'),
 'E35': ('학습 시작 원점 2005→2007 (초기 원점은 시차 피처 결측 많음)', set_top(first_t0=2007), ''),
 'E36': ('XGB 분류 라운드 300→200', set_param('xgbc', rounds=200), ''),
 'E37': ('음이항 성분 학습 표본 확대: 5개국 합 20~49 셀 추가(τ는 적격 표본 유지)', set_param('nb', train_min_total=20), '계수모형은 표본이 많을수록 μ 추정 안정'),
 'E38': ('E37 변형: 음이항 확대 표본 하한 20→10', set_param('nb', train_min_total=10), ''),
 'E39': ('E37 변형: 음이항 확대 표본 하한 20→30', set_param('nb', train_min_total=30), ''),
 'E40': ('학습 원점 확장: 2003·2004 원점 추가(5년 이상 시차 피처는 결측)', set_top(first_t0=2003), 'E35(원점 축소)가 크게 나빠 → 학습 원점 수가 핵심 자원'),
 'E41': ('XGB 분류 학습 표본 확대: 5개국 합 20~49 셀 추가(레이블 = 원점별 적격 표본의 부상 경계 이상)', set_param('xgbc', train_min_total=20), ''),
 'E42': ('로지스틱 학습 표본 확대: 5개국 합 20~49 셀 추가(같은 경계 레이블)', set_param('logit', train_min_total=20), ''),
 'E43': ('음이항 성분 라운드 400→600 (확대 표본 위)', set_param('nb', rounds=600), ''),
 'E44': ('음이항 성분 라운드 600→900', set_param('nb', rounds=900), ''),
 'E45': ('음이항 성분 깊이 3→4', set_param('nb', params=dict(max_depth=4)), ''),
 'E46': ('순위평균 가중 로지스틱 1 : XGB 1 : 음이항 2', lambda cfg: set_top(weights=list(cfg.get('weights', [1, 1, 1]))[:2] + [2] + list(cfg.get('weights', [1, 1, 1]))[3:])(cfg), ''),
 'E47': ('근소 미달 묶음: 음이항 깊이 4 + 라운드 900 (E44+E45)', chain(set_param('nb', params=dict(max_depth=4)), set_param('nb', rounds=900)), ''),
 'E48': ('음이항 깊이 4 + eta 0.05→0.03, 라운드 900', chain(set_param('nb', params=dict(max_depth=4, eta=0.03)), set_param('nb', rounds=900)), ''),
 'E49': ('출원인 유형: KR 대학·공공연구기관 비중, 공공연구 모멘텀, KR 개인 비중·변화, 해외 대학 비중 6피처 → 3성분', feats(['kr_univ_sh', 'kr_gri_sh', 'kr_pub_mom', 'kr_indiv_sh', 'kr_indiv_sh_chg', 'fx_univ_sh']), '기초연구→산업화 파이프라인'),
 'E50': ('KR 등록 신호: 등록률(t0-6..t0-3 출원 중 t0까지 등록)·빠른 등록·등록 모멘텀 3피처 → 3성분', feats(['kr_reg_rate', 'kr_reg_fast', 'kr_reg_mom']), '등록원부 연계, rgst ≤ t0'),
 'E51': ('수요-특허 괴리: 원점 내 z(생산지수 2년 성장) − z(KSIC KR 특허 모멘텀) 1피처 → 3성분', feats(['pi_gap']), '산업 생산은 느는데 특허가 뒤처진 산업'),
 'E52': ('로지스틱 → 스플라인 로지스틱(kr_lvl·kr_mom·US_mom·EP_mom·kr_drawdown·tot_lvl 3차 스플라인 5매듭)', to_spline, ''),
 'E53': ('XGB 분류 확대 표본(5개국 합 10~49, 경계 레이블)', set_param('xgbc', train_min_total=10), ''),
 'E54': ('5개국 수출 경쟁: KSIC별 한국 수출 점유율(KR/CN/JP/US/DE)·변화, 중국 점유율 변화, 한국 RCA, 특허점유율−수출점유율 괴리 5피처 → 3성분', feats(['ex5_kr_sh', 'ex5_kr_sh_chg', 'ex5_cn_sh_chg', 'ex5_kr_rca', 'ex5_pat_gap']), '완결연도 2004·06·09·12만(수집 할당량 초과로 이후 미완)'),
 'E55': ('제목 뜨는 용어 노출도(KR): 셀 최근 3년 제목 단어의 전체 KR 3년 모멘텀 평균·최대, 상위10% 용어 포함 비중 3피처 → 3성분', feats(['kw_expo_mean', 'kw_expo_max', 'kw_hot_sh']), ''),
 'E56': ('제목 뜨는 용어 노출도(US 제목): 평균·상위10% 포함 비중 2피처 → 3성분', feats(['us_kw_expo_mean', 'us_kw_hot_sh']), '해외 용어 선행'),
 'E57': ('근소 미달 묶음: XGB 분류 확대 표본 10~49(E53) + 음이항 900라운드(E44)', chain(set_param('xgbc', train_min_total=10), set_param('nb', rounds=900)), ''),
 'E58': ('생산지수 가속도(1년 성장의 변화) 1피처 → 3성분', feats(['pi_prod_acc']), ''),
 'E59': ('음이항 깊이 3→4 (새 챔피언 위 재시험)', set_param('nb', params=dict(max_depth=4)), ''),
 'E60': ('XGB 분류 라운드 300→450 (확대 표본 위)', set_param('xgbc', rounds=450), ''),
 'E61': ('산업연관 전후방 KSIC 특허 모멘텀 4피처 재시험(새 챔피언 위, 3성분)', feats(IO_K), 'E03 근소 미달'),
 'E62': ('IPC 신설코드 경과연수 3피처 재시험(새 챔피언 위)', feats(AGE), 'E07 근소 미달'),
 # ---- 모형·피처 ----
 'E05': ('로지스틱 성분의 모멘텀 20종을 원점 단면 중앙값 차감값(_dm)으로 교체', replace_raw_with_dm(('logit',)), '원점별 전역 출원 경기 제거'),
 'E06': ('트리 2성분에 원점 중앙값 차감 모멘텀 20피처 추가', feats(DM, comps=('xgbc', 'nb')), ''),
 'E07': ('IPC 신설코드 보정: 셀·메인그룹·KR 첫 출원 후 경과연수 3피처 → 3성분', feats(AGE), '재분류/신설 코드의 과거 0건 인위 성장 구분'),
 'E08': ('국가별 선행 시차: 해외 4개국 합 시차3·4, 국가별 시차3 성장 6피처 → 3성분', feats(LAG), ''),
 'E09': ('KR 0건 셀 처리: KR 과거3년 0건 지시자·0건 셀의 5개국 규모 → 3성분', feats(['kr_zero', 'fx_lvl_krzero']), ''),
 'E10': ('원점 가중: 3성분 모두 최근 원점 가중(반감기 6년)', set_all(half=6), ''),
 'E11': ('학습 창: 최근 6개 원점만 학습(창 이동)', set_top(window=6), ''),
 'E12': ('XGB 분류 5시드 배깅(순위평균)', set_param('xgbc', seeds=5), ''),
 'E13': ('XGB 분류 라운드 300→600, eta 0.05→0.025', chain(set_param('xgbc', rounds=600), set_param('xgbc', params=dict(eta=0.025))), ''),
 'E14': ('4번째 성분: XGB 회귀(연속 성장률 y) 추가', add_xreg, ''),
 'E15': ('순위평균 가중 로지스틱 2 : XGB 1 : 음이항 1', lambda cfg: set_top(weights=[2] + list(cfg.get('weights', [1, 1, 1]))[1:])(cfg), ''),
 'E16': ('로지스틱 C 1.0→0.1', set_param('logit', C=0.1), ''),
 'E17': ('로지스틱에 CN·JP 12피처 복원', feats([f'{c}_{s}' for c in ('CN', 'JP') for s in ('mom', 'mom_l1', 'mom_l2', 'lvl', 'g1', 'slope5')], comps=('logit',)), ''),
}

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; force = '--force' in sys.argv
    done = set(pd.read_csv(C.EXP_CSV, encoding='utf-8-sig').exp_id) if C.os.path.exists(C.EXP_CSV) else set()
    D = C.get_D()
    for e in args:
        if e in done and not force: print('건너뜀', e); continue
        change, f, note = EXPS[e]
        cfg = f(C.load_champ()['cfg'])
        C.challenge(e, change, cfg, note, D=D)
