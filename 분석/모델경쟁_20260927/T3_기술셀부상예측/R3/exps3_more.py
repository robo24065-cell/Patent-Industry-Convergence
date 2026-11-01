# exps3.py 가 exec 로 읽는 추가 실험 정의 (C, feats, set_param, set_top, chain, comp, copy, pd, np, os, HERE, VAR 주입)
import r3lib as L

EARLY = ['kr_early_n', 'kr_early_sh', 'kr_appc_mom', 'fx_appc_mom', 'fx_early_sh']
CNJP = [f'{c}_{s}' for c in ('CN', 'JP') for s in ('mom', 'mom_l1', 'mom_l2', 'lvl', 'g1', 'slope5')]

_WC = {}
def wipo_factors():
    """관청×WIPO분야×연도 커버리지 비율(이 변형의 셀 집계 / WIPO 공개건수). ext3_wipo_coverage_{var}.csv"""
    if 'R' in _WC: return _WC['R']
    c = pd.read_csv(os.path.join(HERE, f'ext3_wipo_coverage_{VAR}.csv'))
    _WC['R'] = c.pivot_table(index=['cc', 'f'], columns='year', values='ratio'); return _WC['R']

def cnjp_corrected(D):
    """CN·JP 셀 집계를 WIPO 커버리지로 보정한 뒤 국가 모멘텀 6종을 다시 계산(원래 이름을 *_wc 로).
    원점 t0 에서 보정계수 = (t0-6..t0-1 비율 중앙값) / (연도 y 비율), y ≤ t0-1 만 보정(WIPO y년 통계는 y+1년 공표), t0년은 보정 없음. 계수 [0.5, 2] 절단."""
    P = L.load(); n = P['A']['n']; R = wipo_factors()
    ug = pd.Series(np.unique(P['mg']))
    conc = pd.read_csv(os.path.join(HERE, '..', '..', '_shared', 'ext', 'quality_wipo_ipc_field_concordance.csv'))
    sub_f = conc[conc.main_group.isna()].set_index('subclass').wipo_field.to_dict()
    mg_f = {(r.subclass, int(r.main_group)): r.wipo_field for r in conc[conc.main_group.notna()].itertuples()}
    fmap = {m: mg_f.get((m.split(' ')[0], int(m.split(' ')[1])), sub_f.get(m.split(' ')[0], np.nan)) for m in ug}
    cf = np.array([fmap[m] for m in P['mg']], dtype=float)
    idx = np.searchsorted(P['cells'], D.cell.values)
    for j, cc in ((2, 'CN'), (3, 'JP')):
        x = n[:, j, :]
        for t0, g in D.groupby('t0'):
            F = np.ones((len(P['cells']), L.Y1 - L.Y0 + 1), dtype=np.float32)
            for f in np.unique(cf[np.isfinite(cf)]):
                if (cc, int(f)) not in R.index: continue
                r = R.loc[(cc, int(f))]
                ref = np.nanmedian([r.get(y, np.nan) for y in range(t0 - 6, t0)])
                for y in range(L.Y0, t0):
                    ry = r.get(y, np.nan)
                    if ry == ry and ry > 0 and ref == ref: F[cf == f, y - L.Y0] = np.clip(ref / ry, 0.5, 2.0)
            xc = x * F; ii = idx[g.index]
            W = L.wsum(xc, t0, -2, 0)
            D.loc[g.index, f'{cc}_mom_wc'] = L._g(xc, t0, 0)[ii]; D.loc[g.index, f'{cc}_mom_l1_wc'] = L._g(xc, t0, 1)[ii]
            D.loc[g.index, f'{cc}_mom_l2_wc'] = L._g(xc, t0, 2)[ii]; D.loc[g.index, f'{cc}_lvl_wc'] = np.log1p(W)[ii]
            D.loc[g.index, f'{cc}_g1_wc'] = np.log((xc[:, t0 - L.Y0] + 1) / (xc[:, t0 - 1 - L.Y0] + 1))[ii]
            D.loc[g.index, f'{cc}_slope5_wc'] = L._slope(xc, t0, 5)[ii]
    return D

def prep(D):
    fn = os.path.join(L.CACHE, 'early_t0.parquet')
    if VAR == 'pub' and os.path.exists(fn) and 'kr_early_n' not in D:
        E = pd.read_parquet(fn); n0 = len(D); D = D.merge(E, on=['cell', 't0'], how='left'); assert len(D) == n0
    if os.environ.get('T3R3_WC') == '1' and 'CN_mom_wc' not in D: D = cnjp_corrected(D)
    return D

def swap_wc(cfg):
    c2 = copy.deepcopy(cfg)
    for c in c2['components']: c['features'] = [f + '_wc' if f in CNJP else f for f in c['features']]
    return c2

OA2 = ['oa_fx_g', 'oa_comp_g']
MORE = {
 'E08': ('조기 공개 신호 5피처(t0년 말까지 공개된 문서를 출원연도로 재색인: KR 조기공개 수·비중, KR·해외 실시간 출원연도 모멘텀, 해외 조기공개 비중) → 3성분', feats(EARLY), '공개연도 변형 전용. 공개 지연 1.5년을 일부 메움'),
 'E09': ('WIPO 커버리지 보정: CN·JP 국가 모멘텀 12피처를 관청×WIPO분야×연도 커버리지 비율로 보정한 값으로 교체(y ≤ t0-1 보정)', swap_wc, 'T3R3_WC=1 로 실행. CN 수록 불규칙·JP 2021 결손 교정층'),
 'E10': ('학습 시작 원점 2005→2007 (공개연도 패널은 출원 2000년 시작이라 공개연도 2000–02 좌측절단)', set_top(first_t0=2007), ''),
 'E11': ('OpenAlex 핵심 2피처만(해외4국 논문 점유 성장, 기업 논문 성장) → 3성분', feats(OA2), 'E02 축소판'),
 'E12': ('OpenAlex 핵심 2피처 → 트리 2성분에만', feats(OA2, comps=('xgbc', 'nb')), ''),
}

EXT_ALL = ['oa_all_g', 'oa_fx_g', 'oa_kr_g', 'oa_comp_g', 'oa_comp_sh', 'oa_kr_sh', 'pr_exp_g3', 'pr_imp_g3', 'pr_ppi_g3', 'pr_ppi_g1', 'pr_tot',
           'us_rship_g3', 'us_emp_g3', 'us_tfp_g3', 'ip5_kr_sh', 'ip5_kr_sh_chg', 'ip5_w_g', 'kp_n', 'kp_lxi', 'kp_lxi_rel']
KS_EXT = ['ksic_kr_mom', 'ksic_tot_mom', 'pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2', 'oa_all_g', 'oa_fx_g', 'oa_kr_g', 'oa_comp_g', 'oa_comp_sh', 'oa_kr_sh',
          'pr_exp_g3', 'pr_imp_g3', 'pr_ppi_g3', 'us_rship_g3', 'us_emp_g3', 'us_tfp_g3']
_KH = {}
def ksic_hier_ext(D):
    """KSIC×원점 부상률을 KSIC 수준 외부자료(+KSIC 특허 모멘텀)로 예측하는 릿지. 원점 t 예측은 결과창이 t 이전에 끝난 원점(≤t-3)만 학습."""
    if 'T' not in _KH:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        fs = [f for f in KS_EXT if f in D.columns]
        K = D.groupby(['t0', 'ksic']).agg(rate=('yb', 'mean'), **{f: (f, 'first') for f in fs}).reset_index(); K['ksic_hier_ext'] = np.nan
        for t in sorted(K.t0.unique()):
            trk = K[(K.t0 <= t - 3)]
            if trk.t0.nunique() < 3: continue
            m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0)).fit(trk[fs].values, trk.rate.values)
            K.loc[K.t0 == t, 'ksic_hier_ext'] = m.predict(K.loc[K.t0 == t, fs].values)
        _KH['T'] = K[['t0', 'ksic', 'ksic_hier_ext']]
    n0 = len(D); D = D.merge(_KH['T'], on=['t0', 'ksic'], how='left'); assert len(D) == n0; return D

_prep0 = prep
def prep(D):
    D = _prep0(D)
    if 'ksic_hier_ext' not in D and 'oa_all_g' in D: D = ksic_hier_ext(D)
    return D

MORE.update({
 'E13': ('외부자료 20피처 묶음(OpenAlex 6·ECOS 물가 5·NBER 3·IP5 3·KPSS 3) → 음이항 성분에만', feats(EXT_ALL, comps=('nb',)), '계수모형만 산업 맥락을 쓰게'),
 'E14': ('계층 결합: KSIC 부상률 릿지(KSIC 특허 모멘텀+생산지수+OpenAlex+물가+NBER 17피처, 원점 ≤t-3 학습) 1피처 → 3성분', feats(['ksic_hier_ext']), '외부자료를 산업 수준 부상률로 요약해 주입'),
 'E15': ('음이항 성분 라운드 900→1300', set_param('nb', rounds=1300), '공개연도 변형 재조정'),
 'E16': ('XGB 분류 라운드 300→450', set_param('xgbc', rounds=450), ''),
 'E17': ('음이항 성분 3시드 배깅(순위평균)', set_param('nb', seeds=3), '분산 감소'),
 'E18': ('로지스틱에 CN·JP 12피처 복원', feats(CNJP, comps=('logit',)), '공개연도 변형에서 해외 신호 재평가'),
})

MORE.update({
 'E19': ('근소 미달 묶음(용량): XGB 분류 450라운드(E16) + 음이항 1300라운드(E15)', chain(set_param('xgbc', rounds=450), set_param('nb', rounds=1300)), '공개연도 변형은 학습 원점이 2년 더 많아 용량 증가 여지'),
 'E20': ('외부자료 20피처 묶음 → 음이항 성분에만 (새 챔피언 위 재시험, E13)', feats(EXT_ALL, comps=('nb',)), ''),
 'E21': ('로지스틱에 CN·JP 12피처 복원 (새 챔피언 위 재시험, E18)', feats(CNJP, comps=('logit',)), ''),
 'E22': ('WIPO 커버리지 보정 CN·JP 모멘텀 교체 (새 챔피언 위 재시험, E09)', swap_wc, 'T3R3_WC=1'),
})

EXT_KS = ['oa_all_g', 'oa_fx_g', 'oa_kr_g', 'oa_comp_g', 'oa_comp_sh', 'oa_kr_sh', 'pr_exp_g3', 'pr_imp_g3', 'pr_ppi_g3', 'pr_ppi_g1', 'pr_tot', 'us_rship_g3', 'us_emp_g3', 'us_tfp_g3']
EXT_CELL = ['ip5_kr_sh', 'ip5_kr_sh_chg', 'ip5_w_g', 'kp_n', 'kp_lxi', 'kp_lxi_rel']
MORE.update({
 'E23': ('외부자료 KSIC 수준 14피처(OpenAlex 6·ECOS 물가 5·NBER 3) → 음이항 성분에만', feats(EXT_KS, comps=('nb',)), 'E20에서 셀 수준(IP5·KPSS) 제외'),
 'E24': ('외부자료 셀 수준 6피처(IP5 3·KPSS 3) → 음이항 성분에만', feats(EXT_CELL, comps=('nb',)), 'E20의 셀 수준 부분'),
 'E25': ('외부자료 20피처 묶음 → 음이항 + XGB 분류 두 트리 성분', feats(EXT_ALL, comps=('nb', 'xgbc')), ''),
 'E26': ('음이항 성분 라운드 1300→1800', set_param('nb', rounds=1800), ''),
 'E27': ('XGB 분류 라운드 450→650', set_param('xgbc', rounds=650), ''),
})

_prep1 = prep
def prep(D):
    D = _prep1(D)
    if 'ix_fx_sci' not in D and 'oa_fx_g' in D:
        D['ix_fx_sci'] = D.fx_mom * D.oa_fx_g                     # 해외 특허 모멘텀 × 해외 과학 성장
        D['ix_us_dem'] = D.US_mom * D.us_rship_g3                 # 미국 특허 모멘텀 × 미국 시장 성장
        D['ix_catch'] = D.kr_share_rel_ksic * (-D.ip5_kr_sh)      # 셀 한국 점유 열세 × 분야 IP5 한국 점유 열세
        D['cg_gap'] = D.kr_share - np.log(D.ip5_kr_sh.clip(lower=1e-4))   # 제공 데이터 셀 한국 점유 − 분야 국제(IP5) 한국 점유 (log)
        D['cg_gap_chg'] = D.kr_share_chg - D.ip5_kr_sh_chg / D.ip5_kr_sh.clip(lower=1e-4)
    return D

MORE.update({
 'E28': ('외부×특허 상호작용 3피처: 해외 특허 모멘텀×해외 논문 성장, 미국 특허 모멘텀×미국 실질출하 성장, 셀 한국점유 열세×분야 IP5 한국점유 열세 → 3성분', feats(['ix_fx_sci', 'ix_us_dem', 'ix_catch']), '조합 C를 셀 신호와 곱해 셀 단위로 내림'),
 'E29': ('국제 위상 격차 2피처: 셀 한국 점유(제공 5개국) − 분야 IP5 한국 점유(log), 그 변화 → 3성분', feats(['cg_gap', 'cg_gap_chg']), '제공 데이터 셀 위상을 국제 고품질 패밀리 위상과 비교'),
})

def add_comp(c_new, w):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c2['components'].append(c_new); c2['weights'] = list(c2.get('weights', [1] * (len(c2['components']) - 1))) + [w]; return c2
    return f
KS_COMP = dict(type='logit', features=['ksic_kr_mom', 'ksic_tot_mom'] + PI3 + EXT_KS, C=0.1)
MORE.update({
 'E30': ('KOSIS 생산지수 성장률 3피처 제거 (E19 챔피언 위 순기여 재확인)', feats(drop=PI3), ''),
 'E31': ('4번째 성분: KSIC 수준 산업 맥락 로지스틱(KSIC 특허 모멘텀 2 + 생산지수 3 + OpenAlex 6 + 물가 5 + NBER 3, C=0.1), 순위평균 가중 0.3', add_comp(KS_COMP, 0.3), '외부자료를 산업 기울기(tilt)로 명시적으로 섞음'),
})
