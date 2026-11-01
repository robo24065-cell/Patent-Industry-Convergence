"""T3 3라운드 A단계 실험 목록. 각 실험 = 현재 챔피언(해당 변형) 설정에 변경 1개.
사용: python exps3.py --var pub E01 E02 ...   (--force: 재실행, --info: 판정 없이 기록)
  E00A = 2라운드 챔피언(E57-audit)을 출원연도 개발 원점 2010–2016 에서 재채점
  E00P = 같은 설정을 공개연도 변형 개발 원점 2010–2018 에서 재채점 (3라운드 최종시험 대상 변형의 출발 챔피언)
"""
import os, sys, copy, json, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--var', default='pub'); ap.add_argument('--force', action='store_true'); ap.add_argument('--info', action='store_true')
ap.add_argument('exps', nargs='+'); a = ap.parse_args()
os.environ['T3R3_VAR'] = a.var
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r3core as C
import pandas as pd, numpy as np

R2SPEC = json.load(open(os.path.join(HERE, '..', 'R2', 'final_spec_r2.json'), encoding='utf-8'))

def comp(cfg, t): return [c for c in cfg['components'] if c['type'] == t][0]
def set_param(t, **kw):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c = comp(c2, t)
        for k, v in kw.items():
            if k == 'params': c['params'].update(v)
            else: c[k] = v
        return c2
    return f
def set_top(**kw):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c2.update(kw); return c2
    return f
def feats(add=(), drop=(), comps=('logit', 'xgbc', 'nb')):
    return lambda cfg: C.with_feats(cfg, add, drop, comps)
def chain(*fs):
    def f(cfg):
        for g in fs: cfg = g(cfg)
        return cfg
    return f

PI3 = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']
OA = ['oa_all_g', 'oa_fx_g', 'oa_kr_g', 'oa_comp_g', 'oa_comp_sh', 'oa_kr_sh']
PR = ['pr_exp_g3', 'pr_imp_g3', 'pr_ppi_g3', 'pr_ppi_g1', 'pr_tot']
US = ['us_rship_g3', 'us_emp_g3', 'us_tfp_g3']
IP5 = ['ip5_kr_sh', 'ip5_kr_sh_chg', 'ip5_w_g']
KP = ['kp_n', 'kp_lxi', 'kp_lxi_rel']
RADAR = ['cr_radar', 'cr_gap_sci', 'cr_gap_dem']

EXPS = {
 'E00A': ('출발 챔피언 = 2라운드 동결 E57-audit(로지스틱42+XGB80+음이항80, 생산지수 성장률 3피처 포함) — 출원연도 개발 원점 2010–2016 재채점', None, '2라운드 dev_result_audit.json 0.4123 재현 점검'),
 'E00P': ('출발 챔피언(공개연도 변형) = E57-audit 설정 그대로, 셀 집계를 공개연도로(t0년 말까지 공개분만)·레이블 KR 공개연도 t0+1..t0+3 성장·생산지수 창 전년12월~당해11월 — 개발 원점 2010–2018', None, '3라운드 최종시험 대상 변형의 출발점'),
 'E01': ('KOSIS 생산지수 성장률 3피처(pi_prod_g2·pi_prod_g1·pi_ship_g2) 3성분에서 제거 — 순기여 확인', feats(drop=PI3), '2라운드 감사 후 남은 외부데이터 3피처의 순기여 = 제거 시 하락분'),
 'E02': ('OpenAlex KSIC 논문 6피처(전세계·해외4국·한국 논문 점유 3년 성장, 기업 논문 성장, 기업 논문 비중, 한국 과학 특화도; 발행연도 ≤t0) → 3성분', feats(OA), '과학 → 특허 선행 가설. 준-정합(현재 스냅샷)'),
 'E03': ('ECOS 물가 5피처(수출·수입·생산자 물가 3년 변화, PPI 1년 변화, 교역조건; 연도 ≤t0-1) → 3성분', feats(PR), '수요(가격) 견인 가설. 수출물가 27 KSIC, PPI 44 KSIC'),
 'E04': ('NBER-CES 미국 실질출하·고용·TFP 3년 성장 3피처(연도 ≤t0-3) → 3성분', feats(US), '미국 시장 수요 견인'),
 'E05': ('OECD IP5 한국 점유(WIPO35, 발명자·우선권연도 ≤t0-3) 수준·변화·세계 IP5 성장 3피처(셀 → 메인그룹의 WIPO 분야) → 3성분', feats(IP5), '따라잡기: 한국이 약한 분야가 이후 더 성장'),
 'E06': ('KPSS 특허가치 xi 셀 3피처(연결 US 특허 수, 평균 log xi, 같은 KSIC 대비 상대 xi; issue_date ≤t0, 출원 t0-5..t0) → 3성분', feats(KP), '시장가치 높은 셀의 부상'),
 'E07': ('조합 C 따라잡기 레이더 3피처: z(−IP5 한국점유)+z(해외 논문 성장)+z(미국 실질출하 성장), 과학 격차=z(해외 논문)−z(KSIC KR 특허 모멘텀), 수요 격차=z(미국 출하)−z(KSIC KR 모멘텀) → 3성분', feats(RADAR), '한국은 아직 약한데 해외 과학·시장이 이미 뜨는 칸. z는 원점별 50 KSIC 단면(IP5는 원점별 셀 단면)'),
}

_RT = {}
def radar(D):
    """조합 C: KSIC 수준 성분은 원점별 KSIC 단면에서 z화, IP5(분야)는 원점별 셀 단면 z → 합성. 확대 표본에도 같은 기준을 쓰도록 표를 저장."""
    if not all(c in D for c in ['ip5_kr_sh', 'oa_fx_g', 'us_rship_g3', 'ksic_kr_mom']): return D
    if 'K' not in _RT:
        K = D.drop_duplicates(['t0', 'ksic'])[['t0', 'ksic', 'oa_fx_g', 'us_rship_g3', 'ksic_kr_mom']].copy()
        for c in ['oa_fx_g', 'us_rship_g3', 'ksic_kr_mom']:
            K['z_' + c] = (K[c] - K.groupby('t0')[c].transform('mean')) / K.groupby('t0')[c].transform('std')
        _RT['K'] = K[['t0', 'ksic', 'z_oa_fx_g', 'z_us_rship_g3', 'z_ksic_kr_mom']]
        _RT['ip'] = D.groupby('t0').ip5_kr_sh.agg(['mean', 'std'])
    n0 = len(D); D = D.drop(columns=[c for c in RADAR if c in D]).merge(_RT['K'], on=['t0', 'ksic'], how='left'); assert len(D) == n0
    ip = _RT['ip']; zi = (D.ip5_kr_sh - D.t0.map(ip['mean'])) / D.t0.map(ip['std'])
    D['cr_radar'] = (-zi).fillna(0) + D.z_oa_fx_g.fillna(0) + D.z_us_rship_g3.fillna(0)
    D['cr_gap_sci'] = D.z_oa_fx_g - D.z_ksic_kr_mom
    D['cr_gap_dem'] = D.z_us_rship_g3 - D.z_ksic_kr_mom
    return D.drop(columns=['z_oa_fx_g', 'z_us_rship_g3', 'z_ksic_kr_mom'])

def load_extra():
    """추가 실험 정의(exps3_more.py)가 있으면 EXPS 갱신."""
    fn = os.path.join(HERE, 'exps3_more.py')
    if os.path.exists(fn):
        ns = dict(C=C, feats=feats, set_param=set_param, set_top=set_top, chain=chain, comp=comp, copy=copy, pd=pd, np=np, radar=radar,
                  PI3=PI3, OA=OA, PR=PR, US=US, IP5=IP5, KP=KP, RADAR=RADAR, os=os, HERE=HERE, VAR=a.var)
        exec(open(fn, encoding='utf-8').read(), ns); EXPS.update(ns.get('MORE', {}))
        return ns
    return {}

if __name__ == '__main__':
    ns = load_extra()
    done = set()
    if os.path.exists(C.EXP_CSV):
        df = pd.read_csv(C.EXP_CSV, encoding='utf-8-sig'); done = set(df[df.variant == a.var].exp_id)
    D = C.get_D(); D = radar(D)
    if 'prep' in ns: D = ns['prep'](D)
    _gw = C.get_wide
    def gw(Dx, mt):
        w = _gw(Dx, mt)
        if 'cr_radar' not in w:
            w = radar(w)
            if 'prep' in ns: w = ns['prep'](w)
            C._WIDE[mt] = w
        return w
    C.get_wide = gw
    for e in a.exps:
        if e in done and not a.force: print('건너뜀', e); continue
        change, f, note = EXPS[e]
        if f is None:
            C.rescore(e, change, R2SPEC['cfg'], note, D=D)
        else:
            cfg = f(C.load_champ()['cfg'])
            C.challenge(e, change, cfg, note, decide=not a.info, D=D)
