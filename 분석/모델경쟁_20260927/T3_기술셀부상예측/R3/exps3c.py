"""T3 3라운드 B단계 재시도 실험 (공개연도 변형, 개발 원점 2010–2018). 각 실험 = 현재 챔피언(champion_pub.json)에 변경 1개.
재시도 수집분: OpenAlex 셀 v2(기업 소속 논문 523 + 균형 보강 ALL 558, r3oa), KOSIS 품목편 공개연도 권장 3피처(r3kosis) → build_ext_r3c.py
사용: python exps3c.py E45 E46 ...   (--force 재실행, --info 판정 없이 기록, --diag 단변량 진단만)
"""
import os, sys, copy, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true'); ap.add_argument('--info', action='store_true')
ap.add_argument('--diag', action='store_true'); ap.add_argument('exps', nargs='*'); a = ap.parse_args()
os.environ['T3R3_VAR'] = 'pub'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import r3core as C
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# exps3b.py 의 prep·헬퍼를 그대로 가져온다(main 블록 제외)
_src = open(os.path.join(HERE, 'exps3b.py'), encoding='utf-8').read().split("if __name__")[0]
_src = _src.replace("ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true'); ap.add_argument('--info', action='store_true')\nap.add_argument('exps', nargs='+'); a = ap.parse_args()", '')
_ns = {'__file__': os.path.join(HERE, 'exps3b.py'), '__name__': 'exps3b_lib'}; exec(_src, _ns)
prepB, feats, set_param, set_top, chain, comp = _ns['prep'], _ns['feats'], _ns['set_param'], _ns['set_top'], _ns['chain'], _ns['comp']
SUFC = os.environ.get('T3R3_EXTC_SUFFIX', '')
def comp_raw(c, tr, te):
    """학습 없는 성분: 피처 값 자체를 점수로(결측은 평가 원점 중앙값). 순위평균에 기울기로만 들어간다."""
    v = te[c['features'][0]].astype(float); return v.fillna(v.median()).fillna(0).values
C.COMP['raw'] = comp_raw

KS_NOW_FEATS = ['kp_gprod2', 'kp_gship2', 'kp_invlvl', 'pi_prod_g2', 'pi_ship_g2', 'ksic_kr_mom', 'ksic_tot_mom']
def ks_now_table(D):
    """KSIC 산업 nowcast 층: KSIC×원점 부상률(적격 셀 중 부상 비율)을 KSIC 수준 산업 피처로 예측하는 릿지.
    원점 t 의 값은 결과창이 t 이전에 끝난 원점(≤ t-3)만으로 학습 → 인과적. 학습 원점 3개 미만이면 NaN."""
    fs = [f for f in KS_NOW_FEATS if f in D.columns]
    K = D.groupby(['t0', 'ksic']).agg(rate=('yb', 'mean'), **{f: (f, 'first') for f in fs}).reset_index()
    K.loc[K.t0 + 3 > C.L.LABEL_MAX_YEAR, 'rate'] = np.nan     # 레이블 미관측 원점(최종·미래 원점)은 학습에 쓰지 않음
    K['ks_now'] = np.nan
    for t in sorted(K.t0.unique()):
        trk = K[(K.t0 <= min(t - 3, C.TRAIN_MAX_T0)) & K.rate.notna()]
        if trk.t0.nunique() < 3: continue
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0)).fit(trk[fs].values, trk.rate.values)
        K.loc[K.t0 == t, 'ks_now'] = m.predict(K.loc[K.t0 == t, fs].values)
    return K[['t0', 'ksic', 'ks_now']]

_KS = {}
def prepC(D, ks_src=None):
    D = prepB(D)
    if 'oa2_co_g3' in D: return D
    cf = pd.read_parquet(os.path.join(HERE, f'extC_cell_pub{SUFC}.parquet')); kf = pd.read_parquet(os.path.join(HERE, f'extC_ksic_pub{SUFC}.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    D['oa2_co_x_krgap'] = D.oa2_co_g3 - D.kr_mom        # 기업 과학 가속 대비 KR 특허 지체(해석층 지표)
    if ks_src is None:          # 적격 표본(레이블 있음)으로 nowcast 층 학습
        _KS['K'] = ks_now_table(D)
    D = D.merge(_KS['K'], on=['t0', 'ksic'], how='left'); assert len(D) == n0
    return D

OA2CO = ['oa2_co_g3', 'oa2_co_share']
OA2 = ['oa2_all_g3', 'oa2_co_g3', 'oa2_co_share']
KP = ['kp_gprod2', 'kp_gship2', 'kp_invlvl']
T2 = ('xgbc', 'nb')

EXPS = {
 'E45': ('OpenAlex 셀 v2 기업 소속 논문 2피처(기업 논문 점유 3년 성장 oa_co_g3, 기업 비중 수준 oa_co_share; 발행 ≤t0, 미수집 NaN) → 트리 2성분',
         feats(OA2CO, comps=T2), 'r3oa 재시도 권고: 공개연도 선행검정 +0.042 13/13, 비중 −0.056 1/13'),
 'E46': ('OpenAlex 셀 v2 기업 2피처 → 3성분(로지스틱은 중앙값 대치)', feats(OA2CO), ''),
 'E47': ('OpenAlex 셀 v2 3피처(균형 보강 558 전체 논문 점유 성장 + 기업 2피처) → 트리 2성분', feats(OA2, comps=T2), ''),
 'E48': ('KOSIS 품목편 공개연도 권장 3피처(연쇄 생산 2년 성장·연쇄 출하 2년 성장·재고/출하 수준, 자료연도 ≤t0-2) → 트리 2성분',
         feats(KP, comps=T2), 'r3kosis 공개연도 검정 +0.18/+0.18/+0.155, Bonferroni 통과'),
 'E49': ('KOSIS 품목편 공개연도 3피처 → 로지스틱 성분에만(KSIC 공변량 선형)', feats(KP, comps=('logit',)), ''),
 'E50': ('KSIC 산업 nowcast 층 ks_now(KSIC 부상률 ~ 품목편 3 + 생산지수 2 + KSIC 특허 모멘텀 2 릿지, 원점 ≤t-3 학습) → 3성분',
         feats(['ks_now']), '산업통계 → KSIC 부상률 → 셀 모형(계층 결합)'),
}

def load_more():
    fn = os.path.join(HERE, 'exps3c_more.py')
    if os.path.exists(fn):
        ns = dict(C=C, feats=feats, set_param=set_param, set_top=set_top, chain=chain, comp=comp, copy=copy, pd=pd, np=np,
                  OA2CO=OA2CO, OA2=OA2, KP=KP, T2=T2, os=os, HERE=HERE)
        exec(open(fn, encoding='utf-8').read(), ns); EXPS.update(ns.get('MORE', {}))

def install_wide():
    _gw = C.get_wide
    def gw(Dx, mt):
        w = _gw(Dx, mt)
        if 'oa2_co_g3' not in w: w = prepC(w, ks_src='D'); C._WIDE[mt] = w
        return w
    C.get_wide = gw

def diag(D):
    """단변량 진단: 원점별 순위 편상관(통제 kr_mom·fx_mom·kr_lvl·tot_lvl) + E40 개발 예측 점수까지 통제한 증분 편상관."""
    from sklearn.metrics import roc_auc_score
    pr = pd.read_parquet(os.path.join(HERE, 'preds', 'dev_preds_r3.parquet'))[['cell', 't0', 'p']]
    D = D[D.t0.isin(C.DEV_T0)].merge(pr, on=['cell', 't0'], how='left')
    rk = lambda v: pd.Series(v).rank().values
    rows = []
    for f in OA2 + ['oa2_all_lv3', 'oa2_co_x_krgap'] + KP + ['ks_now', 'sep_cum', 'sep_new3']:
        a_, pc, pc2 = [], [], []
        for t, g in D.groupby('t0'):
            m = g[f].notna()
            if m.sum() < 100 or g.yb[m].nunique() < 2: continue
            a_.append(roc_auc_score(g.yb[m], g[f][m]))
            for ctl, lst in ((['kr_mom', 'fx_mom', 'kr_lvl', 'tot_lvl'], pc), (['kr_mom', 'fx_mom', 'kr_lvl', 'tot_lvl', 'p'], pc2)):
                X = np.c_[np.ones(m.sum()), g.loc[m, ctl].fillna(0).values]
                res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
                lst.append(np.corrcoef(res(rk(g[f][m].values)), res(rk(g.y[m].values)))[0, 1])
        rows.append(dict(f=f, nonmiss=round(D[f].notna().mean(), 3), auc=round(np.mean(a_), 4), pcor=round(np.mean(pc), 4), pcor_pos=int(np.sum(np.array(pc) > 0)),
                         pcor_given_E40=round(np.mean(pc2), 4), pos_given_E40=int(np.sum(np.array(pc2) > 0)), n_org=len(pc)))
    out = pd.DataFrame(rows); print(out.to_string()); out.to_csv(os.path.join(HERE, 'extC_univariate.csv'), index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    load_more()
    done = set()
    if os.path.exists(C.EXP_CSV):
        df = pd.read_csv(C.EXP_CSV, encoding='utf-8-sig'); done = set(df[df.variant == 'pub'].exp_id)
    D = prepC(C.get_D())
    install_wide()
    if a.diag: diag(D)
    for e in a.exps:
        if e in done and not a.force: print('건너뜀', e); continue
        change, f, note = EXPS[e]
        cfg = f(C.load_champ()['cfg'])
        C.challenge(e, change, cfg, note, decide=not a.info, D=D)
