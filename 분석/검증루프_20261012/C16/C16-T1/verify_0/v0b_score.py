# -*- coding: utf-8 -*-
"""verify_0 (독립 경로 2/2): v0a 모집단 + 저장 점수(FROZEN·baseline_scores·s1F scores)·레이블(cache/labels.parquet)로
 S2·S1(fixed/full)·L-a Δskill을 scipy rankdata Mann–Whitney로 재계산(sklearn·s_score.py·트랙 지표 함수 미사용).
 반박 시도: (r1) v_rgst 가시일 S1 경계(점수 없는 C 쌍은 최악·최선 대입) (r2) v_pub C 쌍 최악·최선 대입 (r3) 공식 법인 재표본으로 짝지은 Δskill(S1full_La − 공식) 분포."""
import sys, json, pickle, pathlib, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding='utf-8')
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"); HERE = pathlib.Path(__file__).resolve().parent; CC = HERE.parent/'cache'
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
t0 = time.time()


def mw_auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def blend(px, pl):
    return 0.9 * (rankdata(px) / len(px)) + (1 - 0.9) * (rankdata(pl) / len(pl))     # rank(pct=True) = 평균순위/n


def skill_of(D):
    y = D.y.values.astype(int); m = mw_auc(blend(D.px.values, D.pl.values), y); b = max(mw_auc(D[c].values, y) for c in BK)
    return (m - b) / (1 - b), m, b


V = pickle.load(open(HERE/'v0_pop.pkl', 'rb')); PP = pickle.load(open(CC/'build'/'pop.pkl', 'rb'))
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'prior_any']); fz = pd.read_parquet(RD/'o2019.parquet', columns=['px', 'pl'])
bo = pd.read_parquet(FC/'baseline_scores.parquet'); L = pd.read_parquet(CC/'labels.parquet'); Lt = L[L.src == 'te'].reset_index(drop=True)
O = te.copy(); O['px'] = fz.px.values; O['pl'] = fz.pl.values; O['y_off'] = Lt.y_u.values
for c in BK: O[c] = bo[c].values
S = pd.read_parquet(CC/'build'/'s1F'/'scores.parquet'); Sk = pd.MultiIndex.from_arrays([S.firm.values, S.k.values]); spos = pd.Series(np.arange(len(S)), index=Sk)
key = pd.MultiIndex.from_arrays([O.firm.values, O.k.values])
res = {}
# 모집단 대조
tef = set(O.firm)
res['pop_compare'] = dict(Foff_indep=len(V['Foff']), te_firms=len(tef), Foff_indep_minus_te=len(V['Foff'] - tef), te_minus_Foff_indep=len(tef - V['Foff']),
                          Fvis_indep=len(V['Fvis']), Fvis_main=len(PP['Fvis']), Fvis_equal=V['Fvis'] == set(PP['Fvis']),
                          C_indep=len(V['C']), C_main=len(PP['C']), C_overlap=len(set(V['C']) & set(map(tuple, PP['C']))),
                          C_indep_in_main_Call=len(set(V['C']) & set(map(tuple, PP['Call']))),
                          prior_vis_indep=len(V['prior_vis']), prior_vis_main=len(PP['prior_vis']), prior_vis_overlap=len(V['prior_vis'] & set(map(tuple, PP['prior_vis']))),
                          prior_any_rebuilt_eq=bool((np.array([(f, k) in V['prior_all'] for f, k in zip(O.firm, O.k)]) == O.prior_any.values).all()))
off_sk, off_m, off_b = skill_of(O[~O.prior_any.values].assign(y=lambda d: d.y_off))
res['OFF'] = dict(skill=off_sk, auc=off_m, M0=off_b)


def build(Fvis, prior_vis, Cset, full, lab='La', cfill=None):
    pv = key.isin(pd.MultiIndex.from_tuples(list(prior_vis))); fin = O.firm.isin(Fvis).values
    Am = ~O.prior_any.values & fin; Bm = O.prior_any.values & fin & ~pv
    A = O[Am | Bm].copy(); A['y'] = np.where(A.prior_any.values, 0 if lab == 'La' else 1, A.y_off.values)
    if full:
        po = spos.reindex(key[Am | Bm]).values.astype(int)
        for c in ['px', 'pl'] + BK: A[c] = S[c].values[po]
    Ck = pd.MultiIndex.from_tuples(list(Cset)) if len(Cset) else pd.MultiIndex.from_arrays([[], []]); have = Ck.isin(Sk)
    Cd = S.iloc[spos.reindex(Ck[have]).values.astype(int)][['firm', 'k', 'px', 'pl'] + BK].copy()
    n_miss = int((~have).sum())
    if n_miss and cfill is not None:
        miss = pd.DataFrame({'firm': [f for f, _ in Ck[~have]], 'k': [k for _, k in Ck[~have]]})
        hi, lo = 2.0, -1.0
        miss['px'] = hi if cfill == 'worst' else lo; miss['pl'] = hi if cfill == 'worst' else lo
        for c in BK: miss[c] = lo if cfill == 'worst' else hi
        Cd = pd.concat([Cd, miss], ignore_index=True)
    Cd['y'] = 0 if lab == 'La' else 1; Cd['prior_any'] = False
    D = pd.concat([A[['firm', 'k', 'y', 'px', 'pl'] + BK], Cd[['firm', 'k', 'y', 'px', 'pl'] + BK]], ignore_index=True)
    return D, dict(A=int(Am.sum()), B=int(Bm.sum()), C_scored=int(have.sum()), C_missing=n_miss, n=int(len(D)), pos=int(D.y.sum()))


for nm, kw in {'S2_indep': None, 'S1fixed_La_indep': dict(full=False), 'S1full_La_indep': dict(full=True)}.items():
    if kw is None:
        fin = O.firm.isin(V['Fvis']).values; D = O[~O.prior_any.values & fin].assign(y=lambda d: d.y_off); info = dict(n=int(len(D)), pos=int(D.y.sum()))
    else:
        D, info = build(V['Fvis'], V['prior_vis'], V['C'], **kw)
    sk, m, b = skill_of(D); res[nm] = dict(skill=sk, delta=sk - off_sk, auc=m, M0=b, **info)
# 작업 경로 결과와 대조(같은 정의에서 차 ≤0.001)
main = {k: json.load(open(CC/'var'/f'{k}.json', encoding='utf-8'))['skill'] for k in ('OFF', 'S2_v_pub', 'S1fixed_La', 'S1full_La')}
res['main'] = main
# 작업 경로 정의(주 C 집합·주 prior_vis)로 같은 계산 → 코드 독립 재현
for nm, full in (('S1fixed_La_mainsets', False), ('S1full_La_mainsets', True)):
    D, info = build(set(PP['Fvis']), set(map(tuple, PP['prior_vis'])), set(map(tuple, PP['C'])), full)
    sk, m, b = skill_of(D); res[nm] = dict(skill=sk, diff_vs_main=sk - main[nm.replace('_mainsets', '')], **info)
# (r1) v_rgst 경계, (r2) v_pub C 최악·최선
for nm, (F_, Cs) in {'r1_vrgst': (V['Fvr'], V['C_vrgst']), 'r2_vpub': (V['Fvis'], V['C'])}.items():
    for fill in ('worst', 'best'):
        if nm == 'r2_vpub':   # 모든 C 점수를 극단값으로
            D, info = build(F_, V['prior_vis'], [], False)
            ext = pd.DataFrame({'firm': [f for f, _ in Cs], 'k': [k for _, k in Cs]}); ext['px'] = 2.0 if fill == 'worst' else -1.0; ext['pl'] = ext.px
            for c in BK: ext[c] = -1.0 if fill == 'worst' else 2.0
            ext['y'] = 0; D = pd.concat([D, ext[D.columns]], ignore_index=True); info['C_extreme'] = len(ext)
        else:
            D, info = build(F_, V['prior_vis'], Cs, False, cfill=fill)
        sk, m, b = skill_of(D); res[f'{nm}_{fill}'] = dict(skill=sk, delta=sk - off_sk, **info)
# (r3) 짝지은 부트스트랩 Δskill(S1full_La − 공식), 공식 법인 목록 재표본(RandomState(5000+i)), 가중 AUC(정렬 누적합 구현)
D1, _ = build(V['Fvis'], V['prior_vis'], V['C'], True); D0 = O[~O.prior_any.values].assign(y=lambda d: d.y_off)[D1.columns]
firms = np.array(sorted(set(O.firm))); fidx = pd.Series(np.arange(len(firms)), index=firms)


def prep(D):
    y = D.y.values.astype(float); f = fidx[D.firm.values].values; cols = [blend(D.px.values, D.pl.values)] + [D[c].values for c in BK]
    pre = []
    for s in cols:
        o = np.argsort(s, kind='mergesort'); ss = s[o]; g = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]); pre.append((o, g, int(g[-1]) + 1))
    return y, f, pre


def wauc(y, w, o, g, G):
    yo = y[o]; wo = w[o]; wp = np.bincount(g, wo * yo, G); wn = np.bincount(g, wo * (1 - yo), G)
    return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


P0, P1 = prep(D0), prep(D1); dl = []
for i in range(200):
    cnt = np.bincount(np.random.RandomState(5000 + i).randint(0, len(firms), len(firms)), minlength=len(firms)).astype(float)
    sk = []
    for (y, f, pre) in (P0, P1):
        w = cnt[f]; a = [wauc(y, w, *p) for p in pre]; b = max(a[1:]); sk.append((a[0] - b) / (1 - b))
    dl.append(sk[1] - sk[0])
dl = np.array(dl); res['r3_paired_delta'] = dict(point=res['S1full_La_indep']['delta'], ci=[float(np.percentile(dl, 2.5)), float(np.median(dl)), float(np.percentile(dl, 97.5))],
                                                share_below_minus_0p0087=float((dl <= -0.0087).mean()), share_below_0=float((dl < 0).mean()))
res['secs'] = round(time.time() - t0)
json.dump(res, open(HERE/'v0b_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float); print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
