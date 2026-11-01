# -*- coding: utf-8 -*-
"""변형 채점(PREREG §2–§5). python s_score.py <변형이름|all>
 변형: OFF(공식 재현), S1full_La, S1full_Lb, S1fixed_La, S1fixed_Lb, S2_v_pub, S2_v_p18, S2_v_rgst, S3_La, S3_Lb
 점추정: AUC·skill(M0=6변형 최강)·신규 skill·상위 100/1,000·상위 1% 포착·SBB 대비 skill.
 CI: 법인 군집 부트스트랩 R=200(RandomState(3000+ci), ci=0..19, 10회씩), 뽑힌 횟수 가중 AUC. → cache/var/<변형>.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
import pickle, time
VD = CC/'var'; VD.mkdir(exist_ok=True)
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
OFFICIAL = dict(skill=0.47267044268107444, auc=0.9002187222427036, top1000=313, ci=[0.45362, 0.48845])


def rank_pct(s): return pd.Series(s).rank(pct=True).values           # fr.rank와 같은 식


def blend(px, pl): return 0.9 * rank_pct(px) + (1 - 0.9) * rank_pct(pl)   # final_run_r3.blend와 같은 식(0.09999999999999998)


class WAUC:
    """가중 AUC(동점 0.5). 점수별 정렬·동점 묶음을 미리 만든다."""
    def __init__(self, s, y):
        self.u, self.inv = np.unique(np.asarray(s, np.float64), return_inverse=True); self.y = np.asarray(y, np.float64); self.G = len(self.u)
    def __call__(self, w):
        wp = np.bincount(self.inv, weights=w * self.y, minlength=self.G); wn = np.bincount(self.inv, weights=w * (1 - self.y), minlength=self.G)
        below = np.cumsum(wn) - wn
        return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def auc(s, y): return WAUC(s, y)(np.ones(len(y)))


def assemble(name):
    """변형 모집단 행렬: firm,k,y,newc,px,pl,base6,sbb"""
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
    fz = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'px', 'pl']); assert (fz.firm.values == te.firm.values).all() and (fz.k.values == te.k.values).all()
    bo = pd.read_parquet(FC/'baseline_scores.parquet'); sb = np.load(CC/'p0'/'sbb_all.npy')
    L = pd.read_parquet(CC/'labels.parquet'); Lt = L[L.src == 'te'].reset_index(drop=True); assert (Lt.firm.values == te.firm.values).all()
    O = te.copy(); O['px'] = fz.px.values; O['pl'] = fz.pl.values; O['sbb'] = sb; O['y_off'] = Lt.y_u.values
    for c in BK: O[c] = bo[c].values
    O['is_C'] = False
    if name == 'OFF':
        D = O[~O.prior_any.values].copy(); D['y'] = D.y_off.values; return D, {}
    if name.startswith('S2_'):
        vcol = name[3:]; Q = pd.read_parquet(CC/f'pop_te_{vcol}.parquet'); assert (Q.firm.values == O.firm.values).all()
        m = ~O.prior_any.values & Q.fin.values; D = O[m].copy(); D['y'] = D.y_off.values
        return D, dict(D_dropped=int((~O.prior_any.values & ~Q.fin.values).sum()), D_dropped_pos=int(O.y_off.values[~O.prior_any.values & ~Q.fin.values].sum()),
                       D_dropped_firms=int(O.firm[~O.prior_any.values & ~Q.fin.values].nunique()))
    PP = pickle.load(open(CC/'build'/'pop.pkl', 'rb')); pv = pd.MultiIndex.from_tuples([tuple(x) for x in PP['prior_vis']])
    key = pd.MultiIndex.from_arrays([O.firm.values, O.k.values]); in_pv = key.isin(pv)
    lab = name.split('_')[-1]
    if name.startswith('S3_'):
        Bm = O.prior_any.values & ~in_pv; m = ~O.prior_any.values | Bm; D = O[m].copy()
        D['y'] = np.where(D.prior_any.values, 0 if lab == 'La' else 1, D.y_off.values)
        return D, dict(B_added=int(Bm.sum()), B_officialdef_pos=int(O.y_off.values[Bm].sum()), B_firms=int(O.firm[Bm].nunique()))
    # S1
    S = pd.read_parquet(CC/'build'/'s1F'/'scores.parquet')
    n_off = len(O); assert (S.firm.values[:0] == O.firm.values[:0]).all()
    Sk = pd.MultiIndex.from_arrays([S.firm.values, S.k.values]); pos = pd.Series(np.arange(len(S)), index=Sk)
    po = pos.reindex(key).values.astype(np.int64)                      # 공식 행의 s1F 위치
    Fvis = set(PP['Fvis']); fin = O.firm.isin(Fvis).values
    Bm = O.prior_any.values & ~in_pv & fin; Am = ~O.prior_any.values & fin
    A = O[Am | Bm].copy(); A['y'] = np.where(A.prior_any.values, 0 if lab == 'La' else 1, A.y_off.values)
    if name.startswith('S1full'):
        for c in ['px', 'pl', 'sbb'] + BK: A[c] = S[c].values[po[Am | Bm]]
    Ck = pd.MultiIndex.from_tuples([tuple(x) for x in PP['C']]); ci_ = pos.reindex(Ck).values.astype(np.int64)
    Cd = S.iloc[ci_][['firm', 'k', 'newc', 'px', 'pl', 'sbb'] + BK].copy(); Cd['prior_any'] = False; Cd['is_C'] = True
    Lc = L[L.src == 'Call'].set_index(['firm', 'k']).y_u.reindex(Ck).values
    Cd['y_off'] = Lc; Cd['y'] = 0 if lab == 'La' else 1
    D = pd.concat([A, Cd[A.columns]], ignore_index=True)
    vis_any = set(PP['vis_any_firms']); D['newc'] = ~D.firm.isin(vis_any).values          # 가시 신규진입자
    chg = None
    if name.startswith('S1full'):
        # C 법인 공식 행의 점수 변화(S1 구성 반영) 규모
        d_px = np.abs(S.px.values[po] - O.px.values); chg = dict(rows_px_changed=int((d_px > 0).sum()), max_abs_px_change=float(d_px.max()),
                                                               rows_unchanged_outside_Cfirms=bool((d_px[~O.firm.isin(set(PP['firms_sel'])).values] == 0).all()))
    info = dict(A=int(Am.sum()), A_pos=int(O.y_off.values[Am].sum()), B_added=int(Bm.sum()), B_officialdef_pos=int(O.y_off.values[Bm].sum()),
                C_added=int(len(Cd)), C_officialdef_pos=int(np.nansum(Lc)), C_firms=int(Cd.firm.nunique()),
                D_dropped=int((~O.prior_any.values & ~fin).sum()), D_dropped_pos=int(O.y_off.values[~O.prior_any.values & ~fin].sum()),
                D_dropped_firms=int(O.firm[~O.prior_any.values & ~fin].nunique()), score_change=chg)
    return D, info


def run(name, R=200, chunk=10):
    fo = VD/f'{name}.json'
    if fo.exists(): print('exists', name); return
    D, info = assemble(name); score_df(name, D, info, R, chunk)


def score_df(name, D, info, R=200, chunk=10):
    fo = VD/f'{name}.json'; t0 = time.time()
    if 'is_C' not in D: D['is_C'] = False
    y = D.y.values.astype(np.int8); s = blend(D.px.values, D.pl.values); nc = D.newc.values.astype(bool); firm = D.firm.values
    ma = auc(s, y); ba = {c: auc(D[c].values, y) for c in BK}; M0 = max(ba.values()); arg = max(ba, key=ba.get)
    man = auc(s[nc], y[nc]); ban = max(auc(D[c].values[nc], y[nc]) for c in BK)
    sbb = auc(D.sbb.values, y)
    o = np.argsort(-s, kind='stable')
    res = dict(variant=name, n=int(len(D)), firms=int(D.firm.nunique()), pos=int(y.sum()), base_rate=float(y.mean()), new_rows=int(nc.sum()), new_pos=int(y[nc].sum()),
               auc=ma, auc_M0=M0, M0_arg=arg, base_aucs=ba, skill=(ma - M0) / (1 - M0), auc_new=man, skill_new=(man - ban) / (1 - ban),
               auc_sbb=sbb, skill_vs_sbb=(ma - sbb) / (1 - sbb),
               top100=int(y[o[:100]].sum()), top1000=int(y[o[:1000]].sum()), capture_top1pct=float(y[o[:len(y) // 100]].sum() / y.sum()),
               top1000_C=int(D.is_C.values[o[:1000]].sum()), info=info)
    if D.is_C.any():
        rC = pd.Series(s).rank(pct=True).values[D.is_C.values]; res['C_score_pct'] = dict(mean=float(rC.mean()), median=float(np.median(rC)), p90=float(np.percentile(rC, 90)))
        rB = pd.Series(D[BK[5]].values).rank(pct=True).values[D.is_C.values]; res['C_xgb_y_u_pct'] = dict(mean=float(rB.mean()), median=float(np.median(rB)))
    # 부트스트랩
    fr_, inv = np.unique(firm, return_inverse=True); nF = len(fr_)
    W = [WAUC(s, y)] + [WAUC(D[c].values, y) for c in BK] + [WAUC(D.sbb.values, y)]
    A = np.zeros((R, len(W)))
    for ci in range(int(np.ceil(R / chunk))):
        rng = np.random.RandomState(3000 + ci)
        for r in range(chunk):
            cnt = np.bincount(rng.randint(0, nF, nF), minlength=nF).astype(np.float64); w = cnt[inv]
            A[ci * chunk + r] = [f(w) for f in W]
    bmax = A[:, 1:7].max(1); sk = (A[:, 0] - bmax) / (1 - bmax); sks = (A[:, 0] - A[:, 7]) / (1 - A[:, 7])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    res['ci'] = q(sk); res['ci_vs_sbb'] = q(sks); res['n_boot'] = R
    res['M0_class'] = 'CI하한>0' if res['ci'][0] > 0 else 'CI하한<=0'; res['SBB_class'] = 'CI하한>0' if res['ci_vs_sbb'][0] > 0 else 'CI하한<=0'
    np.save(VD/f'{name}_boot.npy', A)
    res['secs'] = round(time.time() - t0); res['peak'] = G.rss_peak()
    jdump(res, fo); print(name, 'n', res['n'], 'pos', res['pos'], 'skill', round(res['skill'], 5), [round(x, 5) for x in res['ci']], 'sbb', round(res['skill_vs_sbb'], 4),
                         'top1000', res['top1000'], res['secs'], 's', res['peak'], flush=True)


if __name__ == '__main__':
    names = sys.argv[1:]
    for nm in names: run(nm)
