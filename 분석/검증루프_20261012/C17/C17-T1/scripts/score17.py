# -*- coding: utf-8 -*-
"""변형 채점(PREREG §2–§4). python score17.py <변형...>
 점추정: AUC·skill(M0=6변형 최강)·신규 skill·상위 100/1,000·상위 1% 포착·SBB 대비 skill.
 짝지은 법인 군집 부트스트랩 B=1000(우주 U = 공식 법인 ∪ RC 법인 ∪ 가짜 슬롯 Z00001–Z03912, chunk c=0..9 RandomState(17000+c) 100회).
 → cache/var/<변형>.json, <변형>_boot.npy (B×8: 모형, 기준선 6, SBB)"""
import sys, pathlib, pickle, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
VD = CC/'var'; VD.mkdir(exist_ok=True); BB = 1000; THR = 0.0087
SC = ['px', 'pl', 'sbb'] + BK


def rank_pct(s): return pd.Series(s).rank(pct=True).values


def blend(px, pl): return 0.9 * rank_pct(px) + (1 - 0.9) * rank_pct(pl)


class WAUC:
    def __init__(self, s, y):
        self.u, self.inv = np.unique(np.asarray(s, np.float64), return_inverse=True); self.y = np.asarray(y, np.float64); self.G = len(self.u)
    def __call__(self, w):
        wp = np.bincount(self.inv, weights=w * self.y, minlength=self.G); wn = np.bincount(self.inv, weights=w * (1 - self.y), minlength=self.G)
        below = np.cumsum(wn) - wn
        return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


class FastAUC:
    """법인 가중 AUC를 재표본 여러 개에 한 번에(동점 0.5). 음성을 '양성 고유 점수 사이 구간'으로 묶은 희소 행렬 곱 — WAUC와 수학적으로 같음."""
    def __init__(self, s, y, ui, nU):
        import scipy.sparse as sp
        s = np.asarray(s, np.float64); y = np.asarray(y).astype(bool)
        u = np.unique(s[y]); m = len(u)
        sn = s[~y]; e = np.searchsorted(u, sn, 'left'); t = (e < m) & (u[np.minimum(e, m - 1)] == sn)
        c = e + t.astype(np.int64); fn = ui[~y]; fp = ui[y]; jp = np.searchsorted(u, s[y], 'left')
        one = lambda r, cc, sh: sp.csr_matrix((np.ones(len(r), np.float64), (r, cc)), shape=sh)
        self.Nc = one(c, fn, (m + 1, nU)); self.Nt = one(e[t], fn[t], (m, nU)); self.Pj = one(jp, fp, (m, nU)); self.m = m
    def __call__(self, cnt):
        W = cnt.T.astype(np.float64)                        # nU × R
        wc = self.Nc @ W; wt = self.Nt @ W; wp = self.Pj @ W
        below = np.cumsum(wc, 0)[:self.m]
        num = (wp * (below + 0.5 * wt)).sum(0); return num / (wp.sum(0) * wc.sum(0))


def auc(s, y): return WAUC(s, y)(np.ones(len(y)))


def universe():
    p = CC/'boot_universe.parquet'
    if p.exists(): return pd.read_parquet(p).firm.values
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm']); RC = pd.read_parquet(CC/'rc_firms.parquet').firm
    U = np.array(sorted(set(te.firm) | set(RC) | {f'Z{i + 1:05d}' for i in range(3912)}), dtype=object)
    pd.DataFrame(dict(firm=U)).to_parquet(p, index=False); return U


def counts():
    p = CC/'boot_counts.npy'
    if p.exists(): return np.load(p)
    U = universe(); nU = len(U); C = np.zeros((BB, nU), np.uint8)
    for c in range(BB // 100):
        rng = np.random.RandomState(17000 + c)
        for r in range(100): C[c * 100 + r] = np.bincount(rng.randint(0, nU, nU), minlength=nU)
    np.save(p, C); return C


# ------------------------------------------------------------------ 모집단 조립
def pseudo_rows(base, tag, pool):
    """가짜 법인 행: 공여 법인의 공식 채점 행 점수를 복사. pool: neg(공식 양성 0 신규진입자)|all(신규진입자 전부)"""
    O = base['O']; Ok = O[~O.prior_any.values]
    nw = Ok[Ok.newc.values]; fp = nw.groupby('firm').y_off.max()
    donors = np.array(sorted(fp.index[fp == 0])) if pool == 'neg' else np.array(sorted(fp.index))
    ps = pd.read_parquet(CC/f'pseudo_{tag}.parquet')
    dsel = donors[np.minimum((ps.donor_u.values * len(donors)).astype(int), len(donors) - 1)]
    g = {f: d for f, d in nw.groupby('firm')}
    parts = []
    for slot, dn, pk, pr in zip(ps.slot, dsel, ps.pos_k, ps.prior_k):
        d = g[dn].copy(); pr = set(pr); d = d[~d.k.isin(pr)]
        d['y'] = d.k.isin(set(pk)).astype(np.int8); d['firm'] = slot; d['donor'] = dn; parts.append(d)
    P = pd.concat(parts, ignore_index=True); P['is_new'] = True; P['newc'] = True
    return P, dict(pool=pool, donors=int(len(donors)), pseudo_firms=int(ps.slot.nunique()), pseudo_rows=int(len(P)), pseudo_pos=int(P.y.sum()))


def load_base():
    O = official_frame(); E = pd.read_parquet(CC/'evid_all.parquet'); return dict(O=O, E=E)


def assemble(name, base):
    O = base['O'].drop(columns=['y_reg', 'y_pub']); E = base['E']; Eo = E[E.is_newc_off].set_index('firm')
    Ok = O[~O.prior_any.values].copy(); Ok['y'] = Ok.y_off.values; Ok['is_new'] = False
    key = lambda d: pd.MultiIndex.from_arrays([d.firm.values, d.k.values])
    if name == 'OFF': return Ok, {}
    def n1_drop(col):
        bad = set(Eo.index[~Eo[col]]); return Ok[~(Ok.newc.values & Ok.firm.isin(bad).values)], dict(dropped_firms=len(bad), dropped_rows=int((Ok.newc.values & Ok.firm.isin(bad).values).sum()),
                                                                                                    dropped_pos=int(Ok.y.values[Ok.newc.values & Ok.firm.isin(bad).values].sum()))
    if name in ('N1', 'N1nar', 'N1rl', 'N1wide'):
        col = dict(N1='N1_primary', N1nar='N1_narrow', N1rl='N1_rlins', N1wide='N1_wide')[name]; return n1_drop(col)
    if name == 'N2':
        m = ~Ok.newc.values; return Ok[m], dict(dropped_rows=int((~m).sum()), dropped_pos=int(Ok.y.values[~m].sum()), dropped_firms=int(Ok.firm[~m].nunique()))
    if name in ('N1re', 'N2re'):
        S = pd.read_parquet(CC/'build'/('n1F' if name == 'N1re' else 'n2F')/'scores.parquet')
        D, info = n1_drop('N1_primary') if name == 'N1re' else (Ok[~Ok.newc.values], {})
        pos = pd.Series(np.arange(len(S)), index=key(S)).reindex(key(D)); assert pos.notna().all(), 'build rows missing'
        p = pos.values.astype(np.int64); D = D.copy()
        assert (S.prior_any.values[p] == False).all()
        extra = int((~S.prior_any.values).sum() - len(D)); info = dict(info, build_scored_minus_pop=extra)
        for c in SC: D[c] = S[c].values[p]
        return D, info
    if name.startswith(('N3_', 'N3r_', 'N3x3_')):
        kind, pool, tag = name.split('_')            # 예: N3_neg_s1, N3r_all_s2, N3x3_neg_x3
        P, info = pseudo_rows(base, tag, pool)
        A = Ok if kind in ('N3', 'N3x3') else n1_drop('N1_primary')[0]
        return pd.concat([A, P[A.columns]], ignore_index=True), info
    if name.startswith('R1'):
        S = pd.read_parquet(CC/'build'/'rcF'/'scores.parquet'); L = pd.read_parquet(CC/'lab_rc.parquet')
        RC = set(L.firm); Sr = S[S.firm.isin(RC).values & ~S.prior_any.values].copy()
        Lr = L[~L.prior_any].set_index(['firm', 'k']).reindex(key(Sr)); assert Lr.y_u.notna().all()
        Sr['y'] = Lr.y_u.values.astype(np.int8); Sr['y_off'] = Sr.y.values; Sr['is_new'] = True
        info = dict(rc_rows=int(len(Sr)), rc_pos=int(Sr.y.sum()), rc_firms=int(Sr.firm.nunique()), lab_rows_scored=int((~L.prior_any).sum()))
        if name in ('R1re', 'R1re_worst', 'R1re_best', 'R1re_rand'):
            pos = pd.Series(np.arange(len(S)), index=key(S)).reindex(key(Ok)); assert pos.notna().all(); p = pos.values.astype(np.int64)
            A = Ok.copy()
            for c in SC: A[c] = S[c].values[p]
            assert (S.prior_any.values[p] == False).all()
        else:
            A = Ok
        D = pd.concat([A, Sr[A.columns]], ignore_index=True)
        if name.endswith('_rand'):                      # [사후 탐색, PREREG 밖] RC 행 모형 점수를 공식 채점 행에서 무작위 행(px·pl 함께) 추출로 대체 = 모형이 RC 법인을 구별 못 하는 경우
            m = D.is_new.values; rng = np.random.RandomState(17); ix = rng.randint(0, len(Ok), int(m.sum()))
            for c in ('px', 'pl'):
                v = D[c].values.astype(np.float64).copy(); v[m] = Ok[c].values[ix]; D[c] = v
        if name.endswith(('_worst', '_best')):
            m = D.is_new.values; yy = D.y.values; hi = float(max(D.px.max(), D.pl.max())) + 1.0; lo = float(min(D.px.min(), D.pl.min())) - 1.0
            good = name.endswith('_best')
            for c in ('px', 'pl'):
                v = D[c].values.astype(np.float64).copy(); v[m & (yy == 1)] = hi if good else lo; v[m & (yy == 0)] = lo if good else hi; D[c] = v
        return D, info
    if name.startswith(('S1full', 'S1fixed', 'CMB', 'S1N1')):
        # C16 S1(v_pub, L-a) 조립 — C16 s_score.assemble과 같은 규칙
        PP = pickle.load(open(C16C/'build'/'pop.pkl', 'rb')); pv = pd.MultiIndex.from_tuples([tuple(x) for x in PP['prior_vis']])
        Ko = key(O); in_pv = Ko.isin(pv); Fvis = set(PP['Fvis']); fin = O.firm.isin(Fvis).values
        Bm = O.prior_any.values & ~in_pv & fin; Am = ~O.prior_any.values & fin
        A = O[Am | Bm].copy(); A['y'] = np.where(A.prior_any.values, 0, A.y_off.values).astype(np.int8); A['is_new'] = A.prior_any.values
        Ck = pd.MultiIndex.from_tuples([tuple(x) for x in PP['C']])
        full = name in ('S1full', 'CMB_re', 'S1N1_re')
        src = 's1rcF' if name == 'CMB_re' else None
        S16 = pd.read_parquet(C16C/'build'/'s1F'/'scores.parquet'); pos16 = pd.Series(np.arange(len(S16)), index=key(S16))
        Sx = pd.read_parquet(CC/'build'/'s1rcF'/'scores.parquet') if name.startswith('CMB') else None
        if full:
            Ssrc = Sx if src else S16; ps = pd.Series(np.arange(len(Ssrc)), index=key(Ssrc)).reindex(key(A)); assert ps.notna().all()
            for c in SC: A[c] = Ssrc[c].values[ps.values.astype(np.int64)]
        SC_C = Sx if name == 'CMB_re' else S16
        pc = pd.Series(np.arange(len(SC_C)), index=key(SC_C)).reindex(Ck); assert pc.notna().all()
        Cd = SC_C.iloc[pc.values.astype(np.int64)][['firm', 'k', 'newc'] + SC].copy(); Cd['prior_any'] = False; Cd['y'] = np.int8(0); Cd['y_off'] = np.int8(0); Cd['is_new'] = True
        D = pd.concat([A, Cd[A.columns]], ignore_index=True)
        D['newc'] = ~D.firm.isin(set(PP['vis_any_firms'])).values
        info = dict(A=int(Am.sum()), B=int(Bm.sum()), C=int(len(Cd)))
        if name.startswith(('CMB', 'S1N1')):
            Es = E[E.is_newc_s1].set_index('firm'); bad = set(Es.index[~Es.N1_primary])
            m = ~(D.newc.values & D.firm.isin(bad).values); info.update(n1_dropped_firms=len(bad), n1_dropped_rows=int((~m).sum()), n1_dropped_pos=int(D.y.values[~m].sum())); D = D[m]
        if name.startswith('CMB'):
            L1 = pd.read_parquet(CC/'lab_rc_s1.parquet'); ps = pd.Series(np.arange(len(Sx)), index=key(Sx)).reindex(key(L1)); assert ps.notna().all()
            R = Sx.iloc[ps.values.astype(np.int64)][['firm', 'k'] + SC].copy(); R['y'] = L1.y.values.astype(np.int8); R['y_off'] = R.y.values; R['newc'] = L1.newc_s1.values
            R['prior_any'] = False; R['is_new'] = True
            # RC 법인도 N1(S1판) 적용: 가시 신규진입자 RC 법인은 증거 계산 대상 밖 → 범위안 원점 가시 기록이 없으므로 제외(규칙 일관)
            rbad = set(R.firm[R.newc.values]); R = R[~R.firm.isin(rbad).values]
            info.update(rc_rows=int(len(R)), rc_pos=int(R.y.sum()), rc_newc_firms_dropped=len(rbad))
            D = pd.concat([D, R[D.columns]], ignore_index=True)
        return D, info
    raise ValueError(name)


# ------------------------------------------------------------------ 채점
def score(name, base, Cnt, U):
    fo = VD/f'{name}.json'
    if fo.exists(): print('exists', name); return
    t0 = time.time(); D, info = assemble(name, base)
    y = D.y.values.astype(np.int8); s = blend(D.px.values, D.pl.values); nc = D.newc.values.astype(bool)
    ma = auc(s, y); ba = {c: auc(D[c].values, y) for c in BK}; M0 = max(ba.values()); arg = max(ba, key=ba.get)
    man = auc(s[nc], y[nc]) if nc.any() and 0 < y[nc].sum() < nc.sum() else float('nan'); ban = max(auc(D[c].values[nc], y[nc]) for c in BK) if man == man else float('nan')
    sbb = auc(D.sbb.values, y); o = np.argsort(-s, kind='stable')
    res = dict(variant=name, n=int(len(D)), firms=int(D.firm.nunique()), pos=int(y.sum()), base_rate=float(y.mean()), new_rows=int(nc.sum()), new_pos=int(y[nc].sum()),
               auc=ma, auc_M0=M0, M0_arg=arg, base_aucs=ba, skill=(ma - M0) / (1 - M0), auc_new=man, skill_new=(man - ban) / (1 - ban) if man == man else None,
               auc_sbb=sbb, skill_vs_sbb=(ma - sbb) / (1 - sbb), top100=int(y[o[:100]].sum()), top1000=int(y[o[:1000]].sum()),
               capture_top1pct=float(y[o[:len(y) // 100]].sum() / y.sum()), info=info)
    isn = D.is_new.values.astype(bool)
    if isn.any():
        r = pd.Series(s).rank(pct=True).values; res['added_rows'] = int(isn.sum()); res['added_pos'] = int(y[isn].sum())
        res['added_score_pct'] = dict(mean=float(r[isn].mean()), median=float(np.median(r[isn]))); res['top1000_added'] = int(isn[o[:1000]].sum())
    ui = pd.Series(np.arange(len(U)), index=U).reindex(D.firm.values); assert ui.notna().all(); ui = ui.values.astype(np.int64)
    nU = len(U); WT = None
    A = np.zeros((BB, 8))
    for j, sc_ in enumerate([s] + [D[c].values for c in BK] + [D.sbb.values]):
        fa = FastAUC(sc_, y, ui, nU)
        for r0 in range(0, BB, 250):
            A[r0:r0 + 250, j] = fa(Cnt[r0:r0 + 250])
    np.save(VD/f'{name}_boot.npy', A)
    bmax = A[:, 1:7].max(1); sk = (A[:, 0] - bmax) / (1 - bmax); sks = (A[:, 0] - A[:, 7]) / (1 - A[:, 7])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    res['ci'] = q(sk); res['ci_vs_sbb'] = q(sks); res['n_boot'] = BB
    res['M0_class'] = 'CI하한>0' if res['ci'][0] > 0 else 'CI하한<=0'; res['SBB_class'] = 'CI하한>0' if res['ci_vs_sbb'][0] > 0 else 'CI하한<=0'
    if name != 'OFF':
        Ao = np.load(VD/'OFF_boot.npy'); bo = Ao[:, 1:7].max(1); sko = (Ao[:, 0] - bo) / (1 - bo); sbo = (Ao[:, 0] - Ao[:, 7]) / (1 - Ao[:, 7])
        res['delta'] = res['skill'] - OFFICIAL['skill']; res['delta_ci'] = q(sk - sko); res['delta_sbb_ci'] = q(sks - sbo)
        res['delta_le_thr_share'] = float(np.mean(sk - sko <= -THR)); res['delta_neg_share'] = float(np.mean(sk - sko < 0))
        off = json.load(open(VD/'OFF.json', encoding='utf-8'))
        chg = (res['M0_class'] != off['M0_class']) or (res['SBB_class'] != off['SBB_class'])
        d = res['delta']
        res['verdict'] = '판정 변경' if chg else ('영향 없음' if abs(d) < THR else ('경고(Δ≤−0.0087)' if d <= -THR else '영향 있음(+, 공식이 보수적)'))
    res['secs'] = round(time.time() - t0); res['peak'] = G.rss_peak()
    jdump(res, fo); print(name, 'n', res['n'], 'pos', res['pos'], 'skill', round(res['skill'], 5), [round(x, 5) for x in res['ci']], 'Δ', round(res.get('delta', 0), 5), res.get('delta_ci'), res.get('verdict'), 'top1000', res['top1000'], res['secs'], 's', res['peak'], flush=True)


if __name__ == '__main__':
    U = universe(); Cnt = counts(); base = load_base()
    for nm in sys.argv[1:]: score(nm, base, Cnt, U)
