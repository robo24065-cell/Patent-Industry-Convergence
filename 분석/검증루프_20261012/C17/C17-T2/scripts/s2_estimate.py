# -*- coding: utf-8 -*-
"""C17-T2 s2: PREREG §5 주소 규칙 거짓 불일치 보정(E2)으로 실제 오매칭률과 쌍별 오매칭 확률 p를 추정한다. 부트스트랩 1,000회.
 변형(§7): V1 이전형=정확, V2 2순위→불가, V3 E1(관측/기대 가시율), V4 q=0.5, V5 f 단일값,
          V7 권리목록 판정 쌍만으로 f·w 추정(등록원부 판정 쌍은 구조상 오매칭만 남으므로 — 변경 기록 2), V8a/V8b 동명 검출률 d로 NF 오매칭 추정(변경 기록 2)
 산출: cache/s2_out.json, cache/p_boot_main.npy(1000×P), cache/p_point.parquet(쌍별 점추정 p, 변형별), cache/p_boot_R.npy"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
from sklearn.linear_model import LogisticRegression
t0 = time.time(); out = {}
PJ = pd.read_parquet(CC/'pair_judg.parquet')
DJ = pd.read_parquet(CC/'doc_judg.parquet')
P = PJ[PJ.grp == 'P'].reset_index(drop=True); R = PJ[PJ.grp == 'R'].reset_index(drop=True)
CELLS = [(fo, s) for fo in (False, True) for s in ('S', 'M', 'L')]


def estimate(X, f_mode='cell', w_mode='literal', q=1.0, d=None, tier2=True, transfer_wrong=True, rl_only=False, e1=None, ret_p=True):
    """X: 쌍 표(pv, ptier, ptype, F, foreign, size, det_src). 반환: dict(e, W_F, W_NF, by cell), p 배열."""
    pv = X.pv.values.copy()
    if not transfer_wrong: pv = np.where((pv == 'wrong') & (X.ptype.values == '권리이전형'), 'correct', pv)
    t1c = (pv == 'correct') & (X.ptier.values == 1) if tier2 else (pv == 'correct') & (X.ptier.values == 1)
    if not tier2: pv = np.where((pv == 'correct') & (X.ptier.values == 2), 'undet', pv)
    F = X.F.values; fo = X.foreign.values; sz = X['size'].values; src = X.det_src.values
    cal = np.ones(len(X), bool) if not rl_only else (src != 'A')
    W_F = {}; W_NF = {}; pvec = np.zeros(len(X)); fr = {}
    for (cf, cs) in CELLS:
        c = (fo == cf) & (sz == cs)
        # f
        if f_mode == 'single': base = t1c & cal
        else:
            base = t1c & c & cal
            if base.sum() < 30: base = t1c & (sz == cs) & cal
        f = F[base].mean() if base.sum() else 0.0
        # w_NF
        det = (~F) & ((pv == 'wrong') | t1c) & cal
        dc = det & (sz == cs)
        if dc.sum() < 30: dc = det
        w = (pv[dc] == 'wrong').mean() if dc.sum() else 0.0
        w = min(1.0, w / q)
        nF = int((F & c).sum()); nNF = int((~F & c).sum())
        kw_NF = int(((~F) & c & (pv == 'wrong')).sum()); kw_F = int((F & c & (pv == 'wrong')).sum())
        if w_mode == 'literal' or w_mode == 'q': WNF = nNF * w
        elif w_mode == 'known': WNF = float(kw_NF)
        if rl_only:  # 권리목록 판정 쌍으로 추정한 w를 등록원부 판정 쌍 밖에만 적용, 등록원부 판정 오매칭은 알려진 값으로 더함
            nNF_nonA = int((~F & c & (src != 'A')).sum()); kwA = int((~F & c & (src == 'A') & (pv == 'wrong')).sum())
            WNF = nNF_nonA * w + kwA
        WNF = max(WNF, kw_NF)
        if e1 is not None:
            WF = e1[(cf, cs)]
        else:
            C = (nNF - WNF) / (1 - f) if f < 1 else nNF
            WF = nF - min(nF, f * C)
        if d is not None:  # 동명 검출률 d: 플래그 안 된 오매칭 = W_F(1−d)/d (NF 추정 대체)
            WNF = max(kw_NF, WF * (1 - d) / d)
        WF = max(WF, kw_F)
        W_F[(cf, cs)] = WF; W_NF[(cf, cs)] = WNF; fr[(cf, cs)] = dict(f=f, w=w, nF=nF, nNF=nNF, kw_F=kw_F, kw_NF=kw_NF, WF=WF, WNF=WNF, n_f_base=int(base.sum()), n_w_base=int(dc.sum()))
        if ret_p:
            uF = c & F & (pv == 'undet'); uN = c & (~F) & (pv == 'undet')
            pvec[uF] = np.clip((WF - kw_F) / max(uF.sum(), 1), 0, 1); pvec[uN] = np.clip((WNF - kw_NF) / max(uN.sum(), 1), 0, 1)
    pvec[pv == 'wrong'] = 1.0; pvec[pv == 'correct'] = 0.0
    e = (sum(W_F.values()) + sum(W_NF.values())) / len(X)
    return dict(e=e, W_F=sum(W_F.values()), W_NF=sum(W_NF.values()), cells={f'{"외국" if k[0] else "국내"}-{k[1]}': v for k, v in fr.items()}), pvec


def e1_wf(X):
    """E1: NF(등록원부 판정 제외) 쌍에서 '정확·가시' 확률 π를 로지스틱으로 적합, F 쌍의 기대 가시 정확 수와 관측 비교."""
    Z = X[X.det_src != 'A']
    feats = lambda d: np.column_stack([np.log1p(d.n_docs.values), (d.ay_min.values == 2021), (d.ay_min.values == 2022), (d.ay_min.values == 2023),
                                       d.foreign.values, d['size'].values == 'M', d['size'].values == 'L', d.coapp.values]).astype(float)
    nf = Z[~Z.F]; yv = ((nf.pv == 'correct') & (nf.ptier == 1)).values.astype(int)
    m = LogisticRegression(C=1e4, max_iter=2000).fit(feats(nf), yv)
    fz = Z[Z.F]; pi = m.predict_proba(feats(fz))[:, 1]
    res = {}
    for (cf, cs) in CELLS:
        c = (fz.foreign.values == cf) & (fz['size'].values == cs)
        O = ((fz.pv.values == 'correct') & (fz.ptier.values == 1) & c).sum(); E = pi[c].sum()
        s = min(1.0, O / E) if E > 0 else 1.0
        kA = int(((X.det_src == 'A') & X.F & (X.foreign == cf) & (X['size'] == cs) & (X.pv == 'wrong')).sum())
        res[(cf, cs)] = c.sum() * (1 - s) + kA
    allO = ((fz.pv == 'correct') & (fz.ptier == 1)).sum(); allE = pi.sum()
    return res, dict(O=int(allO), E=float(allE), share_correct=float(min(1, allO / allE)))


# ---- 주 추정 ----
main, p_main = estimate(P)
out['main_E2'] = main
# 동명 검출률 d: 등록원부 코드 확정 '동명·기타·승계' 오매칭 문서(법인번호 둘 다 있고 다름)에서 주소 시도 불일치 비율
X1 = pd.read_csv(CC/'s1b_wrong_docs.csv', dtype={'firm': str, 'appl_no': str})
def d_of(sel):
    hd = DJ.merge(X1[sel][['firm', 'appl_no']].drop_duplicates(), on=['firm', 'appl_no']); hd = hd[hd.e2 != 'none']
    return float((hd.e2 == 'mismatch').mean()), dict(n=int(len(hd)), wilson=wilson(int((hd.e2 == 'mismatch').sum()), int(len(hd))))
d_hom, dinfo_a = d_of(X1.corp_differ & X1.other_same_norm & ~X1.v.str.contains('이전'))   # V8a: 상대 코드가 같은 이름(동명 코드형)
d_hom_b, dinfo_b = d_of(X1.corp_differ & ~X1.v.str.contains('이전'))                       # V8b: 비이전 전체(승계·분할 포함)
d_tr, dinfo_t = d_of(X1.v.str.contains('이전'))
out['d_homonym'] = dict(V8a_same_name=dict(d=d_hom, **dinfo_a), V8b_nontransfer=dict(d=d_hom_b, **dinfo_b), transfer=dict(d=d_tr, **dinfo_t))
e1res, e1diag = e1_wf(P); out['E1_diag'] = e1diag
variants = {'main': (main, p_main)}
variants['V1_transfer_correct'] = estimate(P, transfer_wrong=False)
variants['V2_tier2_undet'] = estimate(P, tier2=False)
variants['V3_E1'] = estimate(P, e1=e1res)
variants['V4_q0.5'] = estimate(P, q=0.5)
variants['V5_f_single'] = estimate(P, f_mode='single')
variants['V7_rl_only'] = estimate(P, rl_only=True)
variants['V8a_d_same_name'] = estimate(P, d=d_hom)
variants['V8b_d_nontransfer'] = estimate(P, d=d_hom_b)
variants['V7+V8a'] = estimate(P, rl_only=True, d=d_hom)
out['variants'] = {k: dict(e=v[0]['e'], W_F=v[0]['W_F'], W_NF=v[0]['W_NF'], expected_removed=float(v[1].sum())) for k, v in variants.items()}
pp = P[['firm', 'k', 'pv', 'ptier', 'ptype', 'F', 'foreign', 'size', 'det_src', 'c16_flag']].copy()
for k, v in variants.items(): pp['p_' + k] = v[1]
pp.to_parquet(CC/'p_point.parquet', index=False)
# ---- 부트스트랩(칸 안 재표본) ----
rs = np.random.RandomState(20261017)
cellid = (P.F.astype(int) * 100 + P.foreign.astype(int) * 10 + P['size'].map({'S': 0, 'M': 1, 'L': 2})).values
groups = [np.where(cellid == g)[0] for g in np.unique(cellid)]
E_b = []; PB = np.zeros((1000, len(P)), np.float32); EV = {k: [] for k in ['V3_E1', 'V7_rl_only', 'V8a_d_same_name', 'V8b_d_nontransfer', 'V7+V8a']}
for b in range(1000):
    idx = np.concatenate([g[rs.randint(0, len(g), len(g))] for g in groups])
    Xb = P.iloc[idx].reset_index(drop=True)
    r, _ = estimate(Xb, ret_p=False); E_b.append(r['e'])
    # p는 원 표본 쌍에 부트스트랩 모수(칸별 W)를 적용: 원 표본의 판정 불가 수로 나눈다
    cellsW = r['cells']
    pv = P.pv.values; pvec = np.where(pv == 'wrong', 1.0, 0.0)
    for (cf, cs) in CELLS:
        key = f'{"외국" if cf else "국내"}-{cs}'; cw = cellsW[key]
        c = (P.foreign.values == cf) & (P['size'].values == cs)
        uF = c & P.F.values & (pv == 'undet'); uN = c & (~P.F.values) & (pv == 'undet')
        kwF = (c & P.F.values & (pv == 'wrong')).sum(); kwN = (c & ~P.F.values & (pv == 'wrong')).sum()
        # 부트스트랩 표본의 W를 원 표본 크기로 환산(비율 유지)
        nF0 = (c & P.F.values).sum(); nN0 = (c & ~P.F.values).sum()
        rF = cw['WF'] / cw['nF'] if cw['nF'] else 0.0; rN = cw['WNF'] / cw['nNF'] if cw['nNF'] else 0.0
        pvec[uF] = np.clip((rF * nF0 - kwF) / max(uF.sum(), 1), 0, 1); pvec[uN] = np.clip((rN * nN0 - kwN) / max(uN.sum(), 1), 0, 1)
    PB[b] = pvec
    if b < 300:
        for k, kw in [('V7_rl_only', dict(rl_only=True)), ('V8a_d_same_name', dict(d=d_hom)), ('V8b_d_nontransfer', dict(d=d_hom_b)), ('V7+V8a', dict(rl_only=True, d=d_hom))]:
            EV[k].append(estimate(Xb, ret_p=False, **kw)[0]['e'])
        EV['V3_E1'].append(estimate(Xb, ret_p=False, e1=e1_wf(Xb)[0])[0]['e'])
    if b % 200 == 0: print('boot', b, round(time.time() - t0), G.rss_now(), flush=True)
np.save(CC/'p_boot_main.npy', PB)
q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)), float(np.percentile(a, 97.5))]
out['main_E2']['boot_ci'] = q(E_b); out['main_E2']['upper95'] = float(np.percentile(E_b, 97.5))
out['main_E2']['verdict'] = '정합' if out['main_E2']['upper95'] <= 0.05 else '경고'
out['variant_boot_ci'] = {k: q(v) for k, v in EV.items()}
# 쌍 오매칭률(코드 확정만, 하한) 및 판정 가능 쌍 기준
out['lower_code_only'] = dict(wrong=int((P.pv == 'wrong').sum()), rate=float((P.pv == 'wrong').mean()))
det = P[P.pv != 'undet']; out['determinable'] = dict(n=int(len(det)), wrong=int((det.pv == 'wrong').sum()), wilson=wilson(int((det.pv == 'wrong').sum()), int(len(det))))
detRL = det[det.det_src != 'A']; out['determinable_RL'] = dict(n=int(len(detRL)), wrong=int((detRL.pv == 'wrong').sum()), wilson=wilson(int((detRL.pv == 'wrong').sum()), int(len(detRL))),
                                                           F_n=int(detRL.F.sum()), F_wrong=int((detRL.F & (detRL.pv == 'wrong')).sum()))
out['share_of_name_positives'] = dict(e_times_P_over_6675=main['e'] * len(P) / 6675)
# ---- R(보조): 코드 확정 + F 칸 E2(권리목록 판정 쌍으로 f·w, 등록원부 오매칭은 알려진 값) ----
rR, pR = estimate(R, rl_only=True)
out['R_secondary'] = dict(e=rR['e'], W_F=rR['W_F'], W_NF=rR['W_NF'], known_wrong=int((R.pv == 'wrong').sum()), expected_readd=float(pR.sum()), cells=rR['cells'])
rsR = np.random.RandomState(20261019); PBR = np.zeros((200, len(R)), np.float32)
cidR = (R.F.astype(int) * 100 + R.foreign.astype(int) * 10 + R['size'].map({'S': 0, 'M': 1, 'L': 2})).values; gR = [np.where(cidR == g)[0] for g in np.unique(cidR)]
for b in range(200):
    idx = np.concatenate([g[rsR.randint(0, len(g), len(g))] for g in gR]); Xb = R.iloc[idx].reset_index(drop=True)
    r, _ = estimate(Xb, rl_only=True, ret_p=False); pv = R.pv.values; pvec = np.where(pv == 'wrong', 1.0, 0.0)
    for (cf, cs) in CELLS:
        cw = r['cells'][f'{"외국" if cf else "국내"}-{cs}']; c = (R.foreign.values == cf) & (R['size'].values == cs)
        for Fm, WW, nn_ in ((True, 'WF', 'nF'), (False, 'WNF', 'nNF')):
            u = c & (R.F.values == Fm) & (pv == 'undet'); kw = (c & (R.F.values == Fm) & (pv == 'wrong')).sum(); n0 = (c & (R.F.values == Fm)).sum()
            rr = cw[WW] / cw[nn_] if cw[nn_] else 0.0; pvec[u] = np.clip((rr * n0 - kw) / max(u.sum(), 1), 0, 1)
    PBR[b] = pvec
np.save(CC/'p_boot_R.npy', PBR); R[['firm', 'k', 'pv', 'F', 'foreign', 'size', 'det_src']].assign(p=pR).to_parquet(CC/'p_point_R.parquet', index=False)
out['secs'] = round(time.time() - t0); out['peak'] = G.rss_peak()
jdump(out, CC/'s2_out.json'); print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
