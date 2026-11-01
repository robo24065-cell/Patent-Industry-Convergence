# -*- coding: utf-8 -*-
"""verify_1: C17-T1 T1-02(N1) 주장 반박 시도. 원천(prep_te, FROZEN o2019, baseline_scores, C16 labels, n1F/n2F scores)을 읽기만 하고
 sklearn roc_auc_score + 자체 법인 군집 부트스트랩(다른 시드)으로 다시 계산. 쓰기는 verify_1 안(v1_out.json)만."""
import sys, pathlib, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'
W = B/'분석/검증루프_20261012/C17/C17-T1'; C16C = B/'분석/검증루프_20261012/C16/C16-T1/cache'; OUT = pathlib.Path(__file__).resolve().parent
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
t0 = time.time(); res = {}
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
fz = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'px', 'pl'])
bo = pd.read_parquet(FC/'baseline_scores.parquet')
L = pd.read_parquet(C16C/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
assert (fz.firm.values == te.firm.values).all() and (fz.k.values == te.k.values).all() and (L.firm.values == te.firm.values).all() and (L.k.values == te.k.values).all()
D = te.copy(); D['px'] = fz.px.values; D['pl'] = fz.pl.values; D['y'] = L.y_u.values.astype(int)
for c in BK: D[c] = bo[c].values
# newc 가 법인 단위 속성인지
g = D.groupby('firm').newc.agg(['min', 'max']); res['newc_firm_level'] = bool((g['min'] == g['max']).all())
D = D[~D.prior_any.values].reset_index(drop=True)
newc_f = set(D.firm[D.newc.values]); res['newc_firms'] = len(newc_f); res['newc_rows'] = int(D.newc.sum()); res['newc_pos'] = int(D.y[D.newc.values].sum())
res['rows'] = len(D); res['pos'] = int(D.y.sum())
res['newc_row_share'] = res['newc_rows'] / len(D); res['newc_pos_share'] = res['newc_pos'] / res['pos']
ev = pd.read_parquet(W/'cache/evid_all.parquet'); evo = ev[ev.is_newc_off]
res['evid_newc_off_eq_te_newc'] = set(evo.firm) == newc_f
vis = set(evo.firm[evo.N1_primary]); bad = newc_f - vis; res['N1_vis'] = len(vis); res['N1_bad'] = len(bad)
# N1_primary == E_pub|E_reg|E_own 정의 확인
res['N1_primary_def_ok'] = bool((evo.N1_primary == (evo.E_pub | evo.E_reg | evo.E_own)).all())


def rk(v): return pd.Series(v).rank(pct=True).values


def metrics(F, s_px=None):
    y = F.y.values; bl = 0.9 * rk(F.px.values) + 0.1 * rk(F.pl.values)
    a = roc_auc_score(y, bl); ba = {c: roc_auc_score(y, F[c].values) for c in BK}; m0 = max(ba.values())
    o = np.argsort(-bl, kind='stable'); t1k = int(y[o[:1000]].sum()); br = y.mean()
    return dict(rows=len(F), firms=int(F.firm.nunique()), pos=int(y.sum()), auc=a, M0=m0, M0_arg=max(ba, key=ba.get), skill=(a - m0) / (1 - m0),
                top1000=t1k, base_rate=br, lift1000=t1k / 1000 / br), bl


off, bl_off = metrics(D); res['OFF'] = off
mN1 = ~(D.newc.values & D.firm.isin(bad).values); N1 = D[mN1].reset_index(drop=True); res['N1_fixed'], _ = metrics(N1)
mN2 = ~D.newc.values; N2 = D[mN2].reset_index(drop=True); res['N2_fixed'], _ = metrics(N2)


def recompute(F, path):
    S = pd.read_parquet(path, columns=['firm', 'k', 'prior_any', 'px', 'pl'] + BK)
    S = S[~S.prior_any.values]
    M = F[['firm', 'k', 'y', 'newc']].merge(S, on=['firm', 'k'], how='left', validate='1:1')
    assert M.px.notna().all(); return M, len(S)


N1r, nS1 = recompute(N1, W/'cache/build/n1F/scores.parquet'); res['N1_re'], _ = metrics(N1r); res['N1_re_scored_minus_pop'] = nS1 - len(N1r)
N2r, nS2 = recompute(N2, W/'cache/build/n2F/scores.parquet'); res['N2_re'], _ = metrics(N2r); res['N2_re_scored_minus_pop'] = nS2 - len(N2r)
for k in ('N1_fixed', 'N1_re', 'N2_fixed', 'N2_re'): res[k]['delta'] = res[k]['skill'] - off['skill']

# 반박 a: 같은 행 수를 무작위 신규진입자 법인에서 빼면(선택 무관 제거) Δ가 비슷한가 — N1 Δ가 '가시성'이 아니라 단순 신규진입 쌍 감소 효과인지
rng = np.random.RandomState(99); nf = sorted(newc_f); dr = []
for r in range(20):
    b2 = set(rng.choice(nf, len(bad), replace=False)); m = ~(D.newc.values & D.firm.isin(b2).values)
    dr.append(metrics(D[m].reset_index(drop=True))[0]['skill'] - off['skill'])
res['rand_drop_same_nfirms_delta'] = dict(mean=float(np.mean(dr)), min=float(np.min(dr)), max=float(np.max(dr)))
# 반박 b: 가시 신규진입자 232곳만 빼기(반대 선택)
m = ~(D.newc.values & D.firm.isin(vis).values); res['drop_visible_only_delta'] = metrics(D[m].reset_index(drop=True))[0]['skill'] - off['skill']

# 자체 법인 군집 부트스트랩(다른 시드 20261099, B=400). 우주 = 공식 법인.
U = np.array(sorted(D.firm.unique())); nU = len(U); BB = 400
uiD = pd.Series(np.arange(nU), index=U).reindex(D.firm.values).values
uiN = pd.Series(np.arange(nU), index=U).reindex(N1r.firm.values).values


def prep(s, y):
    u, inv = np.unique(np.asarray(s, np.float64), return_inverse=True); return inv, len(u), y.astype(np.float64)


def wauc(pre, w):
    inv, G, y = pre
    wp = np.bincount(inv, weights=w * y, minlength=G); wn = np.bincount(inv, weights=w * (1 - y), minlength=G)
    below = np.cumsum(wn) - wn; return (wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


def setup(F, ui):
    y = F.y.values; bl = 0.9 * rk(F.px.values) + 0.1 * rk(F.pl.values)
    return [prep(bl, y)] + [prep(F[c].values, y) for c in BK], ui


PO = setup(D, uiD); PN = setup(N1r, uiN)
rng = np.random.RandomState(20261099); dsk = []; skn = []
for r in range(BB):
    cnt = np.bincount(rng.randint(0, nU, nU), minlength=nU).astype(np.float64)
    out = []
    for P, ui in (PO, PN):
        w = cnt[ui]; a = [wauc(p, w) for p in P]; m0 = max(a[1:]); out.append((a[0] - m0) / (1 - m0))
    dsk.append(out[1] - out[0]); skn.append(out[1])
q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
res['own_boot'] = dict(B=BB, seed=20261099, delta_ci=q(dsk), N1re_skill_ci=q(skn), share_le_m0087=float(np.mean(np.array(dsk) <= -0.0087)))
# 원 작업 부트스트랩 파일에서 Δ CI 재계산
Ao = np.load(W/'cache/var/OFF_boot.npy')
def sk(A): b = A[:, 1:7].max(1); return (A[:, 0] - b) / (1 - b)
for v in ('N1re', 'N1', 'N2', 'N2re', 'N3x3_neg_x3', 'N3_neg_s1', 'N3_neg_s2', 'N3_neg_s3', 'N3_all_s1', 'N3_all_s2', 'N3_all_s3'):
    A = np.load(W/f'cache/var/{v}_boot.npy'); jj = json.load(open(W/f'cache/var/{v}.json', encoding='utf-8'))
    res[f'file_{v}'] = dict(skill=jj['skill'], delta=jj['skill'] - off['skill'], delta_ci=q(sk(A) - sk(Ao)), skill_ci=q(sk(A)), n=jj['n'], pos=jj['pos'], top1000=jj['top1000'])
res['secs'] = round(time.time() - t0)
json.dump(res, open(OUT/'v1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
