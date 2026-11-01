# -*- coding: utf-8 -*-
"""C17-T2 s3: PREREG §6 Δskill. 쌍별 오매칭 확률 p(s2)로 시드 1,000회 확률적 제거.
 (i) 제외: 양성 제거 시 남은 양성의 음성 대비 순위 기여 불변 → 정확 계산(블렌드 백분위는 공식 평가 집합 고정, 10시드 전체 재순위 대조)
 (ii) 음성 전환: 평가 집합 고정 → 순위 고정, 양성 순위합으로 정확 계산
 하한판(코드 확정만), 변형별(점추정 p), V6 점수 배치 경계, C16 플래그 220 재현, R 보조(iii)·결합(iv).
 산출: cache/s3_out.json, sens_results.csv"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
from scipy.stats import rankdata
t0 = time.time(); out = {}
Z = np.load(CC/'eval_base.npz', allow_pickle=True)
s = Z['s']; y = Z['y'].astype(np.int8); base = Z['base']; firm = Z['firm']; kk = Z['k']
OFF = 0.47267044268107444
key = pd.Series(np.arange(len(y)), index=pd.MultiIndex.from_arrays([firm, kk]))
pp = pd.read_parquet(CC/'p_point.parquet')
rows = key.reindex(pd.MultiIndex.from_arrays([pp.firm.astype(str).values, pp.k.values])).values
assert not np.isnan(rows).any(); rows = rows.astype(int)
assert (y[rows] == 1).all() and (Z['ypub'][rows] == 1).all() and (Z['yreg'][rows] == 0).all()
pos = np.where(y == 1)[0]; neg = y == 0; n0 = int(neg.sum()); n1 = len(pos)
posmap = -np.ones(len(y), int); posmap[pos] = np.arange(n1); prow = posmap[rows]


def contrib(sc):
    sn = np.sort(sc[neg]); sp = sc[pos]
    lo = np.searchsorted(sn, sp, 'left'); hi = np.searchsorted(sn, sp, 'right')
    return (lo + 0.5 * (hi - lo)) / n0


Cm = contrib(s); Cb = np.vstack([contrib(b) for b in base])
rm = rankdata(s); rb = np.vstack([rankdata(b) for b in base])
Sm = rm[pos].sum(); Sb = rb[:, pos].sum(1)


def sk_excl(drop_pos_idx):
    keep = np.ones(n1, bool); keep[drop_pos_idx] = False
    a = Cm[keep].mean(); M0 = Cb[:, keep].mean(1).max(); return (a - M0) / (1 - M0)


def sk_relab(flip_rows):
    m = len(flip_rows); n1n = n1 - m; n0n = n0 + m
    a = (Sm - rm[flip_rows].sum() - n1n * (n1n + 1) / 2) / (n1n * n0n)
    bb = (Sb - rb[:, flip_rows].sum(1) - n1n * (n1n + 1) / 2) / (n1n * n0n); M0 = bb.max(); return (a - M0) / (1 - M0)


out['P0_fast'] = dict(excl=sk_excl(np.array([], int)), relab=sk_relab(np.array([], int)))
assert abs(out['P0_fast']['excl'] - OFF) < 1e-6 and abs(out['P0_fast']['relab'] - OFF) < 1e-6


def full_excl(drop_rows):
    """전체 재순위(블렌드 백분위를 남은 집합에서 다시 계산)로 제외 skill."""
    keep = np.ones(len(y), bool); keep[drop_rows] = False
    sc = 0.9 * pd.Series(Z['px'][keep]).rank(pct=True).values + 0.1 * pd.Series(Z['pl'][keep]).rank(pct=True).values
    yy = y[keep]
    def auc(v):
        r = rankdata(v); a1 = yy.sum(); a0 = len(yy) - a1; return (r[yy == 1].sum() - a1 * (a1 + 1) / 2) / (a1 * a0)
    am = auc(sc); M0 = max(auc(b[keep]) for b in base); return (am - M0) / (1 - M0)


# ---- C16 재현 ----
c16 = pp.c16_flag.values
out['C16_flag220'] = dict(n=int(c16.sum()), excl=sk_excl(prow[c16]) - OFF, relab=sk_relab(rows[c16]) - OFF)
# ---- 하한판(코드 확정만) ----
kw = (pp.pv == 'wrong').values
out['lower_code_only'] = dict(n=int(kw.sum()), excl=sk_excl(prow[kw]) - OFF, relab=sk_relab(rows[kw]) - OFF, excl_full=full_excl(rows[kw]) - OFF)
# ---- 주: 부트스트랩 p × Bernoulli ----
PB = np.load(CC/'p_boot_main.npy'); rs = np.random.RandomState(20261017)
D1 = []; D2 = []; NR = []
for b in range(PB.shape[0]):
    w = rs.rand(len(pp)) < PB[b]
    D1.append(sk_excl(prow[w]) - OFF); D2.append(sk_relab(rows[w]) - OFF); NR.append(int(w.sum()))
    if b < 10:
        out.setdefault('full_check', []).append(dict(fast=D1[-1], full=full_excl(rows[w]) - OFF))
q = lambda a: dict(mean=float(np.mean(a)), p2_5=float(np.percentile(a, 2.5)), p50=float(np.median(a)), p97_5=float(np.percentile(a, 97.5)),
                   share_gt_0_002=float(np.mean(np.array(a) > 0.002)))
out['main'] = dict(excl=q(D1), relab=q(D2), n_removed=q(NR))
fc = out['full_check']; out['full_check_maxdiff'] = float(max(abs(x['fast'] - x['full']) for x in fc))
m = out['main']['excl']['mean']
out['verdict_E_C03T1_03'] = ('성립' if abs(m) <= 0.002 else '불성립') + (' (경계: 95% 구간이 0.002를 가로지름)' if out['main']['excl']['p2_5'] <= 0.002 <= out['main']['excl']['p97_5'] else '')
# ---- 변형(점추정 p) ----
var = {}
for c in [x for x in pp.columns if x.startswith('p_')]:
    p = pp[c].values; rv = np.random.RandomState(7); a1 = []; a2 = []
    for _ in range(1000):
        w = rv.rand(len(p)) < p; a1.append(sk_excl(prow[w]) - OFF); a2.append(sk_relab(rows[w]) - OFF)
    var[c[2:]] = dict(excl=q(a1), relab=q(a2), expected_removed=float(p.sum()))
out['variants'] = var
# ---- V6 점수 배치 경계(주 점추정 p): 판정 불가 쌍의 기대 제거 수를 점수 최저/최고 쌍에 몰아 넣음 ----
p = pp.p_main.values; und = (pp.pv == 'undet').values & (p > 0)
nrem = int(round(p[und].sum())); sc_und = s[rows[und]]; ui = np.where(und)[0]
lo = ui[np.argsort(sc_und)[:nrem]]; hi = ui[np.argsort(-sc_und)[:nrem]]
out['V6_bounds'] = dict(n_undet_removed=nrem, worst_lowest_scores=sk_excl(prow[np.r_[np.where(kw)[0], lo]]) - OFF,
                        best_highest_scores=sk_excl(prow[np.r_[np.where(kw)[0], hi]]) - OFF)
# 점수 위치 진단: 판정 불가 F 쌍 vs 판정 정확 쌍 vs 코드 확정 오매칭 쌍(평가 집합 백분위)
pct = pd.Series(s).rank(pct=True).values
grp = np.where(kw, 'code_wrong', np.where(pp.pv == 'correct', 'code_correct', np.where(pp.F, 'undet_F', 'undet_NF')))
out['score_pct_median'] = {g: float(np.median(pct[rows[grp == g]])) for g in np.unique(grp)}
out['score_pct_median']['other_pos'] = float(np.median(pct[np.setdiff1d(pos, rows)]))
# ---- R 보조(iii) 되돌림, (iv) 결합 ----
pr = pd.read_parquet(CC/'p_point_R.parquet'); PBR = np.load(CC/'p_boot_R.npy')
Rkey = pd.Series(np.arange(len(Z['Rfirm'])), index=pd.MultiIndex.from_arrays([Z['Rfirm'], Z['Rk']]))
rrow = Rkey.reindex(pd.MultiIndex.from_arrays([pr.firm.astype(str).values, pr.k.values])).values.astype(int)
Ry = Z['Ry'][rrow].astype(np.int8)
pxs = np.sort(Z['px']); pls = np.sort(Z['pl']); N = len(y)
sR = 0.9 * (np.searchsorted(pxs, Z['Rs_px'][rrow], 'right') / N) + 0.1 * (np.searchsorted(pls, Z['Rs_pl'][rrow], 'right') / N)
bR = Z['Rbase'][:, rrow]
SC = [s] + [base[j] for j in range(6)]; ADD = [sR] + [bR[j] for j in range(6)]
SN = [np.sort(v[neg]) for v in SC]; SP = [np.sort(v[pos]) for v in SC]
CNT = [Cm * n0] + [Cb[j] * n0 for j in range(6)]   # 양성별 '원 음성보다 위' 수(동점 0.5)


def below(sorted_arr, x):   # #sorted_arr < x + 0.5 #==x
    lo = np.searchsorted(sorted_arr, x, 'left'); hi = np.searchsorted(sorted_arr, x, 'right'); return lo + 0.5 * (hi - lo)


def sk_R(sel_R, flip_rows=None):
    """R 선택 행을 평가에 추가하고(점수: 공식 평가 집합 기준 백분위 블렌드) P 전환 행을 음성으로. 쌍 비교를 정확히 센다."""
    fr = np.array([], int) if flip_rows is None else np.asarray(flip_rows, int)
    keep = np.ones(n1, bool); keep[posmap[fr]] = False
    res = []
    for j in range(7):
        v = SC[j]; ad = ADD[j][sel_R]; ay = Ry[sel_R]
        E = np.r_[v[fr], ad[ay == 0]]; Ap = ad[ay == 1]; Fs = np.sort(v[fr])
        num = CNT[j][keep].sum()
        if len(E): num += ((n1 - below(SP[j], E)) - (len(Fs) - below(Fs, E))).sum()   # 남은 양성 중 E보다 위(동점 0.5)
        if len(Ap): num += (below(SN[j], Ap) + below(np.sort(E), Ap)).sum()
        n1n = int(keep.sum()) + len(Ap); n0n = n0 + len(E)
        res.append(num / (n1n * n0n))
    a_ = res[0]; M0 = max(res[1:]); return (a_ - M0) / (1 - M0)


assert abs(sk_R(np.zeros(len(pr), bool)) - OFF) < 1e-6
_t = np.where(pp.c16_flag.values)[0]
assert abs(sk_R(np.zeros(len(pr), bool), rows[_t]) - sk_relab(rows[_t])) < 1e-9


kwR = (pr.pv == 'wrong').values
out['R_lower_code_only'] = dict(n=int(kwR.sum()), pos=int(Ry[kwR].sum()), d_iii=sk_R(kwR) - OFF)
rr = np.random.RandomState(20261019); D3 = []; D4 = []
for b in range(PBR.shape[0]):
    wR = rr.rand(len(pr)) < PBR[b]; wP = rr.rand(len(pp)) < PB[b]
    D3.append(sk_R(wR) - OFF); D4.append(sk_R(wR, rows[wP]) - OFF)
out['R_secondary'] = dict(iii=q(D3), iv=q(D4), seeds=int(PBR.shape[0]))
out['secs'] = round(time.time() - t0); out['peak'] = G.rss_peak()
jdump(out, CC/'s3_out.json')
# ---- sens_results.csv ----
rows_csv = [dict(variant='OFF(공식 재현)', metric='skill', mean=OFF, p2_5='', p97_5='', share_gt_0_002='', expected_removed=0, note='P0 차 −2.5e−9')]
rows_csv.append(dict(variant='C16 플래그 220 전부 제외(재현)', metric='Δ(i)', mean=out['C16_flag220']['excl'], p2_5='', p97_5='', share_gt_0_002='', expected_removed=220, note='C16 +0.01121'))
rows_csv.append(dict(variant='하한판: 코드 확정 오매칭만 제외', metric='Δ(i)', mean=out['lower_code_only']['excl'], p2_5='', p97_5='', share_gt_0_002='', expected_removed=out['lower_code_only']['n'], note=f"전체 재순위 {out['lower_code_only']['excl_full']:.5f}"))
rows_csv.append(dict(variant='하한판: 코드 확정 오매칭만 음성 전환', metric='Δ(ii)', mean=out['lower_code_only']['relab'], p2_5='', p97_5='', share_gt_0_002='', expected_removed=out['lower_code_only']['n'], note=''))
for nm, dd in [('주(E2, 부트스트랩 p, 1,000시드) 제외', out['main']['excl']), ('주(E2) 음성 전환', out['main']['relab'])]:
    rows_csv.append(dict(variant=nm, metric='Δ(i)' if '제외' in nm else 'Δ(ii)', mean=dd['mean'], p2_5=dd['p2_5'], p97_5=dd['p97_5'], share_gt_0_002=dd['share_gt_0_002'], expected_removed=out['main']['n_removed']['mean'], note='판정 지표' if '제외' in nm else ''))
for k, v in var.items():
    rows_csv.append(dict(variant=f'변형 {k}(점추정 p) 제외', metric='Δ(i)', mean=v['excl']['mean'], p2_5=v['excl']['p2_5'], p97_5=v['excl']['p97_5'], share_gt_0_002=v['excl']['share_gt_0_002'], expected_removed=v['expected_removed'], note=''))
rows_csv.append(dict(variant='V6 점수 최저 쌍에 몰기(상한)', metric='Δ(i)', mean=out['V6_bounds']['worst_lowest_scores'], p2_5='', p97_5='', share_gt_0_002='', expected_removed=out['V6_bounds']['n_undet_removed'] + int(kw.sum()), note='경계'))
rows_csv.append(dict(variant='V6 점수 최고 쌍에 몰기(하한)', metric='Δ(i)', mean=out['V6_bounds']['best_highest_scores'], p2_5='', p97_5='', share_gt_0_002='', expected_removed=out['V6_bounds']['n_undet_removed'] + int(kw.sum()), note='경계'))
rows_csv.append(dict(variant='R 보조: 코드 확정 오매칭 R 되돌림', metric='Δ(iii)', mean=out['R_lower_code_only']['d_iii'], p2_5='', p97_5='', share_gt_0_002='', expected_removed=out['R_lower_code_only']['n'], note='판정 외'))
for nm, dd in [('R 보조: 확률 되돌림(200시드)', out['R_secondary']['iii']), ('결합: P 음성 전환 + R 되돌림', out['R_secondary']['iv'])]:
    rows_csv.append(dict(variant=nm, metric='Δ(iii)' if 'R 보조' in nm else 'Δ(iv)', mean=dd['mean'], p2_5=dd['p2_5'], p97_5=dd['p97_5'], share_gt_0_002=dd['share_gt_0_002'], expected_removed='', note='판정 외'))
pd.DataFrame(rows_csv).to_csv(HERE/'sens_results.csv', index=False, encoding='utf-8-sig')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:5000])
