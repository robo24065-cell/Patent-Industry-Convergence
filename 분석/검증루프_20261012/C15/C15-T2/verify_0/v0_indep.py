# -*- coding: utf-8 -*-
"""verify_0 독립 재채점(작업 채점 코드·sklearn roc_auc_score·트랙 라이브러리 미사용).
 python -B v0_indep.py <T1 묶음> <T3 묶음>
 - 점수: T1 블렌드 = 0.9·(rankdata(px)/n) + 0.1·(rankdata(pl)/n) (공식 pandas rank(pct) 와 같은 연산 순서), T3 = 성분 순위 평균(저장된 p 재구성 확인).
 - AUC: Mann–Whitney 순위식. CI: '뽑힌 횟수 = 가중치' 가중 Mann–Whitney AUC(동점 0.5)로 같은 난수 규약 재현.
   T1 M0: RandomState(3000+ci) 10회×20, 기준선 6변형 전부(로지스틱 4개도 공식 r3boot가 아니라 점수에서 직접) 재계산. T1 SBB: RandomState(4242) 200회.
   T3 M0: default_rng(0) 1000회(IPC 메인그룹), SB3: default_rng(11) 1000회.
 입력: 원천 예측·점수 파일(cache/t1/b*/final/o2019.parquet, base_b*.parquet, sbb_b*.npy, VERIFY_R5 labels_c2019_official.parquet, cache/t3/preds_b*_final.parquet)
 → verify_0/v0_result.json"""
import sys, json, pathlib, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
W = pathlib.Path(__file__).resolve().parent.parent; C1 = W/'cache'/'t1'; C3 = W/'cache'/'t3'
T1 = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실")
bT1, bT3 = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time(); OUT = {}


def mw_auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class WAUC:
    """가중 AUC: 점수 정렬·동점 묶음을 한 번만 만들고 가중치만 바꿔 계산"""
    def __init__(self, y, s):
        o = np.argsort(s, kind='mergesort'); self.o = o; ss = s[o]; self.y = y[o].astype(bool)
        self.g = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])])     # 동점 묶음 번호
        self.ng = self.g[-1] + 1
    def __call__(self, w):
        w = w[self.o]; wp = np.where(self.y, w, 0.0); wn = np.where(self.y, 0.0, w)
        P = np.bincount(self.g, wp, self.ng); N = np.bincount(self.g, wn, self.ng)
        below = np.concatenate([[0.0], np.cumsum(N)[:-1]])
        return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
sk = lambda a, b: (a - b) / (1 - b)
# ================= T1 =================
o = pd.read_parquet(C1/f'b{bT1}'/'final'/'o2019.parquet'); lab = pd.read_parquet(T1/'VERIFY_R5'/'labels_c2019_official.parquet')
assert (o.firm.values == lab.firm.values).all() and (o.k.values == lab.k.values).all()
keep = ~o.prior_any.values; oo = o[keep].reset_index(drop=True); y = lab.y_u.values[keep].astype(int); n = len(oo)
bw = 0.9; s = bw * (rankdata(oo.px.values) / n) + (1 - bw) * (rankdata(oo.pl.values) / n)   # 공식 blend 와 같은 가중(1-0.9 = 0.09999999999999998)
base = pd.read_parquet(C1/f'base_b{bT1}.parquet').loc[keep].reset_index(drop=True); sbb = np.load(C1/f'sbb_b{bT1}.npy')
A = {'model': mw_auc(y, s), **{k: mw_auc(y, base[k].values) for k in base.columns}, 'SBB': mw_auc(y, sbb)}
bmax = max(A[k] for k in base.columns)
ref = json.load(open(C1/f'b{bT1}_final.json', encoding='utf-8'))
r1 = dict(bundle=bT1, auc=A['model'], skill=sk(A['model'], bmax), skill_SBB=sk(A['model'], A['SBB']),
          top100=int(y[np.argsort(-s, kind='stable')[:100]].sum()), top1000=int(y[np.argsort(-s, kind='stable')[:1000]].sum()))
r1['d_auc'] = r1['auc'] - ref['auc_champ']; r1['d_skill'] = r1['skill'] - ref['skill']; r1['d_skill_SBB'] = r1['skill_SBB'] - ref['skill_vs_strong']
fr_, inv = np.unique(oo.firm.values, return_inverse=True); nF = len(fr_)
WA = {k: WAUC(y, v) for k, v in [('model', s), ('SBB', sbb)] + [(k, base[k].values) for k in base.columns]}
SK = []
for ci in range(20):
    rng = np.random.RandomState(3000 + ci)
    for r in range(10):
        cnt = np.bincount(rng.randint(0, nF, nF), minlength=nF).astype(float); w = cnt[inv]
        am = WA['model'](w); bm = max(WA[k](w) for k in base.columns); SK.append(sk(am, bm))
r1['ci_M0'] = q(SK); r1['d_ci_M0'] = [r1['ci_M0'][0] - ref['ci'][0], r1['ci_M0'][1] - ref['ci'][2]]
rng = np.random.RandomState(4242); SKs = []
for r in range(200):
    cnt = np.bincount(rng.randint(0, nF, nF), minlength=nF).astype(float); w = cnt[inv]
    SKs.append(sk(WA['model'](w), WA['SBB'](w)))
r1['ci_SBB'] = q(SKs); r1['d_ci_SBB'] = [r1['ci_SBB'][0] - ref['strong_ci'][0], r1['ci_SBB'][1] - ref['strong_ci'][2]]
r1['secs'] = round(time.time() - t0); OUT['T1'] = r1; print('T1', r1, flush=True)
# ================= T3 =================
P = pd.read_parquet(C3/f'preds_b{bT3}_final.parquet'); ref3 = json.load(open(C3/f'b{bT3}_final.json', encoding='utf-8'))
nn = len(P); prec = (rankdata(P.p_logit) / nn + rankdata(P.p_xgbc) / nn + rankdata(P.p_nb) / nn) / 3
yb = P.yb.values.astype(int)
r3 = dict(bundle=bT3, p_recon_maxdiff=float(np.abs(prec - P.p.values).max()), auc=mw_auc(yb, P.p.values), auc_M0=mw_auc(yb, P.p_base.values), auc_SB3=mw_auc(yb, P.SB3.values))
r3['skill'] = sk(r3['auc'], r3['auc_M0']); r3['skill_SB3'] = sk(r3['auc'], r3['auc_SB3'])
o3 = ref3['by_origin']['2021']; r3['d_skill'] = r3['skill'] - o3['skill']; r3['d_skill_SB3'] = r3['skill_SB3'] - ref3['SB3']['skill_vs_sb']
ums, ginv = np.unique(P.mg.values, return_inverse=True)
WM, W0, WS = WAUC(yb, P.p.values), WAUC(yb, P.p_base.values), WAUC(yb, P.SB3.values)
rng = np.random.default_rng(0); K = []
for _ in range(1000):
    smp = rng.choice(ums, len(ums)); cnt = np.bincount(np.searchsorted(ums, smp), minlength=len(ums)).astype(float); w = cnt[ginv]
    K.append(sk(WM(w), W0(w)))
r3['ci_M0'] = q(K); r3['d_ci_M0'] = [r3['ci_M0'][0] - ref3['ci95_M0']['skill'][0], r3['ci_M0'][1] - ref3['ci95_M0']['skill'][1]]
rng = np.random.default_rng(11); K = []
for _ in range(1000):
    smp = rng.choice(ums, len(ums)); cnt = np.bincount(np.searchsorted(ums, smp), minlength=len(ums)).astype(float); w = cnt[ginv]
    K.append(sk(WM(w), WS(w)))
r3['ci_SB3'] = q(K); r3['d_ci_SB3'] = [r3['ci_SB3'][0] - ref3['SB3']['ci95'][0], r3['ci_SB3'][1] - ref3['SB3']['ci95'][1]]
OUT['T3'] = r3; print('T3', r3, flush=True)
dd = [abs(x) for t in ('T1', 'T3') for k, v in OUT[t].items() if k.startswith('d_') for x in (v if isinstance(v, list) else [v])]
OUT['max_abs_diff'] = max(dd); OUT['pass_1e-9'] = OUT['max_abs_diff'] <= 1e-9; OUT['secs'] = round(time.time() - t0)
json.dump(OUT, open(W/'verify_0'/'v0_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('max diff', OUT['max_abs_diff'])
