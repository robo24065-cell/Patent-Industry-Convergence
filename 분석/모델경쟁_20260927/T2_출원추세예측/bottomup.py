# 상향식(bottom-up): (ksic,cc) 를 IPC subclass 성분으로 분해해 성분별 로그드리프트 후 합산.
# 가설: 빠르게 크는 성분의 비중이 커지면 합계 성장률이 가속 → 합계 단일 외삽은 과소예측 (구성 가속 효과)
import sys, json, warnings
import numpy as np, pandas as pd
from common import *

warnings.filterwarnings('ignore')
MAIN = load_panel()
IDX = MAIN.index
d = pd.read_parquet(os.path.join(CACHE, 'panel_kcs.parquet')); d['sc'] = d['sc'].fillna('NA')
SUB = d.pivot_table(index=['ksic', 'cc', 'sc'], columns='year', values='n', aggfunc='sum').reindex(columns=YEARS).fillna(0.0)
GROUPS = {}
for key, g in SUB.groupby(level=['ksic', 'cc']):
    GROUPS[key] = g.values
assert all(np.allclose(GROUPS[k].sum(0), MAIN.loc[k].values) for k in IDX)


def ld(M, k, phi):
    L = np.log1p(M)
    g = (L[:, -1] - L[:, -1 - k]) / k
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return np.expm1(L[:, -1:] + g[:, None] * steps[None, :])


def bu_fn(hist, o, k=8, phi=0.8, minavg=30, recent=3, gcap=None, blend=None):
    T_ = hist.shape[1]
    out = np.zeros((hist.shape[0], H))
    for i, key in enumerate(IDX):
        G = GROUPS[key][:, :T_]
        big = G[:, -recent:].mean(1) >= minavg  # 원점 직전 recent년 평균 기준 (원점 이전 정보)
        comps = G[big]
        rest = G[~big].sum(0, keepdims=True)
        M = np.vstack([comps, rest]) if comps.shape[0] else rest
        F = ld(M, k, phi)
        if gcap is not None:  # 성분 성장률 상한(작은 성분 폭주 방지)
            F = np.minimum(F, M[:, -1:] * np.exp(gcap * np.arange(1, H + 1))[None, :])
        out[i] = F.sum(0)
    if blend is not None:  # 합계 직접 외삽과 혼합
        out = blend * out + (1 - blend) * ld(hist, k, phi)
    return np.maximum(out, 0)


if __name__ == '__main__':
    runs = [('bu_m30', {}), ('bu_m10', {'minavg': 10}), ('bu_m100', {'minavg': 100}),
            ('bu_m30_k5', {'k': 5}), ('bu_m30_p6', {'phi': 0.6}), ('bu_m30_cap', {'gcap': 0.4})]
    if len(sys.argv) > 1:
        runs = [(f'bu_custom{i}', json.loads(a)) for i, a in enumerate(sys.argv[1:])]
    for nm, kw in runs:
        r = evaluate(bu_fn, **kw)
        report(r, nm, seg=True)
        save_pred(r, nm)
