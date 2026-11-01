# -*- coding: utf-8 -*-
"""C17-T2 verify(사후 민감도, 판정 외) V9: 판정 불가 F 쌍 안에서 오매칭을 균등 배치(주)하지 않고 모형 점수로 배치.
 판정 불가 F의 점수 분포 = 정확(가시 정확 쌍의 점수 분포로 근사) × (1−r) + 오매칭 × r. 점수 10분위 막대로 p_i = 1 − (1−r)·g_c(s)/f_u(s)를 구하고
 칸별 기대 제거 수를 주(E2)와 같게 맞춘 뒤 Bernoulli 1,000회 Δ(i). 산출: verify/v_score_informed_out.json"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent/'scripts'))
import guard17 as G
from c17common import *
pp = pd.read_parquet(CC/'p_point.parquet')
Z = np.load(CC/'eval_base.npz', allow_pickle=True); s = Z['s']; y = Z['y']; base = Z['base']
key = pd.Series(np.arange(len(y)), index=pd.MultiIndex.from_arrays([Z['firm'], Z['k']]))
rows = key.reindex(pd.MultiIndex.from_arrays([pp.firm.astype(str).values, pp.k.values])).values.astype(int)
pct = pd.Series(s).rank(pct=True).values[rows]
pos = np.where(y == 1)[0]; neg = y == 0; n0 = neg.sum(); pm = -np.ones(len(y), int); pm[pos] = np.arange(len(pos)); prow = pm[rows]
def contrib(v):
    sn = np.sort(v[neg]); sp = v[pos]; lo = np.searchsorted(sn, sp, 'left'); hi = np.searchsorted(sn, sp, 'right'); return (lo + 0.5 * (hi - lo)) / n0
Cm = contrib(s); Cb = np.vstack([contrib(b) for b in base])
def sk(drop):
    k = np.ones(len(pos), bool); k[drop] = False; a = Cm[k].mean(); M0 = Cb[:, k].mean(1).max(); return (a - M0) / (1 - M0)
OFF = sk(np.array([], int))
edges = np.quantile(pct, np.linspace(0, 1, 11)); edges[0] = 0; edges[-1] = 1.000001
bins = np.digitize(pct, edges) - 1
vis = ((pp.pv == 'correct') & (pp.ptier == 1)).values; und = ((pp.pv == 'undet') & pp.F).values
gc = np.bincount(bins[vis], minlength=10) / vis.sum()
p = pp.p_main.values.copy()
for (fo, sz), _ in pp[und].groupby(['foreign', 'size']):
    c = und & (pp.foreign.values == fo) & (pp['size'].values == sz)
    m = p[c].sum()                       # 주 추정의 칸 기대 제거 수
    if m <= 0: continue
    r = m / c.sum(); fu = np.bincount(bins[c], minlength=10) / c.sum()
    pi = np.clip(1 - (1 - r) * gc[bins[c]] / np.maximum(fu[bins[c]], 1e-9), 0, 1)
    pi = pi * m / pi.sum() if pi.sum() > 0 else pi       # 기대 제거 수 맞춤
    p[c] = np.clip(pi, 0, 1)
rs = np.random.RandomState(13); d = [sk(prow[rs.rand(len(p)) < p]) - OFF for _ in range(1000)]
out = dict(expected_removed=float(p.sum()), main_expected_removed=float(pp.p_main.sum()), delta=dict(mean=float(np.mean(d)), p2_5=float(np.percentile(d, 2.5)), p97_5=float(np.percentile(d, 97.5))),
           undet_F_pct_median=float(np.median(pct[und])), visible_correct_pct_median=float(np.median(pct[vis])))
json.dump(out, open(pathlib.Path(__file__).resolve().parent/'v_score_informed_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
