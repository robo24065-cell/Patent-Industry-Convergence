# -*- coding: utf-8 -*-
r"""C19-T2 s5: 분포 이동 표(famn_shift.csv)와 데이터 점검 수치(cache/s5_check.json).
 - 토큰 연도 2000–2024: 토큰 수, F1·F1x 중복 비중(같은 pid·같은 키 두 번째 토큰 비율)
 - 프레임(frame_cls·frame_haz·ev)·기준일 L 별: 행 수, fam>0 비율, fam_n 평균·분위(원문·F1·F1x), fam>0 중 부풀림 비율, 원문/정규화 비 평균
 - 학습(frame_cls 전체·최근가중) 대 최종(ev) 부풀림 비율 차(사전 서술 기준 10%p)
 - 선택 편향(ev): fam>0 중 F1 부풀림 단위 vs 비부풀림 단위의 포기율·출원인유형·경과연수·청구항·KSIC 상위, fam=0 단위 비교
 - fam_n 구간별 포기율(원문 vs F1)"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import numpy as np, pandas as pd
import pyarrow.parquet as pq

tk = pd.read_parquet(J(C, 's1_tokens_keys.parquet'), columns=['pid', 'yr', 'k1', 'kx', 'cc'])
rows = []
d1 = tk.duplicated(['pid', 'k1']).values; dx = tk.duplicated(['pid', 'kx']).values
for y_ in range(2000, 2025):
    m = tk.yr.values == y_
    rows.append(dict(table='token_year', frame='', L='', year=y_, n=int(m.sum()), dup_share_F1=round(float(d1[m].mean()), 5), dup_share_F1x=round(float(dx[m].mean()), 5),
                     dup_share_F1_US=round(float(d1[m & (tk.cc.values == 'US')].mean()), 5), dup_share_F1_EP=round(float(d1[m & (tk.cc.values == 'EP')].mean()), 5),
                     dup_share_F1_JP=round(float(d1[m & (tk.cc.values == 'JP')].mean()), 5)))
R = {}


def summ(raw, f1, fx, w=None):
    pos = raw > 0
    w = np.ones(len(raw)) if w is None else w
    r = dict(n=int(len(raw)), fam_pos_share=round(float(np.average(pos, weights=w)), 4))
    for nm, a in [('raw', raw), ('F1', f1), ('F1x', fx)]:
        r[f'mean_{nm}'] = round(float(np.average(a, weights=w)), 4)
        q = np.quantile(a[pos], [0.5, 0.75, 0.9, 0.99]) if pos.any() else [0] * 4
        r.update({f'{nm}_p{p}': float(v) for p, v in zip([50, 75, 90, 99], q)})
    if pos.any():
        r['infl_share_F1_famPos'] = round(float(np.average(raw[pos] != f1[pos], weights=w[pos])), 4)
        r['infl_share_F1x_famPos'] = round(float(np.average(raw[pos] != fx[pos], weights=w[pos])), 4)
        r['ratio_raw_F1_famPos'] = round(float(np.average(raw[pos] / np.maximum(f1[pos], 1), weights=w[pos])), 4)
        r['ratio_raw_F1x_famPos'] = round(float(np.average(raw[pos] / np.maximum(fx[pos], 1), weights=w[pos])), 4)
    return r


agg = {}
for fr in ['frame_cls', 'frame_haz', 'ev']:
    d = pq.read_table(J(F22, fr + '.parquet'), columns=['pid', 'L']).to_pandas()
    raw = np.load(J(C, 'famn', f'P0__{fr}.npy')).astype(float); f1 = np.load(J(C, 'famn', f'F1__{fr}.npy')).astype(float); fx = np.load(J(C, 'famn', f'F1x__{fr}.npy')).astype(float)
    Ls = d.L.astype(str).str[:10].values
    for L in sorted(set(Ls)):
        m = Ls == L
        rows.append(dict(table='frame_L', frame=fr, L=L, **summ(raw[m], f1[m], fx[m])))
    rows.append(dict(table='frame_all', frame=fr, L='all', **summ(raw, f1, fx)))
    yr = pd.to_datetime(pd.Series(Ls)).dt.year.values
    w = 1.0 + (yr - 2005) / 2.0                               # 공식 recency_w(linear, base 2005, div 2)
    if fr != 'ev':
        rows.append(dict(table='frame_all_recencyW', frame=fr, L='all(w)', **summ(raw, f1, fx, w)))
    agg[fr] = rows[-1]
sh = pd.DataFrame(rows)
sh.to_csv(J(W, 'famn_shift.csv'), index=False, encoding='utf-8-sig')
ev_r = sh[(sh.table == 'frame_all') & (sh.frame == 'ev')].iloc[0]
for fr in ['frame_cls', 'frame_haz']:
    for t in ['frame_all', 'frame_all_recencyW']:
        r_ = sh[(sh.table == t) & (sh.frame == fr)].iloc[0]
        R[f'shift_{fr}_{t}'] = dict(infl_F1_train=r_.infl_share_F1_famPos, infl_F1_final=ev_r.infl_share_F1_famPos, diff_F1_pp=round(100 * (ev_r.infl_share_F1_famPos - r_.infl_share_F1_famPos), 2),
                                    infl_F1x_train=r_.infl_share_F1x_famPos, infl_F1x_final=ev_r.infl_share_F1x_famPos, diff_F1x_pp=round(100 * (ev_r.infl_share_F1x_famPos - r_.infl_share_F1x_famPos), 2),
                                    ratio_F1_train=r_.ratio_raw_F1_famPos, ratio_F1_final=ev_r.ratio_raw_F1_famPos, mean_raw_train=r_.mean_raw, mean_raw_final=ev_r.mean_raw,
                                    fam_pos_train=r_.fam_pos_share, fam_pos_final=ev_r.fam_pos_share)
R['shift_flag_10pp'] = {k: (abs(v['diff_F1_pp']) >= 10, abs(v['diff_F1x_pp']) >= 10) for k, v in R.items() if k.startswith('shift_')}
out('shift', R)
# ---------------- 선택 편향(ev)
ev = pq.read_table(J(F22, 'ev.parquet'), columns=['pid', 'y', 'atype', 'ksic', 'age', 'claims', 'fam_cc', 'fam_us', 'fam_ep', 'fam_jp']).to_pandas()
raw = np.load(J(C, 'famn', 'P0__ev.npy')).astype(float); f1 = np.load(J(C, 'famn', 'F1__ev.npy')).astype(float); fx = np.load(J(C, 'famn', 'F1x__ev.npy')).astype(float)
grp = np.where(raw == 0, 'fam0', np.where(raw != f1, 'famPos_inflF1', 'famPos_clean'))
ev['grp'] = grp
sel = ev.groupby('grp').agg(n=('y', 'size'), y_rate=('y', 'mean'), age=('age', 'mean'), claims=('claims', 'mean'), fam_cc=('fam_cc', 'mean'),
                            us=('fam_us', 'mean'), ep=('fam_ep', 'mean'), jp=('fam_jp', 'mean')).round(4)
sel['raw_mean'] = pd.Series(raw).groupby(grp).mean().round(3); sel['F1_mean'] = pd.Series(f1).groupby(grp).mean().round(3)
R['selection_ev'] = sel.reset_index().to_dict('records')
R['atype_by_grp'] = pd.crosstab(ev.grp, ev.atype, normalize='index').round(4).to_dict('index')
pos = raw > 0
R['infl_share_by_atype_famPos'] = ev[pos].assign(inf=(raw != f1)[pos]).groupby('atype').inf.agg(['size', 'mean']).round(4).to_dict('index')
ks = ev[pos].assign(inf=(raw != f1)[pos]).groupby('ksic').inf.agg(['size', 'mean'])
R['infl_share_by_ksic_famPos_range'] = [round(float(ks['mean'].min()), 4), round(float(ks['mean'].max()), 4)]
R['infl_share_by_ksic_famPos_top5'] = ks.sort_values('mean', ascending=False).head(5).round(4).to_dict('index')
R['infl_share_by_ksic_famPos_bottom5'] = ks.sort_values('mean').head(5).round(4).to_dict('index')
# within fam_cc strata: inflated vs clean y rate (같은 국가 수 안에서 비교)
st = ev[pos].assign(inf=(raw != f1)[pos]).groupby(['fam_cc', 'inf']).y.agg(['size', 'mean']).round(4)
R['y_by_famcc_infl'] = {f'{int(a)}_{bool(b)}': v for (a, b), v in st.to_dict('index').items() if a <= 6}
# fam_n 구간별 포기율
bins = [-0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 1e9]
lab = ['0', '1', '2', '3-4', '5-8', '9-16', '17+']
for nm, a in [('raw', raw), ('F1', f1), ('F1x', fx)]:
    c = pd.cut(a, bins, labels=lab)
    R[f'y_by_famn_bin_{nm}'] = ev.groupby(c, observed=False).y.agg(['size', 'mean']).round(4).to_dict('index')
from scipy.stats import spearmanr
R['spearman_raw_F1_ev'] = float(spearmanr(raw, f1).statistic); R['spearman_raw_F1x_ev'] = float(spearmanr(raw, fx).statistic)
R['spearman_raw_F1_ev_famPos'] = float(spearmanr(raw[pos], f1[pos]).statistic); R['spearman_raw_F1x_ev_famPos'] = float(spearmanr(raw[pos], fx[pos]).statistic)
jdump(R, J(C, 's5_check.json'))
out('done', R['selection_ev'], R['spearman_raw_F1x_ev'])
