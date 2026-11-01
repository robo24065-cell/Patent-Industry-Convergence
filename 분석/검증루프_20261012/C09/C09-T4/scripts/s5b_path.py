# C09-T4 s5b(사후 진단, 판정 밖): T4 wkd_share–포기 연관의 경로 분해. O12·O16·O18(개발) — 통제 변수를 차례로 더해 계수 변화를 본다.
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
from s2_t4_feats import fdir
import statsmodels.api as sm
hol = load_hol()
p = pd.read_parquet(os.path.join(C, 't4_pay.parquet')); p = p[p.seq > 0]
due = p.due.values.astype('datetime64[D]')
p['wkd'] = dow(due) >= 5; p['hol'] = np.isin(due, hol) & ~p.wkd
pt = pd.read_parquet(os.path.join(C, 't4_pt.parquet'))
out = {}
for o in ['O12', 'O16', 'O18']:
    ev = pd.read_parquet(os.path.join(fdir(o), 'ev.parquet'), columns=['pid', 'L', 'y', 'age', 'n_renew', 'atype', 'rgst_year', 'acd', 'ipc_sec', 'n_late'])
    ev = ev[ev.y.notna()]
    rows = []
    for L, g in ev.groupby('L'):
        q = p[(p.pd.values <= np.datetime64(str(L))) & p.pid.isin(set(g.pid))]
        a = q.groupby('pid').agg(n=('wkd', 'size'), wkd=('wkd', 'mean'), hol=('hol', 'mean'))
        rows.append(g.join(a, on='pid'))
    d = pd.concat(rows, ignore_index=True); d = d[d.n > 0].copy()
    d['wkd_share'] = d.wkd + d.hol
    big = d.acd.value_counts().head(20).index
    d['big'] = d.acd.isin(big).astype(float)
    def fit(cols, extra):
        X = pd.concat([d[cols]] + extra, axis=1).astype(float)
        m = sm.GLM(d.y.astype(float), sm.add_constant(X), family=sm.families.Binomial()).fit()
        return {c: [round(float(m.params[c]), 4), round(float(m.params[c] - 1.96 * m.bse[c]), 4), round(float(m.params[c] + 1.96 * m.bse[c]), 4)] for c in cols}
    nd = pd.get_dummies(d.n.clip(upper=15).astype(int), prefix='n', drop_first=True, dtype=float)
    ad = pd.get_dummies(d.atype, prefix='a', drop_first=True, dtype=float)
    Ld = pd.get_dummies(d.L.astype(str), prefix='L', drop_first=True, dtype=float)
    yd = pd.get_dummies(d.rgst_year, prefix='ry', drop_first=True, dtype=float)
    sd = pd.get_dummies(d.ipc_sec, prefix='s', drop_first=True, dtype=float)
    res = {}
    res['m1_base(n층+age+atype+L)'] = fit(['wkd_share', 'age'], [nd, ad, Ld])
    res['m2_split_wkd_hol'] = fit(['wkd', 'hol', 'age'], [nd, ad, Ld])
    res['m3_+rgst_year'] = fit(['wkd_share', 'age'], [nd, ad, Ld, yd])
    res['m4_+rgst_year+big20+ipc_sec'] = fit(['wkd_share', 'age', 'big'], [nd, ad, Ld, yd, sd])
    res['m5_+n_late'] = fit(['wkd_share', 'age', 'n_late'], [nd, ad, Ld, yd, sd])
    # 대형 권리자 상위 20을 빼고
    out[o] = res
    print(o, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(C, 's5b_path.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
