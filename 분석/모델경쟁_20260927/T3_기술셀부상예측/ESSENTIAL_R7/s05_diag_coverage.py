# -*- coding: utf-8 -*-
"""R7 진단: 제공 셀(KSIC 5자리×메인그룹, 적격 ≥50, 개발 원점)이 연계표 셀(코드×메인그룹)에서 얼마나 '보이는가'.
  - 대응 연계표 셀의 5개국 t0-2..t0 문서 수가 0인 비율, 그 안의 부상 셀 비율
  - 같은 연계표 코드를 공유하는 KSIC 수(해상도 손실): 한 연계표 셀에 몇 개 제공 셀이 겹쳐 같은 피처를 받는가
결과: diag_coverage.json"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
KM = pd.read_csv(os.path.join(HERE, 'conc_ksic_map.csv'), dtype=str).set_index('ksic').code.to_dict()
out = {}
for var, Y1 in [('pub', 2021), ('app', 2019)]:
    P = pd.read_parquet(os.path.join(HERE, 'results', f'{var}_full_preds.parquet'))
    P['alt'] = P.ksic.map(KM) + '|' + P.mg
    for ck in ['conc', 'concm']:
        f = os.path.join(HERE, f'cache_{ck}_{var}', 'panel_ccy.parquet')
        if not os.path.exists(f): continue
        pc = pd.read_parquet(f, columns=['ksic', 'mg', 'cc', 'year', 'n']); pc['cell'] = pc.ksic + '|' + pc.mg
        rows = []
        for t0, g in P.groupby('t0'):
            w = pc[(pc.year >= t0 - 2) & (pc.year <= t0)].groupby('cell').n.sum()
            a = g.alt.map(w).fillna(0)
            rows.append(dict(t0=int(t0), n=len(g), zero_share=float((a == 0).mean()), evt_in_zero=float(g.yb[a == 0].sum() / max(1, g.yb.sum())),
                             evt_rate_zero=float(g.yb[a == 0].mean()) if (a == 0).any() else None,
                             cells_per_alt=float(g.groupby('alt').size().mean())))
        R = pd.DataFrame(rows)
        out[f'{var}_{ck}'] = dict(by_t0=rows, mean_zero_share=float(R.zero_share.mean()), mean_evt_in_zero=float(R.evt_in_zero.mean()),
                                  mean_cells_per_alt=float(R.cells_per_alt.mean()))
        print(var, ck, out[f'{var}_{ck}']['mean_zero_share'], out[f'{var}_{ck}']['mean_evt_in_zero'], out[f'{var}_{ck}']['mean_cells_per_alt'])
json.dump(out, open(os.path.join(HERE, 'diag_coverage.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# 결과(부상 레이블) 일치: 제공 셀 부상 중 대응 연계표 셀도 연계표 과제에서 부상인 비율 / 연계표 부상 셀 중 대응 제공 셀(같은 코드의 KSIC × 같은 메인그룹) 하나라도 부상인 비율
ov = {}
for var in ['pub', 'app', 'pubF']:
    for ck in ['conc', 'concm']:
        fa, fc = os.path.join(HERE, 'results', f'{var}_full_preds.parquet'), os.path.join(HERE, 'results', f'{var}_{ck}_task_preds.parquet')
        if not (os.path.exists(fa) and os.path.exists(fc)): continue
        A = pd.read_parquet(fa); Cc = pd.read_parquet(fc); A['alt'] = A.ksic.map(KM) + '|' + A.mg
        rr = []
        for t0, g in A.groupby('t0'):
            c = Cc[Cc.t0 == t0].set_index('cell'); b = g[g.yb == 1]
            in_set = b.alt.isin(c.index); hit = b.alt.map(c.yb).fillna(0) == 1
            cb = c[c.yb == 1]; prov_b = set(b.alt)
            rr.append(dict(t0=int(t0), prov_breakouts=int(len(b)), alt_cell_eligible=float(in_set.mean()), alt_also_breakout=float(hit.mean()),
                           alt_breakouts=int(len(cb)), alt_breakout_with_prov_breakout=float(pd.Series(cb.index).isin(prov_b).mean())))
        R = pd.DataFrame(rr)
        ov[f'{var}_{ck}'] = dict(by_t0=rr, **{f'mean_{k}': float(R[k].mean()) for k in ['alt_cell_eligible', 'alt_also_breakout', 'alt_breakout_with_prov_breakout']})
        print('label overlap', var, ck, {k: round(v, 3) for k, v in ov[f'{var}_{ck}'].items() if k != 'by_t0'})
out['label_overlap'] = ov
json.dump(out, open(os.path.join(HERE, 'diag_coverage.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
