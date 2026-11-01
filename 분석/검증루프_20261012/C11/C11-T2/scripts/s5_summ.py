# -*- coding: utf-8 -*-
"""R9 s05: (1) 라벨 무관 mg 결과에서 제공·연계표·TXT·맥락없음 비교. R8 s04 의 paired() 그대로(원점 안 mg 복원추출 1,000회 → 9원점 평균 ΔAUC CI).
R8 예측값(NONCIRC_R8/results)은 읽기만, TXT 팔은 TXT_R9/results. 재현 점검: TXT_R9 에서 다시 돌린 mg_ctx-prov 가 R8 값과 같은지."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
# C11-T2 사본: paired()·쌍 순서·seed·B는 R9 s05 그대로. TXT 예측만 팔별 파일로. 사용: python s5_summ.py <arm>  (arm = stored|r8|reg|pub|pure)
import sys; sys.dont_write_bytecode = True
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
T3 = BASE + "/분석/모델경쟁_20260927/T3_기술셀부상예측"; WD = BASE + "/분석/검증루프_20261012/C11/C11-T2"; ARM = sys.argv[1]
R8R = os.path.join(T3, 'NONCIRC_R8', 'results'); R9S = os.path.join(T3, 'TXT_R9', 'results'); R9R = WD + '/res/runs'
B = 1000; rng = np.random.default_rng(20261005); ORIG = list(range(2010, 2019))


def auc(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired(dfA, dfB):
    aA, aB, bs = [], [], np.zeros((len(ORIG), B))
    for i, v in enumerate(ORIG):
        a = dfA[dfA.t0 == v].sort_values('cell'); b = dfB[dfB.t0 == v].sort_values('cell')
        assert (a.cell.values == b.cell.values).all() and (a.yb.values == b.yb.values).all()
        y = a.yb.values; pa = a.p.values; pb = b.p.values
        aA.append(auc(y, pa)); aB.append(auc(y, pb))
        I = rng.integers(0, len(y), (B, len(y)))
        for j in range(B):
            ii = I[j]; yy = y[ii]
            bs[i, j] = np.nan if yy.sum() in (0, len(yy)) else auc(yy, pa[ii]) - auc(yy, pb[ii])
    d = np.array(aA) - np.array(aB); bm = np.nanmean(bs, 0)
    lo, hi = float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5)); w = int((d > 0).sum())
    verdict = 'A 우위' if (d.mean() > 0 and lo > 0 and w >= 6) else ('B 우위' if (d.mean() < 0 and hi < 0 and (9 - w) >= 6) else '같다')
    return dict(aucA=float(np.mean(aA)), aucB=float(np.mean(aB)), dAUC=float(d.mean()), ci=[lo, hi], wins=w, n_orig=len(ORIG),
                by_origin=[round(float(x), 4) for x in d], verdict=verdict)


def L(tag, d):
    return pd.read_parquet(os.path.join(d, f'mg_ctx-{tag}_preds.parquet'))


M = {k: L(k, R8R) for k in ['none', 'prov', 'conc', 'concm', 'prov_lite', 'conc_lite', 'prov_s3', 'conc_s3']}
TXTF = {}
for k in ['txt', 'txt_lite', 'txt_s3']:
    f = os.path.join(R9S, f'mg_ctx-{k}_preds.parquet') if ARM == 'stored' else os.path.join(R9R, f'mg_ctx-{k}__{ARM}_preds.parquet')
    if os.path.exists(f): M[k] = pd.read_parquet(f); TXTF[k] = f
R = {'arm': ARM, 'txt_files': TXTF}
base = M['none'].copy(); base['p'] = base.p_base
rows = []
for k, d in M.items():
    j = json.load(open(TXTF[k].replace('_preds.parquet', '.json') if k.startswith('txt') else os.path.join(R8R, f'mg_ctx-{k}.json'), encoding='utf-8'))
    rows.append(dict(model=k, auc=round(float(np.mean([auc(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG])), 4),
                     skill=round(j['summary']['mean_skill'], 4), p10=round(j['summary']['mean_p10'], 4), n=j['summary']['mean_n'], evt=j['summary']['mean_evt'],
                     ctx_cov=(j['info'].get('ctx_nonmissing_eval') or {}).get('c_ind_kmom')))
R['models'] = rows; print(pd.DataFrame(rows).to_string())
R['baseline_auc'] = float(np.mean([auc(base[base.t0 == v].yb, base[base.t0 == v].p) for v in ORIG]))
pairs = [('prov', 'txt'), ('txt', 'none'), ('txt', 'conc'), ('prov', 'conc'), ('prov', 'none'),
         ('prov_lite', 'txt_lite'), ('txt_lite', 'none'), ('prov_s3', 'txt_s3'), ('txt_s3', 'conc_s3')]
R['pairs'] = {}
for A, Bn in pairs:
    if A in M and Bn in M:
        r = paired(M[A], M[Bn]); R['pairs'][f'{A}_vs_{Bn}'] = r
        print(f"{A} − {Bn}: {r['aucA']:.4f} vs {r['aucB']:.4f} Δ{r['dAUC']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] 우세 {r['wins']}/9 → {r['verdict']}", flush=True)
json.dump(R, open(WD + f'/res/s5_summary_mg__{ARM}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('saved summary_mg_r9.json')
