# -*- coding: utf-8 -*-
r"""C18-T2 verify_0: duckdb 없이 pyarrow/pandas 독립 코드로 재계산.
 (1) US 7/8자리 쌍 수·일치율, EP 짧은/긴 형식 쌍 수·일치율 (s1a·s1c 대조, 차 0)
 (2) 표 2·3 해외↔KR 등록번호 연결: US·EP 쌍 수와 KSIC 일치율을 P0·V1·V1E로 (s3d 대조)
 (3) T3 Δskill: 저장 예측에서 자체 AUC(Mann–Whitney 순위식) — t3res json 대조(차 < 1e-9)
 (4) T3 변형 패널 US·EP 카운트 변화: base_rows(pandas) 차이 vs panel_ccy sum(n) 차이
 (5) 쌍이 원고 3.1 P3 창(출원 2017–19)에 0건인지(→ KR RTA>1·1위 수 구성상 불변)
 사용: python v0_independent.py <step 1..5> → verify_0/v0_<step>.json"""
import os, sys, json, glob, re
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from scipy.stats import rankdata
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T2'
MC = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927'
MD = MC + r'\_shared\meta'
C = W + r'\cache'; step = sys.argv[1]; R = {}
dset = ds.dataset(MD, format='parquet')


def rows(cc, cols):
    return dset.to_table(columns=cols, filter=pc.field('cc') == cc).to_pandas()


def nt(s): return s.fillna('').str.lower().str.replace(r'\s+', ' ', regex=True)


if step == '1':
    u = rows('US', ['appl_no', 'title', 'appl_date', 'main_ipc', 'ksic'])
    s7 = u[u.appl_no.str.len() == 7].copy(); s7['a8'] = s7.appl_no.str.zfill(8)
    l8 = u[u.appl_no.str.len() == 8]
    p = s7.merge(l8, left_on='a8', right_on='appl_no', suffixes=('7', '8'))
    R['us_pairs'] = int(len(p))
    R['us_title_eq'] = float((nt(p.title7) == nt(p.title8)).mean()); R['us_date_eq'] = float((p.appl_date7.fillna('') == p.appl_date8.fillna('')).mean())
    R['us_ksic_eq'] = float((p.ksic7.astype(str) == p.ksic8.astype(str)).mean()); R['us_ksic_diff_n'] = int((p.ksic7.astype(str) != p.ksic8.astype(str)).sum())
    s1a = json.load(open(C + r'\s1a.json', encoding='utf-8'))
    R['diff_vs_s1a'] = dict(pairs=R['us_pairs'] - s1a['n_pairs'][0], title=round(R['us_title_eq'] - s1a['match_rates_all']['title_eq'], 5),
                           date=round(R['us_date_eq'] - s1a['match_rates_all']['appl_date_eq'], 5), ksic=round(R['us_ksic_eq'] - s1a['match_rates_all']['ksic_eq'], 5))
    e = rows('EP', ['appl_no', 'title', 'appl_date', 'ksic'])
    e['k'] = e.appl_no.str.replace(r'[^0-9]', '', regex=True).str.lstrip('0')
    g = e.groupby('k').appl_no.nunique(); kk = set(g[g > 1].index)
    x = e[e.k.isin(kk)].sort_values(['k', 'appl_no'])
    a = x.groupby('k').agg(n=('appl_no', 'size'), t=('title', lambda s: nt(s).nunique() == 1), d=('appl_date', lambda s: s.fillna('').nunique() == 1))
    R['ep_pairs'] = int(len(a)); R['ep_rows_per_key_max'] = int(a.n.max()); R['ep_title_eq'] = float(a.t.mean()); R['ep_date_eq'] = float(a.d.mean())
elif step == '2':
    kr = rows('KR', ['appl_no', 'rgst_no', 'ksic']); kr = kr[kr.rgst_no.notna() & (kr.rgst_no.str.len() >= 9)]
    kr['r9'] = kr.rgst_no.str[:9]; kr = kr[['appl_no', 'r9', 'ksic']].rename(columns={'appl_no': 'kr', 'ksic': 'kk'})
    us7 = set(pd.read_parquet(C + r'\us_dup_pairs.parquet').a7)
    op = pd.read_parquet(C + r'\other_dup_pairs.parquet'); op = op[op.cc == 'EP']; ep9 = set(np.where(op.L1 < op.L2, op.n1, op.n2))
    for cc in ('US', 'EP'):
        f = rows(cc, ['appl_no', 'ksic', 'family']); f = f[f.family.fillna('').str.contains('KR')]
        t = f.assign(t=f.family.str.split('|')).explode('t'); t = t[t.t.str.fullmatch(r'KR(10|20)[0-9]{7}', na=False)]
        t['r9'] = t.t.str[2:]; t = t[['appl_no', 'ksic', 'r9']].drop_duplicates()
        j = t.merge(kr, on='r9')
        for V, drop in (('P0', set()), ('V1', us7 if cc == 'US' else set()), ('V1E', us7 if cc == 'US' else ep9)):
            jj = j[~j.appl_no.isin(drop)]
            R[f'{cc}_{V}'] = [int(len(jj)), int(jj.appl_no.nunique()), float((jj.ksic.astype(str) == jj.kk.astype(str)).mean())]
    s3d = json.load(open(C + r'\s3d_lofam.json', encoding='utf-8'))
    R['s3d'] = {V: [r for r in s3d[V] if r[0] in ('US', 'EP')] for V in ('P0', 'V1', 'V1E')}
elif step == '3':
    def auc(y, s):
        r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    T = C + r'\t3res'
    for f in sorted(glob.glob(T + r'\res_*.json')):
        js = json.load(open(f, encoding='utf-8')); tag = os.path.basename(f)[4:-5]
        pf = T + rf'\preds_{tag}.parquet'
        if not os.path.exists(pf): continue
        p = pd.read_parquet(pf); md = []
        for o, g in p.groupby('t0'):
            a, b = auc(g.yb.values, g.p.values), auc(g.yb.values, g.p_base.values); sk = (a - b) / (1 - b)
            md.append(abs(sk - js['by_origin'][str(o)]['skill']))
        R[tag] = float(max(md))
elif step == '4':
    for V in ('V1', 'V1E', 'V2', 'V12'):
        for m in ('fin',):
            b0 = pd.read_parquet(C + rf'\t3_P0_{m}\..\..\cache\t3_P0_{m}\panel_ccy.parquet') if False else None
            off = MC + r'\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet'
            cnt = lambda f: ds.dataset(f).to_table(columns=['cc', 'year']).to_pandas().groupby('cc').size()
            pn = lambda f: pd.read_parquet(f, columns=['cc', 'n']).groupby('cc').n.sum()
            db = (cnt(off) - cnt(C + rf'\t3_{V}_{m}\base_rows.parquet')).to_dict()
            dp = (pn(C + rf'\t3_P0_{m}\panel_ccy.parquet') - pn(C + rf'\t3_{V}_{m}\panel_ccy.parquet')).to_dict()
            R[f'{V}_{m}'] = dict(base_removed=db, panel_n_removed=dp)
elif step == '5':
    us = pd.read_parquet(C + r'\us_dup_pairs.parquet'); op = pd.read_parquet(C + r'\other_dup_pairs.parquet')
    R['us_pairs_appl_2015_2019'] = int(((us.y8 >= 2015) & (us.y8 <= 2019)).sum())
    R['ep_pairs_appl_2015_2019'] = int(((op.cc == 'EP') & (op.y1 >= 2015) & (op.y1 <= 2019)).sum())
    R['ep_pairs_appl_range'] = [int(op[op.cc == 'EP'].y1.min()), int(op[op.cc == 'EP'].y1.max())]
json.dump(R, open(os.path.join(W, 'verify_0', f'v0_{step}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, default=str)[:3000])
