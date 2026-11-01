# -*- coding: utf-8 -*-
"""T4 의미 검정: 정부R&D 기원 특허(수행주체 유형별)는 같은 연차·KSIC·출원인 유형을 통제해도 3년 내 등록료불납 포기율이 다른가.
- 평가행: T4 R3 개발 원점 프레임(R3/runs/O12..O18/ev.parquet, 기준일 L 8개 2012-01~2018-07, 결과창 끝 ≤2021-07). 읽기만.
- 🔒 F=2022 결과는 쓰지 않는다(프레임 자체가 dev 모드: 소멸 E<2022-01-01).
- 시점 정합: NTIS 성과 레코드 Year ≤ L연도−1 인 연결만 '기준일에 알 수 있던' 정부 기원으로 본다(pit). 스냅샷 전체(any)도 비교.
출력: ntis_T_work/ntis_T4_*.csv/json
"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ntis_T_common import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ORIG = ['O12', 'O14', 'O16', 'O18']
# ---- pid -> rgst_no (t4lib.load 와 같은 규칙: 등록일 있음 & 특허(10))
pt = pd.read_parquet(os.path.join(T4, 'cache', 'patent.parquet'), columns=['rgst_no', 'ksic', 'rgstDate'])
rd = pd.to_datetime(pt.rgstDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
pt = pt[rd.notna().values & (pt.rgst_no.str[:2] == '10').values].reset_index(drop=True)
pt['pid'] = np.arange(len(pt))

# ---- NTIS 연결(특허×성과) → 특허별 주체그룹별 최소 레코드 연도
lk = pd.read_parquet(os.path.join(EXT, 'ntis_patent_link.parquet'), columns=['rgst_no', 'Year', 'PerformAgent', 'ProjectYear'])
lk = lk[lk.rgst_no.notna() & (lk.rgst_no.astype(str).str.len() > 5)].copy()
lk['grp'] = lk.PerformAgent.astype(str).map(GRP_MAP).fillna('oth')
lk['yr'] = pd.to_numeric(lk.Year, errors='coerce')
mn = lk.groupby(['rgst_no', 'grp']).yr.min().unstack()           # 특허 × 그룹 최소 레코드 연도
mn['any'] = mn.min(1)
pt = pt.merge(mn, left_on='rgst_no', right_index=True, how='left')
print('NTIS 연결 T4 특허', int(pt['any'].notna().sum()), '/', len(pt), flush=True)

# ---- 챔피언(X21) 예측 불러오기(없으면 E00) — 파일만 읽음
SPEC0 = json.load(open(os.path.join(T4, 'final_spec.json'), encoding='utf-8'))
CAT = SPEC0['categorical']


def comp_key(spec, comp):
    cfg = spec['components'][comp]
    drop = {'cls_label_rule', 'haz_label_rule', 'cap_rule'} | (set() if comp.startswith('haz') else {'haz_semi'})
    blob = dict(comp=comp, cfg=cfg, CAT=CAT, train={k: v for k, v in spec['train'].items() if k not in drop},
                ext_ver=spec.get('ext_ver', {}))
    return comp + '_' + hashlib.md5(json.dumps(blob, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def champ_pred(o):
    for ex in ('X21', 'E00'):
        sp = json.load(open(os.path.join(T4, 'R3', 'specs', f'{ex}.json'), encoding='utf-8'))['spec']
        ws = sp['ensemble']['weights']
        fs = {k: os.path.join(T4, 'R3', 'runs', o, 'pred', comp_key(sp, k) + '.npy') for k, w in ws.items() if w}
        if all(os.path.exists(f) for f in fs.values()):
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            z = sum(ws[k] * lg(np.load(f)) for k, f in fs.items()) / sum(ws[k] for k in fs)
            return z, ex
    return None, None


rows, reg_rows, ch_rows = [], [], []
for o in ORIG:
    ev = pd.read_parquet(os.path.join(T4, 'R3', 'runs', o, 'ev.parquet'),
                         columns=['pid', 'L', 'ksic', 'age', 'atype', 'acd', 'y', 'next_annl', 'rgst_year', 'claims'])
    assert ev.L.astype(str).str[:10].max() <= '2019-01-01'
    zc, ex = champ_pred(o)
    ev['z_champ'] = zc if zc is not None else np.nan
    chk = pt.ksic.values[ev.pid.values] == ev.ksic.astype(str).values
    print(o, 'pid→ksic 일치율', round(float(chk.mean()), 4), '챔피언 예측', ex, flush=True)
    e = ev.join(pt[['rgst_no'] + GRPS + ['any']].iloc[ev.pid.values].reset_index(drop=True))
    Ly = e.L.astype(str).str[:4].astype(int)
    for g in GRPS + ['any']:
        e[f'g_{g}'] = (e[g].astype(float) <= Ly - 1).astype(float)          # pit: 기준일 전년까지 올라온 레코드
        e[f's_{g}'] = e[g].notna().astype(float)               # 스냅샷 전체
    e = e[e.y.notna()].copy()
    e['ann'] = e.next_annl.fillna(-1).astype(int)
    e['fe'] = e.L.astype(str) + '|' + e.ann.astype(str) + '|' + e.ksic.astype(str) + '|' + e.atype.astype(str)
    e['fe_a'] = e.L.astype(str) + '|' + e.ann.astype(str) + '|' + e.acd.astype(str)
    e['acd_c'] = e.acd.astype(str)
    for samp, m in (('rg2007', e.rgst_year >= 2007), ('all', e.rgst_year > 0)):
        s = e[m.values]
        # 기술통계
        for g in GRPS + ['any']:
            for pre in ('g', 's'):
                v = s[f'{pre}_{g}'] == 1
                rows.append(dict(origin=o, sample=samp, flag=f'{pre}_{g}', n=len(s), n_flag=int(v.sum()), share=round(float(v.mean()), 4),
                                 lapse_flag=round(float(s.y[v].mean()), 4) if v.any() else np.nan,
                                 lapse_other=round(float(s.y[~v].mean()), 4)))
        # 고정효과 LPM: (a) 전체 정부 기원 (b) 주체별
        for pre in ('g', 's'):
            for spec_nm, xs in (('any', [f'{pre}_any']), ('grp', [f'{pre}_{g}' for g in GRPS])):
                for fe_nm, fe, cl in (('L×연차×KSIC×출원인유형', 'fe', 'acd_c'), ('L×연차×출원인(동일 출원인 내)', 'fe_a', 'acd_c')):
                    ss = s if fe == 'fe' else s[s.acd_c != 'NA']
                    r = fe_ols(ss, 'y', xs, fe, cl)
                    for x, rr in r.iterrows():
                        reg_rows.append(dict(origin=o, sample=samp, flags=pre, model=spec_nm, fe=fe_nm, x=x, coef=rr.coef, se=rr.se, t=rr.t,
                                             n=int(rr.n), n_clu=int(rr.n_clu)))
        # 챔피언 대비 증분(로지스틱: y ~ z_champ + 플래그, 같은 원점 표본 in-sample; 파라미터 ≤6)
        if samp == 'all' and s.z_champ.notna().all():
            base = LogisticRegression(C=1e6, max_iter=500).fit(s[['z_champ']], s.y)
            p0 = base.predict_proba(s[['z_champ']])[:, 1]
            xs = ['z_champ'] + [f'g_{g}' for g in GRPS]
            m1 = LogisticRegression(C=1e6, max_iter=500).fit(s[xs], s.y)
            p1 = m1.predict_proba(s[xs])[:, 1]
            a0, a1 = roc_auc_score(s.y, p0), roc_auc_score(s.y, p1)
            ch_rows.append(dict(origin=o, champ=ex, n=len(s), auc_champ=round(a0, 5), auc_plus=round(a1, 5), d_auc=round(a1 - a0, 5),
                                **{f'lor_{g}': round(float(c), 4) for g, c in zip(GRPS, m1.coef_[0][1:])}))
    print(o, 'done', peak_mem_gb(), flush=True)

pd.DataFrame(rows).to_csv(out('ntis_T4_desc.csv'), index=False, encoding='utf-8-sig')
R = pd.DataFrame(reg_rows); R.to_csv(out('ntis_T4_fe.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(ch_rows).to_csv(out('ntis_T4_champ.csv'), index=False, encoding='utf-8-sig')
# 요약: 원점 평균·부호 일관성
S = R.groupby(['sample', 'flags', 'model', 'fe', 'x']).agg(mean_coef=('coef', 'mean'), min_t=('t', 'min'), max_t=('t', 'max'),
                                                          n_neg=('coef', lambda v: int((v < 0).sum())), n_orig=('coef', 'size')).reset_index()
S.to_csv(out('ntis_T4_summary.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(S.to_string())
print(pd.DataFrame(ch_rows).to_string())
