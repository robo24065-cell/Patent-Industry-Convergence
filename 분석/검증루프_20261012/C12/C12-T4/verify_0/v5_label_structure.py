# v5: 제공 KSIC 라벨 구조 점검(기업 결정형인지, 시간 불변인지) — 스냅샷 누수 경로 탐색
import json, numpy as np, pandas as pd
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T4'
C = W + r'\cache'
def H(counts):
    p = counts / counts.sum(); p = p[p > 0]; return -(p * np.log2(p)).sum()
def condH(df, by):
    g = df.groupby(by + ['k']).size().rename('n').reset_index()
    tot = g.groupby(by).n.transform('sum')
    p = g.n / tot
    return float(-(g.n / g.n.sum() * np.log2(p)).sum())
res = {}
for cc in ['CN', 'EP']:
    d = pd.read_parquet(C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'k', 'sc'])
    a = pd.read_parquet(C + rf'\app_{cc}.parquet', columns=['did', 'key'])
    m = a.merge(d, on='did')
    m = m[(m.pub >= 20150101) & m.sc.notna()]
    n3 = m.groupby('key').size()
    big = n3[n3 >= 30].index
    mb = m[m.key.isin(big)].copy()
    r = {'rows': len(mb), 'firms': len(big), 'H_K': float(H(mb.k.value_counts().values)),
         'H_K_given_firm': condH(mb, ['key']), 'H_K_given_sc': condH(mb, ['sc']), 'H_K_given_firm_sc': condH(mb, ['key', 'sc'])}
    pur = mb.groupby('key').k.agg(lambda x: x.value_counts().iloc[0] / len(x))
    r['firm_modal_share_median'] = float(pur.median()); r['firm_modal_share_p25'] = float(pur.quantile(.25))
    # 시간 불변성: 같은 기업·같은 서브클래스의 라벨이 2015-19 대 2020-24 사이에 같은 비율 vs 서로 다른 기업(같은 서브클래스) 기준
    mb['per'] = np.where(mb.pub < 20200101, 0, 1)
    md = mb.groupby(['key', 'sc', 'per']).k.agg(lambda x: x.value_counts().index[0]).unstack('per').dropna()
    r['same_firm_sc_modal_label_early_eq_late'] = float((md[0] == md[1]).mean()); r['n_firm_sc_both'] = int(len(md))
    # 같은 기업 내 다른 서브클래스끼리 최빈 라벨 일치(기업 결정형이면 높음)
    fm = mb.groupby(['key', 'sc']).k.agg(lambda x: x.value_counts().index[0]).reset_index()
    fmod = mb.groupby('key').k.agg(lambda x: x.value_counts().index[0])
    fm['eq_firm_modal'] = fm.k.values == fmod.reindex(fm.key).values
    scmod = mb.groupby('sc').k.agg(lambda x: x.value_counts().index[0])
    fm['eq_sc_modal'] = fm.k.values == scmod.reindex(fm.sc).values
    r['firm_sc_cell_modal_eq_firm_modal'] = float(fm.eq_firm_modal.mean()); r['firm_sc_cell_modal_eq_sc_modal'] = float(fm.eq_sc_modal.mean())
    res[cc] = r
    print(cc, json.dumps(r))
json.dump(res, open(W + r'\verify_0\v5_out.json', 'w'), indent=1)
