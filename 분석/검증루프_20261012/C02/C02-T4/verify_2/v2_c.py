# verify_2: 민감도 — 제공 라벨에 무작위 잡음(문서 단위 p 비율 재배정) 시 자카드, IPC 최빈(제공 라벨 학습) 상한 일치율
import sys, time; sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
import numpy as np, pandas as pd
from lib_cd import load, build, KSIC50, DEV, C
T0=time.time()
own, pat, app, docs, orig = load()
lab = pd.read_parquet(C + r'\kr_publab_src.parquet').set_index('appl_no')
# 문서 단위 제공 라벨(appl_no 기준 최빈)
pk = pd.concat([docs[['appl_no','ksic']], pat[['appl_no','ksic']]]).drop_duplicates()
print('appl_no with >1 prov ksic', pk.appl_no.duplicated().sum())
pk = pk.drop_duplicates('appl_no').set_index('appl_no').ksic
# IPC 최빈 상한(제공 라벨로 학습, 표본 내): sg, sc
sg = lab.sg.reindex(pk.index)
for nm, key in [('sg', sg), ('sc', sg.str[:4])]:
    df = pd.DataFrame({'key': key, 'k': pk}).dropna()
    md = df.groupby('key').k.agg(lambda s: s.value_counts().idxmax())
    print('in-sample IPC-mode agreement with provided', nm, round((df.key.map(md) == df.k).mean(), 4), 'n', len(df))
P = pd.read_parquet(C + r'\dev_pos.parquet')
rng = np.random.default_rng(20261012)
ks = np.array(KSIC50)
for p in [0.05, 0.10, 0.25]:
    flip = rng.random(len(pk)) < p
    newk = pk.values.copy()
    newk[flip] = rng.choice(ks, flip.sum())
    L = pd.Series(newk, index=pk.index)
    agree = (pat.appl_no.map(L) == pat.ksic).mean()
    tot = [0, 0, 0]
    for t0 in DEV:
        Ep = set(zip(P[P.t0 == t0].cd, P[P.t0 == t0].k))
        st, pa, _, _ = build(t0, own, docs, app, orig, docs.appl_no.map(L), dict(zip(pat.rgst_no, pat.appl_no.map(L))))
        Ea = set(zip(pa.cd, pa.k))
        tot[0] += len(Ea); tot[1] += len(Ep); tot[2] += len(Ea & Ep)
    print('noise p', p, 'pat agree', round(agree, 4), 'ev_noise', tot[0], 'ev_prov', tot[1], 'inter', tot[2], 'jac', round(tot[2] / (tot[0] + tot[1] - tot[2]), 4), round(time.time() - T0), flush=True)
