# verify_2: 상한 배정 — 제공 라벨로 학습한 IPC 서브그룹 최빈(표본 내, 비공개 '신탁' 상한)으로 사건 재정의 시 자카드
import sys, time; sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
import numpy as np, pandas as pd
from lib_cd import load, build, KSIC50, DEV, C
T0=time.time()
own, pat, app, docs, orig = load()
lab = pd.read_parquet(C + r'\kr_publab_src.parquet').set_index('appl_no')
pk = pd.concat([docs[['appl_no','ksic']], pat[['appl_no','ksic']]]).drop_duplicates('appl_no').set_index('appl_no').ksic
sg = lab.sg.reindex(pk.index)
df = pd.DataFrame({'key': sg, 'k': pk}).dropna()
md = df.groupby('key').k.agg(lambda s: s.value_counts().idxmax())
L = sg.map(md)
L = L.fillna(pk)  # sg 없는 문서는 제공 라벨 유지(상한에 유리)
P = pd.read_parquet(C + r'\dev_pos.parquet')
print('pat agree', round((pat.appl_no.map(L) == pat.ksic).mean(), 4))
tot=[0,0,0]
for t0 in DEV:
    Ep = set(zip(P[P.t0 == t0].cd, P[P.t0 == t0].k))
    st, pa, _, _ = build(t0, own, docs, app, orig, docs.appl_no.map(L), dict(zip(pat.rgst_no, pat.appl_no.map(L))))
    Ea = set(zip(pa.cd, pa.k)); tot[0]+=len(Ea); tot[1]+=len(Ep); tot[2]+=len(Ea&Ep)
print('ORACLE-sg ev', tot, 'jac', round(tot[2]/(tot[0]+tot[1]-tot[2]),4), 'rec', round(tot[2]/tot[1],4), round(time.time()-T0))
