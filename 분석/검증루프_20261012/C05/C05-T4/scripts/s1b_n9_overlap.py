# s1b: ② N9 문서 겹침 수치를 10자리 차단 후 개발 양성으로 다시 계산(개발 원점만).
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
fl = pd.read_parquet(os.path.join(EXT, 'ntis_patent_flag.parquet'), columns=['cc', 'appl_no', 'gov_rnd', 'min_project_year', 'max_project_year'])
print(fl.cc.value_counts().head())
fl = fl[fl.cc.astype(str).str.upper().isin(['KR'])] if 'KR' in set(fl.cc.astype(str).str.upper()) else fl
ca = load_ca(); nm2b6, _ = name_map(ca)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
pos = pd.read_parquet(os.path.join(C, 'dev_pos.parquet'))
P = []
for s0 in DEV:
    ref = REF[s0]; end = add_months(ref, 6)
    w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end) & docs.b6.notna()].copy(); w['key'] = w.b6 + '|' + w.nname
    P.append(w[w.key.isin(set(pos[pos.s0 == s0].key))][['appl_no', 'key', 'ay']].assign(s0=s0))
P = pd.concat(P).merge(fl.drop(columns='cc').drop_duplicates('appl_no'), on='appl_no', how='left')
P['g'] = P.gov_rnd.fillna(False).astype(bool)
res = dict(pos_docs=int(len(P)), gov_share=round(float(P.g.mean()), 4),
           firm_any_gov_share=round(float(P.groupby(['s0', 'key']).g.any().mean()), 4), n_firms=int(P.groupby(['s0', 'key']).ngroups))
print(res)
json.dump(res, open(os.path.join(C, 's1b_n9.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
