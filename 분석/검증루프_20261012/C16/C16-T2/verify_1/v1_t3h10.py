import sys, json, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
p=sys.argv[1]
df=pd.read_parquet(p, columns=['ksic','mg','cc','year','n'])
df['cell']=df.ksic+'|'+df.mg
w=df[df.year.between(2019,2021)].groupby('cell').n.sum()
elig=w[w>=int(sys.argv[2])].index
e=pd.Series(elig)
mg=e.str.split('|').str[1]
new=mg.str.match(r'^(H10[A-Z]|B64U|G06V|G06Q)')
h10=mg.str.match(r'^H10[A-Z]')
kr=df[(df.cc=='KR')&df.year.between(2019,2021)]
krh10=kr[kr.mg.str.match(r'^H10[A-Z]')].n.sum()
krall=kr.n.sum()
# docs in H10 cells published <=2021 per country
byc=df[(df.year<=2021)&df.mg.str.match(r'^H10[A-Z]')].groupby('cc').n.sum().to_dict()
out=dict(n_elig_2021=int(len(elig)), n_elig_H10=int(h10.sum()), n_elig_B64U=int(mg.str.match(r'^B64U').sum()),
  kr_docs_H10_2019_21=float(krh10), kr_docs_all_2019_21=float(krall), H10_docs_pub_le2021_by_cc={k:float(v) for k,v in byc.items()},
  h10_mg_examples=sorted(set(mg[h10]))[:15])
print(json.dumps(out,ensure_ascii=False,indent=1))
