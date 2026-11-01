import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
d=pd.read_csv(sys.argv[1])
c=d.columns
print(list(c))
print('kind', d[c[2]].value_counts(dropna=False).to_dict())
print('grade', d[c[4]].value_counts(dropna=False).to_dict())
bc=d[d[c[4]].isin(['B','C'])]
print('BC', len(bc), bc[c[7]].value_counts(dropna=False).to_dict())
print('BC trackdoc', bc[c[6]].astype(str).str[:2].value_counts(dropna=False).to_dict())
print('all trackdoc', d[c[6]].astype(str).str[:2].value_counts(dropna=False).to_dict())
print(d[c[0]].value_counts().to_dict())
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',120)
for i,r in d.iterrows():
    print('|'.join(str(x) for x in r.values))
