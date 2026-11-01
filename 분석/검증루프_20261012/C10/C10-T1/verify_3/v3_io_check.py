import sys, pandas as pd, json
xl = sys.argv[1]
g = pd.read_excel(xl, sheet_name='접속산업연관표 이용안내', header=None)
lines = [' | '.join(str(v) for v in r if pd.notna(v)) for r in g.values.tolist()]
lines = [l for l in lines if l.strip()]
z = pd.read_excel(xl, sheet_name='10-A표_총거래표(생산자)', header=None, nrows=8)
hdr = z.iloc[4, 2:].astype(str).str.strip()
n3 = sum(1 for c in hdr if c.isdigit() and len(c)==3)
open(sys.argv[2],"w",encoding="utf-8").write(json.dumps(dict(guide=lines[:60], top_rows=[' | '.join(str(v) for v in r if pd.notna(v))[:200] for r in z.values.tolist()[:5]], n_3digit_cols_2010=n3), ensure_ascii=False, indent=1))
