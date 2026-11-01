# C16-T3 s6: 공동출원(조직 ≥2) 비율 상승 분해 — 조직 문서 조건부 비율, 같은 그룹 의심(키 앞 2글자 같음)·대학 포함 비중
# 사용: python -I s6_joint.py <meta_glob> <cache_dir>
import sys, os, re, json
import pandas as pd, numpy as np, duckdb
M, C = sys.argv[1].replace('\\', '/'), sys.argv[2]
sys.stdout.reconfigure(encoding='utf-8')
d = pd.read_parquet(os.path.join(C, 'cn_docs.parquet'), columns=['appl_no', 'py', 'pub', 'n_orgkey', 'foreign', 'ksic'])
d = d[d.pub.notna() & d.py.between(2015, 2024)]
mu = d[d.n_orgkey >= 2][['appl_no', 'py', 'foreign']]
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
con.register('s', mu[['appl_no']])
ap = con.execute(f"select appl_no, any_value(applicant) applicant from read_parquet('{M}') where cc='CN' and appl_no in (select appl_no from s) group by appl_no").fetchdf()
mu = mu.merge(ap, on='appl_no')
u = pd.read_parquet(os.path.join(C, 'cn_names.parquet')).set_index('nm')


def feat(s):
    nms = [x.strip() for x in re.split(r'[|;]', s) if x.strip()]
    ks = [u.key.get(x) for x in nms if u.orgk.get(x, False)]
    ks = list(dict.fromkeys(ks))
    same = any(a[:2] == b[:2] for i, a in enumerate(ks) for b in ks[i + 1:])
    univ = any(re.search(r'大学|学院', x) for x in nms)
    grid = any(re.search(r'电网|供电|电力', x) for x in nms)
    return same, univ, grid


f = mu.applicant.map(feat)
mu['same2'] = [x[0] for x in f]; mu['univ'] = [x[1] for x in f]; mu['grid'] = [x[2] for x in f]
dd = d[~d.foreign]
g = dd.groupby('py').agg(n=('appl_no', 'size'), multi=('n_orgkey', lambda s: (s >= 2).mean()),
                         multi_given_org=('n_orgkey', lambda s: (s >= 2).sum() / max((s >= 1).sum(), 1)))
h = mu[~mu.foreign].groupby('py')[['same2', 'univ', 'grid']].mean()
g = g.join(h)
g['multi_excl_same2'] = g.multi * (1 - g.same2)
g['multi_excl_grid'] = g.multi * (1 - g.grid)
print(g.round(4).to_string())
g.to_csv(os.path.join(C, 's6_joint.csv'), encoding='utf-8-sig')
