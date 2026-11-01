# C16-T3 s7: KR 출원 분모 1,087,511의 접두 분해(10=특허, 20=실용신안) + NTIS 연결 분자·등록 분해 → kr_prefix.csv
# 사용: python -I s7_kr_prefix.py <meta_glob> <kr_patents.parquet> <ntis_patent_flag.parquet> <registry_full.sqlite> <out_csv> <cache_dir>
import sys, os, json, sqlite3
import duckdb, pandas as pd, numpy as np
M, KRP, NF, REG, OUT, C = sys.argv[1].replace('\\', '/'), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
rows = []
m = con.execute(f"select appl_no, rgst_no, appl_year from read_parquet('{M}') where cc='KR'").fetchdf()
m['p'] = m.appl_no.str.replace(r'\D', '', regex=True).str[:2]
m['rp'] = m.rgst_no.fillna('').str.replace(r'\D', '', regex=True).str[:2]
for p, g in m.groupby('p'):
    rows.append(dict(source='meta KR 행(표 2 KR, C-COUNTRY-SHARE 분모)', prefix=p, n=len(g), n_unique=g.appl_no.nunique(),
                     n_registered=int((g.rp != '').sum()), share=len(g) / len(m)))
rows.append(dict(source='meta KR 행(표 2 KR, C-COUNTRY-SHARE 분모)', prefix='ALL', n=len(m), n_unique=m.appl_no.nunique(), n_registered=int((m.rp != '').sum()), share=1.0))
# 등록 접두 대 출원 접두 교차
xt = pd.crosstab(m.p, m.rp.replace('', 'none'))
# 출원연도별 실용신안 비중
yy = m.groupby('appl_year').p.apply(lambda s: (s == '20').mean())
# kr_patents.parquet
k = pd.read_parquet(KRP)
cand = [c for c in k.columns if '출원번호' in c or c.lower() in ('appl_no', 'application_no')]
kc = cand[0]
kp = k[kc].astype(str).str.replace(r'\D', '', regex=True).str[:2]
for p, n in kp.value_counts().sort_index().items():
    rows.append(dict(source=f'작업데이터 kr_patents.parquet({kc})', prefix=p, n=int(n), n_unique=int(k[kp == p][kc].nunique()), share=n / len(k)))
# NTIS 연결 플래그
f = pd.read_parquet(NF)
f['p'] = f.appl_no.astype(str).str.replace(r'\D', '', regex=True).str[:2]
f['rp'] = f.rgst_no.fillna('').astype(str).str.replace(r'\D', '', regex=True).str[:2]
gcol = 'gov_rnd'
f[gcol] = f[gcol].fillna(0).astype(int)
for p, g in f.groupby('p'):
    rows.append(dict(source='ntis_patent_flag(LINK-NTIS 9.34% 분자·분모)', prefix=p, n=len(g), n_linked=int(g[gcol].sum()), link_rate=g[gcol].mean(),
                     n_registered=int((g.rp != '').sum()), n_linked_registered=int(g.loc[g.rp != '', gcol].sum()),
                     link_rate_registered=g.loc[g.rp != '', gcol].mean()))
rows.append(dict(source='ntis_patent_flag(LINK-NTIS 9.34% 분자·분모)', prefix='ALL', n=len(f), n_linked=int(f[gcol].sum()), link_rate=f[gcol].mean(),
                 n_registered=int((f.rp != '').sum()), n_linked_registered=int(f.loc[f.rp != '', gcol].sum()), link_rate_registered=f.loc[f.rp != '', gcol].mean()))
# 등록 접두 기준 분해(등록 10.95%)
for rp, g in f[f.rp != ''].groupby('rp'):
    rows.append(dict(source='ntis_patent_flag 등록번호 접두(등록 10.95% 분자·분모)', prefix=rp, n=len(g), n_linked=int(g[gcol].sum()), link_rate=g[gcol].mean()))
# registry_full(mode=ro)
uri = 'file:' + REG.replace('\\', '/') + '?mode=ro'
sq = sqlite3.connect(uri, uri=True)
for rp, n in sq.execute("select substr(rgst_no,1,2), count(*) from patent group by 1").fetchall():
    rows.append(dict(source='registry_full.patent(mode=ro)', prefix=rp, n=n))
for rp, n in sq.execute("select substr(applNo,1,2), count(*) from patent group by 1").fetchall():
    rows.append(dict(source='registry_full.patent 출원번호 접두', prefix=rp, n=n))
sq.close()
R = pd.DataFrame(rows)
R.to_csv(OUT, index=False, encoding='utf-8-sig', float_format='%.5f')
pd.set_option('display.width', 250)
print(R.to_string())
print(xt)
print('UM share by appl_year', yy.round(4).to_dict())
json.dump(dict(crosstab=xt.to_dict(), um_by_year={int(a): float(b) for a, b in yy.items() if pd.notna(a)}, kr_patents_cols=list(k.columns)),
          open(os.path.join(C, 's7_extra.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
