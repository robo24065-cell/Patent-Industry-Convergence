# C16-T3 verify_0: 독립 재계산(duckdb 미사용, pyarrow 파일별 읽기 + pandas 벡터 연산)
# 사용: python -I v0_indep.py <meta_dir> <main_csv> <kr_prefix_csv> <out_json>
# (1) CN 조직 비중 연도 계열: C13 정의(공개연도, 조직 키≥1, 외국 포함, 결측=비조직), 출원인1 국내 비중(공개·출원연도), 출원연도 KSIC 2015 표준화
# (2) KR 접두 분해
import sys, os, glob, json, re
import pyarrow.parquet as pq, pandas as pd, numpy as np
MD, MAIN, KRP, OUT = sys.argv[1:5]
sys.stdout.reconfigure(encoding='utf-8')
cols = ['appl_no', 'appl_year', 'pub_date', 'cc', 'applicant', 'prio_cc', 'ksic']
parts = []; kr = []
for f in sorted(glob.glob(os.path.join(MD, '*.parquet'))):
    t = pq.read_table(f, columns=cols).to_pandas()
    t['cc'] = t.cc.astype(str)
    kr.append(t.loc[t.cc == 'KR', ['appl_no']])
    c = t[(t.cc == 'CN') & (t.appl_no.str.len() == 14)]
    c = c[c.appl_no.str.slice(4, 5).isin(['1', '8']) & (c.pub_date.fillna('') >= '20150101')]
    parts.append(c.drop(columns=['cc']))
cn = pd.concat(parts, ignore_index=True); kr = pd.concat(kr, ignore_index=True)
cn['ksic'] = cn.ksic.astype(str); cn['prio_cc'] = cn.prio_cc.astype(object)
# 출원 단위(이미 중복 0이지만 최소 공개일로 축약)
cn['pubi'] = pd.to_numeric(cn.pub_date.where(cn.pub_date.str.len() == 8), errors='coerce')
cn = cn.sort_values('pubi').drop_duplicates('appl_no', keep='first')
cn = cn[cn.pubi.notna() & (cn.pubi < 20250801)]
cn['py'] = (cn.pubi // 10000).astype(int)
# 이름 분해: 벡터 방식
s = cn.applicant.fillna('')
ex = s.str.split(r'[|;]').explode().str.strip()
ex = ex[(ex != '') & ~ex.str.lower().isin(['nan', 'none', '-', 'null'])]
up = ex.str.upper()
orgflag = up.str.contains(r'公司|大学|学院|研究|院|所|厂|集团|中心|局|会|株式会社|LTD|INC|CORP', regex=True)
# 키: 접미 제거 후 영숫자·한자·가나·한글만
SUF = ['股份有限公司', '有限责任公司', '有限公司', '股份公司', '株式会社', '有限会社', '주식회사', '유한책임회사', '유한회사', '합자회사', '합명회사',
       '농업회사법인', '영농조합법인', '(주)', '㈜', '（주）', '(유)', '(株)']
k = up
for x in SUF: k = k.str.replace(x, ' ', regex=False)
k = k.str.replace(r'\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?', ' ', regex=True)
k = k.str.replace(r'[^0-9A-Z一-鿿぀-ヿ가-힣]', '', regex=True)
ok = orgflag & (k.str.len() >= 2)
kk = pd.DataFrame({'idx': ex.index, 'k': k.where(ok)})
norg = kk.dropna().groupby('idx').k.nunique()
cn['n_orgkey'] = norg.reindex(cn.index).fillna(0).values
first = s.str.split(r'[|;]').str[0].str.strip()
fu = first.str.upper()
fk = fu
for x in SUF: fk = fk.str.replace(x, ' ', regex=False)
fk = fk.str.replace(r'\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?', ' ', regex=True)
fk = fk.str.replace(r'[^0-9A-Z一-鿿぀-ヿ가-힣]', '', regex=True)
cn['a1_miss'] = first == ''
cn['a1_org'] = fu.str.contains(r'公司|大学|学院|研究|院|所|厂|集团|中心|局|会|株式会社|LTD|INC|CORP', regex=True) & (fk.str.len() >= 2)
pc = cn.prio_cc.fillna('').astype(str).str.replace('None', '').str.replace('nan', '')
pfor = pc.str.split('|').apply(lambda L: any(x not in ('', 'CN', 'TW', 'HK', 'MO') for x in L))
fmark = first.str.contains(r'株式会社|有限会社|株式會社|合同会社|[぀-ヿ가-힣]', regex=True) | (first.str.match(r'^[\x00-\x7fÀ-ɏ\s]+$') & (first != ''))
cn['foreign'] = (cn.appl_no.str.slice(4, 5) == '8') | pfor | fmark
res = {}
for y in range(2015, 2025):
    g = cn[cn.py == y]; gd = g[~g.foreign & ~g.a1_miss]
    res[f'pub_{y}'] = dict(n=int(len(g)), c13_anyorg_all=float((g.n_orgkey >= 1).mean()), multi_all=float((g.n_orgkey >= 2).mean()),
                           app1org_dom=float(gd.a1_org.mean()), foreign_share=float(g.foreign.mean()))
cn['ay'] = pd.to_numeric(cn.appl_year, errors='coerce')
b = cn[(cn.ay == 2015) & ~cn.foreign & ~cn.a1_miss]; bw = b.ksic.value_counts(normalize=True)
for y in range(2015, 2024):
    gd = cn[(cn.ay == y) & ~cn.foreign & ~cn.a1_miss]
    sk = gd.groupby('ksic').a1_org.mean(); w = bw.reindex(sk.index).fillna(0)
    res[f'appl_{y}'] = dict(n=int((cn.ay == y).sum()), app1org_dom=float(gd.a1_org.mean()), app1org_dom_ksicstd=float((sk * w).sum() / w.sum()))
# 대조
M = pd.read_csv(MAIN, encoding='utf-8-sig')
diffs = []
for _, r in M.iterrows():
    v = res.get(f"{r.basis}_{int(r.year)}")
    if not v: continue
    for kname in v:
        if kname in M.columns:
            diffs.append(dict(key=f"{r.basis}_{int(r.year)}_{kname}", main=float(r[kname]), indep=float(v[kname]), diff=float(v[kname] - r[kname])))
D = pd.DataFrame(diffs)
kr['p'] = kr.appl_no.str.replace(r'\D', '', regex=True).str[:2]
krc = kr.p.value_counts().to_dict()
K = pd.read_csv(KRP, encoding='utf-8-sig')
km = K[K.source.str.startswith('meta KR')].set_index('prefix').n.to_dict()
out = dict(max_abs_diff_share=float(D[~D.key.str.endswith('_n')]['diff'].abs().max()), max_abs_diff_count=float(D[D.key.str.endswith('_n')]['diff'].abs().max()),
           n_compared=int(len(D)), kr_prefix_indep={k: int(v) for k, v in krc.items()}, kr_prefix_main={str(k): int(v) for k, v in km.items()},
           kr_match=bool(all(int(km.get(k, -1)) == v for k, v in krc.items())), res=res, worst=D.reindex(D['diff'].abs().sort_values(ascending=False).index).head(8).to_dict('records'))
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'res'}, ensure_ascii=False, indent=1))
