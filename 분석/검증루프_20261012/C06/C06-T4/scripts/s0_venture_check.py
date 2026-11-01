# s0: 벤처기업명단 재내려받기본 vs 로컬 parquet 동일성, 명단 전체(단위 연결 없음)의 분포·결측·중복·월별 시작일(완결성) 점검.
# 사용: python -I s0_venture_check.py <download_csv>
import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T4'
EXT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
src = sys.argv[1]
raw = open(src, 'rb').read()
for enc in ('utf-8-sig', 'cp949'):
    try:
        txt = raw.decode(enc); break
    except UnicodeDecodeError:
        continue
import io
d = pd.read_csv(io.StringIO(txt), dtype=str)
out = {'download_rows': len(d), 'download_cols': list(d.columns), 'encoding': enc}
rep = [c for c in d.columns if '대표' in c]
dn = d.drop(columns=rep)
loc = pd.read_parquet(EXT + r'\firm_venture_list_202602.parquet')
out['local_rows'] = len(loc); out['local_cols'] = list(loc.columns)
same_cols = list(dn.columns) == list(loc.columns)
out['same_cols'] = same_cols
if same_cols:
    a = dn.fillna('').astype(str).apply(lambda r: '\x1f'.join(r), axis=1)
    b = loc.fillna('').astype(str).apply(lambda r: '\x1f'.join(r), axis=1)
    out['rows_identical_in_order'] = bool((a.values == b.values).all()) if len(a) == len(b) else False
    out['multiset_diff_dl_minus_local'] = int(len(set(a) - set(b))); out['multiset_diff_local_minus_dl'] = int(len(set(b) - set(a)))
v = dn.copy()
cs, ce = [c for c in v.columns if '시작' in c][0], [c for c in v.columns if '종료' in c][0]
cn = [c for c in v.columns if '신규' in c][0]; ct = [c for c in v.columns if '유형' in c][0]
cr = '지역'; cname = '업체명'; cind = [c for c in v.columns if '11' in c][0]
v['s'] = pd.to_datetime(v[cs], errors='coerce'); v['e'] = pd.to_datetime(v[ce], errors='coerce')
out['null_rate'] = {c: round(float(v[c].isna().mean() + (v[c].astype(str).str.strip() == '').mean()), 4) for c in dn.columns}
out['start_min_max'] = [str(v.s.min().date()), str(v.s.max().date())]
out['end_min_max'] = [str(v.e.min().date()), str(v.e.max().date())]
dur = (v.e - v.s).dt.days
out['duration_days_quantiles'] = {str(q): float(dur.quantile(q)) for q in (0, .01, .5, .99, 1)}
out['new_reconf'] = v[cn].value_counts(dropna=False).to_dict()
out['type'] = v[ct].value_counts(dropna=False).to_dict()
out['region'] = v[cr].value_counts(dropna=False).to_dict()
out['confirm_org'] = v[[c for c in v.columns if '기관' in c][0]].value_counts(dropna=False).to_dict()
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
norm = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
v['nname'] = v[cname].map(norm)
out['dup_exact_rows'] = int(v.duplicated(subset=list(dn.columns)).sum())
out['dup_name_rows'] = int(v.duplicated('nname', keep=False).sum())
out['dup_name_addr_rows'] = int(v.duplicated(['nname', '주소'], keep=False).sum())
out['corp_form_share'] = round(float(v[cname].astype(str).str.contains(r'주식회사|㈜|\(주\)|유한').mean()), 4)
m = v.groupby([v.s.dt.to_period('M'), v[cn]]).size().unstack(fill_value=0)
m.index = m.index.astype(str)
m.to_csv(os.path.join(W, 'cache', 's0_venture_monthly_all.csv'), encoding='utf-8-sig')
# 유효 기업 수(명단 기준)를 특정일에: 시작 ≤ d ≤ 종료
for dd in ['2024-12-31', '2025-12-31', '2026-04-30']:
    t = pd.Timestamp(dd); out[f'valid_on_{dd}'] = int(((v.s <= t) & (v.e >= t)).sum())
out['in_list_industry_top10'] = v[cind].value_counts().head(10).to_dict()
v[['nname', 's', 'e', cn, ct, cr, '주소', cind]].rename(columns={cn: 'newre', ct: 'vtype', cr: 'region', '주소': 'addr', cind: 'ind11'}).to_parquet(os.path.join(W, 'cache', 'venture_clean.parquet'), index=False)
json.dump(out, open(os.path.join(W, 'cache', 's0_venture_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: out[k] for k in out if k not in ('region', 'in_list_industry_top10', 'null_rate')}, ensure_ascii=False, indent=1, default=str))
print(m.to_string())
