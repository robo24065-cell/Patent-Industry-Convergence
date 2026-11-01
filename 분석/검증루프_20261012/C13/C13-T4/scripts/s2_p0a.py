# C13-T4 s2: 후보 X1 P0(a) 규모 집계 (PREREG §3·§4). 등록일 < 2025-01-01 행만 읽는다(assert).
import sqlite3, json, sys, re
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
REG = B + r'\수집파이프라인\registry_full.sqlite'
W = B + r'\분석\검증루프_20261012\C13\C13-T4'
CUT = 20250101
con = sqlite3.connect('file:' + REG.replace('\\', '/') + '?mode=ro', uri=True)
p = pd.read_sql('select rgst_no, ksic, rgstDate from patent', con)
p['rd'] = pd.to_numeric(p.rgstDate.astype(str).str.replace(r'\D', '', regex=True).str[:8], errors='coerce')
n_all = len(p); n_na = int(p.rd.isna().sum())
p = p[p.rd < CUT].copy()          # 2025 이후 등록은 버린다
assert p.rd.max() < CUT
a = pd.read_sql('select rgst_no, applicantCd from applicant', con)
a = a[a.rgst_no.isin(p.rgst_no)]
a['d'] = a.applicantCd.astype(str).str[0]
org = a[a.d.isin(['1', '2', '3', '5'])][['rgst_no', 'applicantCd', 'd']].drop_duplicates()
org = org.merge(p[['rgst_no', 'ksic', 'rd']], on='rgst_no')
log = {'n_patent_all': n_all, 'n_rgstdate_na': n_na, 'n_patent_lt2025': len(p), 'max_rd': int(p.rd.max()),
       'org_rows': len(org)}
# 공동(조직 ≥2) 특허
nor = org.groupby('rgst_no').applicantCd.nunique()
org['multi'] = org.rgst_no.map(nor) >= 2
rows = []
pairs_all = []
for t0 in range(2014, 2025):
    end = t0 * 10000 + 1231
    h = org[org.rd <= end]
    cnt = h.groupby('applicantCd').rgst_no.nunique()
    elig = set(cnt[(cnt >= 3) & (cnt.index.str[0] == '1')].index)
    # as-of 최빈 KSIC(동률이면 작은 코드)
    m = h.groupby(['applicantCd', 'ksic']).size().rename('n').reset_index().sort_values(['applicantCd', 'n', 'ksic'], ascending=[True, False, True])
    modal = m.drop_duplicates('applicantCd').set_index('applicantCd').ksic
    r = {'t0': t0, 'risk': len(elig)}
    if t0 <= 2023:
        hm = h[h.multi]
        ex = hm.merge(hm, on='rgst_no')
        ex = ex[ex.applicantCd_x < ex.applicantCd_y]
        exist = set(zip(ex.applicantCd_x, ex.applicantCd_y))
        lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 822
        w = org[(org.rd >= lo) & (org.rd <= hi) & org.multi]
        e = w.merge(w[['rgst_no', 'applicantCd']], on='rgst_no', suffixes=('', '_g'))
        e = e[(e.applicantCd != e.applicantCd_g) & e.applicantCd.isin(elig)]
        key = [tuple(sorted(x)) for x in zip(e.applicantCd, e.applicantCd_g)]
        e['new'] = [k not in exist for k in key]
        e = e[e.new]
        e['mf'] = e.applicantCd.map(modal); e['mg'] = e.applicantCd_g.map(modal)
        loose = e
        strict = e[e.mg.notna() & (e.mg != e.mf)]
        r.update({'win_patents': int(org[(org.rd >= lo) & (org.rd <= hi)].rgst_no.nunique()),
                  'win_multi_patents': int(w.rgst_no.nunique()),
                  'pos_loose_firms': int(loose.applicantCd.nunique()),
                  'pos_strict_firms': int(strict.applicantCd.nunique()),
                  'pos_strict_pairs': int(strict[['applicantCd', 'applicantCd_g']].drop_duplicates().shape[0]),
                  'pos_strict_partner_dom_corp': float((strict.drop_duplicates(['applicantCd', 'applicantCd_g']).applicantCd_g.str[0] == '1').mean()) if len(strict) else None})
        r['rate_loose'] = r['pos_loose_firms'] / r['risk']; r['rate_strict'] = r['pos_strict_firms'] / r['risk']
        pairs_all.append(strict[['applicantCd', 'applicantCd_g']].drop_duplicates().assign(t0=t0))
    rows.append(r)
    print(r, flush=True)
df = pd.DataFrame(rows)
dev = df[df.t0 <= 2023]
lock = df[df.t0 == 2024].iloc[0]
base_s = dev[dev.t0.isin([2022, 2023])].rate_strict.mean()
base_l = dev[dev.t0.isin([2022, 2023])].rate_loose.mean()
pa = pd.concat(pairs_all)
res = {**log,
       'dev_pos_strict_sum_firm_origin': int(dev.pos_strict_firms.sum()),
       'dev_unique_pairs_strict': int(pa[['applicantCd', 'applicantCd_g']].drop_duplicates().shape[0]),
       'dev_pos_loose_sum_firm_origin': int(dev.pos_loose_firms.sum()),
       'lock_risk_t0_2024': int(lock.risk),
       'base_strict_2022_23': base_s, 'base_loose_2022_23': base_l,
       'lock_expected_strict': float(lock.risk * base_s), 'lock_expected_loose_upper': float(lock.risk * base_l)}
res['P0a_pass'] = bool(res['dev_pos_strict_sum_firm_origin'] >= 1000 and res['lock_expected_strict'] >= 1000)
df.to_csv(W + r'\cache\s2_p0a_by_origin.csv', index=False, encoding='utf-8-sig')
json.dump(res, open(W + r'\cache\s2_p0a.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
