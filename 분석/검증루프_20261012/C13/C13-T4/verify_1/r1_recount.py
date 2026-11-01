# verify_1 r1: X1 P0(a) 반박용 독립 재계산 (pandas, 쌍 첫 공동등록일 방식; 작업 캐시·lib13·s2·v1 코드 미사용)
# 등록일 < 2025-01-01 행만 SQL 단계에서 읽는다. sqlite mode=ro.
import sqlite3, json, sys
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
REG = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/수집파이프라인/registry_full.sqlite'
OUT = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C13/C13-T4/verify_1/r1_out.json'
con = sqlite3.connect('file:' + REG + '?mode=ro', uri=True)
o = pd.read_sql("""select a.rgst_no, a.applicantCd cd, p.ksic, cast(p.rgstDate as integer) rd
  from applicant a join patent p on p.rgst_no=a.rgst_no
  where p.rgstDate < '20250101' and a.applicantCd is not null and a.applicantCd<>''""", con)
assert o.rd.max() < 20250101
o['k'] = o.cd.str[0]
allp = o.drop_duplicates(['rgst_no', 'cd'])
org = allp[allp.k.isin(list('1235'))].copy()
# 쌍 첫 공동등록일 (조직 쌍, 무방향 키로 저장 후 양방향 전개)
pm = org.merge(org[['rgst_no', 'cd']], on='rgst_no', suffixes=('', '_g'))
pm = pm[pm.cd != pm.cd_g]
first = pm.groupby(['cd', 'cd_g']).rd.min().rename('first').reset_index()
res = {'n_rows_lt2025': int(len(o)), 'n_patents_lt2025': int(o.rgst_no.nunique()), 'max_rd': int(o.rd.max())}
rows = []
for t0 in range(2014, 2025):
    end = t0 * 10000 + 1231
    h = org[org.rd <= end]
    cnt = h.groupby('cd').rgst_no.nunique()
    risk_set = cnt[(cnt >= 3) & (cnt.index.str[0] == '1')].index
    risk_any1 = cnt[(cnt >= 1) & (cnt.index.str[0] == '1')].index   # 확장 위험집합(≥1건 국내법인)
    risk_org = cnt.index                                             # 확장(모든 조직)
    g = h.groupby(['cd', 'ksic']).size().rename('n').reset_index()
    g = g.sort_values(['cd', 'n', 'ksic'], ascending=[True, False, True]).drop_duplicates('cd')
    modal = g.set_index('cd').ksic
    r = {'t0': t0, 'risk': int(len(risk_set)), 'risk_any1': int(len(risk_any1)), 'risk_org': int(len(risk_org))}
    if t0 <= 2023:
        lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 822
        nw = first[(first['first'] >= lo) & (first['first'] <= hi)].copy()
        nw['mf'] = nw.cd.map(modal); nw['mg'] = nw.cd_g.map(modal)
        st = nw[nw.mg.notna() & nw.mf.notna() & (nw.mg != nw.mf)]
        r['pos_strict'] = int(st[st.cd.isin(risk_set)].cd.nunique())
        r['pos_loose'] = int(nw[nw.cd.isin(risk_set)].cd.nunique())
        r['pos_strict_pairs'] = int(len(st[st.cd.isin(risk_set)]))
        r['pos_loose_any1'] = int(nw[nw.cd.isin(risk_any1)].cd.nunique())
        r['pos_loose_allorg_newfirms'] = int(nw.cd.nunique())   # 이력 무관, 창에서 새 관계 맺은 조직 전체(위험집합 제약 없음)
        w = org[(org.rd >= lo) & (org.rd <= hi)]
        nmo = w.groupby('rgst_no').cd.nunique()
        r['win_multi_patents'] = int((nmo >= 2).sum())
        r['win_patents_org'] = int(w.rgst_no.nunique())
        r['win_patents_all'] = int(allp[(allp.rd >= lo) & (allp.rd <= hi)].rgst_no.nunique())
        r['rate_strict'] = r['pos_strict'] / r['risk']; r['rate_loose'] = r['pos_loose'] / r['risk']
        r['rate_loose_any1'] = r['pos_loose_any1'] / r['risk_any1']
        r['rate_allorg'] = r['pos_loose_allorg_newfirms'] / r['risk_org']
    rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows)
dev = df[df.t0 <= 2023]; L = df[df.t0 == 2024].iloc[0]
b2 = dev[dev.t0.isin([2022, 2023])]
res.update({
    'by_origin': rows,
    'dev_sum_strict': int(dev.pos_strict.sum()),
    'lock_risk': int(L.risk),
    'base_strict': float(b2.rate_strict.mean()),
    'lock_expected_strict': float(L.risk * b2.rate_strict.mean()),
    'lock_expected_loose': float(L.risk * b2.rate_loose.mean()),
    'lock_expected_strict_maxrate': float(L.risk * dev.rate_strict.max()),
    'lock_expected_loose_maxrate': float(L.risk * dev.rate_loose.max()),
    'lock_expected_loose_any1': float(L.risk_any1 * b2.rate_loose_any1.mean()),
    'lock_expected_allorg_newfirms': float(L.risk_org * b2.rate_allorg.mean()),
    'max_dev_allorg_newfirms': int(dev.pos_loose_allorg_newfirms.max()),
})
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print({k: v for k, v in res.items() if k != 'by_origin'})
