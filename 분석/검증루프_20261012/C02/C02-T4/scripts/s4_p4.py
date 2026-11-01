# s4: P4 — 인정 사유 owner 이전의 월별 '건수'(2015-01~2026-08). 잠금 창 포함이지만 월별 집계만 만들고
#     기업·특허 수준 레이블은 만들거나 저장하지 않는다(추가 노출로 공개).
import re, sys, json, time, sqlite3
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
T0 = time.time()
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C02\C02-T4'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
CORP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|코포레이션|인코포레이티드|리미티드|컴퍼니|co\.|ltd|inc\.|corp|gmbh|llc", re.I)
ACC = ('양도', '일부양도', '지분양도', '신탁계약에 의한 양도')
o = pd.read_sql("select o.rgst_no, o.ownerName nm, o.ownerRgstCsDate d, o.ownerRgstCsName cs, o.ownerRgstCsReason rs, p.rgstDate rg "
                "from owner o left join patent p on o.rgst_no=p.rgst_no where o.ownerRgstCsDate >= '20150101'", con)
o['nm'] = o.nm.fillna('')
o['acc'] = o.cs.isin(ACC)
o['m'] = o.d.str[:4] + '-' + o.d.str[4:6]
a = o[o.acc].copy()
a['nn'] = a.nm.map(norm)
a['corp'] = a.nm.map(lambda s: bool(CORP.search(s)))
g = a.groupby(['nn', 'd']).size()
a['bulk'] = a.set_index(['nn', 'd']).index.map(g).values >= 10
a['stock'] = a.rg <= '20241231'
a['dup_pd'] = a.duplicated(['rgst_no', 'd'])
mon = pd.DataFrame({'all': a.groupby('m').size(), 'no_bulk': a[~a.bulk].groupby('m').size(),
                    'corp': a[a.corp].groupby('m').size(), 'stock_le2024': a[a.stock].groupby('m').size(),
                    'patent_date_events': a[~a.dup_pd].groupby('m').size()}).fillna(0).astype(int)
mon.index.name = 'month'
mon.to_csv(W + r'\cache\p4_monthly.csv', encoding='utf-8-sig')
def sm(col, y, m1=1, m2=7):
    return int(mon.loc[[f'{y}-{m:02d}' for m in range(m1, m2 + 1) if f'{y}-{m:02d}' in mon.index], col].sum())
res = {}
for col in mon.columns:
    med_a = float(np.median([mon.loc[f'2026-{m:02d}', col] for m in (5, 6, 7)]))
    med_b = float(np.median([mon.loc[f'2026-{m:02d}', col] for m in (1, 2, 3, 4)]))
    s26, s25 = sm(col, 2026), sm(col, 2025)
    base = np.mean([sm(col, y) for y in (2022, 2023, 2024)])
    res[col] = dict(P4a_med_2026_05_07=med_a, P4a_med_2026_01_04=med_b, P4a_ratio=round(med_a / med_b, 4), P4a_pass=bool(med_a >= 0.7 * med_b),
                    P4b_sum_2026_01_07=s26, P4b_sum_2025_01_07=s25, P4b_ratio=round(s26 / s25, 4), P4b_pass=bool(s26 >= 0.7 * s25),
                    P4c_mean_2022_24_01_07=round(base, 1), P4c_ratio=round(s26 / base, 4), P4c_pass=bool(s26 >= 0.7 * base),
                    jan_jul_by_year={y: sm(col, y) for y in range(2015, 2027)})
# 추세 기울기(2015-01~2024-12 개발 구간 월별, 로그 선형)
dv = mon.loc[(mon.index >= '2015-01') & (mon.index <= '2024-12'), 'all']
x = np.arange(len(dv)); b = np.polyfit(x, np.log(dv.values + 1), 1)[0]
res['dev_trend_log_slope_per_year'] = round(float(b * 12), 4)
# 정정·말소 비율
yr = o.d.str[:4]
res['owner_rows_reason_mallso_jeongjeong'] = {y: int(((o.rs.fillna('').str.contains('말소|정정|경정')) | (o.cs.fillna('').str.contains('말소|정정|경정')))[yr == y].sum()) for y in map(str, range(2015, 2027))}
r = pd.read_sql("select rgstCsDate d, rgstCsName nm from right where rgstCsDate >= '20150101' and (rgstCsName like '%말소%' or rgstCsName like '%경정%' or rgstCsName like '%정정%')", con)
r['y'] = r.d.str[:4]
res['right_mallso_jeongjeong_by_year'] = r.groupby(['y', 'nm']).size().unstack(fill_value=0).to_dict(orient='index')
res['right_direct_correction_by_year'] = r[r.nm.str.contains('직권경정|정정등록|직권말소')].groupby('y').size().to_dict()
# 되돌림 근사: 같은 특허에서 이전 후 365일 안에 직전 권리자 정규화 상호로 다시 인정 사유 이전(연도별 비율)
a2 = a.sort_values(['rgst_no', 'd'])
prev_nn = a2.groupby('rgst_no').nn.shift(2); prev_d = a2.groupby('rgst_no').d.shift(1)
rev = (a2.nn == prev_nn) & ((pd.to_datetime(a2.d) - pd.to_datetime(prev_d)).dt.days <= 365)
res['reversal_share_by_year'] = {y: round(float(rev[a2.d.str[:4] == y].mean()), 4) for y in map(str, range(2015, 2027))}
# 재고 동결: 창 안(원점 말 이후) 등록 특허의 이전 비중
ws = {}
for t0 in list(range(2012, 2023)) + [2024]:
    if t0 + 1 < 2015: continue
    w = a[(a.d >= f'{t0 + 1}0101') & (a.d <= f'{t0 + 2}0731')]
    ws[t0] = dict(n=len(w), share_rg_in_window=round(float((w.rg > f'{t0}1231').mean()), 4))
res['stock_freeze_share_rg_in_window'] = ws
res['patent_rg_by_year'] = dict(con.execute("select substr(rgstDate,1,4), count(*) from patent where rgstDate >= '2015' group by 1").fetchall())
res['owner_max_date'] = con.execute("select max(ownerRgstCsDate) from owner").fetchone()[0]
res['bulk_share_by_year'] = {y: round(float(a.bulk[a.d.str[:4] == y].mean()), 4) for y in map(str, range(2015, 2027))}
res['elapsed'] = round(time.time() - T0, 1)
json.dump(res, open(W + r'\cache\p4_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for k in ['all', 'no_bulk', 'corp', 'stock_le2024', 'patent_date_events']:
    print(k, {kk: v for kk, v in res[k].items()})
for k in res:
    if k not in mon.columns: print(k, res[k])
print(mon.loc['2024-01':].T.to_string())
