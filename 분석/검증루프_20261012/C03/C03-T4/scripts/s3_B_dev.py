# s3_B: C03-B 실시권자 × KSIC 첫 실시권 — 개발 원점 P1 집계. 2025-01-01 이후 시작 실시권은 월별 건수만.
import sys, re, json, time, sqlite3
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
import numpy as np, pandas as pd
from lib_cd import alias_at  # C02 별칭 사전(읽기 전용 import)
T0 = time.time()
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C03\C03-T4'
C2 = B + r'\분석\검증루프_20261012\C02\C02-T4\cache'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
log = {}
lic = []
for t, p, typ in [('elsLcnsList', 'elsLcns', 'excl'), ('noelsLcnsList', 'noelsLcns', 'nonexcl')]:
    log[f'{typ}_monthly_ge2024'] = con.execute(f"select substr({p}StartDate,1,6), count(*) from {t} where {p}PsnNm is not null and {p}StartDate>='20240101' group by 1 order by 1").fetchall()
    x = pd.read_sql(f"select rgst_no, {p}PsnNm nm, {p}StartDate d, {p}RgstCsNm cs from {t} where {p}PsnNm is not null and {p}StartDate < '20250101'", con)
    x['typ'] = typ; lic.append(x)
con.close()
lic = pd.concat(lic, ignore_index=True)
assert lic.d.max() < '20250101'
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
lic['nn'] = lic.nm.fillna('').map(lambda s: STRIP.sub('', str(s)).lower())
# 한 칸에 여러 실시권자가 들어간 경우 진단
log['lic_rows_lt2025'] = len(lic); log['lic_multi_name_share'] = round(float(lic.nm.str.contains(r'[,;/]|외 ', regex=True).mean()), 4)
pat = pd.read_parquet(C2 + r'\pat.parquet', columns=['rgst_no', 'ksic'])
app = pd.read_parquet(C2 + r'\app_reg.parquet', columns=['rgst_no', 'cd', 'nn', 'vis'])
docs = pd.read_parquet(C2 + r'\docs.parquet'); docs = docs[docs.cd.str[0] == '1']
own = pd.read_parquet(C2 + r'\own_lt2025.parquet', columns=['rgst_no', 'seq', 'nn', 'd', 'acc'])
assert own.d.max() < '20250101'
orig = app.groupby('rgst_no').cd.agg(set).to_dict()
lic['k'] = lic.rgst_no.map(dict(zip(pat.rgst_no, pat.ksic.astype(str))))
log['lic_label_missing'] = int(lic.k.isna().sum())
# 같은 날 같은 실시권자 ≥10건 = 일괄
g = lic.groupby(['nn', 'd']).size(); lic['bulk'] = lic.set_index(['nn', 'd']).index.map(g).values >= 10
rows, P = [], []
for t0 in range(2012, 2023):
    T = f'{t0}1231'; ws = f'{t0+1}0101'; we = f'{t0+2}0630'
    n2c, amb = alias_at(app, T)
    dv = docs[docs.vis <= T]; cnt = dv.groupby('cd').appl_no.nunique(); F = set(cnt[cnt >= 3].index)
    b1 = set(zip(dv.cd, dv.ksic.astype(str)))
    pr = own[(own.seq > 0) & (own.d <= T)]
    b2 = set(zip(pr.nn.map(n2c), pr.rgst_no.map(dict(zip(pat.rgst_no, pat.ksic.astype(str))))))
    lp = lic[lic.d <= T]; b3 = set(zip(lp.nn.map(n2c), lp.k))
    blocked = b1 | b2 | b3
    w = lic[(lic.d >= ws) & (lic.d <= we)].copy()
    w['cd'] = w.nn.map(n2c)
    st = dict(t0=t0, win_rows=len(w), win_mapped=int(w.cd.notna().sum()), win_amb=int(w.nn.isin(amb).sum()),
              win_corp_prefix1=int((w.cd.fillna('').str[0] == '1').sum()))
    m = w[w.cd.notna() & w.cd.isin(F) & w.k.notna()].copy()
    m = m[[c not in orig.get(r, set()) for r, c in zip(m.rgst_no, m.cd)]]
    m = m[[(c, k) not in blocked for c, k in zip(m.cd, m.k)]]
    pos = m.sort_values('d').groupby(['cd', 'k']).agg(d=('d', 'first'), n=('rgst_no', 'nunique'), bulk=('bulk', 'max'), typ=('typ', 'first')).reset_index()
    pos['t0'] = t0
    st.update(eligible_firms=len(F), pos=len(pos), pos_firms=int(pos.cd.nunique()))
    rows.append(st); P.append(pos); print(st, round(time.time() - T0), flush=True)
R = pd.DataFrame(rows); P = pd.concat(P, ignore_index=True)
u = P.drop_duplicates(['cd', 'k'])
log['origin_sum'] = len(P); log['unique_cd_k'] = len(u); log['min_per_origin'] = int(R.pos.min()); log['max_per_origin'] = int(R.pos.max())
R.to_csv(W + r'\cache\s3_B_dev_stats.csv', index=False, encoding='utf-8-sig'); P.to_parquet(W + r'\cache\B_dev_pos.parquet', index=False)
log['lic_mapped_share_all_lt2025_at_2024'] = round(float(lic.nn.map(alias_at(app, '20241231')[0]).notna().mean()), 4)
log['elapsed'] = round(time.time() - T0, 1)
json.dump(log, open(W + r'\cache\s3_B_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print({k: v for k, v in log.items() if 'monthly' not in k})
