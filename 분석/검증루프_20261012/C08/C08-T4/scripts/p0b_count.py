# C08-T4 P0-b: 후보 A, A'상한, B, C, D, E 개발 기저율과 잠금 예상. 사건은 날짜 < 2025-01-01만 추출(assert).
import sqlite3, sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import *
T = time.time()
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)


def save(name, sql, dcol):
    df = pd.read_sql(sql, con)
    v = df[dcol].dropna()
    assert (v < CUT).all(), name
    df.to_parquet(os.path.join(C, name + '.parquet'), index=False)
    print(name, df.shape, round(time.time() - T, 1), flush=True)
    return df


pat = pd.read_sql("select rgst_no, ksic, rgstDate, cndrtExptnDate, applNo, ipcCd from patent", con)
pat['rgstDate'] = ok8(pat.rgstDate); pat['exp'] = ok8(pat.cndrtExptnDate)
pat.to_parquet(os.path.join(C, 'pat.parquet'), index=False)
rgt = save('rgt', f"select rgst_no, seq, rgstCsDate d, rgstCsName nm, rgstCsReason rs from right where rgstCsDate < '{CUT}'", 'd')
pay = save('pay', f"select rgst_no, seq, statAnnl, lastAnnl, payAmount, payDate d from pay where payDate < '{CUT}'", 'd')
lic = save('lic', f"select rgst_no, elsLcnsStartDate d, elsLcnsPsnNm nm, 1 ex from elsLcnsList where elsLcnsStartDate < '{CUT}' "
                  f"union all select rgst_no, noelsLcnsStartDate d, noelsLcnsPsnNm nm, 0 ex from noelsLcnsList where noelsLcnsStartDate < '{CUT}'", 'd')
own = save('own', f"select rgst_no, seq, ownerName, ownerNatl, ownerRgstCsDate d, ownerRgstCsName nm from owner where ownerRgstCsDate < '{CUT}'", 'd')
app = pd.read_sql("select rgst_no, applicantName, applicantCd from applicant", con)
app.to_parquet(os.path.join(C, 'app.parquet'), index=False)

st = rgt[rgt.nm.isin(DEAD + ['회복등록'])][['rgst_no', 'd', 'nm']].copy()
st['kind'] = np.where(st.nm == '회복등록', 'rec', 'dead')
st['d'] = ok8(st.d); st = st[st.d.notna()]

# 실시권자 상한용: 특허별 권리자·출원인 정규화 이름 집합
nm_sets = pd.concat([own[['rgst_no', 'ownerName']].rename(columns={'ownerName': 'n'}), app[['rgst_no', 'applicantName']].rename(columns={'applicantName': 'n'})])
nm_sets['n'] = nm_sets.n.map(norm)
nm_pairs = set(zip(nm_sets.rgst_no, nm_sets.n))
lic['d'] = ok8(lic.d); lic = lic[lic.d.notna()].copy()
lic['ntype'] = lic.nm.map(name_type)
lic['self'] = [(r, norm(n)) in nm_pairs for r, n in zip(lic.rgst_no, lic.nm)]
lic['ub'] = (lic.ntype != 4) & (~lic['self'])

part = rgt[(rgt.nm == '권리의 일부말소등록') & rgt.rs.fillna('').str.contains('포기')].copy(); part['d'] = ok8(part.d)
pay['d'] = ok8(pay.d)
for c in ['statAnnl', 'lastAnnl']:
    pay[c] = pd.to_numeric(pay[c], errors='coerce')
p4 = pay[(pay.statAnnl >= 4) & pay.d.notna()].copy()
multi = p4[(p4.lastAnnl - p4.statAnnl) >= 1]
rd = p4.rgst_no.map(pat.set_index('rgst_no').rgstDate)
ok = rd.notna()
due = (rd[ok].str[:4].astype(int) + p4.statAnnl[ok].astype(int) - 1).astype(str) + rd[ok].str[4:]
p4['due'] = None; p4.loc[ok, 'due'] = due
late = p4[p4.due.notna() & (p4.d > p4.due)]

EV = {'A': lic, 'A_UB': lic[lic.ub], 'B': part, 'D': multi, 'E': late}
rows = []
for Y in DEV + [2025]:
    t0, t1 = f'{Y}0101', f'{Y + 1}0101'
    m = alive_mask(pat, st, t0)
    alive = set(pat.rgst_no[m])
    r = {'origin': Y, 'risk': int(m.sum())}
    if Y < 2025:
        for k, e in EV.items():
            hit = set(e.rgst_no[(e.d >= t0) & (e.d < t1)]) & alive
            r[f'{k}_pos'] = len(hit); r[f'{k}_rate'] = len(hit) / r['risk']
    rows.append(r); print(r, round(time.time() - T, 1), flush=True)
R = pd.DataFrame(rows)

# C: 불납 소멸 코호트의 365일 내 회복
ext = rgt[(rgt.nm == '소멸등록') & (rgt.rs == '등록료불납')][['rgst_no', 'd']].copy(); ext['d'] = ok8(ext.d)
ext = ext.dropna().sort_values('d').drop_duplicates('rgst_no', keep='last')
rec = st[st.kind == 'rec'][['rgst_no', 'd']].rename(columns={'d': 'dr'})
x = ext.merge(rec, on='rgst_no', how='left')
dd = (pd.to_datetime(x.dr, format='%Y%m%d') - pd.to_datetime(x.d, format='%Y%m%d')).dt.days
x['hit'] = (dd > 0) & (dd <= 365)
x = x.groupby(['rgst_no', 'd']).hit.max().reset_index()
x['y'] = x.d.str[:4].astype(int)
cC = x.groupby('y').agg(risk=('hit', 'size'), pos=('hit', 'sum'))
cC['rate'] = cC.pos / cC.risk
lag = dd[(dd > 0)]
print('C cohort\n', cC.loc[2013:2024], flush=True)
OUT = {'table_ABDE': R.to_dict('records'), 'C_cohort': cC.reset_index().to_dict('records'),
       'C_recovery_lag_days_quantiles': lag.quantile([.5, .9, .95, .99]).to_dict(), 'C_recovery_lag_gt365_share': float((lag > 365).mean()),
       'lic_rows_lt2025': int(len(lic)), 'lic_ub_share': float(lic.ub.mean()), 'lic_individual_share': float((lic.ntype == 4).mean()), 'lic_self_share': float(lic['self'].mean())}
dev = R[R.origin < 2025]; lockrisk = int(R[R.origin == 2025].risk.iloc[0])
feas = []
for k in EV:
    rt = dev[f'{k}_rate']
    main = float(dev[dev.origin.isin([2021, 2022])][f'{k}_rate'].mean())
    feas.append({'cand': k, 'dev_pos_total': int(dev[f'{k}_pos'].sum()), 'dev_rate_mean': float(rt.mean()), 'dev_rate_min': float(rt.min()), 'dev_rate_max': float(rt.max()),
                 'rate_2021_22': main, 'lock_risk': lockrisk, 'lock_exp_main': lockrisk * main, 'lock_exp_alldev': lockrisk * float(rt.mean())})
cd = cC.loc[2015:2022]
lock_risk_C = 0.5 * float(cC.loc[[2023, 2024], 'risk'].mean())
mainC = float(cC.loc[[2021, 2022], 'rate'].mean())
feas.append({'cand': 'C', 'dev_pos_total': int(cd.pos.sum()), 'dev_rate_mean': float(cd.rate.mean()), 'dev_rate_min': float(cd.rate.min()), 'dev_rate_max': float(cd.rate.max()),
             'rate_2021_22': mainC, 'lock_risk': lock_risk_C, 'lock_exp_main': lock_risk_C * mainC, 'lock_exp_alldev': lock_risk_C * float(cd.rate.mean())})
F = pd.DataFrame(feas)
F['pass_rate'] = F.dev_rate_mean >= 0.003; F['pass_lock'] = F.lock_exp_main >= 1000; F['P0'] = F.pass_rate & F.pass_lock
OUT['feas'] = F.to_dict('records'); OUT['sec'] = round(time.time() - T, 1)
json.dump(OUT, open(os.path.join(C, 'p0b_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(R.round(5)); print(F.round(5)); print(json.dumps({k: v for k, v in OUT.items() if k.startswith(('C_rec', 'lic'))}, ensure_ascii=False, default=float))
