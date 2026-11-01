# C08-T4 s1: E* 레이블(원점 2012–2022) + 잠금 위험집합 + P1 + P2(a)(b)(c)(d)(e) + P3(b)(d). 캐시(사건 < 2025-01-01)만.
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import *
T = time.time()
pat = pd.read_parquet(os.path.join(C, 'pat.parquet'))
rgt = pd.read_parquet(os.path.join(C, 'rgt.parquet'))
pay = pd.read_parquet(os.path.join(C, 'pay.parquet'))
app = pd.read_parquet(os.path.join(C, 'app.parquet'))
for df in (rgt, pay):
    assert (df.d.dropna() < CUT).all()
LOG = {}
# --- 납부행 정리
LOG['pay_rows'] = len(pay)
pay['d8'] = ok8(pay.d)
LOG['pay_date_bad_share'] = float(pay.d8.isna().mean())
for c in ['statAnnl', 'lastAnnl']:
    pay[c] = pd.to_numeric(pay[c], errors='coerce')
LOG['pay_dup_rgst_statAnnl_share'] = float(pay.duplicated(['rgst_no', 'statAnnl']).mean())
LOG['pat_rgstDate_null_share'] = float(pat.rgstDate.isna().mean())
pay = pay[pay.d8.notna() & pay.statAnnl.notna() & pay.lastAnnl.notna()].copy()
rd = pay.rgst_no.map(pat.set_index('rgst_no').rgstDate)
pay = pay[rd.notna()].copy(); rd = rd[rd.notna()]
due = (rd.str[:4].astype(int) + pay.statAnnl.astype(int) - 1).astype(str) + rd.str[4:]
due = due.where(~due.str.endswith('0229'), due.str[:4] + '0228')
pay['due'] = pd.to_datetime(due, format='%Y%m%d', errors='coerce')
pay['dt'] = pd.to_datetime(pay.d8, format='%Y%m%d')
pay['lag'] = (pay.dt - pay.due).dt.days
pay['late7'] = (pay.statAnnl >= 4) & (pay.lag > 7)
pay['late3'] = (pay.statAnnl >= 4) & (pay.lag > 3)
pay['late0'] = (pay.statAnnl >= 4) & (pay.lag > 0)
pay[['rgst_no', 'statAnnl', 'lastAnnl', 'd8', 'lag', 'late7', 'late3', 'late0']].to_parquet(os.path.join(C, 's1_pay.parquet'), index=False)
# --- 상태 사건
st = rgt[rgt.nm.isin(DEAD + ['회복등록'])][['rgst_no', 'd', 'nm']].copy()
st['kind'] = np.where(st.nm == '회복등록', 'rec', 'dead'); st['d'] = ok8(st.d); st = st[st.d.notna()]
# --- 첫 출원인
a1 = app.drop_duplicates('rgst_no').set_index('rgst_no')
pat['cd'] = pat.rgst_no.map(a1.applicantCd).fillna('NA')
# --- 원점별 레이블
rows, lab = [], []
for Y in list(range(2012, 2023)) + [2025]:
    t0, t1 = f'{Y}0101', f'{Y + 1}0101'
    m = alive_mask(pat, st, t0)
    r = {'origin': Y, 'risk': int(m.sum())}
    if Y <= 2022:
        w = pay[(pay.d8 >= t0) & (pay.d8 < t1)]
        df = pd.DataFrame({'rgst_no': pat.rgst_no[m].values})
        for k in ['late7', 'late3', 'late0']:
            hit = set(w.rgst_no[w[k]])
            df[k] = df.rgst_no.isin(hit).astype(np.int8)
            r[f'{k}_pos'] = int(df[k].sum()); r[f'{k}_rate'] = r[f'{k}_pos'] / r['risk']
        df['origin'] = Y
        lab.append(df)
    rows.append(r); print(r, round(time.time() - T, 1), flush=True)
R = pd.DataFrame(rows)
L = pd.concat(lab, ignore_index=True)
L.to_parquet(os.path.join(C, 's1_labels.parquet'), index=False)
dev = R[R.origin.between(2015, 2022)]
lock_risk = int(R[R.origin == 2025].risk.iloc[0])
P1 = {'pos_by_origin': dict(zip(dev.origin, dev.late7_pos)), 'rate_by_origin': dict(zip(dev.origin, dev.late7_rate.round(6))),
      'pos_total': int(dev.late7_pos.sum()), 'unique_patents': int(L[(L.origin.between(2015, 2022)) & (L.late7 == 1)].rgst_no.nunique()),
      'min_pos': int(dev.late7_pos.min()), 'dev_rate_mean': float(dev.late7_rate.mean()), 'rate_2021_22': float(dev[dev.origin >= 2021].late7_rate.mean()),
      'lock_risk': lock_risk}
P1['lock_exp_main'] = lock_risk * P1['rate_2021_22']; P1['lock_exp_alldev'] = lock_risk * P1['dev_rate_mean']
P1['pass_a'] = bool(P1['min_pos'] >= 200 and P1['pos_total'] >= 1000)
P1['pass_b'] = bool(P1['dev_rate_mean'] >= 0.003 and P1['lock_exp_main'] >= 1000)
P1['E3'] = {'rate_mean': float(dev.late3_rate.mean()), 'lock_exp_main': lock_risk * float(dev[dev.origin >= 2021].late3_rate.mean())}
P1['E0_strict'] = {'rate_mean': float(dev.late0_rate.mean()), 'lock_exp_main': lock_risk * float(dev[dev.origin >= 2021].late0_rate.mean())}
P1['rate_max_min_ratio'] = float(dev.late7_rate.max() / dev.late7_rate.min())
LOG['P1'] = P1; print(json.dumps(P1, ensure_ascii=False, default=float), flush=True)

# --- P2(a) 모집단 대조
kr = pd.read_parquet(os.path.join(B, '공모전데이터', '작업데이터', 'kr_registered.parquet'), columns=['등록번호', '등록일자', '세세분류코드'])
kr['등록번호'] = kr['등록번호'].astype(str).str.strip()
s_kr, s_rg = set(kr['등록번호']), set(pat.rgst_no)
LOG['P2a'] = {'kr_n': len(kr), 'kr_unique': len(s_kr), 'reg_n': len(pat), 'kr_in_reg': len(s_kr & s_rg) / len(s_kr), 'reg_in_kr': len(s_kr & s_rg) / len(s_rg)}
mk = kr.drop_duplicates('등록번호').set_index('등록번호')['세세분류코드'].astype(str)
LOG['P2a']['ksic_agree'] = float((pat.rgst_no.map(mk) == pat.ksic.astype(str)).mean())
LOG['P2a']['pass'] = bool(min(LOG['P2a']['kr_in_reg'], LOG['P2a']['reg_in_kr']) >= 0.99)

# --- P2(b) 납부 기록 완결성
pr = pat.set_index('rgst_no')
comp, compk = {}, []
for Y in range(2015, 2023):
    t0, t1 = f'{Y}0101', f'{Y + 1}0101'
    a0 = set(L[L.origin == Y].rgst_no)
    m2 = alive_mask(pat, st, f'{Y + 2}0101')
    a2 = set(pat.rgst_no[m2])
    S = pat[pat.rgst_no.isin(a0 & a2)].copy()
    S = S[S.exp.isna() | (S.exp >= t1)]
    S['kw'] = Y - S.rgstDate.str[:4].astype(int) + 1
    S = S[S.kw >= 4]
    lim = (pd.Timestamp(t1) + pd.Timedelta(days=214)).strftime('%Y%m%d')
    cov = pay[pay.d8 < lim].groupby('rgst_no').lastAnnl.max()
    S['cov'] = S.rgst_no.map(cov).fillna(0)
    S['ok'] = S['cov'] >= S.kw
    comp[Y] = {'n': int(len(S)), 'covered': float(S.ok.mean())}
    compk.append(S.groupby('ksic').ok.mean().rename(Y))
CK = pd.concat(compk, axis=1)
LOG['P2b'] = {'by_origin': comp, 'min_origin': float(min(v['covered'] for v in comp.values())), 'ksic_min': float(CK.min().min()),
              'ksic_min_where': str(CK.stack().idxmin())}
LOG['P2b']['pass'] = bool(LOG['P2b']['min_origin'] >= 0.95); LOG['P2b']['warn_ksic'] = bool(LOG['P2b']['ksic_min'] < 0.90)
print('P2b', json.dumps(LOG['P2b'], ensure_ascii=False, default=float), flush=True)
# --- P2(c)
LOG['P2c'] = {'dup': LOG['pay_dup_rgst_statAnnl_share'], 'bad_date': LOG['pay_date_bad_share'], 'rgst_null': LOG['pat_rgstDate_null_share']}
LOG['P2c']['pass'] = bool(max(LOG['P2c'].values()) <= 0.01)
# --- P2(d) 집중도
D = L[L.origin.between(2015, 2022)].merge(pat[['rgst_no', 'cd', 'ksic']], on='rgst_no')
pos = D[D.late7 == 1]
top10 = pos.groupby('cd').size().sort_values(ascending=False)
LOG['P2d'] = {'top10_cd_share': float(top10.head(10).sum() / len(pos)), 'top10': {str(k): int(v) for k, v in top10.head(10).items()},
              'ksic_present': int(pos.ksic.nunique()), 'top5_ksic_share_pos': float(pos.ksic.value_counts(normalize=True).head(5).sum()),
              'top5_ksic_share_risk': float(D.ksic.value_counts(normalize=True).head(5).sum()),
              'ksic_rate_min': float(D.groupby('ksic').late7.mean().min()), 'ksic_rate_max': float(D.groupby('ksic').late7.mean().max()),
              'type_rate': {str(k): float(v) for k, v in D.groupby(D.cd.str[0]).late7.mean().items()},
              'type_share_pos': {str(k): float(v) for k, v in pos.cd.str[0].value_counts(normalize=True).items()},
              'type_share_risk': {str(k): float(v) for k, v in D.cd.str[0].value_counts(normalize=True).items()}}
LOG['P2d']['pass'] = bool(LOG['P2d']['top10_cd_share'] <= 0.50); LOG['P2d']['warn'] = bool(LOG['P2d']['top10_cd_share'] > 0.30)
D.groupby(['origin', 'ksic']).late7.agg(['size', 'sum']).reset_index().to_parquet(os.path.join(C, 's1_ksic_origin.parquet'), index=False)
# --- P2(e)
LOG['P2e_warn'] = bool(P1['rate_max_min_ratio'] > 2.0)
# --- P3(b) 소급 기록
cov_all = pay.groupby('rgst_no').lastAnnl.max()
ext = rgt[(rgt.nm == '소멸등록') & (rgt.rs == '등록료불납')][['rgst_no', 'd']].copy(); ext['d'] = ok8(ext.d)
ext = ext.dropna().drop_duplicates('rgst_no', keep='last')
ext['rd'] = ext.rgst_no.map(pr.rgstDate); ext['cv'] = ext.rgst_no.map(cov_all); ext = ext.dropna()
ext = ext[(ext.d >= '20150101') & (ext.d < '20230101') & (ext.rd.str[4:] != '0229')]
end = pd.to_datetime((ext.rd.str[:4].astype(int) + ext.cv.astype(int)).astype(str) + ext.rd.str[4:], format='%Y%m%d', errors='coerce')
gap = (pd.to_datetime(ext.d, format='%Y%m%d') - end).dt.days
LOG['P3b'] = {'n': int(gap.notna().sum()), 'share_gap_le_183': float((gap <= 183).mean()), 'median_gap': float(gap.median())}
LOG['P3b']['pass'] = bool(LOG['P3b']['share_gap_le_183'] < 0.05)
# --- P3(d) 양성 중 t0 연체 몫
od = {}
for Y in range(2015, 2023):
    t0 = f'{Y}0101'
    cv0 = pay[pay.d8 < t0].groupby('rgst_no').lastAnnl.max()
    x = D[(D.origin == Y)][['rgst_no', 'late7']].copy()
    x['rd'] = x.rgst_no.map(pr.rgstDate); x['cv'] = x.rgst_no.map(cv0).fillna(3)
    nd = (x.rd.str[:4].astype(int) + x.cv.astype(int)).astype(str) + x.rd.str[4:].where(x.rd.str[4:] != '0229', '0228')
    x['over'] = nd < t0
    od[Y] = {'pos_overdue_share': float(x[x.late7 == 1].over.mean()), 'risk_overdue_share': float(x.over.mean()),
             'rate_if_overdue': float(x[x.over].late7.mean()), 'rate_if_not': float(x[~x.over].late7.mean())}
LOG['P3d'] = od
LOG['sec'] = round(time.time() - T, 1)
R.to_csv(os.path.join(C, 's1_origin_table.csv'), index=False)
json.dump(LOG, open(os.path.join(C, 's1_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in LOG.items() if k not in ('P1',)}, ensure_ascii=False, indent=1, default=float))
