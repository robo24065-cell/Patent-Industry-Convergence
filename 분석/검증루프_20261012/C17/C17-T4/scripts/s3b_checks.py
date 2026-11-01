# C17-T4 s3b: 데이터 점검 d1–d5, E_lock, P0(a). 개발 구역만 읽는다(LOCKED 금지).
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib17 import *
import numpy as np, pandas as pd, duckdb
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
R = {}
# ---------- d1 총량 ----------
full = json.load(open(C + r'\s3a_full_20250303.json'))
OFFICIAL = {  # USPTO 연간 실용특허 등록(공개 집계; 2016 PTMT 303,051, 2017 예비 318,849; 나머지 USPTO us_stat 표 값 — 원 페이지 404, 기억 값으로 표시)
    2011: 224505, 2012: 253155, 2013: 277835, 2014: 300678, 2015: 298407, 2016: 303051, 2017: 318849}
d1 = {}
for y in range(2009, 2025):
    b = full['by_grant_year'][str(y)]
    d1[y] = {'n': b['n'], 'span_ratio': round(b['n'] / b['span'], 4),
             'official': OFFICIAL.get(y), 'rel_diff': (round(b['n'] / OFFICIAL[y] - 1, 4) if y in OFFICIAL else None)}
R['d1'] = d1
R['d1_pass'] = all(d1[y]['span_ratio'] >= 0.97 and abs(d1[y]['rel_diff']) <= 0.03 for y in range(2011, 2018))
# 단계별 소멸 비율 근사(EXP/(EXP+PAY)) 전체 대 제공연결
evc = full['events']
lap = {}
for grp in ('link', 'other'):
    for y in range(2016, 2025):
        for s in (1, 2, 3):
            ex = evc.get(f'{grp}|{y}|{s}|EXP', 0); pa = evc.get(f'{grp}|{y}|{s}|PAY', 0)
            lap[f'{grp}|{y}|{s}'] = round(ex / (ex + pa), 4) if ex + pa else None
R['lapse_ratio_eventyear'] = lap
# ---------- d2 연결 정확성 ----------
docs = pd.read_parquet(C + r'\us_docs.parquet', columns=['appl8', 'appl_year', 'rgst_no', 'rgst_date', 'family', 'prio_cc', 'applicant', 'ksic', 'pub_date', 'main_ipc'])
EV = (C + r'\ev_dev_20250303.parquet').replace('\\', '/')
pt = con.execute(f"select appl8, any_value(pat) pat, any_value(G) G from '{EV}' where regexp_matches(pat,'^[0-9]+$') group by appl8").fetchdf()
docs = docs.merge(pt, on='appl8', how='left')
docs['linked'] = docs.pat.notna()
hr = docs[docs.linked & docs.rgst_no.notna() & (docs.rgst_no.str.len() == 8) & docs.rgst_no.str.isdigit()]
eq = (hr.rgst_no.astype(np.int64) == hr.pat.astype(np.int64))
gdiff = (hr.rgst_date.astype(int) == hr.G)
R['d2'] = {'n_linked_with_rgst': int(len(hr)), 'pat_eq_rgst': round(float(eq.mean()), 5), 'G_eq_rgst_date': round(float(gdiff.mean()), 5),
           'mismatch_examples': hr[~eq][['appl8', 'rgst_no', 'pat']].head(5).values.tolist()}
nr = docs[docs.rgst_no.notna() & (docs.rgst_no.str.len() == 8) & docs.rgst_no.str.isdigit()]
R['d2']['rgst_docs_linked_rate'] = round(float(nr.linked.mean()), 4)
R['d2']['rgst_docs_linked_rate_by_rgst_year'] = nr.groupby(nr.rgst_date.str[:4]).linked.mean().round(4).to_dict()
# 표본 300: 특허번호가 family 토큰에 있는가 / 공개번호 형식 토큰과 혼동되는가
rng = np.random.default_rng(SEED)
smp = docs[docs.linked].sample(300, random_state=SEED)
def fam_has(pat, fam):
    if not isinstance(fam, str):
        return None
    n = int(pat); toks = set(fam.split('|'))
    return any(t in toks for t in (f'US{n}', f'US{n:08d}', f'US0{n}', f'US{n:09d}'))
fh = [fam_has(p, f) for p, f in zip(smp.pat, smp.family)]
fh2 = [x for x in fh if x is not None]
R['d2']['family_token_has_pat_rate_sample300'] = round(float(np.mean(fh2)), 4) if fh2 else None
R['d2']['family_missing_sample300'] = int(sum(x is None for x in fh))
# 공개번호 형식(US + 연도4 + 7자리) 토큰이 특허번호로 오인될 가능성: 연결은 출원번호로 하므로 0. 제공 rgst_no 중 11자리 이상 형식 수
R['d2']['rgst_no_len_dist'] = docs.rgst_no.dropna().str.len().value_counts().to_dict()
R['d2']['rgst_no_nondigit'] = docs.rgst_no.dropna().str.contains(r'\D').sum().item()
R['d2_pass'] = R['d2']['pat_eq_rgst'] >= 0.97
# ---------- d3 선택 편향 ----------
m = docs[(docs.appl_year >= 2005) & (docs.appl_year <= 2015)].copy()
m['org'] = m.applicant.fillna('').str.upper().str.contains(r'\b(?:INC|CORP|LTD|LLC|GMBH|CO|COMPANY|LIMITED|AG|SA|KK|KAISHA|UNIV|INSTITUT|PLC|BV|NV|AB|OY|SPA|SE)\b|UNIVERSIT|INSTITUTE', regex=True)
pc = m.prio_cc.fillna('').str.split('|').str[0]
m['prio_g'] = np.where(pc == '', 'none', np.where(pc.isin(['US', 'JP', 'KR', 'CN', 'DE']), pc, 'other'))
kr = m.groupby('ksic').linked.mean()
R['d3'] = {'n': int(len(m)), 'link_rate': round(float(m.linked.mean()), 4),
           'by_appl_year': m.groupby('appl_year').linked.mean().round(4).to_dict(),
           'by_ksic_min_med_max': [round(float(kr.min()), 4), round(float(kr.median()), 4), round(float(kr.max()), 4)],
           'ksic_min_code': kr.idxmin(), 'by_prio': m.groupby('prio_g').linked.mean().round(4).to_dict(),
           'by_org': m.groupby('org').linked.mean().round(4).to_dict(),
           'composition_linked_vs_not': {
               'org_share': [round(float(m[m.linked].org.mean()), 4), round(float(m[~m.linked].org.mean()), 4)],
               'prio_share_linked': m[m.linked].prio_g.value_counts(normalize=True).round(4).to_dict(),
               'prio_share_unlinked': m[~m.linked].prio_g.value_counts(normalize=True).round(4).to_dict()},
           'tvd_ksic_linked_vs_unlinked': round(float(0.5 * (m[m.linked].ksic.value_counts(normalize=True) - m[~m.linked].ksic.value_counts(normalize=True)).abs().fillna(0).sum()), 4)}
R['d3_pass'] = bool(kr.min() >= 0.5 * kr.median())
# ---------- d4 시점 정합 ----------
E1 = (C + r'\ev_dev_20241216.parquet').replace('\\', '/')
ex1 = con.execute(f"select distinct pat, e from '{E1}' where code='EXP.'").fetchdf()
ex2 = con.execute(f"select distinct pat, G, e from '{EV}' where code='EXP.'").fetchdf()
mx1 = int(ex1.e.max())
R['d4'] = {'max_exp_entry_20241216': mx1}
lag_days = (ymd2date(20241216) - ymd2date(mx1)).days
lag_w = int(np.ceil(max(lag_days, 0) / 7.0) * 7)
# 마감 D 추정(가장 가까운 기념일)
def nearest_D(G, e):
    best = None
    for s in (1, 2, 3):
        D = add_years(G, 4 * s); dd = (ymd2date(e) - ymd2date(D)).days
        if best is None or abs(dd) < abs(best[1]):
            best = (D, dd)
    return best
nd = [nearest_D(g, e) for g, e in zip(ex2.G, ex2.e)]
ex2['D'] = [a for a, b in nd]; ex2['dd'] = [b for a, b in nd]
r18 = ex2[(ex2.e >= 20180101) & (ex2.e < 20250101)]
R['d4']['delay_p99_2018_2024'] = float(np.percentile(r18.dd, 99)); R['d4']['delay_p50'] = float(np.percentile(r18.dd, 50))
R['d4']['delay_share_le45'] = round(float((r18.dd.between(-7, 45)).mean()), 4)
w = ex2[(ex2.D >= 20240101) & (ex2.D <= 20241015)]
k1 = set(zip(ex1.pat, ex1.e))
R['d4']['present_in_1216_given_0303'] = round(float(np.mean([(p, e) in k1 for p, e in zip(w.pat, w.e)])), 5)
k2 = set(zip(ex2.pat, ex2.e))
e1w = ex1[ex1.e < 20241101]
R['d4']['vanished_1216_to_0303'] = round(float(np.mean([(p, e) not in k2 for p, e in zip(e1w.pat, e1w.e)])), 5)
R['d4']['vanished_n'] = int(sum((p, e) not in k2 for p, e in zip(e1w.pat, e1w.e)))
U = pd.read_parquet(C + r'\units.parquet', columns=['pat', 's', 'D', 'y', 'expx_after', 'late_prev', 'ent_prev', 'paid_s', 'ksic', 'prio_cc', 'G'])
dv = U[U.D < LOCK_START]
R['d4']['expx_after_rate_among_pos'] = round(float(dv[dv.y == 1].expx_after.mean()), 4)
R['d4']['late_prev_rate'] = round(float(U.late_prev.mean()), 4)
R['d4']['y1_and_paid'] = int(((dv.y == 1) & (dv.paid_s == 1)).sum())
E_lock = add_years(0, 0)  # placeholder
E_lock = int((pd.Timestamp('2025-03-03') - pd.Timedelta(days=45 + lag_w)).strftime('%Y%m%d'))
R['d4']['lag_post_days'] = lag_days; R['d4']['lag_post_weeks_days'] = lag_w; R['E_lock'] = E_lock
R['d4_pass'] = (R['d4']['delay_p99_2018_2024'] <= 45 and R['d4']['present_in_1216_given_0303'] >= 0.99 and R['d4']['vanished_1216_to_0303'] <= 0.01)
# ---------- d5 분포·중복 ----------
s2 = json.load(open(C + r'\s2_log.json'))
R['d5'] = {'distinct_removed_linked_rows': s2['ev_numeric_rows_raw'] - s2['ev_numeric_rows_distinct'],
           'entity_prev_by_s_Dyear': dv.groupby(['s', dv.D // 10000]).ent_prev.value_counts(normalize=True).round(3).unstack().fillna(0).reset_index().astype(str).values.tolist(),
           'ksic_top5_units': U.ksic.value_counts().head(5).to_dict(), 'ksic_n': int(U.ksic.nunique()),
           'prio_missing_units': round(float(U.prio_cc.isna().mean()), 4)}
R['d5_pass'] = True
R['P0d_pass'] = bool(R['d1_pass'] and R['d2_pass'] and R['d3_pass'] and R['d4_pass'] and R['d5_pass'])
# ---------- P0(a) ----------
lk = U[(U.D >= LOCK_START) & (U.D <= E_lock)]
base = {s: float(dv[(dv.s == s) & (dv.D >= 20230101) & (dv.D <= 20241115)].y.mean()) for s in (2, 3)}
nlk = lk.s.value_counts().to_dict()
exp_pos = sum(nlk.get(s, 0) * base[s] for s in (2, 3))
R['P0a'] = {'lock_window': [LOCK_START, E_lock], 'n_lock_by_s': {int(k): int(v) for k, v in nlk.items()}, 'base_rate': base,
            'expected_pos': round(exp_pos, 1), 'pass': exp_pos >= 1000}
json.dump(R, open(C + r'\s3b_checks.json', 'w'), indent=1, default=str)
out(json.dumps({k: R[k] for k in ['d1_pass', 'd2', 'd2_pass', 'd3', 'd3_pass', 'd4', 'd4_pass', 'P0d_pass', 'E_lock', 'P0a']}, indent=1, default=str))
