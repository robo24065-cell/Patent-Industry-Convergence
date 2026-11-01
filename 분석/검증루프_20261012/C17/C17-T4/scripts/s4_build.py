# C17-T4 s4: 단위 피처(BASE용)와 팔별 산업 라벨. 결과(y)는 개발 단위만 있고, 잠금 단위는 y 없음.
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib17 import *
import numpy as np, pandas as pd, duckdb
t_start = time.time()
U = pd.read_parquet(C + r'\units.parquet')
# ---- BASE 피처 ----
U['org'] = U.applicant.fillna('').str.upper().str.contains(
    r'\b(?:INC|CORP|LTD|LLC|GMBH|CO|COMPANY|LIMITED|AG|SA|KK|KAISHA|UNIV|INSTITUT|PLC|BV|NV|AB|OY|SPA|SE)\b|UNIVERSIT|INSTITUTE', regex=True).astype(int)
U['n_app'] = U.applicant.fillna('').str.count(r'\|') + 1
pc = U.prio_cc.fillna('').str.split('|').str[0]
U['prio_g'] = np.where(pc == '', 'none', np.where(pc.isin(['US', 'JP', 'KR', 'CN', 'DE']), pc, 'other'))
fam = U.family.fillna('')
U['fam_n'] = fam.map(lambda x: len(set(x.split('|'))) if x else 0)
U['fam_cc'] = fam.map(lambda x: len({t[:2] for t in x.split('|')}) if x else 0)
U['pend'] = [(ymd2date(g) - ymd2date(int(a))).days / 365.25 if isinstance(a, str) and len(a) == 8 else np.nan for g, a in zip(U.G, U.appl_date)]
U['gyear'] = U.G // 10000
ip = U.main_ipc.fillna('').str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
ex = ip.str.extract(r'^([A-H]\d\d[A-Z])\s?0*(\d+)(?:/(\d+))?')
U['sc'] = ex[0]; U['mg'] = (ex[0] + ' ' + ex[1]).where(ex[1].notna()); U['sg'] = (U.mg + '/' + ex[2]).where(ex[2].notna())
U['sec'] = U.sc.str[0].fillna('NA')
# 기업 키(군집용): 첫 출원인 정규화
LAT = re.compile(r'\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?')
NON = re.compile(r'[^0-9A-Z]')
U['firm'] = U.applicant.fillna('').str.split('|').str[0].str.upper().map(lambda s: NON.sub('', LAT.sub(' ', s)))
U.loc[U.firm == '', 'firm'] = 'NA_' + U.appl8
U['k_prov'] = U.ksic
# ---- 공개 배정: FREE-IPC / H-sc-1 (KR TXT-FREE 배정 → IPC 최빈표) ----
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
TXTFREE = (MC + r'\T1_산업대기실\TXT_R9\cache\kr_txtfree.parquet').replace('\\', '/')
kr = con.execute(f"select appl_no, cast(pub_date as integer) pub, main_ipc from read_parquet('{META}') where cc='KR' and length(pub_date)=8 and pub_date >= '19000101' and pub_date < '20250101'").fetchdf().drop_duplicates('appl_no')
tf = con.execute(f"select appl_no, k_free from read_parquet('{TXTFREE}')").fetchdf().drop_duplicates('appl_no').set_index('appl_no').k_free
kr['kt'] = kr.appl_no.map(tf)
kip = kr.main_ipc.fillna('').str.upper().str.replace(r'\s+', ' ', regex=True).str.strip().str.extract(r'^([A-H]\d\d[A-Z])\s?0*(\d+)(?:/(\d+))?')
kr['sc'] = kip[0]; kr['mg'] = (kip[0] + ' ' + kip[1]).where(kip[1].notna()); kr['sg'] = (kr.mg + '/' + kip[2]).where(kip[2].notna())
log = {'kr_docs': len(kr), 'kr_kt_cov': round(float(kr.kt.notna().mean()), 4)}


def tables(t0):
    d = kr[(kr.pub < t0) & kr.kt.notna()]
    tabs = {}
    for lev in ['sg', 'mg', 'sc']:
        g = d.groupby([lev, 'kt']).size().rename('n').reset_index()
        tot = g.groupby(lev).n.sum()
        g = g.sort_values([lev, 'n', 'kt'], ascending=[True, False, True]).drop_duplicates(lev)
        g['tot'] = g[lev].map(tot)
        tabs[lev] = g.set_index(lev)
    return tabs


for Y in (2021, 2022, 2023, 2025):
    tb = tables(Y * 10000 + 101)
    res = pd.Series(np.nan, index=U.index, dtype=object)
    for lev in ['sg', 'mg', 'sc']:
        t = tb[lev]; m = U[lev].map(t.kt.where(t.tot >= 3))
        res = res.where(res.notna(), m)
    U[f'k_free_{Y}'] = res.fillna('UNK')
    t = tb['sc']; U[f'k_hsc_{Y}'] = U.sc.map(t.kt.where(t.tot >= 1)).fillna('UNK')
    log[f'agree_free_{Y}'] = round(float((U[f'k_free_{Y}'] == U.k_prov).mean()), 4)
    log[f'agree_hsc_{Y}'] = round(float((U[f'k_hsc_{Y}'] == U.k_prov).mean()), 4)
# ---- 공식 KSIC–IPC 연계표(2013, 산업 3자리) ----
off = pd.read_excel(B + r'\공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx', sheet_name='전체연계표')
off.columns = ['code', 'ipc', 'n']
oi = off.ipc.astype(str).str.upper().str.replace(r'\s+', ' ', regex=True).str.strip().str.extract(r'^([A-H]\d\d[A-Z])\s?0*(\d+)(?:/(\d+))?')
off['sc'] = oi[0]; off['mg'] = (oi[0] + ' ' + oi[1]).where(oi[1].notna()); off['sg'] = (off.mg + '/' + oi[2]).where(oi[2].notna())
res = pd.Series(np.nan, index=U.index, dtype=object)
for lev in ['sg', 'mg', 'sc']:
    g = off.groupby([lev, 'code']).n.sum().reset_index().sort_values([lev, 'n', 'code'], ascending=[True, False, True]).drop_duplicates(lev).set_index(lev).code
    res = res.where(res.notna(), U[lev].map(g))
U['k_off'] = res.fillna('UNK')
log['off_cov'] = round(float((U.k_off != 'UNK').mean()), 4); log['off_ncodes'] = int(U.k_off.nunique())
keep = ['appl8', 'pat', 'firm', 's', 'D', 't0', 'y', 'expx_after', 'G', 'appl_year', 'ent_prev', 'late_prev', 'n_pet', 'n_expx', 'n_stat', 'n_exp_prior', 'pay_prev_lead',
        'org', 'n_app', 'prio_g', 'fam_n', 'fam_cc', 'pend', 'gyear', 'sec', 'sc', 'k_prov', 'k_off'] + [c for c in U.columns if c.startswith('k_free_') or c.startswith('k_hsc_')]
U[keep].to_parquet(C + r'\feat_units.parquet', index=False)
# 귀무 라벨용: 제공 US 문서 전체의 (서브클래스 × 출원연도) 칸
docs = pd.read_parquet(C + r'\us_docs.parquet', columns=['appl8', 'appl_year', 'main_ipc', 'ksic'])
di = docs.main_ipc.fillna('').str.upper().str.replace(r'\s+', ' ', regex=True).str.strip().str.extract(r'^([A-H]\d\d[A-Z])')
docs['cell'] = di[0].fillna('NA') + '|' + docs.appl_year.astype(str)
docs[['appl8', 'cell', 'ksic']].to_parquet(C + r'\null_cells.parquet', index=False)
log['n_units'] = len(U); log['sec'] = round(time.time() - t_start, 1)
json.dump(log, open(C + r'\s4_log.json', 'w'), indent=1)
out(log)
