# -*- coding: utf-8 -*-
"""s05 (a): right_list 디자인·실용신안 등록연도별 건수 대 KOSIS 지식재산통계(내려받은 json, 인자로 받은 폴더에서 읽기만).
 python -I s05_official.py <download/kosis 폴더>
 → official_compare.csv, cache/s05_out.json"""
import sys, json, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
KD = pathlib.Path(sys.argv[1])
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'
def J(f): return pd.DataFrame(json.load(open(KD/f, encoding='utf-8')))
OUT = {}
# --- KOSIS 법인 등록: A045(2004–07), A040_1(2015–25)
a45 = J('data_TX_13801_A045_1990_2016.json'); a40 = J('data_TX_13801_A040_1_2010_2025.json')
a46 = J('data_TX_13801_A046_2005_2025.json'); a06 = J('data_TX_13801_A006_2005_2025.json')
for df in (a45, a40, a46, a06): df['DT'] = pd.to_numeric(df.DT, errors='coerce')
def show(df, tag):
    cols = [c for c in df.columns if c.endswith('_NM') and c not in ('TBL_NM', 'UNIT_NM', 'ITM_NM')]
    OUT[f'{tag}_dims'] = {c: sorted(df[c].dropna().unique().tolist()) for c in cols}
for df, t in ((a45, 'A045'), (a40, 'A040_1'), (a46, 'A046'), (a06, 'A006')): show(df, t)
rows = []
# A040_1: C1=권리별, C2=개인·법인별, C3=내외국인별
for kind, nm in (('design', '디자인'), ('um', '실용신안')):
    for basis in (['청구서기준'] if kind == 'design' else [None]):
        pass
def pick40(kind_nm, corp='법인', nat='계', sub=None):
    x = a40[(a40.C2_NM == corp) & (a40.C3_NM == nat)]
    if sub is None: x = x[x.C1_NM == kind_nm]
    else: x = x[x.C1_NM == sub]
    return x.groupby('PRD_DE').DT.sum()
OUT['A040_1_C1_values'] = sorted(a40.C1_NM.unique().tolist())
# --- right_list 측
fl = pd.read_parquet(HERE/'cache/dsum_flags.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'kind', 'pert', 'tp'])
aq = pd.read_parquet(HERE/'cache/aq.parquet', columns=['applicant_cd', 'status'])
fm = set(pd.read_parquet(T1/'cache/firm_meta.parquet', columns=['applicant_cd']).applicant_cd)
fl = fl[fl.pert == '권리자란'].copy()
fl['ry'] = pd.to_numeric(fl.rgst_date.astype(str).str[:4], errors='coerce')
fl['base9'] = fl.rgst_no.astype(str).str[:9]; fl['suf'] = fl.rgst_no.astype(str).str[9:]
OUT['design_suffix_nonzero_share'] = float((fl[fl.kind == 'design'].suf != '0000').mean())
OUT['um_suffix_nonzero_share'] = float((fl[fl.kind == 'um'].suf != '0000').mean())
OUT['tp_dist'] = fl.tp.value_counts().to_dict()
domestic = fl.tp.astype(str).str.startswith('국내') | (fl.tp == '국가기관')
res = []
C1 = {'design': dict(a45='15138AG1ADAI', a40='A0003', a06='15138AG1AD'), 'um': dict(a45='15138AG1AB', a40='A003', a06='15138AG1AB')}
def corp45(c1, dom=False):
    x = a45[a45.C1 == c1]; out = {}
    for y in ('2004', '2005', '2006'):
        v = x[(x.PRD_DE == y) & x.C2.isin(['15138AI4ABAB'] if dom else ['15138AI4ABAB', '15138AI4ADAB'])].DT.sum(); out[y] = float(v)
    v = x[(x.PRD_DE == '2007') & (x.C2 == ('15138AI4ACAA' if dom else '15138AI4AC00'))].DT.sum(); out['2007'] = float(v)
    return out
for kind in ('design', 'um'):
    x = fl[fl.kind == kind]
    n_all = x.groupby('ry').rgst_no.nunique(); n_b9 = x.groupby('ry').base9.nunique()
    dm = domestic.loc[x.index] & (x.tp != '국내자연인')
    n_dom = x[dm].groupby('ry').base9.nunique() if kind == 'design' else x[dm].groupby('ry').rgst_no.nunique()
    n_fm = x[x.applicant_cd.isin(fm)].groupby('ry').base9.nunique() if kind == 'design' else x[x.applicant_cd.isin(fm)].groupby('ry').rgst_no.nunique()
    nb = n_b9 if kind == 'design' else n_all
    c = C1[kind]
    k40 = a40[(a40.C1 == c['a40']) & (a40.C2 == 'B003') & (a40.C3 == 'C001')].groupby('PRD_DE').DT.sum()
    k40d = a40[(a40.C1 == c['a40']) & (a40.C2 == 'B003') & (a40.C3 == 'C002')].groupby('PRD_DE').DT.sum()
    k45 = corp45(c['a45']); k45d = corp45(c['a45'], True)
    k06 = a06[a06.C1 == c['a06']].groupby('PRD_DE').DT.sum()
    for y in range(1995, 2026):
        s_ = str(y)
        kc = float(k40[s_]) if s_ in k40.index else (k45.get(s_, np.nan))
        kcd = float(k40d[s_]) if s_ in k40d.index else (k45d.get(s_, np.nan))
        res.append(dict(kind=kind, year=y, rl_count=int(nb.get(y, 0)), rl_unique_rgst=int(n_all.get(y, 0)), rl_domestic=int(n_dom.get(y, 0)), rl_t1_firms=int(n_fm.get(y, 0)),
                        kosis_corp=kc, kosis_corp_src=('A040_1' if s_ in k40.index else ('A045' if s_ in k45 else '')), kosis_corp_domestic=kcd,
                        kosis_total=float(k06[s_]) if s_ in k06.index else np.nan))
R = pd.DataFrame(res)
R['ratio_corp'] = R.rl_count / R.kosis_corp; R['ratio_corp_domestic'] = R.rl_domestic / R.kosis_corp_domestic; R['ratio_total'] = R.rl_count / R.kosis_total
R['ratio_t1_total'] = R.rl_t1_firms / R.kosis_total
R.to_csv(HERE/'official_compare.csv', index=False, encoding='utf-8-sig')
for kind in ('design', 'um'):
    r = R[(R.kind == kind)]
    rc = r[(r.year <= 2019) & r.ratio_corp.notna()]
    med = float(rc.ratio_corp.median()) if len(rc) else None
    OUT[f'{kind}_ratio_corp_median_le2019'] = med
    OUT[f'{kind}_ratio_corp_by_year'] = {int(a): round(float(b), 4) for a, b in zip(r.year, r.ratio_corp) if np.isfinite(b)}
    OUT[f'{kind}_ratio_total_by_year'] = {int(a): round(float(b), 4) for a, b in zip(r.year, r.ratio_total) if np.isfinite(b)}
    OUT[f'{kind}_ratio_corp_domestic_by_year'] = {int(a): round(float(b), 4) for a, b in zip(r.year, r.ratio_corp_domestic) if np.isfinite(b)}
    if med: OUT[f'{kind}_years_below_0.7med'] = [int(a) for a, b in zip(rc.year, rc.ratio_corp) if b < 0.7 * med]
    rt = r[(r.year <= 2019) & r.ratio_total.notna()]
    mt = float(rt.ratio_total.median()); OUT[f'{kind}_ratio_total_median_le2019'] = mt
    OUT[f'{kind}_years_total_below_0.7med'] = [int(a) for a, b in zip(rt.year, rt.ratio_total) if b < 0.7 * mt]
json.dump(OUT, open(HERE/'cache/s05_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=0)[:6000])
