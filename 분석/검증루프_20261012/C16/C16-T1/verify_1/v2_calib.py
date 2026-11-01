# -*- coding: utf-8 -*-
"""verify_1 v2: 주소 시도 규칙 거짓 불일치율이 법인 등록원부 규모(시도 수·문서 수)에 따라 달라지는지, P-미등록 근거 문서 분포로 재가중.
 거짓 불일치율 ≈ 관측 불일치율이면 의심 P 플래그 대부분이 정확 매칭(이전·다지역) → '공식이 보수적' 해석 약화."""
import sys, json, pathlib, sqlite3, re
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
HERE = B/'분석/검증루프_20261012/C16/C16-T1'; CC = HERE/'cache'; OUT = HERE/'verify_1'
CACHE = B/'분석/모델경쟁_20260927/T1_산업대기실/cache'
src = open(HERE/'scripts'/'a_alias.py', encoding='utf-8').read()
ns = {'re': re}; exec(src[src.index('SIDO = ['):src.index('# ---- 레이블·쌍 ----')], ns); region = ns['region']
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ra = pd.read_sql('select rgst_no, applicantCd, applicantAddr from applicant', con); con.close()
ra['applicantCd'] = ra.applicantCd.astype(str); ra['reg_r'] = ra.applicantAddr.map(region)
codes = ra.groupby('rgst_no').applicantCd.apply(set)
rr = ra.dropna(subset=['reg_r'])
fr_by = {f: g for f, g in rr.groupby('applicantCd')}
# 캘리브레이션 문서: app_pub 등록, 출원 2020-23, 이름매칭 코드 = 원출원인 코드
a = pd.read_parquet(CACHE/'app_pub.parquet'); a = a[a.reg.astype(bool) & (a.ay > 2019) & (a.ay <= 2023)].copy(); a['appl_no'] = a.appl_no.astype(str)
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '출원인명', '출원인주소']); kp['출원번호'] = kp['출원번호'].astype(str)
a = a.merge(kp, left_on='appl_no', right_on='출원번호', how='left'); a['rg'] = a['등록번호'].fillna('').astype(str)
a['applicant_cd'] = a.applicant_cd.astype(str)
a = a[[c in codes.get(r, set()) for c, r in zip(a.applicant_cd, a.rg)]]
firms = pd.read_parquet(CACHE/'firms.parquet'); firms = firms[firms.status == 'OK']
app = pd.read_parquet(CACHE/'ins_applicant.parquet')
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(firms.applicant_cd)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]; n2c = dict(zip(al.n, al.applicant_cd.astype(str)))


def e2_of(names_raw, addrs_raw, firm, excl_rg):
    names = str(names_raw).split('|'); addrs = str(addrs_raw).split('|')
    posi = [i for i, nm in enumerate(names) if n2c.get(norm(nm)) == firm]
    sa = ({region(addrs[posi[0]])} if posi and len(addrs) == len(names) else {region(x) for x in addrs}) - {None}
    g = fr_by.get(firm)
    if g is None: return 'none', 0, 0
    gg = g[g.rgst_no != excl_rg]; sf = set(gg.reg_r)
    nd = gg.rgst_no.nunique(); nr = len(sf)
    if not sa or not sf: return 'none', nd, nr
    return ('match' if sa & sf else 'mismatch'), nd, nr


cal = a.sample(n=min(30000, len(a)), random_state=7)
R = [e2_of(r.출원인명, r.출원인주소, r.applicant_cd, r.rg) for r in cal.itertuples(index=False)]
cal = cal.assign(e2=[x[0] for x in R], nd=[x[1] for x in R], nr=[x[2] for x in R])
# P-미등록 근거 문서: 같은 규모 지표(법인 등록원부 전체; 미등록이라 제외할 문서 없음)
DOC = pd.read_parquet(CC/'alias_docs.parquet'); pu = DOC[(DOC.grp == 'P') & ~DOC.is_reg].copy(); pu['firm'] = pu.firm.astype(str)
pu['nd'] = [fr_by[f].rgst_no.nunique() if f in fr_by else 0 for f in pu.firm]; pu['nr'] = [fr_by[f].reg_r.nunique() if f in fr_by else 0 for f in pu.firm]
bins = [0, 1, 2, 5, 10, 30, 100, 10**9]; lab = ['1', '2', '3-5', '6-10', '11-30', '31-100', '>100']
cal['b'] = pd.cut(cal.nd, bins, labels=lab); pu['b'] = pd.cut(pu.nd, bins, labels=lab)
cd = cal[cal.e2 != 'none']; pd_ = pu[pu.e2 != 'none']
tab = pd.DataFrame({'cal_n': cd.groupby('b').size(), 'cal_false_mm': cd.groupby('b').e2.apply(lambda s: (s == 'mismatch').mean()),
                    'Punreg_n': pd_.groupby('b').size(), 'Punreg_mm': pd_.groupby('b').e2.apply(lambda s: (s == 'mismatch').mean())})
w = tab.Punreg_n.fillna(0) / tab.Punreg_n.sum()
fm_rw = float((w * tab.cal_false_mm.fillna(0)).sum())
obs = float((pd_.e2 == 'mismatch').mean())
# 시도 수 기준
cal['br'] = pd.cut(cal.nr, [-1, 1, 2, 4, 100], labels=['1', '2', '3-4', '5+']); pu['br'] = pd.cut(pu.nr, [-1, 1, 2, 4, 100], labels=['1', '2', '3-4', '5+'])
cd = cal[cal.e2 != 'none']; pd_ = pu[pu.e2 != 'none']
tab2 = pd.DataFrame({'cal_n': cd.groupby('br').size(), 'cal_false_mm': cd.groupby('br').e2.apply(lambda s: (s == 'mismatch').mean()),
                     'Punreg_n': pd_.groupby('br').size(), 'Punreg_mm': pd_.groupby('br').e2.apply(lambda s: (s == 'mismatch').mean())})
w2 = tab2.Punreg_n.fillna(0) / tab2.Punreg_n.sum(); fm_rw2 = float((w2 * tab2.cal_false_mm.fillna(0)).sum())
out = dict(cal_docs=int(len(cal)), cal_determinable=int((cal.e2 != 'none').sum()), cal_false_mm_raw=float((cal[cal.e2 != 'none'].e2 == 'mismatch').mean()),
           Punreg_docs=int(len(pu)), Punreg_determinable=int(len(pd_)), Punreg_obs_mm=obs,
           false_mm_reweighted_by_doccount=fm_rw, false_mm_reweighted_by_regions=fm_rw2,
           implied_true_error_share_of_flags_doccount=max(0.0, 1 - fm_rw / obs), implied_true_error_share_of_flags_regions=max(0.0, 1 - fm_rw2 / obs),
           tab_doccount=tab.reset_index().astype(str).to_dict('records'), tab_regions=tab2.reset_index().astype(str).to_dict('records'),
           median_nd=dict(cal=float(cal.nd.median()), Punreg=float(pu.nd.median())))
json.dump(out, open(OUT/'v2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
