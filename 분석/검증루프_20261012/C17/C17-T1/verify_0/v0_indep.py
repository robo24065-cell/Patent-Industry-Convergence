# -*- coding: utf-8 -*-
"""verify_0(PREREG §6): 트랙 캐시(ins_firm·app_pub·rl_pat·oos)·작업 스크립트(score17·n1_prep) 없이 sqlite 직접 + 별도 정규화로
 신규진입자 법인·N1 증거를 다시 만들고, N1·N2 Δskill(고정판)을 scipy rankdata Mann–Whitney로 저장 점수에서 다시 계산.
 점수·레이블: FROZEN_final o2019(px·pl), baseline_scores, C16 labels(공식 y_u) — 읽기 전용. → verify_0/v0_out.json"""
import sys, pathlib, re, json, sqlite3, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent/'scripts'))
import guard17 as G
import numpy as np, pandas as pd
from scipy.stats import rankdata
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'
PIPE = B/'수집파이프라인'; C16C = B/'분석/검증루프_20261012/C16/C16-T1/cache'; OUT = pathlib.Path(__file__).resolve().parent
t0 = time.time(); res = {}
ro = lambda p: sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
rg = ro(PIPE/'registry_full.sqlite'); br = ro(PIPE/'business_rights.sqlite')
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
FIRMS = set(te.firm)
# (1) 등록원부 직접: 원출원인 코드별 범위안 등록특허 출원연도·등록일 (patent.applDate는 국내단계일 → kr_registered 출원일과 PCT에서 다름)
ap = pd.read_sql("select a.applicantCd cd, p.rgst_no, p.applDate ad, p.rgstDate rd from applicant a join patent p on a.rgst_no=p.rgst_no where a.applicantCd is not null", rg)
ap['ay'] = pd.to_numeric(ap.ad.astype(str).str[:4], errors='coerce'); ap['rdn'] = pd.to_numeric(ap.rd.astype(str).str.replace('-', '').str[:8], errors='coerce')
apf = ap[ap.cd.isin(FIRMS) & (ap.ay >= 2000)]
inc_reg = set(apf[apf.ay <= 2019].cd)
newc_ind = FIRMS - inc_reg
off_newc = set(te.firm[te.newc.values])
res['newc_indep_registry_applDate'] = len(newc_ind); res['newc_official'] = len(off_newc); res['newc_sym_diff'] = len(newc_ind ^ off_newc)
res['newc_ind_minus_off'] = len(newc_ind - off_newc); res['newc_off_minus_ind'] = len(off_newc - newc_ind)
# 원인 확인: 차이 법인의 범위안 첫 출원연도(등록원부 applDate)
d1 = apf[apf.cd.isin(newc_ind - off_newc)].groupby('cd').ay.min(); res['ind_minus_off_first_applDate_year'] = d1.value_counts().sort_index().to_dict()
res['newc_registry_rgst_le_origin'] = int(apf[apf.cd.isin(off_newc) & (apf.rdn <= 20191231)].cd.nunique())
# (2) 증거: 별도 정규화(법인격 표기·기호 제거, 영문 소문자) — 트랙 STRIP 식과 독립 작성
LEG = ['주식회사', '(주)', '㈜', '유한회사', '유한책임회사', '재단법인', '사단법인', '학교법인', '가부시키가이샤', '가부시끼가이샤', '가부시기가이샤', '가부시키카이샤', '코포레이션', '인코포레이티드', '리미티드', '컴퍼니']
def nz(s):
    s = str(s)
    for t in LEG: s = s.replace(t, '')
    return ''.join(ch for ch in s.lower() if ch.isalnum())
aq = pd.read_sql("select applicant_cd, applicant_name, status from applicant_query", br)
okc = set(aq.applicant_cd[aq.status == 'OK'])
nm = pd.concat([aq[aq.status == 'OK'][['applicant_cd', 'applicant_name']].rename(columns={'applicant_cd': 'cd', 'applicant_name': 'n'}),
                pd.read_sql("select applicantCd cd, applicantName n from applicant", rg).query('cd in @okc')])
nm['z'] = nm.n.map(nz); nm = nm[nm.z.str.len() >= 2].drop_duplicates(['cd', 'z'])
amb = nm.groupby('z').cd.nunique(); u = nm[nm.z.map(amb) == 1]; z2c = dict(zip(u.z, u.cd))
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드'])
kp.columns = ['ay', 'pub', 'reg', 'names', 'ksic']
K50 = set(pd.read_sql("select distinct ksic from patent", rg).ksic.astype(str)); kp = kp[kp.ksic.astype(str).isin(K50)]
def d8(x):
    s = pd.Series(x).fillna('').astype(str).str.replace('-', '').str[:8]; return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).values
kp['v'] = np.minimum(d8(kp.pub.values), d8(kp.reg.values)); kp = kp[(kp.ay >= 2000) & (kp.ay <= 2019) & (kp.v <= 20191231)]
kp = kp.assign(n=kp.names.fillna('').str.split('|')).explode('n'); kp['cd'] = kp.n.map(nz).map(z2c)
E_pub = set(kp.cd.dropna()) & off_newc
# E_own: business_rights right_list(json pertDiv=권리자란) 범위안·등록≤원점 + registry owner 이전등록일≤원점·이름 일치
ph = ','.join('?' * len(off_newc))
rl = pd.read_sql(f"select applicant_cd cd, rgst_no, rgst_date, json_extract(raw,'$.pertDiv') pert from right_list where applicant_cd in ({ph}) and substr(rgst_no,1,2)='10'", br, params=sorted(off_newc))
rl = rl[(rl.pert == '권리자란') & (d8(rl.rgst_date.values) <= 20191231)]
insno = set(pd.read_sql("select rgst_no from patent", rg).rgst_no.astype(str)); rl = rl[rl.rgst_no.astype(str).isin(insno)].drop_duplicates(['cd', 'rgst_no'])
rn = sorted(rl.rgst_no.astype(str).unique()); ow = []
for i in range(0, len(rn), 900):
    ch = rn[i:i + 900]; ow.append(pd.read_sql(f"select rgst_no, ownerName, ownerRgstCsDate from owner where rgst_no in ({','.join('?' * len(ch))})", rg, params=ch))
ow = pd.concat(ow); ow['z'] = ow.ownerName.map(nz); ow['d'] = d8(ow.ownerRgstCsDate.values)
al_n = pd.concat([aq[aq.applicant_cd.isin(off_newc)][['applicant_cd', 'applicant_name']].rename(columns={'applicant_cd': 'cd', 'applicant_name': 'n'}),
                  pd.read_sql(f"select applicantCd cd, applicantName n from applicant where applicantCd in ({ph})", rg, params=sorted(off_newc))])
al_n['z'] = al_n.n.map(nz); al = al_n.groupby('cd').z.agg(set).to_dict()
m = rl.merge(ow, on='rgst_no'); m['ok'] = [isinstance(z, str) and (z in al.get(c, set()) or any(len(a) >= 3 and a in z for a in al.get(c, set()))) for c, z in zip(m.cd, m.z)]
E_own = set(m[m.ok & (m.d <= 20191231)].cd)
E_reg = set(apf[apf.cd.isin(off_newc) & (apf.rdn <= 20191231)].cd)
N1_vis = E_pub | E_own | E_reg
res['E_pub'] = len(E_pub); res['E_own'] = len(E_own); res['E_reg'] = len(E_reg); res['N1_vis'] = len(N1_vis)
ev = pd.read_parquet(OUT.parent/'cache'/'evid_all.parquet'); ev = ev[ev.is_newc_off]; main_vis = set(ev.firm[ev.N1_primary])
res['N1_vis_main'] = len(main_vis); res['N1_vis_sym_diff'] = len(N1_vis ^ main_vis); res['E_pub_sym_diff'] = len(E_pub ^ set(ev.firm[ev.E_pub])); res['E_own_sym_diff'] = len(E_own ^ set(ev.firm[ev.E_own]))
# (3) 채점: scipy rankdata Mann–Whitney
fz = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'px', 'pl']); bo = pd.read_parquet(FC/'baseline_scores.parquet')
L = pd.read_parquet(C16C/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
assert (fz.firm.values == te.firm.values).all() and (L.firm.values == te.firm.values).all()
keep = ~te.prior_any.values
def mw(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def skill_of(mask):
    y = L.y_u.values[mask].astype(int); px = fz.px.values[mask]; pl = fz.pl.values[mask]
    bl = 0.9 * (rankdata(px) / len(px)) + 0.1 * (rankdata(pl) / len(pl))           # pandas rank(pct=True)와 같은 식(평균 순위/n)
    a = mw(bl, y); m0 = max(mw(bo[c].values[mask], y) for c in bo.columns); return (a - m0) / (1 - m0), a, m0
s_off = skill_of(keep); res['off_skill'] = s_off[0]
for nm_, drop in (('N1_indep_def', set(off_newc) - N1_vis), ('N1_main_def', set(off_newc) - main_vis), ('N2_official_newc', set(off_newc)), ('N2_indep_newc', newc_ind)):
    msk = keep & ~(te.firm.isin(drop).values & (te.newc.values if nm_ != 'N2_indep_newc' else np.ones(len(te), bool)))
    sk = skill_of(msk); res[nm_] = dict(rows=int(msk.sum()), pos=int(L.y_u.values[msk].sum()), skill=sk[0], delta=sk[0] - s_off[0], auc=sk[1], M0=sk[2])
res['secs'] = round(time.time() - t0); res['mem_peak'] = G.rss_peak()
json.dump(res, open(OUT/'v0_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float); print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
