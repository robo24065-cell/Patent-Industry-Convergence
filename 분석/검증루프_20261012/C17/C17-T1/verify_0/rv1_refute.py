# -*- coding: utf-8 -*-
"""반박 검증(C17-T1 발견, rv1): 원천(sqlite mode=ro, kr_registered/kr_patents, FROZEN o2019, prep_te, C16 labels)만 읽어
 (1) applicant_query 대 등록원부 applicantCd, (2) 공식 신규진입자 첫 범위안 출원·등록 연도(연도 필터 없이),
 (3) 채점쌍·양성 비중, (4) 원점 가시 증거(별도 경로: owner 표 직접 이름매칭, kr_patents 느슨/엄격 매칭), (5) 상위 1,000 구성·동점 민감도
 를 다시 계산한다. 쓰기: verify_0/rv1_out.json 만."""
import sys, pathlib, json, sqlite3, time, re
import numpy as np, pandas as pd
from scipy.stats import rankdata
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
t0 = time.time(); R = {}
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'
PIPE = B/'수집파이프라인'; OUT = pathlib.Path(__file__).resolve().parent
ro = lambda p: sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
rg = ro(PIPE/'registry_full.sqlite'); br = ro(PIPE/'business_rights.sqlite')
def d8(x):
    s = pd.Series(x).fillna('').astype(str).str.replace('-', '').str.replace('.', '').str.strip().str[:8]
    return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).astype(np.int64).values

# (1) applicant_query vs registry
aq = pd.read_sql("select applicant_cd, applicant_name, status from applicant_query", br)
regc = pd.read_sql("select distinct applicantCd c from applicant", rg).c
reg_ne = set(regc[regc.notna() & (regc.astype(str) != '')].astype(str))
aqs = set(aq.applicant_cd.astype(str))
R['aq_rows'] = len(aq); R['aq_distinct'] = len(aqs); R['aq_null_or_empty'] = int((aq.applicant_cd.isna() | (aq.applicant_cd.astype(str) == '')).sum())
R['reg_distinct_nonempty'] = len(reg_ne); R['reg_has_null_or_empty'] = bool(regc.isna().any() or (regc.astype(str) == '').any())
R['reg_minus_aq'] = len(reg_ne - aqs); R['aq_minus_reg'] = len(aqs - reg_ne); R['aq_status'] = aq.status.value_counts().to_dict()

# (2) 공식 신규진입자
o = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'newc', 'y_u', 'prior_any', 'px', 'pl'])
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
R['o_rows'] = len(o); R['te_rows'] = len(te)
R['o_te_align'] = bool((o.firm.values == te.firm.values).all() and (o.k.values == te.k.values).all())
R['o_te_newc_equal'] = bool((o.newc.values.astype(bool) == te.newc.values.astype(bool)).all())
R['o_te_prior_equal'] = bool((o.prior_any.values.astype(bool) == te.prior_any.values.astype(bool)).all())
NF = set(te.firm[te.newc.values.astype(bool)]); ALLF = set(te.firm)
R['newc_firms'] = len(NF); R['all_firms'] = len(ALLF)
# 신규진입자 = 법인 전체 행이 newc 인지(법인 단위 일관성)
g_ = te.groupby('firm').newc.agg(['min', 'max']); R['newc_firm_inconsistent'] = int((g_['min'] != g_['max']).sum())
R['newc_status'] = aq[aq.applicant_cd.isin(NF)].status.value_counts().to_dict()
import pyarrow.parquet as pq
cols = pq.ParquetFile(B/'공모전데이터/작업데이터/kr_registered.parquet').schema_arrow.names
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=[cols[i] for i in (1, 2, 3, 4, 6, 17)]); kr.columns = ['ad', 'ay', 'rn', 'rd', 'pd', 'ksic']
R['kr_cols_used'] = [cols[i] for i in (1, 2, 3, 4, 6, 17)]
kr['rn'] = kr.rn.astype(str); kr['ay'] = pd.to_numeric(kr.ay, errors='coerce'); kr['rdn'] = d8(kr.rd.values); kr['pdn'] = d8(kr['pd'].values); kr['adn'] = d8(kr.ad.values)
ap = pd.read_sql("select rgst_no, applicantCd cd from applicant where applicantCd is not null", rg); ap['rgst_no'] = ap.rgst_no.astype(str)
pt = pd.read_sql("select rgst_no, applDate, rgstDate from patent", rg); pt['rgst_no'] = pt.rgst_no.astype(str)
R['registry_patents'] = len(pt); R['kr_rows'] = len(kr); R['kr_unique_rn'] = kr.rn.nunique()
R['registry_rn_not_in_kr'] = int((~pt.rgst_no.isin(set(kr.rn))).sum()); R['kr_rn_not_in_registry'] = int((~kr.rn.isin(set(pt.rgst_no))).sum())
x = ap.merge(kr[['rn', 'ay', 'rdn', 'pdn', 'adn']], left_on='rgst_no', right_on='rn', how='left').merge(pt, on='rgst_no', how='left')
x['reg_rd'] = d8(x.rgstDate.values); x['reg_ay'] = pd.to_numeric(x.applDate.astype(str).str[:4], errors='coerce')
xn = x[x.cd.isin(NF)]
R['newc_rows_without_kr'] = int(xn.rn.isna().sum()); R['newc_firms_without_registry'] = len(NF - set(xn.cd))
gn = xn.groupby('cd').agg(fay=('ay', 'min'), frd=('rdn', 'min'), fpd=('pdn', 'min'), freg_rd=('reg_rd', 'min'), freg_ay=('reg_ay', 'min'), npat=('rgst_no', 'nunique'))
R['newc_first_kr_ay'] = {int(k): int(v) for k, v in gn.fay.value_counts().sort_index().items()}
R['newc_first_kr_rgst_year'] = {int(k): int(v) for k, v in (gn.frd // 10000).value_counts().sort_index().items()}
R['newc_first_registry_rgst_year'] = {int(k): int(v) for k, v in (gn.freg_rd // 10000).value_counts().sort_index().items()}
R['newc_first_registry_applDate_year'] = {int(k): int(v) for k, v in gn.freg_ay.value_counts().sort_index().items()}
R['newc_any_kr_ay_le2019'] = int((gn.fay <= 2019).sum()); R['newc_any_rgst_le_origin'] = int(((gn.frd <= 20191231) | (gn.freg_rd <= 20191231)).sum())
R['newc_any_pub_le_origin'] = int((gn.fpd <= 20191231).sum())
R['newc_first_kr_ay_minmax'] = [float(gn.fay.min()), float(gn.fay.max())]; R['newc_first_rgst_minmax'] = [int(gn.frd.min()), int(gn.frd.max())]
# 역방향: 비신규진입자 법인은 2000–2019 범위안 출원(kr 출원연도)이 있어야 함
xi = x[x.cd.isin(ALLF - NF)]; hi = xi[(xi.ay >= 2000) & (xi.ay <= 2019)].cd.unique()
R['incumbent_firms'] = len(ALLF - NF); R['incumbent_without_2000_2019_ins'] = len((ALLF - NF) - set(hi))
R['newc_with_ay_lt2000'] = int((gn.fay < 2000).sum())

# (3) 채점쌍·양성
keep = ~te.prior_any.values.astype(bool); nc = te.newc.values.astype(bool)
L = pd.read_parquet(B/'분석/검증루프_20261012/C16/C16-T1/cache/labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
R['labels_align'] = bool((L.firm.values == te.firm.values).all() and (L.k.values == te.k.values).all())
y = L.y_u.values.astype(int); R['labels_vs_o_yu_equal'] = bool((y == o.y_u.values.astype(int)).all())
R['pairs_kept'] = int(keep.sum()); R['newc_pairs_kept'] = int((keep & nc).sum()); R['newc_pair_share'] = R['newc_pairs_kept'] / R['pairs_kept']
R['pos_kept'] = int(y[keep].sum()); R['newc_pos_kept'] = int(y[keep & nc].sum()); R['newc_pos_share'] = R['newc_pos_kept'] / R['pos_kept']

# (4) 원점 가시 증거: 별도 경로
LEG = r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|corporation|corp|incorporated|inc|limited|ltd|co"
def nz(s):
    s = str(s).lower(); s = re.sub(r"[^0-9a-z가-힣]", '', s)
    return s
def nz_leg(s):
    s = str(s).lower(); s = re.sub(LEG, '', s); return re.sub(r"[^0-9a-z가-힣]", '', s)
rgnames = pd.read_sql("select applicantCd cd, applicantName n from applicant where applicantCd is not null", rg)
names = pd.concat([aq[['applicant_cd', 'applicant_name']].rename(columns={'applicant_cd': 'cd', 'applicant_name': 'n'}), rgnames]).dropna().drop_duplicates()
okc = set(aq.applicant_cd[aq.status == 'OK']); names = names[names.cd.isin(okc)]
names['z'] = names.n.map(nz_leg); names = names[names.z.str.len() >= 2].drop_duplicates(['cd', 'z'])
nz_amb = names.groupby('z').cd.nunique()
# 엄격: 모호 이름 제외 / 느슨: 모호 이름도 해당 코드 모두에 귀속
strict = names[names.z.map(nz_amb) == 1]; z2c_s = dict(zip(strict.z, strict.cd))
z2c_l = names.groupby('z').cd.agg(list).to_dict()
alias_nf = names[names.cd.isin(NF)].groupby('cd').z.agg(set).to_dict()
# (4a) owner 표 직접: 범위안 등록특허 owner 행 중 이름이 신규진입자 별칭과 같고 이전등록일 ≤ 2019-12-31
ow = pd.read_sql("select rgst_no, ownerName, ownerRgstCsDate, ownerRgstCsReason from owner", rg)
ow['z'] = ow.ownerName.map(nz_leg); ow['d'] = d8(ow.ownerRgstCsDate.values)
ow19 = ow[ow.d <= 20191231]
zset = {}
for c, zs in alias_nf.items():
    for z in zs: zset.setdefault(z, set()).add(c)
E_own_exact = set()
for z in ow19.z.unique():
    if z in zset: E_own_exact |= zset[z]
R['E_own_alt_exact_owner_table'] = len(E_own_exact)
R['owner_rows_le2019'] = len(ow19)
# (4b) kr_patents 이름매칭(출원 ≤2019, min(공개,등록) ≤ 2019-12-31), 범위안 KSIC
kc = pq.ParquetFile(B/'공모전데이터/작업데이터/kr_patents.parquet').schema_arrow.names
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=[kc[i] for i in (2, 4, 6, 8, 17)]); kp.columns = ['ay', 'rd', 'pd', 'nm', 'ksic']; R['kp_cols_used'] = [kc[i] for i in (2, 4, 6, 8, 17)]
K50 = set(kr.ksic.astype(str)); R['kp_rows'] = len(kp); kp = kp[kp.ksic.astype(str).isin(K50)]; R['kp_rows_inscope_ksic'] = len(kp)
kp['ay'] = pd.to_numeric(kp.ay, errors='coerce'); kp['v'] = np.minimum(d8(kp['pd'].values), d8(kp.rd.values))
kv = kp[(kp.ay >= 2000) & (kp.ay <= 2019) & (kp.v <= 20191231)].copy()
kv = kv.assign(n=kv.nm.fillna('').astype(str).str.split('|')).explode('n'); kv['z'] = kv.n.map(nz_leg)
E_pub_s = set(kv.z.map(z2c_s).dropna()) & NF
E_pub_l = set(c for z in kv.z.unique() if z in z2c_l for c in z2c_l[z]) & NF
R['E_pub_alt_strict'] = len(E_pub_s); R['E_pub_alt_loose'] = len(E_pub_l)
# 비교: 작업 evid_all
ev = pd.read_parquet(OUT.parent/'cache/evid_all.parquet'); ev = ev[ev.is_newc_off]
R['evid_newc_firms'] = len(ev); R['evid_set_equals_NF'] = set(ev.firm) == NF
wE_pub = set(ev.firm[ev.E_pub]); wE_own = set(ev.firm[ev.E_own]); wN1 = set(ev.firm[ev.N1_primary])
R['work_E_pub'] = len(wE_pub); R['work_E_own'] = len(wE_own); R['work_N1'] = len(wN1); R['work_E_reg'] = int(ev.E_reg.sum())
R['E_pub_strict_vs_work'] = [len(E_pub_s - wE_pub), len(wE_pub - E_pub_s)]
R['E_own_alt_vs_work'] = [len(E_own_exact - wE_own), len(wE_own - E_own_exact)]
alt_N1 = E_pub_s | E_own_exact; alt_N1_l = E_pub_l | E_own_exact
R['alt_N1_strict'] = len(alt_N1); R['alt_N1_loose'] = len(alt_N1_l); R['alt_N1_vs_work'] = [len(alt_N1 - wN1), len(wN1 - alt_N1)]
R['work_wide_N1'] = int(ev.N1_wide.sum()); R['work_E_oos_vis'] = int(ev.E_oos_vis.sum()); R['work_E_rl_any'] = int(ev.E_rl_any.sum())
fir = te.firm.values
for nm_, S in (('work_N1', wN1), ('alt_N1_strict', alt_N1), ('alt_N1_loose', alt_N1_l)):
    m = keep & nc & np.isin(fir, list(S)); R[f'pairs_{nm_}'] = int(m.sum()); R[f'pos_{nm_}'] = int(y[m].sum())

# (5) 상위 1,000
px = o.px.values[keep]; pl = o.pl.values[keep]; yk = y[keep]; nck = nc[keep]; fk = fir[keep]
s = 0.9 * rankdata(px) / len(px) + 0.1 * rankdata(pl) / len(pl)
inv = np.isin(fk, list(NF - wN1)) & nck; inv_alt = np.isin(fk, list(NF - alt_N1)) & nck
ordr = np.argsort(-s, kind='stable')
for K_ in (100, 1000):
    t = ordr[:K_]; thr = s[ordr[K_ - 1]]
    R[f'top{K_}'] = dict(hits=int(yk[t].sum()), newc_rows=int(nck[t].sum()), newc_hits=int((yk[t] & nck[t]).sum()), invis_newc_hits=int((yk[t] & inv[t]).sum()),
                         invis_alt_newc_hits=int((yk[t] & inv_alt[t]).sum()), ties_at_boundary=int((s == thr).sum()), n_above=int((s > thr).sum()))
# 원점 비가시 = '범위안 기록 증거 없음' 정의일 뿐인지: 원점 가시 범위밖 등록이 있는 법인 수(작업 E_oos_vis)
R['secs'] = round(time.time() - t0)
json.dump(R, open(OUT/'rv1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v))
print(json.dumps(R, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
