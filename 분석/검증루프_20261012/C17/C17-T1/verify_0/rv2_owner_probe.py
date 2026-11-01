# -*- coding: utf-8 -*-
"""rv2: rv1에서 owner 표 직접 매칭으로 늘어난 신규진입자 법인(작업 E_own 밖)의 성격 점검 → rv2_out.json"""
import sys, pathlib, json, sqlite3, re
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'; PIPE = B/'수집파이프라인'; OUT = pathlib.Path(__file__).resolve().parent
ro = lambda p: sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True); rg = ro(PIPE/'registry_full.sqlite'); br = ro(PIPE/'business_rights.sqlite')
def d8(x):
    s = pd.Series(x).fillna('').astype(str).str.replace('-', '').str.strip().str[:8]
    return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).astype(np.int64).values
LEG = r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|corporation|corp|incorporated|inc|limited|ltd|co"
nz = lambda s: re.sub(r"[^0-9a-z가-힣]", '', re.sub(LEG, '', str(s).lower()))
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any']); NF = set(te.firm[te.newc.values])
aq = pd.read_sql("select applicant_cd cd, applicant_name n, status from applicant_query", br)
rgn = pd.read_sql("select applicantCd cd, applicantName n from applicant where applicantCd is not null", rg)
names = pd.concat([aq[aq.status == 'OK'][['cd', 'n']], rgn[rgn.cd.isin(set(aq.cd[aq.status == 'OK']))]]).dropna().drop_duplicates()
names['z'] = names.n.map(nz); names = names[names.z.str.len() >= 2].drop_duplicates(['cd', 'z'])
amb = names.groupby('z').cd.nunique()
al = names[names.cd.isin(NF)]
z2 = {}
for c, z in zip(al.cd, al.z): z2.setdefault(z, set()).add(c)
ow = pd.read_sql("select rgst_no, ownerName, ownerRgstCsDate, ownerRgstCsReason, finalOwnerYn from owner", rg)
ow['z'] = ow.ownerName.map(nz); ow['d'] = d8(ow.ownerRgstCsDate.values)
m = ow[ow.z.isin(z2.keys()) & (ow.d <= 20191231)].copy()
m['cds'] = m.z.map(z2); m = m.explode('cds')
ev = pd.read_parquet(OUT.parent/'cache/evid_all.parquet'); ev = ev[ev.is_newc_off]; wown = set(ev.firm[ev.E_own]); wN1 = set(ev.firm[ev.N1_primary])
extra = m[~m.cds.isin(wown)]
R = {}
R['extra_firms'] = int(extra.cds.nunique()); R['extra_firms_not_in_N1'] = int(len(set(extra.cds) - wN1))
R['extra_reason'] = extra.ownerRgstCsReason.value_counts().head(10).to_dict()
R['extra_name_len_dist'] = extra.drop_duplicates('cds').z.str.len().describe().round(1).to_dict()
R['extra_name_ambiguous_firms'] = int(extra[extra.z.map(amb).fillna(1) > 1].cds.nunique())
R['extra_rows'] = len(extra); R['extra_patents'] = int(extra.rgst_no.nunique())
# 이름 길이 ≥4·비모호 로만
ex2 = extra[(extra.z.str.len() >= 4) & (extra.z.map(amb).fillna(1) == 1)]
R['extra_firms_len4_unamb'] = int(ex2.cds.nunique()); R['extra_firms_len4_unamb_not_in_N1'] = int(len(set(ex2.cds) - wN1))
# 그 특허의 원출원인 코드가 이 법인인지(원출원인이면 E_reg인데 0이어야 함) / 원출원인이 다른 법인인지
ap = pd.read_sql("select rgst_no, applicantCd from applicant", rg)
ex2 = ex2.merge(ap.groupby('rgst_no').applicantCd.agg(set).rename('apps').reset_index(), on='rgst_no', how='left')
R['ex2_rows_where_firm_is_original_applicant'] = int(sum(c in (a if isinstance(a, set) else set()) for c, a in zip(ex2.cds, ex2.apps)))
# right_list에 해당 (법인, 특허)가 있는지 — 없으면 2026 이전에 이전해 나간 것으로 해석
rl = pd.read_sql(f"select applicant_cd, rgst_no from right_list where applicant_cd in ({','.join('?'*len(set(ex2.cds)))})", br, params=sorted(set(ex2.cds)))
rls = set(zip(rl.applicant_cd, rl.rgst_no.astype(str)))
ex2['in_rl'] = [(c, str(r)) in rls for c, r in zip(ex2.cds, ex2.rgst_no)]
R['ex2_rows_in_right_list'] = int(ex2.in_rl.sum()); R['ex2_rows'] = len(ex2)
R['ex2_finalOwner'] = ex2.finalOwnerYn.value_counts().to_dict()
# 법인이 나중에 그 특허를 넘겼는지(같은 특허에 더 늦은 owner 행 존재)
lat = ow.groupby('rgst_no').d.max(); ex2['later_owner'] = ex2.rgst_no.map(lat) > ex2.d
R['ex2_rows_with_later_owner'] = int(ex2.later_owner.sum())
# 표본 20건(이름 정규화형·사유·연도만; 원자료 외부 전송 없음, 로컬 기록)
smp = ex2.drop_duplicates('cds').head(20)
R['sample'] = [dict(cd=c, owner_norm=z, n_alias=sorted(z2c for z2c, cs in z2.items() if c in cs)[:3], reason=r, year=int(d // 10000), in_rl=bool(i)) for c, z, r, d, i in zip(smp.cds, smp.z, smp.ownerRgstCsReason, smp.d, smp.in_rl)]
# 채점쌍 영향
keep = ~te.prior_any.values; nc = te.newc.values
L = pd.read_parquet(B/'분석/검증루프_20261012/C16/C16-T1/cache/labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True); y = L.y_u.values.astype(int)
S = wN1 | set(ex2.cds); msk = keep & nc & te.firm.isin(S).values
R['N1_plus_ex2_firms'] = len(S); R['N1_plus_ex2_pairs'] = int(msk.sum()); R['N1_plus_ex2_pos'] = int(y[msk].sum())
json.dump(R, open(OUT/'rv2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str); print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
