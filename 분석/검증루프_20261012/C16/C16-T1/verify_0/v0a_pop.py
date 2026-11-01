# -*- coding: utf-8 -*-
"""verify_0 (독립 경로 1/2): 트랙 캐시(ins_firm·oos·app_pub)·kr_registered·작업 스크립트를 쓰지 않고 원천 DB에서 모집단을 다시 만든다.
 범위안: registry_full.sqlite patent(ksic, applDate, rgstDate, opnDate) × applicant(applicantCd), 법인 = business_rights.applicant_query status OK
 범위밖: business_rights.sqlite right_list(권리자란, '10' 특허, 범위안 아님, 등록 ≤2025-08-22, 자사출원 = 별칭 ⊂ applicantInfo; prep2.py 정의를 새로 구현)
 이름매칭 선행: kr_patents 출원인명 → 별칭(모호 제거) 새 구현.
 가시일: 범위안 min(rgstDate, opnDate) [kr_registered 공개일자 대신 등록원부 opnDate], 범위밖 min(rgst_date, appl+18개월), 선행 min(공개일, 등록일)
 → v0_pop.pkl, v0a_out.json"""
import sys, os, re, json, pickle, sqlite3, pathlib, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"); HERE = pathlib.Path(__file__).resolve().parent
t0 = time.time()
ro = lambda rel: sqlite3.connect((B/rel).as_uri() + '?mode=ro', uri=True)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
nz = lambda s: STRIP.sub('', str(s)).lower()


def d8(x):
    s = pd.Series(x).fillna('').astype(str).str.replace('-', '').str.strip().str[:8]
    return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).astype(np.int64).values


def plus18(x):
    d = pd.to_datetime(pd.Series(x).astype(str).str.replace('-', '').str[:8], format='%Y%m%d', errors='coerce') + pd.DateOffset(months=18)
    return pd.to_numeric(d.dt.strftime('%Y%m%d'), errors='coerce').fillna(99999999).astype(np.int64).values


aq = pd.read_sql("select applicant_cd, applicant_name, status from applicant_query", ro('수집파이프라인/business_rights.sqlite'))
F = set(aq[aq.status == 'OK'].applicant_cd)
rg = ro('수집파이프라인/registry_full.sqlite')
pt = pd.read_sql('select rgst_no, ksic, applDate, rgstDate, opnDate from patent', rg)
ap = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', rg); rg.close()
pt['ay'] = pd.to_numeric(pt.applDate.astype(str).str[:4], errors='coerce')
pt['vr'] = d8(pt.rgstDate); pt['vp'] = np.minimum(pt.vr, d8(pt.opnDate))
ins = ap[ap.applicantCd.isin(F)][['rgst_no', 'applicantCd']].drop_duplicates().merge(pt, on='rgst_no')
ins = ins[(ins.ay >= 2000) & (pt.vr.max() >= 0)]
out = dict(ins_rows=int(len(ins)), ins_firms=int(ins.applicantCd.nunique()), registry_patents=int(len(pt)), rgst_max=int(pt.vr[pt.vr < 99999999].max()))
# 별칭(조회명 + 등록원부 이름)
al = pd.concat([aq[aq.status == 'OK'][['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                ap[ap.applicantCd.isin(F)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(nz); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
# 범위밖(권리목록 원천 직접)
q = """select applicant_cd, rgst_no, rgst_date, appl_date, json_extract(raw,'$.pertDiv') pert, json_extract(raw,'$.applicantInfo') ainfo
       from right_list where substr(rgst_no,1,2)='10'"""
parts = []
br = ro('수집파이프라인/business_rights.sqlite')
for ch in pd.read_sql(q, br, chunksize=400000):
    ch = ch[(ch.pert == '권리자란') & ch.applicant_cd.isin(F)]; parts.append(ch)
br.close()
rl = pd.concat(parts, ignore_index=True); insset = set(pt.rgst_no)
rl = rl[~rl.rgst_no.isin(insset)].copy(); rl['vr'] = d8(rl.rgst_date); rl = rl[rl.vr <= 20250822]
rl['ay'] = pd.to_numeric(rl.appl_date.astype(str).str[:4], errors='coerce')


def is_self(cd, ainfo):
    a = nz(ainfo)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(nz(p) == n for p in str(ainfo).split(',')): return True
    return False


rl = rl[rl.ay >= 2000]
rl['self'] = [is_self(c, a) for c, a in zip(rl.applicant_cd.values, rl.ainfo.values)]
oos = rl[rl.self].copy(); oos['vp'] = np.minimum(oos.vr.values, plus18(oos.appl_date.values))
out.update(oos_rows=int(len(oos)), secs_oos=round(time.time() - t0))
# 이름매칭 선행·공개(kr_patents 직접)
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호', '등록일자', '공개일자', '출원인명', '세세분류코드'])
amb = al.groupby('n').applicant_cd.nunique(); al1 = al[al.n.map(amb) == 1]; n2c = dict(zip(al1.n, al1.applicant_cd))
kp['nm'] = kp['출원인명'].fillna('').str.split('|'); kp = kp.explode('nm'); kp['cd'] = kp.nm.map(nz).map(n2c)
kp = kp.dropna(subset=['cd']).drop_duplicates(['출원번호', 'cd'])
kp = kp[kp['출원연도'] >= 2000]; kp['v'] = np.minimum(d8(kp['공개일자']), d8(kp['등록일자']))
K = sorted(pt.ksic.astype(str).unique()); ki = {k: i for i, k in enumerate(K)}
kp['k'] = kp['세세분류코드'].astype(str).map(ki); kp = kp.dropna(subset=['k']); kp['k'] = kp.k.astype(int)
ins['k'] = ins.ksic.astype(str).map(ki)
CUT, V = 2019, 20191231
ip = ins[ins.ay <= CUT]; op = oos[oos.ay <= CUT]


def cnt(vcol):
    i = ip if vcol is None else ip[ip[vcol] <= V]; o = op if vcol is None else op[op[vcol] <= V]
    return pd.concat([i.drop_duplicates(['applicantCd', 'rgst_no']).applicantCd, o.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd]).value_counts()


c0 = cnt(None); Foff = set(c0[c0 >= 3].index); cv = cnt('vp'); Fvis = set(cv[cv >= 3].index); cr = cnt('vr'); Fr = set(cr[cr >= 3].index)
ipo = ip[ip.applicantCd.isin(Foff)]; held_off = set(zip(ipo.applicantCd, ipo.k)); held_vis = set(zip(ipo[ipo.vp <= V].applicantCd, ipo[ipo.vp <= V].k))
held_vr = set(zip(ipo[ipo.vr <= V].applicantCd, ipo[ipo.vr <= V].k))
apr = kp[kp['출원연도'] <= CUT]; prior_all = set(zip(apr.cd, apr.k)); prior_vis = set(zip(apr[apr.v <= V].cd, apr[apr.v <= V].k))
pairs_off = len(Foff) * 50 - len(held_off)
# 공식 te 쌍과 대조할 수 있도록 집합 저장
C = sorted((f, k) for (f, k) in held_off - held_vis if f in Fvis and (f, k) not in prior_vis)
Cr = sorted((f, k) for (f, k) in held_off - held_vr if f in Fr and (f, k) not in prior_vis)
pickle.dump(dict(Foff=Foff, Fvis=Fvis, Fvr=Fr, held_off=held_off, prior_all=prior_all, prior_vis=prior_vis, C=C, C_vrgst=Cr, vis_any=set(ipo[ipo.vp <= V].applicantCd)), open(HERE/'v0_pop.pkl', 'wb'))
out.update(Foff=len(Foff), pairs_off=pairs_off, Fvis=len(Fvis), Fvr=len(Fr), C=len(C), C_vrgst=len(Cr), prior_all_pairs=len(prior_all), secs=round(time.time() - t0))
json.dump(out, open(HERE/'v0a_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(out)
