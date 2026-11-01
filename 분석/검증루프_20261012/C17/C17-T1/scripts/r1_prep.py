# -*- coding: utf-8 -*-
"""R1 준비(PREREG §2 R1): RC:001 법인 코드(1·2·3·5)를 OK 법인과 같은 규칙으로 되살린 원천 행.
 → cache/rc_ins.parquet(ins_firm과 같은 열), cache/rc_app.parquet(app_pub과 같은 열, RC 코드에 걸린 행만), cache/rc_fm.parquet(firm_meta 열),
   cache/rc_s1.pkl(S1판 RC Call·C·Fvis·held), cache/r1_prep.json(자기 점검: OK만 별칭으로 다시 돌린 매칭이 공식 app_pub과 같은지 포함)"""
import sys, pathlib, re, pickle; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
out = {}
firms = pd.read_parquet(CACHE/'firms.parquet'); OK = firms[firms.status == 'OK']
RCc = firms[(firms.status == 'RC:001') & firms.applicant_cd.str[0].isin(list('1235'))]; RC = set(RCc.applicant_cd)
app = pd.read_parquet(CACHE/'ins_applicant.parquet'); kr = pd.read_parquet(CACHE/'ins_pat.parquet')
# (1) 범위안 원출원인 행 — prep2.py:52-53과 같은 식
ins = app[app.applicantCd.isin(RC)][['rgst_no', 'applicantCd']].rename(columns={'applicantCd': 'applicant_cd'})
ins = ins.merge(kr[['rgst_no', 'ksic', 'ay', 'appl_date', 'rgst_date', 'title', 'mipc']], on='rgst_no').drop_duplicates(['rgst_no', 'applicant_cd'])
off = pd.read_parquet(CACHE/'ins_firm.parquet', columns=None)
assert list(ins.columns) == list(off.columns), (list(ins.columns), list(off.columns))
for c in off.columns: ins[c] = ins[c].astype(off[c].dtype)
ins.to_parquet(CC/'rc_ins.parquet', index=False)
out['rc_ins_rows'] = int(len(ins)); out['rc_ins_firms'] = int(ins.applicant_cd.nunique()); out['rc_ins_overlap_rgst_with_OK'] = int(ins.rgst_no.isin(set(off.rgst_no)).sum())
# (2) 이름매칭 출원 — prep_app.py와 같은 식, 별칭 집합만 OK∪RC
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
k = pd.read_parquet(KP, columns=['출원번호', '출원일', '출원연도', '등록번호', '등록일자', '공개일자', '출원인명', '세세분류코드'])
k.columns = ['appl_no', 'appl_date', 'ay', 'rgst_no', 'rgst_date', 'pub_date', 'names', 'ksic']
k['ksic'] = k.ksic.astype(str); k['reg'] = k.rgst_no.fillna('').astype(str).str.len() > 3
k['nm'] = k.names.fillna('').str.split('|'); k = k.explode('nm'); k['nn'] = k.nm.map(norm)


def match(fset):
    fr_ = firms[firms.applicant_cd.isin(fset)]
    al = pd.concat([fr_[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                    app[app.applicantCd.isin(fset)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
    al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
    amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]
    n2c = dict(zip(al.n, al.applicant_cd))
    kk = k.assign(applicant_cd=k.nn.map(n2c))
    m = kk.dropna(subset=['applicant_cd']).drop_duplicates(['appl_no', 'applicant_cd'])
    return m[['applicant_cd', 'appl_no', 'ay', 'appl_date', 'pub_date', 'rgst_date', 'reg', 'ksic']]


ap_off = pd.read_parquet(CACHE/'app_pub.parquet')
m_ok = match(set(OK.applicant_cd))
key = lambda d: set(zip(d.applicant_cd, d.appl_no.astype(str)))
out['selfcheck_OKonly_rows'] = int(len(m_ok)); out['official_app_pub_rows'] = int(len(ap_off)); out['selfcheck_eq_official'] = bool(key(m_ok) == key(ap_off))
m_all = match(set(OK.applicant_cd) | RC)
rc_app = m_all[m_all.applicant_cd.isin(RC)].copy()
for c in ap_off.columns: rc_app[c] = rc_app[c].astype(ap_off[c].dtype)
rc_app.to_parquet(CC/'rc_app.parquet', index=False)
lost = key(ap_off) - key(m_all[~m_all.applicant_cd.isin(RC)])
out['rc_app_rows'] = int(len(rc_app)); out['rc_app_firms'] = int(rc_app.applicant_cd.nunique())
out['ok_rows_that_would_turn_ambiguous'] = len(lost); out['ok_rows_ambiguous_share'] = len(lost) / len(ap_off)
# (3) 법인 유형: OK 법인의 코드 첫 자리별 최빈(비결측) 유형
fm = pd.read_parquet(CACHE/'firm_meta.parquet'); fm['p'] = fm.applicant_cd.str[0]
mode = fm.dropna(subset=['tp']).groupby('p').tp.agg(lambda x: x.value_counts().index[0]).to_dict()
rf = RCc.copy(); rf['tp'] = rf.applicant_cd.str[0].map(mode); rf['corp_rgstno'] = None
rf = rf[[c for c in fm.columns if c != 'p']]
for c in rf.columns:
    if c in ('n_in_fusion', 'total_count'): rf[c] = pd.to_numeric(rf[c], errors='coerce').fillna(0).astype(fm[c].dtype)
rf.to_parquet(CC/'rc_fm.parquet', index=False); out['tp_mode_by_prefix'] = mode
# (4) S1판(v_pub) RC 위험쌍 — C16 popdef와 같은 규칙(범위밖 없음)
K, ki = kidx()
kp = pd.read_parquet(KR, columns=['등록번호', '공개일자']); km = dict(zip(kp['등록번호'].astype(str), kp['공개일자'].astype(str)))
x = ins[ins.ay >= 2000].copy(); x['v_pub'] = np.minimum(s8(x.rgst_date.values), s8(x.rgst_no.astype(str).map(km).values))
ip = x[x.ay <= CUT].drop_duplicates(['applicant_cd', 'rgst_no'])
c_off = ip.groupby('applicant_cd').size(); Foff = set(c_off[c_off >= 3].index)
c_vis = ip[ip.v_pub <= CAPV].groupby('applicant_cd').size(); Fvis = set(c_vis[c_vis >= 3].index)
ipo = ip[ip.applicant_cd.isin(Foff)].assign(k=lambda d: d.ksic.map(ki))
held_off = set(zip(ipo.applicant_cd, ipo.k)); held_vis = set(zip(ipo[ipo.v_pub <= CAPV].applicant_cd, ipo[ipo.v_pub <= CAPV].k))
a2 = pd.concat([ap_off[ap_off.applicant_cd.isin(Foff)], rc_app[rc_app.applicant_cd.isin(Foff)]]); a2 = a2[a2.ay >= 2000]
a2['k'] = a2.ksic.astype(str).map(ki); a2 = a2.dropna(subset=['k']); a2['k'] = a2.k.astype(int); a2['v'] = np.minimum(s8(a2.pub_date.values), s8(a2.rgst_date.values))
apr = a2[a2.ay <= CUT]; prior_all = set(zip(apr.applicant_cd, apr.k)); prior_vis = set(zip(apr[apr.v <= CAPV].applicant_cd, apr[apr.v <= CAPV].k))
Call = sorted((f, kk) for (f, kk) in held_off - held_vis if f in Fvis)
C = [(f, kk) for (f, kk) in Call if (f, kk) not in prior_vis]
vis_any = set(ipo[ipo.v_pub <= CAPV].applicant_cd)
pickle.dump(dict(Foff=sorted(Foff), Fvis=sorted(Fvis), held_off=sorted(held_off), held_vis=sorted(held_vis), prior_all=sorted(prior_all), prior_vis=sorted(prior_vis), Call=Call, C=C, vis_any=sorted(vis_any)), open(CC/'rc_s1.pkl', 'wb'))
out['rc_Foff'] = len(Foff); out['rc_Fvis'] = len(Fvis); out['rc_off_riskpairs'] = int(len(Foff) * 50 - len(held_off)); out['rc_off_prior_any_pairs'] = int(sum(1 for f in Foff for kk in range(50) if (f, kk) not in held_off and (f, kk) in prior_all))
out['rc_s1_Call'] = len(Call); out['rc_s1_C'] = len(C); out['mem_peak'] = G.rss_peak()
jdump(out, CC/'r1_prep.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
