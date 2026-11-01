# -*- coding: utf-8 -*-
"""레이블·가짜 법인 준비(PREREG §2). 트랙 r3.reg_lab·pub_lab과 같은 정의를 직접 병합으로 계산하고, 공식 행에서 공식 레이블과 같은지 먼저 대조한다.
 → cache/lab_rc.parquet(RC 법인 공식 정의 위험쌍: firm,k,prior_any,y_reg,y_pub,y_u), cache/lab_rc_s1.parquet(RC S1판 행·L-a),
   cache/pseudo_s{1,2,3}.parquet, cache/pseudo_x3.parquet(가짜 법인 코드·이름·레이블 쌍), cache/lab17.json"""
import sys, pathlib, re, pickle; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
out = {}; K, ki = kidx(); ENDS = '20250822'


def reg_pos(ins):
    e = ins[(ins.ay >= 2020) & (ins.ay <= 2023)][['applicant_cd', 'ksic']].drop_duplicates(); return set(zip(e.applicant_cd, e.ksic.map(ki).astype(int)))


def app_k(a):
    a = a.copy(); a['k'] = a.ksic.astype(str).map(ki); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
    a['pub_date'] = a.pub_date.fillna('').astype(str); a['rgst_date'] = a.rgst_date.fillna('').astype(str); return a[a.ay >= 2000]


def pub_pos(a):
    vis = ((a.pub_date.str.len() == 8) & (a.pub_date <= ENDS)) | ((a.rgst_date.str.len() == 8) & (a.rgst_date <= ENDS))
    e = a[vis & (a.ay >= 2020) & (a.ay <= 2023)]; return set(zip(e.applicant_cd, e.k))


def prior_set(a):
    e = a[a.ay <= CUT]; return set(zip(e.applicant_cd, e.k))


# (0) 공식 행 대조
O = official_frame()
ins0 = pd.read_parquet(CACHE/'ins_firm.parquet'); ins0 = ins0[ins0.ay >= 2000]; a0 = app_k(pd.read_parquet(CACHE/'app_pub.parquet'))
R0, P0s, Q0 = reg_pos(ins0), pub_pos(a0), prior_set(a0)
kk = list(zip(O.firm, O.k))
yr = np.array([x in R0 for x in kk], np.int8); yp = np.array([x in P0s for x in kk], np.int8); pa = np.array([x in Q0 for x in kk])
out['official_check'] = dict(y_reg_eq=bool((yr == O.y_reg.values).all()), y_pub_eq=bool((yp == O.y_pub.values).all()), prior_any_eq=bool((pa == O.prior_any.values).all()))
assert all(out['official_check'].values()), out['official_check']
# (1) RC 법인 공식 정의 위험쌍
rci = pd.read_parquet(CC/'rc_ins.parquet'); rci = rci[rci.ay >= 2000]; rca = app_k(pd.read_parquet(CC/'rc_app.parquet'))
RC3 = set(pd.read_parquet(CC/'rc_firms.parquet').firm)
held = set(zip(rci[rci.ay <= CUT].applicant_cd, rci[rci.ay <= CUT].ksic.map(ki).astype(int)))
rows = [(f, k) for f in sorted(RC3) for k in range(50) if (f, k) not in held]
L = pd.DataFrame(rows, columns=['firm', 'k']); Rr, Pr, Qr = reg_pos(rci), pub_pos(rca), prior_set(rca)
kk = list(zip(L.firm, L.k)); L['prior_any'] = [x in Qr for x in kk]; L['y_reg'] = np.array([x in Rr for x in kk], np.int8); L['y_pub'] = np.array([x in Pr for x in kk], np.int8)
L['y_u'] = np.maximum(L.y_reg, L.y_pub).astype(np.int8); L.to_parquet(CC/'lab_rc.parquet', index=False)
Lk = L[~L.prior_any]
out['rc'] = dict(firms=len(RC3), pairs=int(len(L)), prior_any=int(L.prior_any.sum()), scored=int(len(Lk)), pos=int(Lk.y_u.sum()), pos_reg=int(Lk.y_reg.sum()), pos_pub=int(Lk.y_pub.sum()),
                 base_rate=float(Lk.y_u.mean()), firms_with_pos=int(Lk[Lk.y_u == 1].firm.nunique()))
# (2) RC S1판(L-a): 가시 법인, 가시 보유 없음·가시 선행 없음. 숨은 보유(C형)·숨은 선행(B형)은 0
S = pickle.load(open(CC/'rc_s1.pkl', 'rb')); Fv = set(S['Fvis']); hv = set(map(tuple, S['held_vis'])); ho = set(map(tuple, S['held_off'])); pv = set(map(tuple, S['prior_vis'])); pall = set(map(tuple, S['prior_all']))
rs = [(f, k) for f in sorted(Fv) for k in range(50) if (f, k) not in hv and (f, k) not in pv]
L1 = pd.DataFrame(rs, columns=['firm', 'k']); kk = list(zip(L1.firm, L1.k))
L1['typ'] = ['C' if x in ho else ('B' if x in pall else 'A') for x in kk]
yu = dict(zip(zip(L.firm, L.k), L.y_u))
L1['y'] = [int(yu.get(x, 0)) if t == 'A' else 0 for x, t in zip(kk, L1.typ)]
L1['newc_s1'] = ~L1.firm.isin(set(S['vis_any'])); L1.to_parquet(CC/'lab_rc_s1.parquet', index=False)
out['rc_s1'] = dict(firms=int(L1.firm.nunique()), rows=int(len(L1)), typ=L1.typ.value_counts().to_dict(), pos=int(L1.y.sum()))
# (3) 가짜 법인(N3): CA 법인 코드 ∉ applicant_query, 국민연금 2019-12 가입자>0
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
ca = pd.read_parquet(CA); ca.columns = ['applicant_cd', 'name', 'jurir_raw', 'bizr_raw', 'nm', 'jurir', 'bizr']; ca['applicant_cd'] = ca.applicant_cd.astype(str)
aq = pd.read_parquet(CACHE/'firms.parquet', columns=['applicant_cd', 'applicant_name']); AQ = set(aq.applicant_cd)
ca1 = ca[ca.applicant_cd.str[0].isin(list('1235'))].drop_duplicates('applicant_cd')
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp', 'indcd']); P19 = P[(P.snap == '2019-12') & (P.n_emp.fillna(0) > 0)].drop_duplicates('applicantCd').set_index('applicantCd')
frame = ca1[~ca1.applicant_cd.isin(AQ) & ca1.applicant_cd.isin(P19.index)].sort_values('applicant_cd').reset_index(drop=True)
# 이름 유일성: CA 이름 + applicant_query 이름(OK·RC) + 등록원부 이름 전체에서 정규화 이름이 코드 하나에만 걸릴 때
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['applicantCd', 'applicantName'])
al = pd.concat([ca1[['applicant_cd', 'name']].rename(columns={'name': 'nm'}), aq.rename(columns={'applicant_name': 'nm'}), app.rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n']); amb = al.groupby('n').applicant_cd.nunique()
frame['n'] = frame.name.map(norm); frame['uniq'] = frame.n.map(amb).fillna(0).eq(1) & (frame.n.str.len() >= 2)
k = pd.read_parquet(KP, columns=['출원연도', '등록일자', '공개일자', '출원인명', '세세분류코드']); k.columns = ['ay', 'rgst_date', 'pub_date', 'names', 'ksic']
k['nm'] = k.names.fillna('').str.split('|'); k = k.explode('nm'); k['n'] = k.nm.map(norm)
fn = frame[frame.uniq]; n2c = dict(zip(fn.n, fn.applicant_cd))
k['applicant_cd'] = k.n.map(n2c); km = app_k(k.dropna(subset=['applicant_cd'])[['applicant_cd', 'ay', 'rgst_date', 'pub_date', 'ksic']])
PP_, QQ_ = pub_pos(km), prior_set(km)
out['n3_frame'] = dict(frame=int(len(frame)), name_unique=int(frame.uniq.sum()), firms_with_inscope_app=int(km.applicant_cd.nunique()),
                       firms_with_pos_2020_23=len(set(f for f, _ in PP_)), firms_with_prior_le2019=len(set(f for f, _ in QQ_)),
                       prefix=frame.applicant_cd.str[0].value_counts().to_dict(), emp_median=float(P19.n_emp.reindex(frame.applicant_cd).median()))
pos_by = {}; pri_by = {}
for f, kq in PP_: pos_by.setdefault(f, set()).add(kq)
for f, kq in QQ_: pri_by.setdefault(f, set()).add(kq)
for tag, s, n in [('s1', 1, 1304), ('s2', 2, 1304), ('s3', 3, 1304), ('x3', 1, 3912)]:
    rng = np.random.RandomState(20261017 + s); idx = np.sort(rng.choice(len(frame), n, replace=False))
    sub = frame.iloc[idx][['applicant_cd', 'name']].reset_index(drop=True); sub['slot'] = [f'Z{i + 1:05d}' for i in range(n)]
    sub['pos_k'] = [sorted(pos_by.get(c, set())) for c in sub.applicant_cd]; sub['prior_k'] = [sorted(pri_by.get(c, set())) for c in sub.applicant_cd]
    sub['donor_u'] = rng.random_sample(n)                # 공여 법인 선택용 균등 난수(같은 RNG 다음 값)
    sub.to_parquet(CC/f'pseudo_{tag}.parquet', index=False)
    out[f'pseudo_{tag}'] = dict(n=n, with_pos=int((sub.pos_k.str.len() > 0).sum()), pos_pairs=int(sub.pos_k.str.len().sum()), with_prior=int((sub.prior_k.str.len() > 0).sum()))
out['mem_peak'] = G.rss_peak(); jdump(out, CC/'lab17.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
