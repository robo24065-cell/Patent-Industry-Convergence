# (e)-P8 보안과제 누락분(공식−수집, 수행주체별 비율)로 재가중 시 N6·N9 개발 구간 양성률·KSIC 순위 변화, N7 보안 경로 첫 진입 누락 시나리오
# 개발 원점만 사용: N9 t0 2014–2020, N6 t0 2014–2019, N7 t0 2015–2020. 공식 성적 불변.
import os, re, sys, json
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
EXT = os.path.join(MC, r"_shared\ext"); EXTF = EXT.replace("\\", "/"); R3I = os.path.join(MC, "_ideas_R3")
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
Pf = pd.read_csv(os.path.join(OUT, '_tab_a_performer.csv'))
R = Pf.groupby(['year', 'performer'])[['official_eok', 'collected_eok']].sum()
R['r'] = R.official_eok / R.collected_eok
rr = R.r.unstack()
res = {}
# ---------- N9
D = pd.read_parquet(os.path.join(MC, r"N9_정부RnD과제_특허성과예측\cache\panel_dev.parquet"), columns=['t0', 'y', 'perform_grp', 'top_ksic'])
D = D[D.t0.between(2014, 2020)].copy()
# panel perform_grp는 build_raw 오표기 그대로: '중견기업'=실제 정부부처(06), '기타'=기타+실제 중견기업(08)
mp = {'국공립연': '국공립연구소', '출연연': '출연연구소', '대학': '대학', '대기업': '대기업', '중소기업': '중소기업', '중견기업': '정부부처'}
def w9(row_t0, g):
    if g in mp: return rr.loc[row_t0, mp[g]]
    o = R.loc[(row_t0, ['기타', '중견기업']), ['official_eok', 'collected_eok']].sum(); return o.official_eok / o.collected_eok
wt = {(t, g): w9(t, g) for t in D.t0.unique() for g in D.perform_grp.astype(str).unique()}
D['w'] = [wt[(t, g)] for t, g in zip(D.t0, D.perform_grp.astype(str))]
base = D.y.mean(); rw = np.average(D.y, weights=D.w)
k = D.groupby('top_ksic').apply(lambda g: pd.Series({'n': len(g), 'rate': g.y.mean(), 'rate_w': np.average(g.y, weights=g.w)}), include_groups=False)
k = k[k.n >= 200]
rho9 = spearmanr(k.rate, k.rate_w).statistic
res['N9'] = dict(dev_rate=float(base), dev_rate_reweighted=float(rw), rel_change=float(rw / base - 1), ksic_n=int(len(k)), ksic_spearman=float(rho9),
                 max_ksic_rel_change=float((k.rate_w / k.rate - 1).abs().max()), weights_range=[float(D.w.min()), float(D.w.max())])
print('N9', res['N9'])
# ---------- N6
PN = pd.read_parquet(os.path.join(MC, r"N6_정부RnD기업_첫특허전환\cache\panel_dev.parquet"), columns=['bz', 't0', 'y', 'sme'])
PN = PN[PN.t0.between(2014, 2019)].copy()
def w6(t0, sme):
    big = R.loc[(t0, ['대기업', '중견기업']), ['official_eok', 'collected_eok']].sum()
    return sme * rr.loc[t0, '중소기업'] + (1 - sme) * (big.official_eok / big.collected_eok)
PN['w'] = [w6(t, s) for t, s in zip(PN.t0, PN.sme.fillna(1))]
b6 = PN.y.mean(); r6 = np.average(PN.y, weights=PN.w)
# 양성 기업의 첫 결과 출원 KSIC(최빈)
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno"])
b2a = ns.dropna().groupby('bizno').applicantCd.apply(list)
af = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "appl_date", "ksic"]); af = af[af.appl_date < 20220101]
pos = PN[PN.y == 1][['bz', 't0', 'w']].copy()
pos['codes'] = pos.bz.map(b2a)
pe = pos.explode('codes').dropna(subset=['codes']).merge(af, left_on='codes', right_on='applicantCd')
pe = pe[(pe.appl_date >= (pe.t0 + 1) * 10000 + 101) & (pe.appl_date <= (pe.t0 + 2) * 10000 + 1231)]
pk = pe.groupby(['bz', 't0']).ksic.agg(lambda s: s.value_counts().index[0]).rename('ksic').reset_index().merge(pos[['bz', 't0', 'w']], on=['bz', 't0'])
kc = pk.groupby('ksic').agg(n=('w', 'size'), nw=('w', 'sum'))
kc = kc[kc.n >= 10]
rho6 = spearmanr(kc.n, kc.nw).statistic
res['N6'] = dict(dev_rate=float(b6), dev_rate_reweighted=float(r6), rel_change=float(r6 / b6 - 1), pos_ksic_covered=float(len(pk) / len(pos)), ksic_n=int(len(kc)),
                 ksic_count_spearman=float(rho6), share_rel_change_max=float(((kc.nw / kc.nw.sum()) / (kc.n / kc.n.sum()) - 1).abs().max()))
print('N6', res['N6'])
# ---------- N7: 보안과제 경로 첫 진입 누락 시나리오
nt = duckdb.sql(f"""select year, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, perform_agent, ministry, gov_funds from read_parquet('{EXTF}/ntis_project_raw.parquet')
  where year between 2010 and 2022 and secret_flag = false""").df()
nt = nt[nt.bz.str.len() == 10]
first = nt.groupby('bz').year.min()
nt['is_first_year'] = nt.year == nt.bz.map(first)
firmtypes = ['대기업', '중견기업', '중소기업']
sc = []
for y in range(2016, 2023):
    w = nt[(nt.year == y) & nt.perform_agent.isin(firmtypes)]
    for g in firmtypes:
        ww = w[w.perform_agent == g]
        if len(ww) == 0: continue
        avg_f = ww.gov_funds.mean() / 1e8
        miss = max(R.loc[(y, g), 'official_eok'] - R.loc[(y, g), 'collected_eok'], 0)
        est_proj = miss / avg_f if avg_f > 0 else 0
        firms = ww.bz.nunique(); new_firms = ww[ww.is_first_year].bz.nunique()
        proj_per_firm = len(ww) / firms
        est_firms = est_proj / proj_per_firm
        sc.append(dict(year=y, performer=g, missing_eok=miss, avg_funds_eok=avg_f, est_secret_projects=est_proj, est_secret_firms=est_firms,
                       new_firm_share=new_firms / firms, est_secret_new_firms=est_firms * new_firms / firms, observed_new_firms=new_firms))
S = pd.DataFrame(sc)
print(S.groupby('year')[['est_secret_projects', 'est_secret_firms', 'est_secret_new_firms', 'observed_new_firms']].sum().round(1))
# N7 양성과 전체 신규 진입 기업 대비 비율로 환산
P7 = pd.read_parquet(os.path.join(MC, r"N7_특허기업_정부RnD첫수행\cache\dev_panel.parquet"), columns=['applicantCd', 't0', 'y', 'bizno', 'first_ntis'])
P7 = P7[P7.t0.between(2015, 2020)]
newall = nt[nt.is_first_year].groupby('year').bz.nunique()      # 모든 주체 포함 신규 사업자번호(비보안)
pos7 = P7[P7.y == 1].groupby('t0').size()
rows = []
Sy = S.groupby('year').est_secret_new_firms.sum()
for t0 in range(2015, 2021):
    yrs = [t0 + 1, t0 + 2]
    new_obs = newall.reindex(yrs).sum(); est_sec = Sy.reindex(yrs).fillna(Sy.mean()).sum()
    capture = pos7.get(t0, 0) / new_obs      # 신규 진입 기업 중 N7 양성으로 잡히는 비율
    est_missing_pos = est_sec * capture
    rows.append(dict(t0=t0, n7_pos=int(pos7.get(t0, 0)), new_firms_obs=int(new_obs), est_secret_new_firms=float(est_sec), capture=float(capture),
                     est_missing_pos=float(est_missing_pos), missing_share=float(est_missing_pos / (pos7.get(t0, 0) + est_missing_pos))))
M7 = pd.DataFrame(rows); print(M7.round(4).to_string())
# 상한 시나리오: 보안과제 수행 기업이 전부 첫 진입이라고 가정
S['est_secret_firms_all_new'] = S.est_secret_firms
up = []
for t0 in range(2015, 2021):
    yrs = [t0 + 1, t0 + 2]; new_obs = newall.reindex(yrs).sum(); capture = pos7.get(t0, 0) / new_obs
    e = S[S.year.isin(yrs)].est_secret_firms.sum() * capture
    up.append(e / (pos7.get(t0, 0) + e))
# KSIC: 국방 부처(방위사업청·국방부) 비보안 과제로 첫 진입한 N7 양성의 KSIC 분포에 누락 양성을 배분
af1 = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "known_date", "ksic"]); af1 = af1[af1.known_date.notna()]
mk = []
for t0 in range(2015, 2021):
    a = af1[af1.known_date <= t0 * 10000 + 1231]
    m = a[a.applicantCd.isin(set(P7[P7.t0 == t0].applicantCd))].groupby('applicantCd').ksic.agg(lambda s: s.value_counts().index[0])
    mk.append(P7[P7.t0 == t0].assign(ksic=lambda d: d.applicantCd.map(m)))
P7k = pd.concat(mk)
fy = nt.sort_values('year').drop_duplicates('bz')[['bz', 'ministry', 'perform_agent']]
P7k['bz'] = P7k.bizno.astype(str).str.replace(r'\D', '', regex=True)
P7k = P7k.merge(fy, on='bz', how='left')
defpos = P7k[(P7k.y == 1) & P7k.ministry.astype(str).isin(['방위사업청', '국방부'])]
if len(defpos) == 0:   # 국방 부처 비보안 과제가 없으므로 대체 근사: 첫 과제가 대기업·중견기업 수행인 양성
    defpos = P7k[(P7k.y == 1) & P7k.perform_agent.isin(['대기업', '중견기업'])]
kk = P7k.groupby('ksic').agg(n=('y', 'size'), pos=('y', 'sum'))
dshare = defpos.ksic.value_counts(normalize=True)
tot_missing = M7.est_missing_pos.sum()
kk['pos_adj'] = kk.pos + tot_missing * dshare.reindex(kk.index).fillna(0)
kk = kk[kk.n >= 100]
rho7 = spearmanr(kk.pos / kk.n, kk.pos_adj / kk.n).statistic
res['N7'] = dict(dev_pos_total=int(P7.y.sum()), est_missing_pos_total=float(tot_missing), missing_share_mean=float(M7.missing_share.mean()),
                 missing_share_upper_all_new=float(np.mean(up)), n_def_ministry_first_pos=int(len(defpos)), ksic_n=int(len(kk)), ksic_rate_spearman=float(rho7),
                 ksic_max_rel_change=float(((kk.pos_adj / kk.pos.clip(lower=1)) - 1).max()), top_def_ksic=dshare.head(5).round(3).to_dict())
print('N7', res['N7'])
S.to_csv(os.path.join(OUT, '_tab_e_n7_secret_scenario.csv'), index=False, encoding='utf-8-sig')
json.dump(res, open(os.path.join(OUT, '_tab_e_reweight.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
