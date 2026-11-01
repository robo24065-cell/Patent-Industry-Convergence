# (c) 연결 선택 편향 SMD
#  A. 과제 단위: 제공 KR 특허(50 KSIC)와 연결된 과제 vs 비연결 과제 (과제연도 2009-2021, 비보안)
#  B. N6 모집단(개발 원점 2014-2019 재현): 사업자번호→출원인코드 연결 기업 vs 비연결 기업
#  C. N7 모집단 근원: 제공 KR 출원 국내법인(코드 1…) 중 국세청 사업자번호 연결 vs 비연결 (원점 2019, 알려진 출원 기준)
import os, re, sys, json
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
EXT = os.path.join(MC, r"_shared\ext"); EXTF = EXT.replace("\\", "/"); R3I = os.path.join(MC, "_ideas_R3")
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
def smd_cont(a, b):
    a = pd.Series(a, dtype=float).dropna(); b = pd.Series(b, dtype=float).dropna()
    s = np.sqrt((a.var() + b.var()) / 2); return float((a.mean() - b.mean()) / s) if s > 0 else 0.0
rows = []
def add(block, var, a, b, kind='cont'):
    a = pd.Series(a, dtype=float); b = pd.Series(b, dtype=float)
    rows.append(dict(block=block, variable=var, mean_linked=float(a.mean()), mean_unlinked=float(b.mean()), smd=smd_cont(a, b), n_linked=int(a.notna().sum()), n_unlinked=int(b.notna().sum())))
# ---------- A
P = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['pjt_no', 'year', 'ministry', 'perform_agent', 'secret_flag', 'dev_phase', 'region_cd', 'gov_funds', 'continuous'])
P = P[(~P.secret_flag) & P.year.between(2009, 2021)].copy()
L = pd.read_parquet(os.path.join(EXT, 'ntis_patent_link.parquet'), columns=['ProjectID'])
P['linked'] = P.pjt_no.isin(set(L.ProjectID.astype(str)))
fam = {'과학기술부': 'MSIT', '교육과학기술부': 'MSIT', '미래창조과학부': 'MSIT', '과학기술정보통신부': 'MSIT', '교육인적자원부': 'EDU', '교육부': 'EDU',
       '산업자원부': 'MOTIE', '지식경제부': 'MOTIE', '산업통상자원부': 'MOTIE', '정보통신부': 'MSIT', '방송통신위원회': 'MSIT', '중소기업청': 'SME', '중소벤처기업부': 'SME',
       '보건복지부': 'HEALTH', '보건복지가족부': 'HEALTH', '방위사업청': 'DEFENSE', '국방부': 'DEFENSE'}
P['mfam'] = P.ministry.astype(str).map(fam).fillna('OTHER')
a, b = P[P.linked], P[~P.linked]
print('A linked share', P.linked.mean(), 'by year', P.groupby('year').linked.mean().round(4).to_dict())
add('A_project', 'log_gov_funds', np.log1p(a.gov_funds.fillna(0)), np.log1p(b.gov_funds.fillna(0)))
add('A_project', 'year', a.year, b.year)
for g in ['대학', '출연연구소', '국공립연구소', '대기업', '중견기업', '중소기업', '정부부처', '기타']:
    add('A_project', f'performer={g}', a.perform_agent == g, b.perform_agent == g)
for g in ['MSIT', 'MOTIE', 'EDU', 'SME', 'HEALTH', 'DEFENSE', 'OTHER']:
    add('A_project', f'ministry_family={g}', a.mfam == g, b.mfam == g)
for g in ['기초연구', '응용연구', '개발연구', '기타']:
    add('A_project', f'dev_phase={g}', a.dev_phase.astype(str) == g, b.dev_phase.astype(str) == g)
rc = P.region_cd.astype(str).str.strip().str.zfill(2)
add('A_project', 'region=대전', rc[P.linked] == '06', rc[~P.linked] == '06')
add('A_project', 'region=수도권', rc[P.linked].isin(['01', '04', '08']), rc[~P.linked].isin(['01', '04', '08']))
# 연구비 가중 연결률(참고)
lw = P[P.linked].groupby('year').gov_funds.sum() / P.groupby('year').gov_funds.sum()
del P, a, b
# ---------- B: N6 모집단 (공식 정의 재현, 개발 원점 2014-2019)
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
nt = duckdb.sql(f"""select pjt_no, year, research_agency nm, regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz, perform_agent, perform_grp, gov_funds, ministry, dev_phase
  from read_parquet('{EXTF}/ntis_project_raw.parquet') where year between 2008 and 2019 and perform_grp in ('중소기업','중견기업','대기업')""").df()
nt = nt[nt.bz.str.len() == 10]
ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(norm); uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
firm = nt[nt.year >= 2010].groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()
rest = nt[~nt.bz.isin(set(firm.bz))]
firm = pd.concat([firm, rest.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()], ignore_index=True)
firm["by_bz"] = firm.bz.map(lambda z: isinstance(b2a.get(z), list)); firm["by_name"] = firm.nm.map(norm).map(uq).notna()
firm["linked"] = firm.by_bz | firm.by_name
nt['mfam'] = nt.ministry.astype(str).map(fam).fillna('OTHER')
W = []
for t0 in range(2014, 2020):
    w = nt[nt.year.between(t0 - 2, t0)]
    g = w.groupby('bz').agg(n_proj=('pjt_no', 'nunique'), funds=('gov_funds', 'sum'), sme=('perform_agent', lambda s: (s == '중소기업').mean()),
                            big=('perform_agent', lambda s: (s == '대기업').mean()), dev=('dev_phase', lambda s: (s.astype(str) == '개발연구').mean()),
                            sme_min=('mfam', lambda s: (s == 'SME').mean()), motie=('mfam', lambda s: (s == 'MOTIE').mean()))
    g['t0'] = t0; W.append(g.reset_index())
W = pd.concat(W).merge(firm[['bz', 'linked', 'by_bz', 'by_name']], on='bz', how='left')
W['corp_type'] = pd.to_numeric(W.bz.str[3:5], errors='coerce'); W['is_corp'] = W.corp_type.between(81, 88)
# 위험집합(기존 출원 없음)은 연결 기업에서만 판정 가능 -> 전체 수행기업 기준으로 비교
a, b = W[W.linked == True], W[W.linked != True]
print('B N6 firm-origin linked share', W.linked.mean(), 'by_bz', W.by_bz.mean(), 'by_name_only', (W.by_name & ~W.by_bz).mean())
for v in ['n_proj', 'funds', 'sme', 'big', 'dev', 'sme_min', 'motie', 'is_corp']:
    x = np.log1p(a[v]) if v in ('n_proj', 'funds') else a[v]; y = np.log1p(b[v]) if v in ('n_proj', 'funds') else b[v]
    add('B_N6_firm', ('log_' + v) if v in ('n_proj', 'funds') else v, x, y)
# N6 공식 패널의 linked(점검용 컬럼) 비율
try:
    PN = pd.read_parquet(os.path.join(MC, r"N6_정부RnD기업_첫특허전환\cache\panel_dev.parquet"), columns=['t0', 'y', 'linked'])
    n6_linked = {str(k): v for k, v in PN.groupby('t0').agg(linked=('linked', 'mean'), y=('y', 'mean')).round(4).to_dict('index').items()}
    n6_y_by_link = {str(k): float(v) for k, v in PN.groupby('linked').y.mean().round(4).items()}
except Exception as e:
    n6_linked = str(e); n6_y_by_link = None
print('N6 panel linked', n6_linked, n6_y_by_link)
# ---------- C: N7 모집단 근원
af = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "appl_no", "known_date", "ksic", "ay"])
af = af[(af.applicantCd.astype(str).str[0] == '1') & af.known_date.notna() & (af.known_date <= 20191231)]
nsr = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "registered"])
nsr = nsr[nsr.registered.fillna(False).astype(bool) & nsr.bizno.notna()]
g = af.groupby('applicantCd').agg(n_app=('appl_no', 'nunique'), n_ksic=('ksic', 'nunique'), last=('ay', 'max'), first=('ay', 'min'),
                                  ksic=('ksic', lambda s: s.value_counts().index[0]))
g['linked'] = g.index.isin(set(nsr.applicantCd))
g['k2'] = g.ksic.astype(str).str[:2]
a, b = g[g.linked], g[~g.linked]
print('C N7 root linked share', g.linked.mean(), len(g))
add('C_N7_applicant', 'log_n_app', np.log1p(a.n_app), np.log1p(b.n_app))
add('C_N7_applicant', 'n_ksic', a.n_ksic, b.n_ksic)
add('C_N7_applicant', 'last_appl_year', a['last'], b['last'])
add('C_N7_applicant', 'first_appl_year', a['first'], b['first'])
for k in sorted(g.k2.unique()):
    add('C_N7_applicant', f'ksic2={k}', a.k2 == k, b.k2 == k)
# KSIC 5자리 분포 총변동거리
pa = a.ksic.astype(str).value_counts(normalize=True); pb = b.ksic.astype(str).value_counts(normalize=True)
tv = float(0.5 * (pa.reindex(pa.index.union(pb.index)).fillna(0) - pb.reindex(pa.index.union(pb.index)).fillna(0)).abs().sum())
lr_ksic = g.groupby('ksic').linked.mean()
S = pd.DataFrame(rows)
S['flag_abs_gt_0.25'] = S.smd.abs() > 0.25
S.to_csv(os.path.join(OUT, 'bias_smd.csv'), index=False, encoding='utf-8-sig')
print(S.round(3).to_string())
nn_all = set(ns.nn); unl = firm[~firm.linked]; amb = float(unl.nm.map(norm).isin(nn_all).mean())
json.dump(dict(A_funds_weighted_link_rate={str(k): float(v) for k, v in lw.round(4).items()}, N6_unlinked_firms=int(len(unl)), N6_unlinked_with_ambiguous_name_match_share=amb,
               N6_panel_linked=n6_linked, N6_panel_y_by_linked=n6_y_by_link,
               C_N7_root_linked_share=float(g.linked.mean()), C_N7_root_n=int(len(g)), C_ksic5_TV=tv,
               C_link_rate_by_ksic_min_max=[float(lr_ksic.min()), float(lr_ksic.max())], C_link_rate_by_ksic_q={str(k): float(v) for k, v in lr_ksic.quantile([.1, .5, .9]).round(4).items()}),
          open(os.path.join(OUT, '_tab_c_link.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
