# C03-T1 s5: 이름 매칭 레이블 표본 점검(P5) — 공개 경로가 결정한 양성(y_pub=1, y_reg=0) 100쌍, 음성(y_u=0) 100쌍. 시드 20261012.
#            규칙: (i) 권리목록(2026 스냅샷)에서 그 출원이 그 법인 원출원인으로 확인 → 정확
#                  (ii) 정규화 이름 일치·길이≥4·등록원부 전체에서 같은 정규화 이름이 다른 코드에 쓰인 적 없음 → 정확 추정
#                  (iii) 정규화 이름 길이≤3, 또는 등록원부/조회목록에서 같은 정규화 이름이 다른 코드(법인 집합 밖 포함)에 쓰임 → 의심
#            음성: 같은 KSIC·출원 2020–23 출원인명 중 그 법인 정규화 이름(길이≥4)을 포함하거나 그 안에 포함되는 이름(완전 일치 제외) → 누락 의심
import sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
T1 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실'; TC = T1/'cache'
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; WC = W/'cache'
def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique())
ev = pd.read_parquet(WC/'eval_o2019.parquet', columns=['firm', 'k', 'newc', 'yA_reg', 'yA_pub'])
ap = pd.read_parquet(WC/'app_pub_src_2020_23.parquet')
nm = pd.read_parquet(WC/'kr_names_2020_23.parquet')
al = pd.read_parquet(WC/'alias_src.parquet')
# 등록원부 전체·조회목록에서 정규화 이름 → 코드 집합(법인 집합 밖 포함)
reg = pd.read_sql('select distinct applicantCd, applicantName from applicant', ro(ROOT/'수집파이프라인/registry_full.sqlite'))
aq = pd.read_sql('select applicant_cd, applicant_name, status from applicant_query', ro(ROOT/'수집파이프라인/business_rights.sqlite'))
allnm = pd.concat([reg.rename(columns={'applicantCd': 'cd', 'applicantName': 'nm'}), aq[['applicant_cd', 'applicant_name']].rename(columns={'applicant_cd': 'cd', 'applicant_name': 'nm'})])
allnm['n'] = allnm.nm.map(norm); ncodes = allnm.groupby('n').cd.agg(lambda x: set(x)).to_dict()
del reg
rl = pd.read_parquet(TC/'rl_pat.parquet', columns=['applicant_cd', 'appl_no', 'pert', 'ainfo'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
def is_self(cd, ainfo):
    a = norm(ainfo)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(ainfo).split(',')): return True
    return False
rng = np.random.RandomState(20261012)
pos = ev[(ev.yA_pub == 1) & (ev.yA_reg == 0)].reset_index(drop=True)
neg = ev[(ev.yA_pub == 0) & (ev.yA_reg == 0)].reset_index(drop=True)
ps = pos.iloc[rng.choice(len(pos), 100, replace=False)]; ns = neg.iloc[rng.choice(len(neg), 100, replace=False)]
fname = aq.set_index('applicant_cd').applicant_name
def judge_pos(firm, k):
    m = ap[(ap.applicant_cd == firm) & (ap.k == k)]
    appls = list(m.appl_no.unique())
    names = nm[nm.appl_no.isin(appls)]
    hit = names[names.nn.isin(set(alias.get(firm, [])))]
    nn = hit.nn.iloc[0] if len(hit) else ''; raw = hit.nm.iloc[0] if len(hit) else ''
    r = rl[rl.appl_no.isin(appls) & (rl.applicant_cd == firm)]
    conf = bool(len(r) and any(is_self(firm, a) for a in r.ainfo.values))
    other = sorted(ncodes.get(nn, set()) - {firm})
    if conf: v = '정확(권리목록 확인)'
    elif len(nn) <= 3 or other: v = '의심'
    else: v = '정확 추정(이름 규칙)'
    return dict(n_appl=len(appls), appl_example=appls[0] if appls else '', applicant_name_raw=raw, norm_name=nn, norm_len=len(nn),
                rl_confirmed=conf, same_name_other_codes=len(other), other_code_example=other[0] if other else '', verdict=v)
rows = []
for x in ps.itertuples():
    d = judge_pos(x.firm, int(x.k)); d.update(group='양성(y_pub만)', firm=x.firm, firm_name=fname.get(x.firm, ''), ksic=K[int(x.k)], newc=bool(x.newc)); rows.append(d)
print('pos judged', round(time.time() - t0), flush=True)
kn = nm.merge(pd.read_parquet(ROOT/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '세세분류코드']).rename(columns={'출원번호': 'appl_no', '세세분류코드': 'ksic'}).astype({'ksic': str}), on='appl_no')
for x in ns.itertuples():
    ks = K[int(x.k)]; cand = kn[kn.ksic == ks]
    own = [n for n in alias.get(x.firm, []) if len(n) >= 4]
    found = cand[[any((o in n_ or (len(n_) >= 4 and n_ in o)) and n_ != o for o in own) for n_ in cand.nn.values]] if own else cand.iloc[0:0]
    found = found[~found.nn.isin(set(alias.get(x.firm, [])))]
    rows.append(dict(group='음성', firm=x.firm, firm_name=fname.get(x.firm, ''), ksic=ks, newc=bool(x.newc), n_appl=int(len(found)),
                     appl_example=found.appl_no.iloc[0] if len(found) else '', applicant_name_raw=found.nm.iloc[0] if len(found) else '',
                     norm_name=own[0] if own else '', norm_len=len(own[0]) if own else 0, rl_confirmed=False, same_name_other_codes=0, other_code_example='',
                     verdict=('누락 의심(부분 일치 이름 있음)' if len(found) else ('판정 불가(별칭 길이<4)' if not own else '누락 없음'))))
S = pd.DataFrame(rows); S.to_csv(W/'name_match_sample.csv', index=False, encoding='utf-8-sig')
# 모집단 전체(양성 y_pub만 2,990쌍)에도 같은 규칙 적용(표본 오차 보완)
full = [judge_pos(x.firm, int(x.k))['verdict'] for x in pos.itertuples()]
vc_full = pd.Series(full).value_counts().to_dict()
out = dict(sample_pos=S[S.group != '음성'].verdict.value_counts().to_dict(), sample_neg=S[S.group == '음성'].verdict.value_counts().to_dict(),
           population_pos_n=int(len(pos)), population_pos=vc_full, suspicious_rate_sample=float((S[S.group != '음성'].verdict == '의심').mean()),
           suspicious_rate_population=float(np.mean([v == '의심' for v in full])))
p_ = out['suspicious_rate_sample']; out['suspicious_rate_sample_wilson95'] = [float((p_ + 1.96**2 / 200 - 1.96 * np.sqrt(p_ * (1 - p_) / 100 + 1.96**2 / 40000)) / (1 + 1.96**2 / 100)),
                                                                            float((p_ + 1.96**2 / 200 + 1.96 * np.sqrt(p_ * (1 - p_) / 100 + 1.96**2 / 40000)) / (1 + 1.96**2 / 100))]
json.dump(out, open(WC/'s5_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)); print('done', round(time.time() - t0))
