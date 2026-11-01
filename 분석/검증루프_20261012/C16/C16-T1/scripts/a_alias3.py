# -*- coding: utf-8 -*-
"""별칭 감사 3단계(사후 보정, 변경 기록 [1](b)): 주소 시도 규칙의 거짓 불일치율을 '선택 안 된' 등록 문서 중 코드가 맞는 문서(정답 정확)에서 잰다.
 출원 2020–23 app_pub 등록 문서 중 이름매칭 코드 = 원출원인 코드인 문서에 a_alias.py와 같은 E2 규칙 적용 → 불일치 비율 = 거짓 불일치율."""
import sys, pathlib, sqlite3, re; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
src = open(pathlib.Path(__file__).resolve().parent/'a_alias.py', encoding='utf-8').read()
exec(src[src.index('SIDO = ['):src.index('# ---- 레이블·쌍 ----')])          # a_alias.py의 region()·SIDO·CTRY 그대로


def wilson(x, n, z=1.959963984540054):
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ra = pd.read_sql('select rgst_no, applicantCd, applicantAddr from applicant', con); con.close()
ra['reg_r'] = ra.applicantAddr.map(region)
codes = ra.groupby('rgst_no').applicantCd.apply(lambda s: set(s.dropna().astype(str)))
a = load_app(); a = a[a.reg.astype(bool) & (a.ay > 2019) & (a.ay <= 2023)].copy(); a['appl_no'] = a.appl_no.astype(str)
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '출원인명', '출원인주소']); kp['출원번호'] = kp['출원번호'].astype(str)
a = a.merge(kp, left_on='appl_no', right_on='출원번호', how='left'); a['rg'] = a['등록번호'].fillna('').astype(str)
a['code_ok'] = [(str(c) in codes.get(r, set())) for c, r in zip(a.applicant_cd, a.rg)]
ok = a[a.code_ok].sample(n=min(20000, int(a.code_ok.sum())), random_state=20261017)
# 법인 주소 시도 집합: 그 법인 코드의 등록원부 주소 중 '이 문서 자신'은 뺀다(같은 문서 주소로 맞히는 순환 방지)
firm_rows = ra[ra.applicantCd.isin(set(ok.applicant_cd.astype(str)))].dropna(subset=['reg_r'])
STRIP2 = STRIP
firms = pd.read_parquet(CACHE/'firms.parquet'); firms = firms[firms.status == 'OK']
app = pd.read_parquet(CACHE/'ins_applicant.parquet')
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(firms.applicant_cd)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]; n2c = dict(zip(al.n, al.applicant_cd))
fr_by = {f: g for f, g in firm_rows.groupby('applicantCd')}
res = dict(match=0, mismatch=0, none=0)
for r in ok.itertuples(index=False):
    names = str(r.출원인명).split('|'); addrs = str(r.출원인주소).split('|')
    posi = [i for i, nm in enumerate(names) if n2c.get(norm(nm)) == str(r.applicant_cd)]
    if posi and len(addrs) == len(names): sa = {region(addrs[posi[0]])} - {None}
    else: sa = {region(x) for x in addrs} - {None}
    g = fr_by.get(str(r.applicant_cd)); sf = set(g.reg_r[g.rgst_no != r.rg]) if g is not None else set()
    e2 = 'none' if (not sa or not sf) else ('match' if sa & sf else 'mismatch')
    res[e2] += 1
n_det = res['match'] + res['mismatch']
out = dict(docs_code_ok_2020_23=int(a.code_ok.sum()), sampled=int(len(ok)), e2=res, false_mismatch_rate_wilson=wilson(res['mismatch'], n_det))
summ = json.load(open(CC/'alias_summary.json', encoding='utf-8')); summ['posthoc_addr_false_mismatch'] = out
# 보정: P-미등록 표본 의심률 − 거짓 불일치율(주소 규칙으로 판정된 문서 기준)
s = summ['sample_final']['P-미등록']; p_obs = s['wrong'] / s['n']; fm = out['false_mismatch_rate_wilson'][0]
summ['posthoc_P_unreg_corrected'] = dict(observed=p_obs, false_mismatch=fm, corrected=max(0.0, (p_obs - fm) / (1 - fm)))
jdump(summ, CC/'alias_summary.json'); print(json.dumps({k: summ[k] for k in ('posthoc_addr_false_mismatch', 'posthoc_P_unreg_corrected')}, ensure_ascii=False, indent=1), G.rss_peak())
