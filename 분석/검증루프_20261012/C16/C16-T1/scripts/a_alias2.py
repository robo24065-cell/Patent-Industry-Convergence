# -*- coding: utf-8 -*-
"""별칭 감사 2단계(PREREG 변경 기록 [1]):
 (a) P-미등록 56쌍 추가 추출 → alias_audit.csv 320행
 (b) 사후 진단(판정 미사용): 선택 안 된 등록 문서의 이름매칭–원출원인 코드 불일치율, 현재 권리자 표기 가설
 (c) 민감도 Δskill(공식 모집단·공식 점수, 법인 군집 CI): i) 오매칭 의심 P 양성 제외 ii) 음성으로 iii) 오매칭 R 쌍 되돌림 iv) ii+iii"""
import sys, pathlib, sqlite3, re; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
import s_score as SS
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()


def wilson(x, n, z=1.959963984540054):
    if n == 0: return [None, None, None]
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


summ = json.load(open(CC/'alias_summary.json', encoding='utf-8'))
DOC = pd.read_parquet(CC/'alias_docs.parquet')
S1 = pd.read_csv(HERE/'alias_audit.csv', encoding='utf-8-sig', dtype={'firm': str, 'appl_no': str, 'rgst_no': str})
if 'sample_wave' not in S1.columns:
    S1['sample_wave'] = 1
    rs = np.random.RandomState(20261017); DOC['_u'] = rs.rand(len(DOC))                 # a_alias.py와 같은 쌍별 문서 선택 재현
    one = DOC.sort_values(['grp', 'firm', 'k', '_u']).groupby(['grp', 'firm', 'k'], sort=False).head(1).reset_index(drop=True)
    one['stratum'] = np.where(one.grp == 'R', 'R', np.where(one.is_reg, 'P-등록', 'P-미등록'))
    g = one[one.stratum == 'P-미등록']; used = set(zip(S1.firm.astype(str), S1.k.astype(int)))
    g = g[[(str(f), int(k)) not in used for f, k in zip(g.firm, g.k)]]
    add = g.sample(n=56, random_state=20261018).assign(stratum_size=int((one.stratum == 'P-미등록').sum()), sample_wave=2)
    add['firm'] = add.firm.astype(str); add['appl_no'] = add.appl_no.astype(str)
    S = pd.concat([S1, add[S1.columns]], ignore_index=True)
    S.to_csv(HERE/'alias_audit.csv', index=False, encoding='utf-8-sig')
else:
    S = S1
S['wrong'] = S.verdict.str.startswith('오매칭'); S['undet'] = S.verdict.eq('판정 불가')
st = {}
groups = list(S.groupby('stratum')) + [('P-미등록 1차', S[(S.stratum == 'P-미등록') & (S.sample_wave == 1)]), ('P-미등록 추가', S[(S.stratum == 'P-미등록') & (S.sample_wave == 2)]),
                                       ('P 전체(표본)', S[S.stratum != 'R']), ('표본 전체', S)]
for gname, d in groups:
    n_det = int((~d.undet).sum()); x = int(d.wrong.sum())
    st[gname] = dict(n=int(len(d)), wrong=x, undet=int(d.undet.sum()), rate_wilson=wilson(x, n_det), verdicts=d.verdict.value_counts().to_dict())
# P 모집단 가중(층 크기: P-등록 24쌍, P-미등록 2,966쌍) 오류율
nP_reg = int(S[S.stratum == 'P-등록'].stratum_size.iloc[0]); nP_unr = int(S[S.stratum == 'P-미등록'].stratum_size.iloc[0])
pr_reg = st['P-등록']['rate_wilson'][0]; pr_unr = st['P-미등록']['rate_wilson'][0]
st['P 층가중'] = dict(rate=(nP_reg * pr_reg + nP_unr * pr_unr) / (nP_reg + nP_unr), weights=[nP_reg, nP_unr])
summ['sample_final'] = st
# ---- (b) 사후 진단 ----
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ra = pd.read_sql('select rgst_no, applicantCd from applicant', con)
ow = pd.read_sql("select rgst_no, ownerName, finalOwnerYn from owner", con); con.close()
codes = ra.groupby('rgst_no').applicantCd.apply(lambda s: set(s.dropna().astype(str)))
a = load_app()
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '출원인명']); kp['출원번호'] = kp['출원번호'].astype(str)
reg_docs = a[a.reg.astype(bool)].copy(); reg_docs['appl_no'] = reg_docs.appl_no.astype(str)
reg_docs = reg_docs.merge(kp[['출원번호', '등록번호']], left_on='appl_no', right_on='출원번호', how='left')
reg_docs['rg'] = reg_docs['등록번호'].fillna('').astype(str)
reg_docs['in_reg'] = reg_docs.rg.isin(codes.index)
reg_docs['code_ok'] = [(c in codes.get(r, set())) for c, r in zip(reg_docs.applicant_cd.astype(str), reg_docs.rg)]
diag = {}
for nm, m in (('출원 2020–23', (reg_docs.ay > 2019) & (reg_docs.ay <= 2023)), ('출원 ≤2019', reg_docs.ay <= 2019), ('출원 2000–2009', reg_docs.ay <= 2009), ('출원 2015–2019', (reg_docs.ay >= 2015) & (reg_docs.ay <= 2019))):
    d = reg_docs[m & reg_docs.in_reg]; x = int((~d.code_ok).sum())
    diag[nm] = dict(docs=int(len(d)), code_mismatch=x, rate_wilson=wilson(x, len(d)))
summ['posthoc_unselected_registered_mismatch'] = diag
# 현재 권리자 표기 가설: 코드 불일치 등록 근거 문서(P·R)에서 kr_patents 매칭 이름이 최종 권리자 이름과 같은 비율
fo = ow[ow.finalOwnerYn.astype(str).str.upper().isin(['Y', '1', 'TRUE'])].copy(); fo['n'] = fo.ownerName.map(norm)
fon = fo.groupby('rgst_no').n.apply(set)
w = DOC[DOC.is_reg & DOC.wrong]
eq = [nm in fon.get(r, set()) for nm, r in zip(w.name_norm, w.rgst_no)]
summ['posthoc_current_owner_hypothesis'] = dict(wrong_registered_docs=int(len(w)), name_equals_final_owner=int(np.sum(eq)), share=float(np.mean(eq)) if len(eq) else None,
                                                owner_rows_final=int(len(fo)))
# ---- (c) 민감도 Δskill ----
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
fz = pd.read_parquet(RD/'o2019.parquet', columns=['px', 'pl']); bo = pd.read_parquet(FC/'baseline_scores.parquet'); sb = np.load(CC/'p0'/'sbb_all.npy')
L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
O = te.copy(); O['px'] = fz.px.values; O['pl'] = fz.pl.values; O['sbb'] = sb; O['y_off'] = L.y_u.values
for c in SS.BK: O[c] = bo[c].values
key = pd.MultiIndex.from_arrays([O.firm.values, O.k.values])
pP = pd.read_parquet(CC/'alias_pairs_P.parquet'); pR = pd.read_parquet(CC/'alias_pairs_R.parquet')
flagP = key.isin(pd.MultiIndex.from_arrays([pP.firm[pP.all_wrong].values, pP.k[pP.all_wrong].values]))
flagR = key.isin(pd.MultiIndex.from_arrays([pR.firm[pR.all_wrong].values, pR.k[pR.all_wrong].values]))
kept = ~O.prior_any.values
sens = {}
V = {'ALIAS_i_exclude': (kept & ~flagP, None), 'ALIAS_ii_relabel0': (kept, 'relabel'), 'ALIAS_iii_addR': (kept | flagR, None), 'ALIAS_iv_ii_iii': (kept | flagR, 'relabel')}
for nm, (m, how) in V.items():
    D = O[m].copy(); D['y'] = D.y_off.values
    if how == 'relabel': D.loc[key[m].isin(key[flagP & kept]), 'y'] = 0
    info = dict(flagP_pairs=int((flagP & kept).sum()), flagR_pairs=int(flagR.sum()), flagR_officialdef_pos=int(O.y_off.values[flagR].sum()))
    SS.score_df(nm, D, info)
    r = json.load(open(SS.VD/f'{nm}.json', encoding='utf-8')); sens[nm] = dict(skill=r['skill'], delta=r['skill'] - SS.OFFICIAL['skill'], ci=r['ci'], skill_vs_sbb=r['skill_vs_sbb'], ci_vs_sbb=r['ci_vs_sbb'], pos=r['pos'], n=r['n'], top1000=r['top1000'])
summ['sensitivity'] = sens
summ['peak2'] = G.rss_peak()
jdump(summ, CC/'alias_summary.json')
print(json.dumps({k: summ[k] for k in ('sample_final', 'posthoc_unselected_registered_mismatch', 'posthoc_current_owner_hypothesis', 'sensitivity')}, ensure_ascii=False, indent=1, default=str))
