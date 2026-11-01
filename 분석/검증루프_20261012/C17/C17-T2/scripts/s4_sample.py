# -*- coding: utf-8 -*-
"""C17-T2 s4: PREREG §4 층화 표본(≥600쌍) → alias_audit_v2.csv. 쌍별 근거 문서 요약(판정 문서·보유자·원출원인명·등록원부 코드·주소 시도)과 p, 점수 백분위.
 산출: alias_audit_v2.csv, cache/s4_out.json"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
t0 = time.time(); out = {}
PJ = pd.read_parquet(CC/'pair_judg.parquet'); P = PJ[PJ.grp == 'P'].reset_index(drop=True)
pp = pd.read_parquet(CC/'p_point.parquet'); P = P.merge(pp[['firm', 'k', 'p_main']], on=['firm', 'k'])
DJ = pd.read_parquet(CC/'doc_judg.parquet'); DJ = DJ[DJ.grp == 'P']
RL = pd.read_parquet(CC/'rl_hits.parquet'); RL['holder_cd'] = RL.holder_cd.astype(str)
Z = np.load(CC/'eval_base.npz', allow_pickle=True); pct = pd.Series(Z['s']).rank(pct=True).values
key = pd.Series(np.arange(len(pct)), index=pd.MultiIndex.from_arrays([Z['firm'], Z['k']]))
P['score_pct'] = pct[key.reindex(pd.MultiIndex.from_arrays([P.firm.values, P.k.values])).values.astype(int)]
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique())
con = ro(BIZ); aq = pd.read_sql('select applicant_cd, applicant_name from applicant_query', con); con.close(); aqn = dict(zip(aq.applicant_cd.astype(str), aq.applicant_name))
rs = np.random.RandomState(20261017)
P['_u'] = rs.rand(len(P)); P['det'] = P.pv != 'undet'
P['cell'] = np.where(P.F, 'F', 'NF') + '-' + np.where(P.foreign, '외국', '국내') + '-' + P['size']
S_F = P[P.F]
quota = 60
while True:
    parts = []
    for (fo, sz), g in P[~P.F].groupby(['foreign', 'size']):
        g = g.sort_values(['det', '_u'], ascending=[False, True]); parts.append(g.head(quota))
    S = pd.concat([S_F] + parts)
    if len(S) >= 600: break
    quota += 1
out['quota_NF_per_cell'] = quota; out['n_sample'] = len(S)
S = S.copy(); S['stratum_size'] = S.cell.map(P.cell.value_counts())
# 근거 문서 요약: 판정 문서(정확·오매칭 우선) 1건
pri = {'correct': 0, 'wrong': 1, 'undet': 2}
DJ = DJ.assign(_r=DJ.st.map(pri)).sort_values(['firm', 'k', '_r', 'appl_no'])
top = DJ.groupby(['firm', 'k']).head(1).set_index(['firm', 'k'])
agg = DJ.groupby(['firm', 'k']).agg(doc_verdicts=('v', lambda v: '; '.join(f'{a}×{b}' for a, b in v.value_counts().items())),
                                    addr_rules=('addr_rule', lambda v: '/'.join(sorted(set(v)))))
S = S.join(top[['appl_no', 'ay', 'is_reg', 'rgst_no', 'name_raw', 'v', 'e2', 'app_regions', 'firm_regions', 'e1_codes', 'n_coapp', 'homonym_codes']], on=['firm', 'k']).join(agg, on=['firm', 'k'])
hold = RL.groupby('appl_no').agg(rl_holders=('holder_cd', lambda v: '/'.join(sorted(set(v)))), rl_holder_corp=('corp_rgstno', lambda v: '/'.join(sorted(set(map(str, v.dropna()))))[:60]),
                                 rl_ainfo=('ainfo', lambda v: str(v.dropna().iloc[0])[:120] if v.notna().any() else ''), rl_status=('st', 'first'))
S = S.join(hold, on='appl_no')
S['firm_name'] = S.firm.map(aqn); S['ksic'] = [K[int(x)] for x in S.k]
S['sample_role'] = np.where(S.F, 'F 전수', 'NF 할당(판정 가능 우선)')
cols = ['cell', 'stratum_size', 'sample_role', 'firm', 'firm_name', 'k', 'ksic', 'n_docs', 'n_det', 'pv', 'ptier', 'ptype', 'p_main', 'score_pct', 'c16_flag', 'F', 'foreign', 'size', 'nd',
        'appl_no', 'ay', 'is_reg', 'rgst_no', 'name_raw', 'v', 'doc_verdicts', 'addr_rules', 'e2', 'app_regions', 'firm_regions', 'e1_codes', 'rl_holders', 'rl_holder_corp', 'rl_ainfo', 'rl_status',
        'n_coapp', 'homonym_codes']
S = S.rename(columns={'v': 'key_doc_verdict'}); cols = [c if c != 'v' else 'key_doc_verdict' for c in cols]
S[cols].sort_values(['cell', 'pv', 'firm']).to_csv(HERE/'alias_audit_v2.csv', index=False, encoding='utf-8-sig')
# 표본 요약(층별 판정 분포, 판정 가능 표본 층가중 오매칭률)
tab = S.groupby('cell').agg(n=('pv', 'size'), det=('det', 'sum'), wrong=('pv', lambda v: int((v == 'wrong').sum())), correct=('pv', lambda v: int((v == 'correct').sum())),
                            N=('stratum_size', 'first'), p_mean=('p_main', 'mean')).reset_index()
out['sample_table'] = tab.astype(str).to_dict('records')
# 층가중 기대 오매칭률(표본 p 평균 × 층 크기) — 모집단 전체 추정과 같은 p를 쓰므로 일치 확인용
out['sample_weighted_expected_rate'] = float((tab.p_mean * tab.N).sum() / tab.N.sum())
out['pop_expected_rate'] = float(P.p_main.mean())
out['secs'] = round(time.time() - t0)
jdump(out, CC/'s4_out.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
