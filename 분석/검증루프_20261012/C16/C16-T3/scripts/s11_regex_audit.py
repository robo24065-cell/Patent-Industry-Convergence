# C16-T3 s11: 정규식 감사 표 → regex_audit.csv
# 판정자 수작업 판정: 독립 규칙(s4)과 C13 판정이 다른 문서·독립 규칙 보류(unk) 문서를 모두 원문 이름으로 검토(2026-10-17).
#   unk 문서는 전부 '公司/大学/研究院/株式会社' 포함 기관(독립 규칙이 괄호·라틴 혼합 이름을 보류했을 뿐) → C13 판정 정답.
#   아래 ADJ에 C13 판정이 틀린 문서만 적는다(truth 값). 비조직 일치 85건(2015 54·2019 25·2024 6)도 전부 개인 이름으로 확인.
# 사용: python -I s11_regex_audit.py <cache_dir> <out_csv>
import sys, os, math
import pandas as pd, numpy as np
C, OUT = sys.argv[1], sys.argv[2]
ADJ = {  # appl_no: (truth_doc, 유형, 메모)
    '201510329021.2': ('org', 'FN', '商行(개인사업체 상호) — 정규식 어휘 없음'),
    '201510148035.4': ('org', 'FN', '中国人民解放军…部队 — 정규식 어휘 없음'),
    '201810996049.5': ('org', 'FN', '中国人民解放军…部队'),
    '201680089777.7': ('org', 'FN', '连株式会社(LINE) — 정규식은 맞췄으나 접미 제거 후 키 1글자로 탈락'),
    '201910144011.X': ('nonorg', 'FP', '冯会仓 개인 — 이름의 会'),
    '201910228448.1': ('org', 'FN', '广西壮族自治区药用植物园 — 园 없음'),
    '202410215541.X': ('org', 'FN', '季华实验室 — 实验室 없음'),
    '202411089859.4': ('org', 'FN', '中国人民解放军…部队'),
    '202311813994.4': ('org', 'FN', '湖北九峰山实验室 — 实验室 없음'),
}
s = pd.read_csv(os.path.join(C, 'regex_sample.csv'), encoding='utf-8-sig', dtype={'appl_no': str})
s['truth_doc'] = s.c13_doc
s['truth_app1'] = s.c13_app1
for a, (t, typ, memo) in ADJ.items():
    assert a in set(s.appl_no), a
    s.loc[s.appl_no == a, 'truth_doc'] = t; s.loc[s.appl_no == a, 'truth_app1'] = t
s.loc[s.appl_no == '201710504264.4', 'truth_app1'] = 'nonorg'   # 출원인1 钟博文(개인), 공동 기관 있음 → 문서 org, 출원인1 nonorg (C13과 같음)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


pop = pd.read_csv(os.path.join(C, 's5_regex_pop.csv'), encoding='utf-8-sig').set_index('py')
rows = []
for y, g in s.groupby('py'):
    e = g.c13_doc != g.truth_doc; e1 = g.c13_app1 != g.truth_app1
    lo, hi = wilson(int(e.sum()), len(g))
    rows.append(dict(kind='sample_summary', pub_year=int(y), n=len(g), c13_org_rate=float((g.c13_doc == 'org').mean()), truth_org_rate=float((g.truth_doc == 'org').mean()),
                     err_doc=int(e.sum()), fp_doc=int(((g.c13_doc == 'org') & (g.truth_doc == 'nonorg')).sum()), fn_doc=int(((g.c13_doc == 'nonorg') & (g.truth_doc == 'org')).sum()),
                     err_rate_doc=float(e.mean()), err_ci_lo=lo, err_ci_hi=hi, err_rate_app1=float(e1.mean()),
                     verdict_measurement_error=bool(e.mean() > 0.05),
                     pop_fp_cand=float(pop.fp_cand.get(y, np.nan)), pop_fn_lex=float(pop.fn_lex.get(y, np.nan)), pop_fn_key=float(pop.fn_key.get(y, np.nan)),
                     pop_app1_org=float(pop.app1_org.get(y, np.nan)), pop_app1_org_corrected=float(pop.app1_org_corrected.get(y, np.nan))))
for y, r in pop.iterrows():
    if y in (2015, 2019, 2024): continue
    rows.append(dict(kind='population_only', pub_year=int(y), pop_fp_cand=r.fp_cand, pop_fn_lex=r.fn_lex, pop_fn_key=r.fn_key, pop_app1_org=r.app1_org, pop_app1_org_corrected=r.app1_org_corrected))
for a, (t, typ, memo) in ADJ.items():
    g = s[s.appl_no == a].iloc[0]
    rows.append(dict(kind='error_case', pub_year=int(g.py), appl_no=a, applicant=g.applicant, c13_doc=g.c13_doc, truth_doc=t, error_type=typ, note=memo))
R = pd.DataFrame(rows).sort_values(['kind', 'pub_year'])
R.to_csv(OUT, index=False, encoding='utf-8-sig', float_format='%.5f')
sys.stdout.reconfigure(encoding='utf-8')
print(R[R.kind == 'sample_summary'].dropna(axis=1, how='all').to_string())
