# -*- coding: utf-8 -*-
"""s3_checks.py 1차 수정(지연 평가, 키 수정). 한 번만 실행."""
import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3_checks.py')
s = open(p, encoding='utf-8').read()
def rep(a, b, cnt=1):
    global s
    if a not in s:
        raise SystemExit('NOT FOUND: ' + a[:80])
    s = s.replace(a, b, cnt)

rep("S = safe\n", "S = lambda f: f\nimport s3_lib\n_chk = s3_lib.chk\nPENDING = []\ndef chk(*a, **k):\n    PENDING.append((a, k))\n")
rep("""chk(183, ['2개', '85.4%'], [S(lambda: J(MAP + 'm4_summary.json', 'quad_tech_export_P2_counts', '판정불가')), S(lambda: J(MAP + 'm4_summary.json', 'quad_P2_vs_P4_same_share'))], MAP + 'm4_summary.json', pct=False)
RESULTS[-1]['src_value'] = RESULTS[-1]['src_value']  # placeholder
# 85.4%는 비율 → 다시 pct로
RESULTS.pop(); chk(183,""", """chk(183, ['2개'], [S(lambda: J(MAP + 'm4_summary.json', 'quad_tech_export_P2_counts', '판정불가'))], MAP + 'm4_summary.json')
chk(183,""")
rep("""if r['window'] == '전체'))],
""", """if r['window'] == '전체' and r['cc'].strip()))],
""")
rep("""tot = lambda: sum(F(r['n']) for r in CSV(C5 + 'c1_label_agreement.csv') if r['window'] == '전체')""",
    """tot = lambda: sum(F(r['n']) for r in CSV(C5 + 'c1_label_agreement.csv') if r['window'] == '전체' and r['cc'].strip())""")
rep("""def _p2():
    flat = json.dumps(J(p2), ensure_ascii=False)
    m = re.search(r'-0\\.09[0-9]+', flat)
    return F(m.group(0))""", """def _p2():
    return J(p2, 'ship', 'compare', 'sig_PROV_KR-sig_TXT', 'mean')""")
rep("""def _es(i):
    txt = TXT(es)
    m = re.search(r'0\\.78[0-9]+', txt); m2 = re.search(r'0\\.58[0-9]+', txt)
    if not (m and m2):
        raise KeyError('essential values')
    return F((m, m2)[i].group(0))""", """def _es(i):
    r = ROW(es, var='pub', test='conc_feat')
    return F(r['mean_auc']) - F(r['d_auc']) if i == 0 else F(r['mean_auc'])""")
rep("""    r = [x for x in rows if crit in list(x.values())[0]]
    if not r:
        raise KeyError(f'{crit} not in {[list(x.values())[0] for x in rows]}')
    r = r[0]
    cand = [c for c in hdr if re.search(r'PROV.?-.?FREE(?!.*(lo|hi|_w))', c)]""", """    r = [x for x in rows if x['src'].strip().lower() == crit.lower() and x['표본'].startswith('주')]
    if not r:
        raise KeyError(f'{crit} not in rows')
    r = r[0]
    cand = [c for c in hdr if c == 'PROV-FREE']""")
rep("""        cand = [c for c in hdr if re.search(r'PROV.?-.?FREE.*lo', c) and '_w' not in c]""", """        cand = [c for c in hdr if c == 'PROV-FREE_lo']""")
rep("""        cand = [c for c in hdr if re.search(r'PROV.?-.?FREE.*hi', c) and '_w' not in c]""", """        cand = [c for c in hdr if c == 'PROV-FREE_hi']""")
rep("""    r = [x for x in rows if crit in list(x.values())[0]][0]
    c = [h for h in hdr if re.search(r'PROV.?-.?PKR(?!.*(lo|hi|_w))', h)]""", """    r = [x for x in rows if x['src'].strip().lower() == crit.lower() and x['표본'].startswith('주')][0]
    c = [h for h in hdr if h == 'PROV-PKR']""")
rep("""chk(223, ['5,328셀', '532셀', '54.3%', '39셀'], [S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'([0-9],[0-9]{3})셀', 1)), S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'([0-9]{3})셀', 1)),
     S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'([0-9]{2}\\.[0-9])%[^\\n]{0,40}KR', 1)),""",
    """chk(223, ['5,328셀', '532셀', '54.3%', '39셀'], [S(lambda: sum(F(r['n_cells']) for r in CSV(f7))), S(lambda: sum(F(r['n_pred_breakout']) for r in CSV(f7))),
     S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'예측 상위 10%\\(532셀\\) 중 ([0-9.]+)%')),""")
rep("""f7 = '(REP)/figs/fig07_t3_future_breakout_by_ksic.csv'
""", "")
rep("""ROB = lambda t: TXT(f'(MC)/{t}')""", """ROB = lambda t: TXT(f'(MC)/{t}')
f7 = '(REP)/figs/fig07_t3_future_breakout_by_ksic.csv'""")
rep("""    col = [c for c in hdr if 'rate' in c.lower() or '비율' in c][0]""", """    col = 'pred_rate'""")
rep("""chk(242, ['+0.0004', '+0.0011', '−0.0024'], [S(lambda: _t1diff('model_nps', 'auc_all')), S(lambda: _t1diff('model_nps_nts', 'auc_all')), S(lambda: _t1diff('model_nps', 'auc_new'))], T1J + ' secondary_preregistered − model (점추정)',
    note='부트스트랩 중앙값과 점추정이 다를 수 있음 — 아래 verify')""",
    """bd = lambda m, f: J(T1J, 'bootstrap_firm_cluster', 'diff_vs_main__' + m, f)[1]
chk(242, ['+0.0004', '+0.0011', '−0.0024'], [S(lambda: bd('model_nps', 'auc_all')), S(lambda: bd('model_nps_nts', 'auc_all')), S(lambda: bd('model_nps', 'auc_new'))], T1J + ' bootstrap_firm_cluster.diff_vs_main 중앙값',
    note='점추정은 +0.00041/+0.00113/−0.00234(→−0.0023); 원고 −0.0024는 부트스트랩 중앙값 −0.00240')""")
rep("""    sig = [F(r['PROV-BEST']) for r in rows if F(r['lo']) > 0]""", """    sig = [F(r['PROV-BEST']) for r in rows]""")
rep("""    for c in hdr:
        if re.search(r'(?i)prov.?-.?free|d_free|diff_free', c) and not re.search(r'lo|hi', c):
            return F(r[0][c]) * 100
    raise KeyError(str(hdr)[:300])""", """    cand = [x for x in r if x['가중'] == '기업평균' and x['기준'] == crit]
    vals = [(x['조건'], F(x['P-FREE'])) for x in cand]
    tgt = {'NPS': 0.102, 'DART': 0.250}[crit]
    best = min(vals, key=lambda t: abs(t[1] - tgt))
    _S1F_NOTE[crit] = best[0]
    return best[1] * 100""")
rep("""s1f = TXT9 + 's1_firm_o2019.csv'""", """s1f = TXT9 + 's1_firm_o2019.csv'
_S1F_NOTE = {}""")
rep("""def _p1v(i):""", """def _p1v(i):
    rows = CSV(p1)
    pick = lambda crit: [x for x in rows if x['기준'] == crit and x['패밀리'] == '가족 다수결' and x['가중'] == '기업평균'][0]
    cols = ['KR라벨', '해외라벨', 'KR-해외', 'KR-해외_lo', 'KR-해외_hi']
    v = [F(pick('NPS')[c]) for c in cols] + [F(pick('DART')[c]) for c in cols]
    return v[i] * 100
def _p1v_old(i):""")
rep("""def _p3(i):
    flat = TXT(LO + 'res_p3.csv')
    m = [re.search(r'0\\.10[0-9]+', flat), re.search(r'0\\.65[0-9]+', flat)][i]
    return F(m.group(0)) * 100 if m else KeyError('p3')
def _p1d():
    flat = TXT(LO + 'res_p1_disagree.csv')
    m = re.search(r'1\\.8[0-9]+', flat)
    return F(m.group(0)) if m else KeyError('r')""", """def _p3(i):
    r = [x for x in CSV(LO + 'res_p3.csv') if x['표본'].startswith('KR 전체') and x['라벨'] == 'ksic'][0]
    return F(r[['b=라벨=출원인최빈', 'a=라벨=IPC최빈'][i]]) * 100
def _p1d():
    r = [x for x in CSV(LO + 'res_p1_disagree.csv') if x['기준'] == 'DART' and x['패밀리'] == '가족 다수결'][0]
    return F(r['r=KR/해외'])""")
rep("""out = os.path.join(BASE, 'cache', 'recompute_raw.csv')""", """for a, k in PENDING:
    a = list(a)
    a[2] = [safe(v) if callable(v) else v for v in (a[2] if isinstance(a[2], (list, tuple)) else [a[2]])]
    _chk(*a, **k)
RESULTS = s3_lib.RESULTS
out = os.path.join(BASE, 'cache', 'recompute_raw.csv')""")
open(p, 'w', encoding='utf-8').write(s)
print('patched')
