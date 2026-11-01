# -*- coding: utf-8 -*-
"""s3_checks.py 2차 수정. 한 번만 실행."""
import os
d = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(d, 's3_checks.py')
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    if a not in s:
        raise SystemExit('NOT FOUND: ' + a[:90])
    s = s.replace(a, b, 1)
rep("""[S(lambda: [x for x in J(LO + 'res_fam_check.json', 'links') if x[0] == c][0][1]) for c in ('US', 'JP', 'EP')]""",
    """[S(lambda c=c: [x for x in J(LO + 'res_fam_check.json', 'links') if x[0] == c][0][1]) for c in ('US', 'JP', 'EP')]""")
rep("""def _kr_maj():
    d = J(LO + 'res_tests.json')""", """def _kr_maj():
    return J(LO + 'res_tests.json', 'P0', 'agree_kr_vs_famMajor') * 100
def _kr_maj_old():
    d = J(LO + 'res_tests.json')""")
rep("""S(lambda: md_num('(MC)/T1_산업대기실/R3/FINAL_REPORT_R3.md', r'교란 시험.{0,80}?최대 차 ([0-9.]+)') if re.search(r'교란 시험', TXT('(MC)/T1_산업대기실/R3/FINAL_REPORT_R3.md')) else 0.0)""",
    """S(lambda: md_num('(MC)/T1_산업대기실/R3/FINAL_REPORT_R3.md', r'\\*\\*교란 시험\\*\\*[^\\n]*?최대 차[^0-9]{0,12}([0-9.]+)'))""")
rep("""    m = re.search(r'"[^"]*rho[^"]*"\\s*:\\s*(0\\.99[0-9]*)', flat)
    return F(m.group(1)) if m else KeyError('rho')""", """    return J(MAP + 'm7_summary.json', 'jp_2021_ksic_rank_raw_vs_corr_spearman')""")
rep("""def _kosis(k):
    d = J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json')""", """def _kosis(k):
    return J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json', k + '_사업체_0809_1617')
def _kosis_old(k):
    d = J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json')""")
rep("""    row = ROW(SP + 't5_cases.csv', ksic=k)
    cols = list(row)
    return F(row[cols[12]])""", """    return F(ROW(SP + 't5_cases.csv', ksic=k)['KOSIS출하_연성장%'])""")
rep("""    r = [x for x in CSV(nc) if x['set'].startswith('n6') and 'frozen' in x['set'] and x['comparison'] == 'PROV-TXT']""",
    """    r = [x for x in CSV(nc) if x['set'] == 'n6_prov_y' and x['comparison'] == 'PROV-TXT']""")
rep("""    rows = CSV('(LOOP)/C06/C06-T1/rescore_B.csv')
    r = [x for x in rows if list(x.values())[0].startswith('B1')][0]
    nums_ = [F(v) for v in list(r.values())[1:] if re.fullmatch(r'-?[0-9.]+', v.strip() or 'x')]
    cand = [x for x in nums_ if 0.01 < x < 0.12]
    tgt = [0.0618, 0.0145, 0.1095][i]
    best = min(cand, key=lambda x: abs(x - tgt))
    return best * 100""", """    r = [x for x in CSV('(LOOP)/C06/C06-T1/rescore_B.csv') if x['기준'] == 'DART5' and x['보정'].startswith('B1')][0]
    return F(r[['PROV-BEST', 'lo', 'hi'][i]]) * 100""")
rep("""def _fnl(w):
    flat = TXT(fnl)""", """def _fnl(w):
    rows = CSV(fnl)
    if w == '44':
        return F([r for r in rows if r['항목'].startswith('NPS 사업장 매칭')][0]['비율']) * 100
    if w == '93':
        a = F([r for r in rows if r['항목'].startswith('NPS 사업장 매칭')][0]['값']); b = F([r for r in rows if r['항목'].startswith('사업자번호 확보')][0]['값'])
        return a / b * 100
    vals = []
    for r in rows:
        if re.match(r'\\d{5}', r['항목'].strip()) and r['비율'].strip():
            vals.append(F(r['비율']))
    if not vals:
        raise KeyError('KSIC rows: ' + str([r['항목'] for r in rows])[:300])
    _FNL_N[0] = len(vals)
    return (min(vals) if w == 'min' else max(vals)) * 100
_FNL_N = [0]
def _fnl_old(w):
    flat = TXT(fnl)""")
open(p, 'w', encoding='utf-8').write(s)
# s3_lib: TXT/CSV 캐시 키 분리
pl = os.path.join(d, 's3_lib.py')
t = open(pl, encoding='utf-8').read()
t = t.replace("""    if p not in _cache:
        _cache[p] = open(p, encoding='utf-8', errors='replace').read()
    return _cache[p]""", """    if ('TXT', p) not in _cache:
        _cache[('TXT', p)] = open(p, encoding='utf-8-sig', errors='replace').read()
    return _cache[('TXT', p)]""")
t = t.replace("""    if p not in _cache:
        _cache[p] = list(csv.DictReader(open(p, encoding='utf-8-sig')))
    return _cache[p]""", """    if ('CSV', p) not in _cache:
        _cache[('CSV', p)] = list(csv.DictReader(open(p, encoding='utf-8-sig')))
    return _cache[('CSV', p)]""")
open(pl, 'w', encoding='utf-8').write(t)
print('patched b')
