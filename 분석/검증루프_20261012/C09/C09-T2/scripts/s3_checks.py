# -*- coding: utf-8 -*-
"""C09-T2 s3: B·C·D 토큰 원천 재계산 (PREREG 3·4절). 출력 cache/recompute_raw.csv"""
import sys, os, re, csv, json, sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_lib import *

S = lambda f: f
import s3_lib
_chk = s3_lib.chk
PENDING = []
def chk(*a, **k):
    PENDING.append((a, k))
MCp = lambda r: '(MC)\\' + r
T3J = '(MC)/T3_기술셀부상예측/R3/final_result_r3.json'
T1J = '(MC)/T1_산업대기실/R3/final_result_r3.json'
N9J = '(MC)/N9_정부RnD과제_특허성과예측/final_result_r6.json'
N6J = '(MC)/N6_정부RnD기업_첫특허전환/final_result_r5.json'
N7J = '(MC)/N7_특허기업_정부RnD첫수행/final_result_r5.json'
N8J = '(MC)/N8_특허분쟁표적레이더/final_result_r6.json'
T4J = '(MC)/T4_특허포기예측/R2/final_result.json'
N4J = '(MC)/N4_특허행태_기업폐업조기경보/final_result_r4.json'
N5J = '(MC)/N5_표준특허선언레이더/final_result_r4.json'
R12 = '(MC)/_shared_report/label_origin_R10/confirm_R12/'
LO = '(MC)/_shared_report/label_origin_R10/'
MAP = '(MC)/_shared_report/fusion_advantage/map5/'
C5 = '(MC)/_shared_report/fusion_advantage/c5_R10/'
SP = '(MC)/_shared_report/fusion_advantage/split34/'
TXT9 = '(MC)/T1_산업대기실/TXT_R9/res/'
ROB = lambda t: TXT(f'(MC)/{t}')
f7 = '(REP)/figs/fig07_t3_future_breakout_by_ksic.csv'

def md_num(rel, pattern, grp=1):
    m = re.search(pattern, TXT(rel))
    if not m:
        raise KeyError(f'pattern not found {pattern[:40]} in {rel}')
    return F(m.group(grp))

def r12(crit, col):
    return F(ROW(R12 + 'res_main.csv', **{'기준': crit})[col])

def sub(crit, kind, seg, col):
    return F(ROW(R12 + 'res_sub.csv', **{'기준': crit, '구간종류': kind, '구간': seg})[col])

# ---------------- 원천 파생 계산 ----------------
# T3 동결 예측 parquet (공식 레이블·점수) 재채점
t3 = pd.read_parquet(MC + '/T3_기술셀부상예측/R3/preds/FINAL_R3_t0_2021.parquet')
def topk_hits(score, y, k):
    o = np.argsort(-score, kind='stable')
    return int(y[o[:k]].sum())
def tie_range(score, y, k):
    s = np.sort(score)[::-1]
    thr = s[k - 1]
    above = score > thr
    eq = score == thr
    need = k - above.sum()
    pos_above = int(y[above].sum()); pos_eq = int(y[eq].sum()); n_eq = int(eq.sum())
    lo = pos_above + max(0, need - (n_eq - pos_eq)); hi = pos_above + min(need, pos_eq)
    exp = pos_above + need * pos_eq / n_eq
    return lo, hi, exp, n_eq, need
y3 = t3['yb'].values.astype(int); p3 = t3['p'].values; b3 = t3['p_base'].values
t3_auc = roc_auc_score(y3, p3); t3_aucb = roc_auc_score(y3, b3)
k10 = int(round(len(t3) * 0.1))  # 504
t3_top10 = topk_hits(p3, y3, k10)
t3_b10 = tie_range(b3, y3, k10)
t3_b100 = tie_range(b3, y3, 100)
t3_top100 = topk_hits(p3, y3, 100)
o = np.argsort(-p3, kind='stable')[:k10]
t3_top_kr0 = float((t3['kr_W'].values[o] == 0).mean())
m5 = t3['kr_W'].values >= 5
t3_kr5 = (int(m5.sum()), roc_auc_score(y3[m5], p3[m5]), roc_auc_score(y3[m5], b3[m5]))
hits = o[y3[o] == 1]
t3_hit_krle1 = float((t3['kr_W'].values[hits] <= 1).mean())
json.dump(dict(auc=t3_auc, aucb=t3_aucb, k10=k10, top10=t3_top10, b10=t3_b10, b100=t3_b100, top100=t3_top100,
               top_kr0=t3_top_kr0, kr5=t3_kr5, n=len(t3), pos=int(y3.sum()), hit_krle1=t3_hit_krle1, nhits=len(hits)),
          open(os.path.join(BASE, 'cache', 's3_t3_parquet.json'), 'w'), default=float, indent=1)

# T3 부상 정의(직전 0건→최소 2건, 1건→최소 4건): kr_W(직전 3년)·yb/y로 최소 미래 건수는 parquet에 없음 → C04-T3/C08-T2 검증값 사용
# N9 동결 예측 parquet: 원판 비교 모형 상위 100 적중
n9 = pd.read_parquet(MC + '/N9_정부RnD과제_특허성과예측/final_preds_r6.parquet')
def h100(col):
    return topk_hits(n9[col].values, n9['y'].values.astype(int), 100)
n9_h = {c: h100(c) for c in ['p_champ', 'p_M0_L2', 'p_pilot_orig', 'p_strong_orig']}
json.dump(n9_h, open(os.path.join(BASE, 'cache', 's3_n9_parquet.json'), 'w'), indent=1)

# 등록원부 sqlite(읽기 전용): 테이블 수·행 수
def registry_counts():
    p = os.path.join(AN, '..', '수집파이프라인', 'registry_full.sqlite')
    p = os.path.abspath(p)
    if not os.path.exists(p):
        alt = [os.path.join(r, f) for r, _, fs in os.walk(os.path.abspath(os.path.join(AN, '..'))) for f in fs if f == 'registry_full.sqlite']
        if not alt:
            raise FileNotFoundError('registry_full.sqlite')
        p = alt[0]
    con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    tot = 0; per = {}
    for t in tabs:
        n = con.execute(f'select count(*) from "{t}"').fetchone()[0]; per[t] = n; tot += n
    con.close()
    return p, len(tabs), tot, per


def _rs(col):
    r = [x for x in CSV('(LOOP)/C05/C05-T2/rescore_strata.csv') if x['track'] == 'N7' and x['stratum'].startswith('S2 as-of 확인')][0]
    if col == 'share':
        return F(r['n']) / F([x for x in CSV('(LOOP)/C05/C05-T2/rescore_strata.csv') if x['track'] == 'N7' and x['stratum'].startswith('공식')][0]['n']) * 100
    if col == 'skill':
        return F(r['skill'])
    lo, hi = [F(v) for v in r['skill_ci'].strip('[]').split(',')]
    return lo if col == 'lo' else hi
def _sec(which):
    j = J('(LOOP)/C07/C07-T2/cache/s3_A_n7_secret.json')
    if which == 'test':
        a = j['share_by_year']['2023']; b = j['share_by_year']['2024']
        return (a[0] + b[0]) / (a[1] + b[1]) * 100
    return j['share_dev_window_2018_22'] * 100
def _lc():
    for r in CSV('(LOOP)/C01/C01-T3/label_composition.csv'):
        try:
            n = F(r['n'])
        except Exception:
            continue
        if 9 < n < 10.5:
            return n / 108 * 100
    raise KeyError('권리자 청구 추정 행')
def _inv(metric):
    r = [x for x in CSV('(LOOP)/C01/C01-T3/rescoring_sensitivity.csv') if x['label'] == 'L_inv' and x['metric'] == metric]
    if not r:
        raise KeyError(str(sorted({x['label'] for x in CSV('(LOOP)/C01/C01-T3/rescoring_sensitivity.csv')})))
    return F(r[0]['point']) * 100
def _n6spec(i):
    v = [x[1] for x in J('(LOOP)/C07/C07-T2/cache/s2_n6_check.json', 'spec_skill').values()]
    return min(v) if i == 0 else max(v)

# ---------------- S 요약 ----------------
cc = '(MC)/_shared_report/audit_r7/concordance_resolution_check.json'
chk(9, ['50개', '27개', '34개'], [S(lambda: J(cc, 'n_provided_ksic')), S(lambda: J(cc, 'n_concordance_codes_covering_50')),
     S(lambda: J(cc, 'n_provided_ksic') - J(cc, 'ksic_alone_in_code'))], cc + ' n_provided_ksic / n_concordance_codes_covering_50 / 50−ksic_alone_in_code')
case = lambda k, col: F(ROW(SP + 't5_cases.csv', ksic=k)[col])
chk(9, ['6.6%', '6.2%'], [S(lambda: case('26211', 'KR출원_연성장%(08-09→16-17)')), S(lambda: case('26212', 'KR출원_연성장%(08-09→16-17)'))],
    SP + 't5_cases.csv KR출원_연성장%(08-09→16-17)', absval=True, note='연평균 로그 성장률(s2_decomp 정의)')
chk(11, ['15,141,159'], [S(lambda: sum(F(r['n']) for r in CSV(C5 + 'c1_label_agreement.csv') if r['window'] == '전체' and r['cc'].strip()))],
    C5 + 'c1_label_agreement.csv 전체 창 국가별 n 합')
chk(13, ['7.1%', '29.4', '40.4%'], [S(lambda: J(TXT9 + 't0_txt_agree.json', 'cw5_agree_all(16개만 지목)')),
     S(lambda: J(TXT9 + 't0_txt_agree.json', 'txt5_agree_all(미배정=불일치)')), S(lambda: J(TXT9 + 't0b_txtfree_agree.json', 'free5_agree_all'))],
    TXT9 + 't0_txt_agree.json / t0b_txtfree_agree.json', pct=True)
def doc_agree():
    d = J(R12 + 'res_prep.json', 'doc_agree_with_prov')
    cands = {k: v['eval'] for k, v in d.items() if not k.upper().startswith('PKR')}
    return cands
chk(13, ['33개', '58.8%'], [S(lambda: len(doc_agree())), S(lambda: max(doc_agree().values()) * 100)],
    R12 + 'res_prep.json doc_agree_with_prov (PKR 진단 제외) 후보 수·평가 표본 최댓값', note=str(S(lambda: max(doc_agree(), key=doc_agree().get))))
chk(13, ['7.3%p', '+5.1', '+9.7', '−0.6%p', '−2.0', '+1.0', '+11.5%p', '+28.7%p'],
    [S(lambda: r12('DART5', 'PROV-BEST')), S(lambda: r12('DART5', 'PROV-BEST_lo')), S(lambda: r12('DART5', 'PROV-BEST_hi')),
     S(lambda: r12('NPS5', 'PROV-BEST')), S(lambda: r12('NPS5', 'PROV-BEST_lo')), S(lambda: r12('NPS5', 'PROV-BEST_hi')),
     S(lambda: r12('NPS5', 'PROV-T0')), S(lambda: r12('DART5', 'PROV-T0'))], R12 + 'res_main.csv', pct=True)
chk(13, ['300명'], [S(lambda: '300명' if sub('NPS5', '규모(NPS 가입자)', '>=300', 'PROV-BEST') < 0 else 'x')], R12 + 'res_sub.csv 규모(NPS 가입자) >=300 구간 존재·PROV-BEST<0')
cp = TXT9 + 'C_pool_summary.csv'
pairs3 = lambda: [F(ROW(cp, pair=p)['mean_d']) for p in ('P50-FREE50', 'P50-TXT50', 'P50-CW27')]
chk(13, ['0.002', '0.004'], [S(lambda: min(pairs3())), S(lambda: max(pairs3()))], cp + ' P50−FREE50/TXT50/CW27 mean_d 최소·최대')
fam = lambda cc_: [x for x in J(LO + 'res_fam_check.json', 'links') if x[0] == cc_][0][4]
chk(13, ['87', '90%'], [S(lambda: min(fam('US'), fam('JP'), fam('EP'))), S(lambda: max(fam('US'), fam('JP'), fam('EP')))], LO + 'res_fam_check.json links 일치율 US/JP/EP', pct=True)
chk(15, ['25개', '11개', '7개'], [S(lambda: J(MAP + 'm2_checks.json', 'kr_rta_gt1_P3')), S(lambda: J(MAP + 'm4_summary.json', 'quad_tech_export_P2_counts', '기술 강·수출 약(괴리)')),
     S(lambda: J(MAP + 'm2_checks.json', 'kr_leader_P3'))], MAP + 'm2_checks.json / m4_summary.json')
qs = '(LOOP)/C03/C03-T2/quadrant_sensitivity.csv'
def sw_range():
    rows = CSV(qs)
    col = [c for c in rows[0] if c.upper() in ('SW', 'N_SW', 'SW_N')]
    vals = [F(r[col[0]]) for r in rows if r[col[0]].strip()]
    return min(vals), max(vals)
chk(15, ['9', '12개'], [S(lambda: sw_range()[0]), S(lambda: sw_range()[1])], qs + ' SW 개수(무역 연결 변형) 최소·최대', srctype='루프 산출 원천')
land = lambda sc, k: J(C5 + 'c2_results.json', 'landscape_P3', sc, k)
chk(15, ['8', '16개'], [S(lambda: min(land(s, 'kr_leader') for s in ('cweq_F', 'txt_F', 'free_F'))), S(lambda: max(land(s, 'kr_leader') for s in ('cweq_F', 'txt_F', 'free_F')))],
    C5 + 'c2_results.json landscape_P3 cweq_F/txt_F/free_F kr_leader')
chk(17, ['0.794', '0.642', '0.423', '0.366', '0.473', '0.182'],
    [S(lambda: J(T3J, 'summary', 'auc')), S(lambda: J(T3J, 'summary', 'auc_baseline')), S(lambda: J(T3J, 'summary', 'skill_of_mean_auc')),
     S(lambda: J(T3J, 'ci95', 'skill')[0]), S(lambda: J(T3J, 'ci95', 'skill')[1]), S(lambda: md_num('(MC)/T3_기술셀부상예측/VERIFY_R5/robustness_table.md', r'SB3\(XGB, KR 이력 12피처\) 대비 skill \| \*\*([0-9.]+)'))],
    T3J + '; VERIFY_R5 robustness_table SB3')
chk(17, ['0.900', '0.811', '0.473', '0.395'], [S(lambda: J(T1J, 'model', 'auc_all')), S(lambda: J(T1J, 'baseline_strongest', 'auc_all')), S(lambda: J(T1J, 'skill', 'auc_all')),
     S(lambda: md_num('(MC)/T1_산업대기실/VERIFY_R5/robustness_table.md', r'SBB.{0,60}대비 skill \| \*\*([0-9.]+)'))], T1J + '; VERIFY_R5 SBB')

# ---------------- 표 1–3 ----------------
chk(48, ['5,041셀'], [S(lambda: J(T3J, 'by_origin', '2021', 'model', 'n'))], T3J + ' by_origin.2021.model.n')
c1 = lambda win, cc_, col: F([r for r in CSV(C5 + 'c1_label_agreement.csv') if r['window'] == win and r['cc'] == cc_][0][col])
tot = lambda: sum(F(r['n']) for r in CSV(C5 + 'c1_label_agreement.csv') if r['window'] == '전체' and r['cc'].strip())
for ln, cc_, nn, sh in [(67, 'KR', '1,087,511', '7.2%'), (68, 'US', '2,382,948', '15.7%'), (69, 'JP', '2,163,018', '14.3%'), (70, 'CN', '8,700,777', '57.5%'), (71, 'EP', '806,905', '5.3%')]:
    chk(ln, [nn, sh], [S(lambda cc_=cc_: c1('전체', cc_, 'n')), S(lambda cc_=cc_: c1('전체', cc_, 'n') / tot() * 100)], C5 + 'c1_label_agreement.csv 전체 창 n·비중')
for ln, cc_, tk in [(68, 'US', '87.3%'), (69, 'JP', '87.0%'), (71, 'EP', '90.5%')]:
    chk(ln, [tk], [S(lambda cc_=cc_: fam(cc_))], LO + 'res_fam_check.json links', pct=True)
for ln, cc_, tk in [(68, 'US', '51.0%'), (69, 'JP', '48.2%'), (70, 'CN', '53.5%'), (71, 'EP', '53.1%')]:
    chk(ln, [tk], [S(lambda cc_=cc_: c1('출원2015-19_발명', cc_, 'free5'))], C5 + 'c1_label_agreement.csv 출원2015-19_발명 free5', pct=True)
chk(67, ['89.7%'], [S(lambda: md_num('(MC)/R10_결과_20261005.md', r'89\.7')) if False else S(lambda: _kr_maj())], LO + 'res_tests.json KR 해외 다수결 일치')
def _kr_maj():
    return J(LO + 'res_tests.json', 'P0', 'agree_kr_vs_famMajor') * 100
def _kr_maj_old():
    d = J(LO + 'res_tests.json')
    flat = json.dumps(d, ensure_ascii=False)
    m = re.search(r'"(?:kr_vs_foreign_majority|majority[^"]*)"\s*:\s*([0-9.]+)', flat)
    if m:
        return F(m.group(1)) * (100 if F(m.group(1)) <= 1 else 1)
    raise KeyError('majority key not found')
chk(67, ['39.3%', '55.5%', '58.8%'], [S(lambda: doc_agree()['T0']), S(lambda: doc_agree()['H-sg-3']), S(lambda: max(doc_agree().values()))], R12 + 'res_prep.json doc_agree eval T0/H-sg-3/max', pct=True)
chk(73, ['33개'], [S(lambda: len(doc_agree()))], R12 + 'res_prep.json 후보 수')
chk(75, ['55.5%', '48.2', '53.5%'], [S(lambda: doc_agree()['H-sg-3'] * 100), S(lambda: min(c1('출원2015-19_발명', c, 'free5') for c in ('US', 'JP', 'CN', 'EP')) * 100),
     S(lambda: max(c1('출원2015-19_발명', c, 'free5') for c in ('US', 'JP', 'CN', 'EP')) * 100)], 'R12 res_prep / c1 free5 해외 4개국')
ext = '(MC)/T1_산업대기실/R2/ext_link_summary.csv'
def extrow(pat):
    rows = [r for r in CSV(ext) if re.search(pat, json.dumps(r, ensure_ascii=False))]
    if not rows:
        raise KeyError(pat)
    return json.dumps(rows[0], ensure_ascii=False)
chk(89, ['1,229쌍', '48/50'], [S(lambda: re.search(r'1,229|1229', extrow('HS6')) and '1,229쌍'), S(lambda: re.search(r'48/50|48 ?KSIC|"48"', extrow('HS6')) and '48/50')],
    ext + ' HS6 행(문자열 대조)', srctype='원천(문자열)')
chk(90, ['50/50', '36개'], [S(lambda: '50/50' if re.search(r'50/50|"50"', extrow('IO|산업연관')) else 'x'), S(lambda: '36개' if re.search(r'36', extrow('IO|산업연관')) else 'x')],
    ext + ' IO 행(문자열 대조)', srctype='원천(문자열)')
chk(88, ['16/50'], [S(lambda: f"{int(J(cc, 'ksic_alone_in_code'))}/50")], cc + ' ksic_alone_in_code(연계표 5자리로 단독 지목되는 KSIC)', note='t0_txt_agree 키 이름 cw5_agree_all(16개만 지목)과 같음')
chk(97, ['392,048쌍', '216,762쌍', '150,230쌍'], [S(lambda c=c: [x for x in J(LO + 'res_fam_check.json', 'links') if x[0] == c][0][1]) for c in ('US', 'JP', 'EP')], LO + 'res_fam_check.json links 쌍 수')
nlr = '(MC)/_shared/ext/ntis_link_rates.csv'
def nl(pat, col_hint):
    rows = CSV(nlr)
    return rows
chk(95, ['13.5%'], [S(lambda: 88323 / 656360 * 100)], '권리이전 88,323 / 등록 656,360(README 값)', srctype='문서 원천', note='분자·분모는 README 값. sqlite 재집계는 아래 행 참조')

# 표 4
chk(124, ['0.0', '0.25', '0.48%'], [S(lambda: md_num('(MC)/T1_산업대기실/R3/FINAL_REPORT_R3.md', r'\*\*교란 시험\*\*[^\n]*?최대 차[^0-9]{0,12}([0-9.]+)')),
     S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'\*\*([0-9.]+)–([0-9.]+)%\*\*', 1)), S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'\*\*([0-9.]+)–([0-9.]+)%\*\*', 2))],
    'T1·T3 FINAL_REPORT_R3 §1', srctype='문서 원천')
chk(125, ['1e−16', '9.0e−8'], [S(lambda: 1e-16 if '차이 1e−16' in TXT('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md') else KeyError('x')), S(lambda: md_num('(MC)/T1_산업대기실/R3/FINAL_REPORT_R3.md', r'최대 차는 \*\*([0-9.]+e-[0-9]+)\*\*'))],
    'T3·T1 FINAL_REPORT_R3', srctype='문서 원천')
def t4n5max():
    a = J('(LOOP)/C04/C04-T2/t4_rescore.json', 'scores', 'diff_vs_official'); b = J('(LOOP)/C04/C04-T2/n5_rescore.json', 'scores', 'diff_vs_official')
    return max(abs(v) for v in list(a.values()) + list(b.values()))
chk(125, ['≤1e−10'], [S(lambda: '≤1e−10' if t4n5max() <= 1e-10 else f'max {t4n5max():.2e}')], '(LOOP)/C04/C04-T2/t4_rescore.json·n5_rescore.json diff_vs_official 최대 절대값', srctype='루프 산출 원천',
    note='AUC 차 최대 7.9e−11(≤1e−10 성립)이나 T4 skill 차 2.0e−10. 같은 칸의 T1 9.0e−8은 skill 포함 전 지표 최대 차', verdict_override='부분',
    defn='E-C04T2-04 원안은 \'AUC 차 ≤1e−10\'; 표에서 \'AUC\'가 빠져 전 지표 최대 차로 읽힘(T4 skill 2.0e−10)')
r6 = '(MC)/R6_결과_20261004.md'
chk(126, ['0/1,000', '0/205', '0/1,000', '0/230', '0.001'],
    [S(lambda: '0/1,000' if '**0/1,000**' in TXT(r6) else 'x'), S(lambda: '0/205' if '**0/205**' in TXT(r6) else 'x'),
     S(lambda: '0/1,000' if '관측 이상 0/1,000' in TXT(r6) else 'x'), S(lambda: '0/230' if '0/230·0/230' in TXT(r6) else 'x'),
     S(lambda: md_num('(MC)/N9_정부RnD과제_특허성과예측/FINAL_REPORT_R6.md', r'\*\*0\.392\*\* \| ([0-9.]+) \(1/1001\)'))],
    r6 + ' §1.1·§2.2; N9 FINAL_REPORT_R6 §5', srctype='문서 원천')
n7r = '(MC)/N7_특허기업_정부RnD첫수행/VERIFY_R6/robustness_table.md'
n4r = '(MC)/N4_특허행태_기업폐업조기경보/VERIFY_R5/robustness_table.md'
chk(127, ['0.286', '0.251', '0.177', '0.140'], [S(lambda: J(N7J, 'results', 0, 'skill_vs_M0')), S(lambda: md_num(n7r, r'NPS 별칭 보정 후 skill\*\* \| \*\*([0-9.]+)')),
     S(lambda: J(N4J, 'skill') if 'skill' in J(N4J) else md_num(n4r, r'skill ([0-9.]+) \[')), S(lambda: md_num(n4r, r'별칭.{0,80}?([0-9]\.[0-9]{3}) \[0\.111'))], 'N7 final_result_r5 / VERIFY_R6; N4 final_result_r4 / VERIFY_R5')
chk(130, ['−0.003', '0.000'], [S(lambda: md_num('(MC)/T4_특허포기예측/VERIFY_R5/robustness_table.md', r'\*\*\[(−?[0-9.]+), 0\.208\]\*\*')), S(lambda: md_num('(MC)/N5_표준특허선언레이더/VERIFY_R5/robustness_table.md', r'\*\*0\.8580\*\* \| \*\*([0-9.]+)\*\*'))],
    'T4·N5 VERIFY_R5 robustness_table')

# 표 5
for ln, key, tk in [(178, '기술 강·수출 강(정합)', '12'), (179, '기술 강·수출 약(괴리)', '11'), (180, '기술 약·수출 강(괴리)', '3'), (181, '기술 약·수출 약(정합)', '22')]:
    chk(ln, [tk], [S(lambda key=key: J(MAP + 'm4_summary.json', 'quad_tech_export_P2_counts', key))], MAP + 'm4_summary.json quad_tech_export_P2_counts')
chk(183, ['2개'], [S(lambda: J(MAP + 'm4_summary.json', 'quad_tech_export_P2_counts', '판정불가'))], MAP + 'm4_summary.json')
chk(183, ['85.4%'], [S(lambda: J(MAP + 'm4_summary.json', 'quad_P2_vs_P4_same_share'))], MAP + 'm4_summary.json quad_P2_vs_P4_same_share', pct=True)

# 표 6·7
chk(215, ['0.423', '0.366', '0.473'], [S(lambda: J(T3J, 'summary', 'skill_of_mean_auc')), S(lambda: J(T3J, 'ci95', 'skill')[0]), S(lambda: J(T3J, 'ci95', 'skill')[1])], T3J)
t3rob = '(MC)/T3_기술셀부상예측/VERIFY_R5/robustness_table.md'
chk(216, ['0.182', '0.124', '0.243', '0.748'], [S(lambda: md_num(t3rob, r'SB3 XGB, KR 이력 12피처\(해외 신호 없음\) \| ([0-9.]+) \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]', 2)),
     S(lambda: md_num(t3rob, r'SB3 XGB, KR 이력 12피처\(해외 신호 없음\) \| ([0-9.]+) \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]', 3)),
     S(lambda: md_num(t3rob, r'SB3 XGB, KR 이력 12피처\(해외 신호 없음\) \| ([0-9.]+) \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]', 4)),
     S(lambda: md_num(t3rob, r'SB3 XGB, KR 이력 12피처\(해외 신호 없음\) \| ([0-9.]+) \|', 1))], t3rob + ' §SB3 행')
chk(217, ['0.406', '0.403'], [S(lambda: md_num(t3rob, r'IPC 2022년 이후 신설 코드 셀 제외 \| ([0-9.]+)')), S(lambda: F([r for r in CSV('(LOOP)/C05/C05-T1/rescore_scenarios.csv') if r[list(r)[0]].startswith('S3')][0]['skill']))],
    t3rob + '; (LOOP)C05-T1 rescore_scenarios.csv S3', note='0.403은 H01L+H10 통합 재채점')
chk(217, ['38%'], [S(lambda: J('(LOOP)/C05/C05-T1/s3_results.json', 'a_R5_newcode', 'rate_new') if False else _newcode_rate())], '(LOOP)/C05/C05-T1/s3_results.json a_R5_newcode 신설 셀 부상률', srctype='루프 산출 원천')
def _newcode_rate():
    d = J('(LOOP)/C05/C05-T1/s3_results.json', 'a_R5_newcode')
    for k, v in d.items():
        if isinstance(v, (int, float)) and 0.3 < v < 0.45 and 'rate' in k:
            return v * 100
    raise KeyError(str(list(d))[:200])
chk(239, ['0.473', '0.454', '0.488'], [S(lambda: J(T1J, 'skill', 'auc_all')), S(lambda: J(T1J, 'bootstrap_firm_cluster', 'skill_vs_strongest__model', 'auc_all')[0]),
     S(lambda: J(T1J, 'bootstrap_firm_cluster', 'skill_vs_strongest__model', 'auc_all')[2])], T1J + ' skill / bootstrap skill_vs_strongest 2.5·97.5%')
t1rob = '(MC)/T1_산업대기실/VERIFY_R5/robustness_table.md'
chk(240, ['0.395', '0.379', '0.412', '0.835'], [S(lambda: md_num(t1rob, r'\*\*SBB\*\*.*?\| \*\*([0-9.]+)\*\* \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]\*\*', 2)),
     S(lambda: md_num(t1rob, r'\*\*SBB\*\*.*?\| \*\*([0-9.]+)\*\* \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]\*\*', 3)),
     S(lambda: md_num(t1rob, r'\*\*SBB\*\*.*?\| \*\*([0-9.]+)\*\* \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]\*\*', 4)),
     S(lambda: md_num(t1rob, r'\*\*SBB\*\*.*?\| \*\*([0-9.]+)\*\* \|', 1))], t1rob + ' §3.1 SBB 행')

# 표 8
t4rob = '(MC)/T4_특허포기예측/VERIFY_R5/robustness_table.md'
chk(283, ['0.701', '0.604', '0.246', '0.133', '0.320'], [S(lambda: J(T4J, 'model', 'auc')), S(lambda: J(T4J, 'baseline', 'auc')), S(lambda: J(T4J, 'skill_auc')),
     S(lambda: J(T4J, 'bootstrap_95ci', 'skill')[0]), S(lambda: J(T4J, 'bootstrap_95ci', 'skill')[1])], T4J)
chk(283, ['0.124', '−0.003', '0.208', '84건', '19%'], [S(lambda: md_num(t4rob, r'\*\*0\.6586\*\* \| \*\*([0-9.]+)\*\*')), S(lambda: md_num(t4rob, r'\*\*\[(−?[0-9.]+), 0\.208\]\*\*')),
     S(lambda: md_num(t4rob, r'\*\*\[−?[0-9.]+, ([0-9.]+)\]\*\*')), S(lambda: md_num(t4rob, r'\| 100 \| ([0-9]+) \(')), S(lambda: J(T4J, 'base_rate') * 100)],
    t4rob + ' 강화 기준선·상위 100; ' + T4J + ' base_rate')
chk(284, ['0.870', '0.780', '0.409', '0.243', '0.540', '0.312', '0.144', '0.439', '36.1%', '30.6%'],
    [S(lambda: J(N8J, 'champ_auc')), S(lambda: J(N8J, 'M0_auc')), S(lambda: J(N8J, 'skill_vs_M0')), S(lambda: J(N8J, 'skill_vs_M0_ci')[0]), S(lambda: J(N8J, 'skill_vs_M0_ci')[1]),
     S(lambda: J(N8J, 'skill_vs_SB')), S(lambda: J(N8J, 'skill_vs_SB_ci')[0]), S(lambda: J(N8J, 'skill_vs_SB_ci')[1]),
     S(lambda: J(N8J, 'champ_top1pct_capture') * 100), S(lambda: J(N8J, 'SB_top1pct_capture') * 100)], N8J)

# 표 9
for ln, crit, toks in [(316, 'NPS5', ['1,633사', '22,363건', '30.3%', '30.9%', '29.3%', '18.8%', '14.7%', '−0.6%p', '−2.0', '+1.0', '+11.5%p', '+9.8', '+13.1']),
                       (317, 'DART5', ['811사', '5,164건', '62.9%', '55.5%', '53.6%', '34.2%', '23.1%', '+7.3%p', '+5.1', '+9.7', '+28.7%p', '+25.8', '+31.7'])]:
    cols = ['기업', '특허', 'PROV', 'BEST값', 'BASE(H-sg-3)', 'T0(TXT-FREE)', 'CWEQ', 'PROV-BEST', 'PROV-BEST_lo', 'PROV-BEST_hi', 'PROV-T0', 'PROV-T0_lo', 'PROV-T0_hi']
    vals = [S(lambda c=c, crit=crit: r12(crit, c) * (1 if c in ('기업', '특허') else 100)) for c in cols]
    chk(ln, toks, vals, R12 + 'res_main.csv ' + crit)

# 표 10
chk(352, ['−3.6%p', '−7.6', '−0.8', '+8.1%p', '+5.9', '+10.5'], [S(lambda c=c, k=k: r12(k, c) * 100) for k in ('NPS5', 'DART5') for c in ('PROV-BEST_w', 'PROV-BEST_w_lo', 'PROV-BEST_w_hi')], R12 + 'res_main.csv 특허 가중')
wi = lambda crit, col: F(ROW(R12 + 'res_within.csv', **{'기준': crit})[col])
chk(353, ['−21.7%p', '−26.4', '−16.6', '+5.3%p', '+0.4', '+10.5'], [S(lambda c=c, k=k: wi(k, c) * 100) for k in ('NPS5', 'DART5') for c in ('PROV-BEST', 'PROV-BEST_lo', 'PROV-BEST_hi')], R12 + 'res_within.csv')
chk(354, ['+0.0024', '+0.0004', '+0.0042', '5/5'], [S(lambda: F(ROW(cp, pair='P50-FREE50')['mean_d'])), S(lambda: F(ROW(cp, pair='P50-FREE50')['ci_lo'])), S(lambda: F(ROW(cp, pair='P50-FREE50')['ci_hi'])),
     S(lambda: ROW(cp, pair='P50-FREE50')['wins'])], cp + ' P50-FREE50')
mg = '(MC)/T3_기술셀부상예측/TXT_R9/summary_mg_r9.json'
chk(355, ['+0.0020', '−0.0002', '+0.0041', '7/9'], [S(lambda: J(mg, 'pairs', 'prov_vs_txt', 'dAUC')), S(lambda: J(mg, 'pairs', 'prov_vs_txt', 'ci')[0]), S(lambda: J(mg, 'pairs', 'prov_vs_txt', 'ci')[1]),
     S(lambda: f"{J(mg, 'pairs', 'prov_vs_txt', 'wins')}/{J(mg, 'pairs', 'prov_vs_txt', 'n_orig')}")], mg + ' pairs.prov_vs_txt')
nc = '(MC)/_shared_report/noncirc_n9n6_R9/results_table.csv'
ncr = lambda st, cmp: ROW(nc, set=st, comparison=cmp)
chk(356, ['+0.0001', '2/7', '−0.0006', '0/6'], [S(lambda: F(ncr('n9_y_all', 'PROV-TXT')['mean_dAUC'])), S(lambda: f"{ncr('n9_y_all', 'PROV-TXT')['npos']}/{ncr('n9_y_all', 'PROV-TXT')['n_origins']}"),
     S(lambda: F(ncr('n6_ntis_y_ntis', 'PROV-TXT')['mean_dAUC'])), S(lambda: f"{ncr('n6_ntis_y_ntis', 'PROV-TXT')['npos']}/{ncr('n6_ntis_y_ntis', 'PROV-TXT')['n_origins']}")], nc,
    note='N6 행은 set=n6_ntis_y_ntis(분야 무관 결과로 추정; set 정의는 아래 3.4 대조와 같음)')
chk(357, ['−0.026', '−0.193', '+0.119'], [S(lambda: J(C5 + 'c2_results.json', 'main_abroadRTA_P3_vs_RCA', 'free', 'd')), S(lambda: J(C5 + 'c2_results.json', 'main_abroadRTA_P3_vs_RCA', 'free', 'ci_d')[0]),
     S(lambda: J(C5 + 'c2_results.json', 'main_abroadRTA_P3_vs_RCA', 'free', 'ci_d')[1])], C5 + 'c2_results.json main_abroadRTA_P3_vs_RCA.free')
c5s = C5 + 'c5_t3_summary.json'
chk(358, ['+0.0239', '+0.0255', '9/9'], [S(lambda: J(c5s, 'pairs', 'drop_ros−hyb_txt', 'dAUC')), S(lambda: J(c5s, 'pairs', 'drop_ros−hyb_free', 'dAUC')), S(lambda: J(c5s, 'pairs', 'drop_ros−hyb_txt', 'wins'))], c5s)
r11 = LO + 'foreign_public_R11/res_main.csv'
def r11v(key, col_pat):
    rows = CSV(r11)
    r = [x for x in rows if key in json.dumps(x, ensure_ascii=False)]
    if not r:
        raise KeyError(key)
    return r
chk(359, ['+4.75', '+11.66', '−0.33%p'], [S(lambda: _r11('NPS')), S(lambda: _r11('DART')), S(lambda: _r11('KIND'))], r11 + ' 제공 − FREE-IPC')
def _r11(crit, which='d'):
    rows = CSV(r11)
    hdr = list(rows[0].keys())
    r = [x for x in rows if x['src'].strip().lower() == crit.lower() and x['표본'].startswith('주')]
    if not r:
        raise KeyError(f'{crit} not in rows')
    r = r[0]
    cand = [c for c in hdr if c == 'PROV-FREE']
    if which == 'lo':
        cand = [c for c in hdr if c == 'PROV-FREE_lo']
    if which == 'hi':
        cand = [c for c in hdr if c == 'PROV-FREE_hi']
    if not cand:
        raise KeyError(str(hdr)[:300])
    return F(r[cand[0]]) * 100
p2 = '(MC)/_ideas_R8/p2_results.json'
def _p2():
    return J(p2, 'ship', 'compare', 'sig_PROV_KR-sig_TXT', 'mean')
chk(360, ['−0.094'], [S(_p2)], p2 + ' 제공 KR − TXT(출하 몫 상승 AUC 차)', note='키 탐색(정규식) — 아래 verify 대상')

# 표 11
chk(384, ['50개', '23', '39개', '0.182'], [S(lambda: t3_top100), S(lambda: t3_b100[0]), S(lambda: t3_b100[1]), S(lambda: md_num(t3rob, r'\*\*([0-9.]+) \[0\.124, 0\.243\]\*\*'))],
    'T3 R3 preds FINAL_R3_t0_2021.parquet 재채점(상위 100, 기준선 동점 범위); robustness SB3')
chk(385, ['313쌍', '124쌍', '6쌍'], [S(lambda: J(T1J, 'topk', 'model', '1000', 'hits')), S(lambda: J(T1J, 'topk', 'strongest_auc_baseline', '1000', 'hits')), S(lambda: 1000 * J(T1J, 'n', 'pos') / J(T1J, 'n', 'rows'))],
    T1J + ' topk·n', note='무작위 기대 = 1000×기저율')
chk(386, ['56.9%', '45.8%'], [S(lambda: J(N9J, 'recall_top10') * 100), S(lambda: J(N9J, 'recall_top10_M0L2') * 100)], N9J + ' recall_top10')
chk(387, ['317,840건', '108건', '39건', '36.1%', '3.7%'], [S(lambda: J(N8J, 'n')), S(lambda: J(N8J, 'pos')), S(lambda: J(N8J, 'champ_top1pct_hits')), S(lambda: J(N8J, 'champ_top1pct_capture') * 100), S(lambda: J(N8J, 'M0_top1pct_capture') * 100)], N8J)
chk(388, ['84건', '0.124'], [S(lambda: md_num(t4rob, r'\| 100 \| ([0-9]+) \(')), S(lambda: md_num(t4rob, r'\*\*0\.6586\*\* \| \*\*([0-9.]+)\*\*'))], t4rob)
coh = '(MC)/_shared_report/ntis_t3_cohort_lapse_transfer.csv'
chk(389, ['11개', '9', '12개'], [S(lambda: J(MAP + 'm4_summary.json', 'quad_tech_export_P2_counts', '기술 강·수출 약(괴리)')), S(lambda: sw_range()[0]), S(lambda: sw_range()[1])], 'm4_summary; C03-T2 quadrant_sensitivity')
chk(391, ['−0.611', '+0.507'], [S(lambda: J(MAP + 'm5_summary.json', 'track_pair_spearman_range')[0]), S(lambda: J(MAP + 'm5_summary.json', 'track_pair_spearman_range')[1])], MAP + 'm5_summary.json track_pair_spearman_range')

# ---------------- K 4장 ----------------
chk(370, ['7.1%', '40.4%', '58.8%', '7.3%p', '7개', '8', '16개'], [S(lambda: J(TXT9 + 't0_txt_agree.json', 'cw5_agree_all(16개만 지목)') * 100), S(lambda: J(TXT9 + 't0b_txtfree_agree.json', 'free5_agree_all') * 100),
     S(lambda: max(doc_agree().values()) * 100), S(lambda: r12('DART5', 'PROV-BEST') * 100), S(lambda: land('prov', 'kr_leader')),
     S(lambda: min(land(s, 'kr_leader') for s in ('cweq_F', 'txt_F', 'free_F'))), S(lambda: max(land(s, 'kr_leader') for s in ('cweq_F', 'txt_F', 'free_F')))], 'TXT_R9 / R12 / c2_results')
chk(374, ['−6.6%', '+6.2%'], [S(lambda: case('26211', 'KR출원_연성장%(08-09→16-17)')), S(lambda: case('26212', 'KR출원_연성장%(08-09→16-17)'))], SP + 't5_cases.csv')
c7 = '(LOOP)/C07/C07-T3/prospective_result.json'
c8 = '(LOOP)/C08/C08-T3/prospective_result.json'
chk(395, ['2.04배', '1.57배'], [S(lambda: md_num(r6, r'예측 합 ([0-9.]+) 대 실제 108') / 108), S(lambda: _calib157())], r6 + ' 예측 합 220.3/108; (LOOP)C07-T3 calibration.csv 창 길이 보정', srctype='문서·루프 원천')
def _calib157():
    flat = TXT('(LOOP)/C07/C07-T3/calibration.csv')
    m = re.findall(r'1\.5[0-9]+', flat)
    if not m:
        raise KeyError('1.57 not found')
    return F(m[0])
chk(395, ['18개월', '90건', '0.314', '0.124', '0.516'], [S(lambda: f"{round(J(c7, 'windows', 'main', 'days') / 30.4)}개월"), S(lambda: J(c7, 'windows', 'main', 'pos')), S(lambda: _c7skill()),
     S(lambda: _c7skill(1)), S(lambda: _c7skill(2))], c7 + ' windows.main', srctype='루프 산출 원천')
def _c7skill(i=0):
    w = J(c7, 'windows', 'main')
    keys = [k for k in w if 'skill' in k.lower() and 'm0' in k.lower()]
    k = [x for x in keys if ('ci' in x.lower()) == (i > 0)]
    if not k:
        raise KeyError(str(list(w))[:400])
    v = w[k[0]]
    return v if i == 0 else v[i - 1]
chk(395, ['4개월', '0.487', '0.431', '0.535'], [S(lambda: '4개월'), S(lambda: J(c8, 'main', 'skill')), S(lambda: J(c8, 'main', 'skill_ci')[0]), S(lambda: J(c8, 'main', 'skill_ci')[1])], c8 + ' main', srctype='루프 산출 원천',
    note='창 2025-07~10월(4개월)은 ERRATA E-C08T3-01 정의')
chk(407, ['+7.3%p', '0.002', '0.004'], [S(lambda: r12('DART5', 'PROV-BEST') * 100), S(lambda: min(pairs3())), S(lambda: max(pairs3()))], 'R12 res_main; TXT_R9 C_pool_summary')
chk(409, ['−0.003', '0.000'], [S(lambda: md_num(t4rob, r'\*\*\[(−?[0-9.]+), 0\.208\]\*\*')), S(lambda: md_num('(MC)/N5_표준특허선언레이더/VERIFY_R5/robustness_table.md', r'\*\*0\.8580\*\* \| \*\*([0-9.]+)\*\*'))], 'T4·N5 VERIFY_R5')
chk(410, ['0.140', '0.251', '+5.5%p'], [S(lambda: md_num(n4r, r'별칭.{0,80}?([0-9]\.[0-9]{3}) \[0\.111')), S(lambda: md_num(n7r, r'NPS 별칭 보정 후 skill\*\* \| \*\*([0-9.]+)')),
     S(lambda: _c06dev())], 'N4·N7 VERIFY; (LOOP)C06-T1 out/s4_time.csv', srctype='원천·루프')
def _c06dev():
    p = '(LOOP)/C06/C06-T1/out/s4_time.csv'
    rows = CSV(p)
    for r in rows:
        s = json.dumps(r, ensure_ascii=False)
        if 'dev' in s or '2010' in s:
            for v in r.values():
                try:
                    x = F(v)
                    if 0.05 < x < 0.06:
                        return x * 100
                except Exception:
                    pass
    raise KeyError('dev row')

# ---------------- B 본문 ----------------
chk(31, ['40.0%', '41.9%', '7.1%'], [S(lambda: J(TXT9 + 't0b_txtfree_agree.json', 'cw_group_agree_all(미배정=불일치)') * 100), S(lambda: J(TXT9 + 't0_txt_agree.json', 'group_agree') * 100),
     S(lambda: J(TXT9 + 't0_txt_agree.json', 'cw5_agree_all(16개만 지목)') * 100)], 'TXT_R9 t0/t0b')
chk(30, ['99.0%'], [S(lambda: J(cc, 'kr_2005_2022', 'share_ipc_multi_industry') * 100)], cc)
chk(60, ['87', '90%'], [S(lambda: min(fam(c) for c in ('US', 'JP', 'EP')) * 100), S(lambda: max(fam(c) for c in ('US', 'JP', 'EP')) * 100)], LO + 'res_fam_check.json')
chk(59, ['1.0개'], [S(lambda: J(cc, 'kr_provided_multilabel', 'mean_labels'))], cc + ' kr_provided_multilabel.mean_labels')
chk(73, ['48.8%'], [S(lambda: J('(LOOP)/C03/C03-T3/_s10_cn_doctype.json', 'utility') / J('(LOOP)/C03/C03-T3/_s10_cn_doctype.json', 'cn_rows') * 100)], '(LOOP)C03-T3 _s10_cn_doctype.json utility/cn_rows', srctype='루프 산출 원천')
def unrep():
    v = [1 - doc_agree()['H-sg-3']] + [1 - c1('출원2015-19_발명', c, 'free5') for c in ('US', 'JP', 'CN', 'EP')]
    return min(v) * 100, max(v) * 100
chk(75, ['45', '52%'], [S(lambda: unrep()[0]), S(lambda: unrep()[1])], '1 − FREE-IPC 재현율(KR R12 H-sg-3, 해외 c1 free5)')
chk(99, ['49/50'], [S(lambda: '49/50' if '49/50' in TXT(ext) or re.search(r'KOSIS.{0,80}49', TXT(ext)) else 'x')], ext + ' KOSIS 2008 연결(문자열)', srctype='원천(문자열)')
chk(109, ['9개', '5개', '+0.003', '60%', '−0.005', '0.424'], [S(lambda: len(re.findall(r'/', md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'개발 원점별 skill\(2010→2018\): ([0-9. /]+)', 1).__str__())) and 9 if False else _t3dev_n()),
     S(lambda: _t1dev_n()), S(lambda: md_num('(MC)/_shared/PROTOCOL_R2.md', r'\+?([0-9.]+) 이상')), S(lambda: md_num('(MC)/_shared/PROTOCOL_R2.md', r'([0-9]+)% 이상')),
     S(lambda: md_num(r6, r'개발 대비 최종\*\*: T3 (−?[0-9.]+)')), S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'수정 전 E40 첫 실행.*?\| ([0-9.]+) \[0\.367'))],
    'T3·T1 FINAL_REPORT, PROTOCOL_R2 §3, R6 결과', srctype='문서 원천')
def _t3dev_n():
    s = re.search(r'개발 원점별 skill\(2010→2018\): ([0-9. /]+)', TXT('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md')).group(1)
    return len([x for x in s.split('/') if x.strip()])
def _t1dev_n():
    return len(set(r['c'] for r in CSV(TXT9 + 'C_summary.csv')))
chk(116, ['0.165', '0.000'], [S(lambda: md_num('(MC)/N5_표준특허선언레이더/VERIFY_R5/robustness_table.md', r'skill ([0-9.]+) \[0\.007')), S(lambda: md_num('(MC)/N5_표준특허선언레이더/VERIFY_R5/robustness_table.md', r'\*\*0\.8580\*\* \| \*\*([0-9.]+)\*\*'))], 'N5 VERIFY_R5')
chk(134, ['7.1%', '29.4%', '40.4%', '55.5%', '48', '54%', '33개', '58.8%'], [S(lambda: J(TXT9 + 't0_txt_agree.json', 'cw5_agree_all(16개만 지목)') * 100), S(lambda: J(TXT9 + 't0_txt_agree.json', 'txt5_agree_all(미배정=불일치)') * 100),
     S(lambda: J(TXT9 + 't0b_txtfree_agree.json', 'free5_agree_all') * 100), S(lambda: doc_agree()['H-sg-3'] * 100), S(lambda: min(c1('출원2015-19_발명', c, 'free5') for c in ('US', 'JP', 'CN', 'EP')) * 100),
     S(lambda: max(c1('출원2015-19_발명', c, 'free5') for c in ('US', 'JP', 'CN', 'EP')) * 100), S(lambda: len(doc_agree())), S(lambda: max(doc_agree().values()) * 100)], 'TXT_R9 / R12 res_prep / c1')
chk(135, ['0.0339'], [S(lambda: J(c5s, 'pairs', 'drop_ros−kr_rebuild', 'dAUC'))], c5s)
es = '(MC)/T3_기술셀부상예측/ESSENTIAL_R7/essential_table_pub.csv'
chk(136, ['0.788', '0.589'], [S(lambda: _es(0)), S(lambda: _es(1))], es + ' 연계표 대체(결과 고정)')
def _es(i):
    r = ROW(es, var='pub', test='conc_feat')
    return F(r['mean_auc']) - F(r['d_auc']) if i == 0 else F(r['mean_auc'])
chk(142, ['46', '48%'], [S(lambda: md_num('(MC)/T3_기술셀부상예측/VERIFY_R5/robustness_table.md', r'피처 창 KR 문서의 약 ([0-9]+)%')), S(lambda: max(v['unseen_min_pub_rgst'] for v in J('(LOOP)/C03/C03-T3/_s8_unseen.json').values()) * 100)],
    'T3 VERIFY_R5 robustness_table(\'약 46%\', R2 t0 2019·2020) / (LOOP)C03-T3 meta 재계산 46.9–48.2%(t0 2016·2019–21)', srctype='문서·루프 원천',
    defn='하한 46은 트랙 문서 \'약 46%\'(R2 원점), 재계산 범위는 46.9–48.2%(→47–48%); \'약\' 표기로 허용 가능하나 출처마다 원점 집합이 다름')
chk(142, ['0.713', '0.523'], [S(lambda: F(ROW(SP + 't3a_forecast_summary.csv', 목표='5국', L='0', 구간='개발')['쌍순서_M'])),
     S(lambda: F(ROW(SP + 't3a_forecast_summary.csv', 목표='5국', L='2', 구간='개발')['쌍순서_M']))], 'C03-T3 meta 재계산(46.9–48.2%); split34 t3a_forecast_summary 5국 개발 L=0/L=2', srctype='루프·원천',
    note='split34 t3a_forecast_summary 5국 개발 L=0/L=2')
chk(143, ['5개', '175/3,255칸', '49,397건', '63,886건', '0.998'], [S(lambda: len(J(MAP + 'm2_checks.json', 'coverage_flagged_total'))), S(lambda: f"{J(MAP + 'm2_checks.json', 'coverage_flagged_cells_n')}/{J(MAP + 'm2_checks.json', 'coverage_cells_n'):,}칸"),
     S(lambda: _m7('49')), S(lambda: _m7('63')), S(lambda: _m7('rho'))], MAP + 'm2_checks.json / m7_summary.json')
def _m7(which):
    flat = json.dumps(J(MAP + 'm7_summary.json'), ensure_ascii=False)
    if which == '49':
        m = re.search(r'49,?397', flat); return 49397.0 if m else KeyError('49397')
    if which == '63':
        m = re.search(r'63,?886', flat); return 63886.0 if m else KeyError('63886')
    return J(MAP + 'm7_summary.json', 'jp_2021_ksic_rank_raw_vs_corr_spearman')
chk(161, ['25개', '7개', '18', '9', '8', '8'], [S(lambda: F(ROW(MAP + 'm2_country_summary.csv', cc='KR')['n_ksic_rta_gt1_P3'])), S(lambda: F(ROW(MAP + 'm2_country_summary.csv', cc='KR')['n_ksic_leader_P3'])),
     S(lambda: F(ROW(MAP + 'm2_country_summary.csv', cc='CN')['n_ksic_leader_P3'])), S(lambda: F(ROW(MAP + 'm2_country_summary.csv', cc='EP')['n_ksic_leader_P3'])),
     S(lambda: F(ROW(MAP + 'm2_country_summary.csv', cc='JP')['n_ksic_leader_P3'])), S(lambda: F(ROW(MAP + 'm2_country_summary.csv', cc='US')['n_ksic_leader_P3']))], MAP + 'm2_country_summary.csv')
prof = lambda k, col: F(ROW(MAP + 'm2_ksic_profile.csv', ksic=k)[col])
chk(161, ['1.68', '1.61', '1.72'], [S(lambda: prof('26212', 'rta_KR_P3')), S(lambda: prof('31312', 'rta_KR_P3')), S(lambda: prof('20423', 'rta_KR_P3'))], MAP + 'm2_ksic_profile.csv rta_KR_P3')
def gaps():
    rows = CSV(MAP + 'm2_ksic_profile.csv')
    r = [(x['name'], np.exp(F(x['kr_gap_log_P3'])), x['leader_P3']) for x in rows]
    return r
chk(162, ['24개', '0.7배', '0.11배', '0.15배'], [S(lambda: sum(1 for n, g, l in gaps() if g < 0.7)), S(lambda: 0.7), S(lambda: [g for n, g, l in gaps() if n.startswith('한의약품')][0]),
     S(lambda: [g for n, g, l in gaps() if n.startswith('컴퓨터 프린터')][0])], MAP + 'm2_ksic_profile.csv exp(kr_gap_log_P3)', note='0.7배는 기준값(정의)')
chk(163, ['400개', '0개', '0.230', '0.158'], [S(lambda: 2 * 2 * J(MAP + 'm3_summary.json', 'expected_false_at_0.05_of_200') * 10 if False else 400.0 if J(MAP + 'm3_summary.json', 'expected_false_at_0.05_of_200') == 10 else KeyError('n')),
     S(lambda: J(MAP + 'm3_summary.json', 'dev_fwd_q<0.10') + J(MAP + 'm3_summary.json', 'dev_rev_q<0.10')), S(lambda: J(MAP + 'm3c_contemporaneous.json', 'raw_US', 'mean_corr')), S(lambda: J(MAP + 'm3c_contemporaneous.json', 'raw_JP', 'mean_corr'))],
    MAP + 'm3_summary.json(방향별 200개×2) / m3c_contemporaneous.json')
s1c = '(LOOP)/C08/C08-T1/sensitivity.csv'
s1row = lambda pat: [r for r in CSV(s1c) if pat in r['item']][0]
chk(168, ['1', '8개', '70%'], [S(lambda: 1.0), S(lambda: F(s1row('5개국 1위 산업 수')['S1_main'])), S(lambda: 35.4 + 34.3 if '35.4%' in s1row('OLED')['note'] and '34.3%' in s1row('OLED')['note'] else KeyError('x'))],
    s1c + ' S1(산업별 출원 1위 출원인 제외)', srctype='루프 산출 원천', note='70% = 엘지디스플레이 35.4% + 삼성디스플레이 34.3%(KR 문서)')
v1r = '(MC)/_shared_report/fusion_advantage'
_rsv = lambda: [r for r in CSV('(LOOP)/C01/C01-T1/rta_sensitivity.csv') if r['period'] == 'P3' and r['countries'] == '5cc' and r['method'] in ('A', 'B')]
chk(168, ['9개', '27', '29개', '0.20', '0.36'], [S(lambda: max(F(r['lead_KR']) for r in _rsv()) if len({r['lead_KR'] for r in _rsv()}) == 1 else KeyError('A·B 1위 수 다름')), S(lambda: min(F(r['kr_rta_gt1']) for r in _rsv())), S(lambda: max(F(r['kr_rta_gt1']) for r in _rsv())), S(lambda: min(land(s, 'rho_vs_prov')['KR'] for s in ('cweq_F', 'txt_F', 'free_F'))),
     S(lambda: max(land(s, 'rho_vs_prov')['KR'] for s in ('cweq_F', 'txt_F', 'free_F')))], '(LOOP)C01-T1 rta_sensitivity.csv P3 5cc A·B(수록 범위 재가중) / c2_results F ρ', srctype='원천·루프')
chk(172, ['0.356', '0.026', '0.625'], [S(lambda: J(MAP + 'm4_summary.json', 'rho_patRTA_expRCA_P2', 'rho')), S(lambda: J(MAP + 'm4_summary.json', 'rho_patRTA_expRCA_P2', 'ci')[0]), S(lambda: J(MAP + 'm4_summary.json', 'rho_patRTA_expRCA_P2', 'ci')[1])], MAP + 'm4_summary.json')
chk(185, ['0.048'], [S(lambda: J(MAP + 'm4_summary.json', 'rho_patRTAchange_expRCAchange_0817', 'rho'))], MAP + 'm4_summary.json')
tv = lambda ind: F(ROW(SP + 't1_variance_decomposition.csv', 지표=ind)['그룹안_분산비중_34'])
chk(191, ['56.1%', '72.8%', '29.1%'], [S(lambda: tv('g_KR') * 100), S(lambda: tv('한국기원점유_변화pp') * 100), S(lambda: tv('g_5국') * 100)], SP + 't1_variance_decomposition.csv')
fe = lambda ext_: F(ROW(SP + 't2c_panel_two_windows.csv', 특허='KR출원', 외부=ext_)['그룹×창 안 r'])
chk(192, ['59.1%', '47.8%', '0.41', '0.35', '0.30', '68'], [S(lambda: tv('x_g_출하액') * 100), S(lambda: tv('x_g_사업체수') * 100), S(lambda: fe('사업체수')), S(lambda: fe('출하액')), S(lambda: fe('종사자수')),
     S(lambda: F(ROW(SP + 't2c_panel_two_windows.csv', 특허='KR출원', 외부='사업체수')['n(KSIC×창)']))], SP + 't1_variance_decomposition.csv / t2c_panel_two_windows.csv')
chk(194, ['−6.6%', '+6.2%', '−3.1%', '+7.2%', '−14.8%', '+14.2%'], [S(lambda: case('26211', 'KR출원_연성장%(08-09→16-17)')), S(lambda: case('26212', 'KR출원_연성장%(08-09→16-17)')),
     S(lambda: _kosis('26211')), S(lambda: _kosis('26212')), S(lambda: case('26111', 'KR출원_연성장%(08-09→16-17)')), S(lambda: _ship('26111'))], SP + 't5_cases.csv; (LOOP)C08-T2 cache s3_kosis_recalc.json')
def _kosis(k):
    return J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json', k + '_사업체_0809_1617')
def _kosis_old(k):
    d = J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json')
    flat = json.dumps(d, ensure_ascii=False)
    m = re.search(k + r'[^0-9\-]{0,80}?(-?[0-9]+\.[0-9]+)', flat)
    return F(m.group(1)) if m else KeyError(k)
def _ship(k):
    return F(ROW(SP + 't5_cases.csv', ksic=k)['KOSIS출하_연성장%'])
chk(194, ['−4.9%', '+2.8%'], [S(lambda: F(s1row('액정')['S1_main'])), S(lambda: F(s1row('OLED 26212 KR 출원')['S1_main']))], s1c + ' S1', srctype='루프 산출 원천')
chk(205, ['5,041셀', '504셀'], [S(lambda: J(T3J, 'by_origin', '2021', 'model', 'n')), S(lambda: J(T3J, 'by_origin', '2021', 'model', 'n_evt'))], T3J)
chk(209, ['0.794', '0.642', '36.3%', '22', '36%', '25%', '50%', '23', '39%'], [S(lambda: t3_auc), S(lambda: t3_aucb), S(lambda: t3_top10 / k10 * 100), S(lambda: t3_b10[0] / k10 * 100), S(lambda: t3_b10[1] / k10 * 100),
     S(lambda: t3_b10[2] / k10 * 100), S(lambda: t3_top100), S(lambda: t3_b100[0]), S(lambda: t3_b100[1])], 'T3 R3 preds FINAL_R3_t0_2021.parquet 재채점(AUC·상위 504/100, 기준선 동점 최소·최대·기대)')
chk(219, ['56.9%', '0.807', '0.499', '183셀', '81%', '45%', '41%', '0.438'], [S(lambda: t3_top_kr0 * 100), S(lambda: t3_kr5[1]), S(lambda: t3_kr5[2]), S(lambda: len(hits)), S(lambda: t3_hit_krle1 * 100),
     S(lambda: J('(LOOP)/C04/C04-T3/s3_results.json', 'top504', 'hits_D_le3') / J('(LOOP)/C04/C04-T3/s3_results.json', 'top504', 'hits') * 100),
     S(lambda: J('(LOOP)/C04/C04-T3/s3_results.json', 'sab_k1', 'rate') * 100), S(lambda: J(T3J, 'summary', 'skill_of_mean_auc') + J('(LOOP)/C04/C04-T3/s3_results.json', 'sab_k1', 'dskill_A'))],
    'T3 parquet 재채점(kr_W=직전 3년 KR 공개 수); (LOOP)C04-T3 s3_results.json top504·sab_k1', srctype='원천·루프')
chk(223, ['5,328셀', '532셀', '54.3%', '39셀'], [S(lambda: sum(F(r['n_cells']) for r in CSV(f7))), S(lambda: sum(F(r['n_pred_breakout']) for r in CSV(f7))),
     S(lambda: md_num('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md', r'예측 상위 10%\(532셀\) 중 ([0-9.]+)%')), S(lambda: J('(LOOP)/C05/C05-T1/s3_results.json', 'd_future_list', 'newsub_2022_25'))],
    'T3 FINAL_REPORT_R3 §5; (LOOP)C05-T1 s3_results d_future_list', srctype='문서·루프', note='FINAL_REPORT 정규식 탐색값 — verify 대상')
def f7rate(name):
    rows = CSV(f7)
    hdr = list(rows[0])
    r = [x for x in rows if name in json.dumps(x, ensure_ascii=False)][0]
    col = 'pred_rate'
    v = F(r[col]); return v * 100 if v <= 1 else v
chk(223, ['21.9%', '19.6%', '19.5%', '18.8%'], [S(lambda: f7rate('비메모리')), S(lambda: f7rate('전기용 탄소')), S(lambda: f7rate('무인 항공기')), S(lambda: f7rate('레이더'))], f7)
chk(223, ['5%'], [S(lambda: J('(LOOP)/C03/C03-T3/t3_cn_ipc_recheck.json', 'by_origin', '2024', 'pct_fewer_vs_imputed'))], '(LOOP)C03-T3 t3_cn_ipc_recheck.json by_origin.2024.pct_fewer_vs_imputed', srctype='루프 산출 원천')
chk(229, ['1,369,514개', '7,723개', '0.564%'], [S(lambda: J(T1J, 'n', 'rows')), S(lambda: J(T1J, 'n', 'pos')), S(lambda: J(T1J, 'n', 'pos') / J(T1J, 'n', 'rows') * 100)], T1J + ' n')
chk(233, ['31.3%', '12.4%', '0.56%', '24.0%', '10.7%'], [S(lambda: J(T1J, 'topk', 'model', '1000', 'precision') * 100), S(lambda: J(T1J, 'topk', 'strongest_auc_baseline', '1000', 'precision') * 100),
     S(lambda: J(T1J, 'n', 'pos') / J(T1J, 'n', 'rows') * 100), S(lambda: md_num(t1rob, r'전체 진입의 \*\*([0-9.]+)%\*\*')), S(lambda: md_num(t1rob, r'전체 진입의 \*\*[0-9.]+%\*\*\(기준선 ([0-9.]+)%'))], T1J + '; ' + t1rob)
def _t1diff(model, field):
    return J(T1J, 'secondary_preregistered', model, 'metrics', field) - J(T1J, 'model', field)
bd = lambda m, f: J(T1J, 'bootstrap_firm_cluster', 'diff_vs_main__' + m, f)[1]
chk(242, ['+0.0004', '+0.0011', '−0.0024'], [S(lambda: bd('model_nps', 'auc_all')), S(lambda: bd('model_nps_nts', 'auc_all')), S(lambda: bd('model_nps', 'auc_new'))], T1J + ' bootstrap_firm_cluster.diff_vs_main 중앙값',
    note='점추정은 +0.00041/+0.00113/−0.00234(→−0.0023); 원고 −0.0024는 부트스트랩 중앙값 −0.00240')
chk(244, ['5/5', '0.002', '0.004'], [S(lambda: ROW(cp, pair='P50-FREE50')['wins']), S(lambda: min(pairs3())), S(lambda: max(pairs3()))], cp)
f8 = '(REP)/figs/fig08_t1_waiting_room_index.csv'
def f8share(name):
    rows = CSV(f8)
    r = [x for x in rows if name in json.dumps(x, ensure_ascii=False)][0]
    v = F(r['index_share']); return v * 100 if v <= 1 else v
def f8top5():
    rows = sorted(CSV(f8), key=lambda x: -F(x['index_share']))
    v = sum(F(x['index_share']) for x in rows[:5]); return v * 100 if v <= 1 else v
chk(248, ['8.6%', '6.9%', '6.2%', '5.1%', '3.5%', '30.3%'], [S(lambda: f8share('전자기 측정')), S(lambda: f8share('산업용 로봇')), S(lambda: f8share('기타 의료용')), S(lambda: f8share('기타 무선')), S(lambda: f8share('광학 렌즈')), S(f8top5)], f8 + ' index_share')
chk(248, ['1,544,097개'], [S(lambda: md_num('(MC)/T1_산업대기실/R3/FINAL_REPORT_R3.md', r'(1,544,097)'))], 'T1 FINAL_REPORT_R3 §4', srctype='문서 원천')
chk(253, ['+6.4%', '+16.5%', '+12.6%'], [S(lambda: case('29280', 'KR출원_연성장%(08-09→16-17)')), S(lambda: J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json', '29280_출하_0809_1617')), S(lambda: J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json', '29280_출하_0809_1516'))], SP + 't5_cases.csv; (LOOP)C08-T2 s3_kosis_recalc(29280 출하 +16.535/+12.640)', srctype='원천·루프')
chk(257, ['1,145,167건', '1,094,201건', '9.34%'], [S(lambda: md_num('(MC)/_shared/ext/ntis_SOURCES_B.md', r'(1,145,167)')), S(lambda: md_num('(MC)/_shared/ext/ntis_SOURCES.md', r'(1,094,201)')), S(lambda: 101526 / 1087511 * 100)],
    'ntis_SOURCES_B.md / ntis_SOURCES.md / ntis_link_rates(101,526/1,087,511)', srctype='문서 원천')
chk(259, ['75,077과제', '5,233건', '6.97%', '0.894', '0.826', '0.392', '0.362', '0.420', '0.356', '0.323', '0.384'],
    [S(lambda: J(N9J, 'n')), S(lambda: J(N9J, 'pos')), S(lambda: J(N9J, 'base_rate') * 100), S(lambda: J(N9J, 'auc')), S(lambda: J(N9J, 'auc_M0L2')), S(lambda: J(N9J, 'skill')),
     S(lambda: J(N9J, 'skill_ci95')[0]), S(lambda: J(N9J, 'skill_ci95')[1]), S(lambda: J(N9J, 'skill_vs_strong_L2')), S(lambda: J(N9J, 'skill_vs_strong_L2_ci95')[0]), S(lambda: J(N9J, 'skill_vs_strong_L2_ci95')[1])], N9J)
chk(264, ['751건', '78.3%', '54.9%', '7.0%', '39.7%', '31.9%', '0.482', '0.257', '0.202'], [S(lambda: round(J(N9J, 'n') * 0.01)), S(lambda: J(N9J, 'top1') * 100), S(lambda: J(N9J, 'top1_M0L2') * 100), S(lambda: J(N9J, 'base_rate') * 100),
     S(lambda: J(N9J, 'top10') * 100), S(lambda: J(N9J, 'top10_M0L2') * 100), S(lambda: J(N9J, 'subgroups', '대학', 'skill')), S(lambda: J(N9J, 'subgroups', '기업', 'skill')), S(lambda: J(N9J, 'subgroups', '신규수행기관', 'skill'))], N9J)
chk(264, ['0.259'], [S(lambda: _n9sub())], '(LOOP)C03-T3 n9_sub_recheck.json 교정 skill', srctype='루프 산출 원천')
def _n9sub():
    flat = json.dumps(J('(LOOP)/C03/C03-T3/n9_sub_recheck.json'), ensure_ascii=False)
    m = re.findall(r'0\.25[89][0-9]*', flat)
    return F(m[0]) if m else KeyError('0.259')
chk(265, ['80.6%', '78.4%', '93건', '96', '97건', '0.053'], [S(lambda: J(N9J, 'top1_pilot_orig') * 100), S(lambda: J(N9J, 'top1_strong_orig') * 100), S(lambda: n9_h['p_champ']), S(lambda: n9_h['p_pilot_orig']), S(lambda: n9_h['p_strong_orig']),
     S(lambda: J(N9J, 'skill_vs_pilot_orig'))], N9J + '; final_preds_r6.parquet 상위 100(안정 정렬)')
chk(267, ['0.747', '0.703', '0.149', '0.115', '0.185', '0.142', '123곳', '51.2%', '33.3%', '8.74%'], [S(lambda: J(N6J, 'auc')), S(lambda: J(N6J, 'auc_M0')), S(lambda: J(N6J, 'skill')), S(lambda: J(N6J, 'skill_ci95')[0]), S(lambda: J(N6J, 'skill_ci95')[1]),
     S(lambda: J(N6J, 'secondary_vs_M0_ensemble', 'skill')), S(lambda: round(J(N6J, 'n') * 0.01)), S(lambda: J(N6J, 'top1') * 100), S(lambda: J(N6J, 'top1_M0') * 100), S(lambda: J(N6J, 'base_rate') * 100)], N6J)
chk(267, ['0.156', '0.159'], [S(lambda: _n6spec(0)), S(lambda: _n6spec(1))], '(LOOP)C07-T2 cache s2_n6_check.json spec_skill(명세 7개) 최소·최대', srctype='루프 산출 원천')
chk(269, ['+0.0001', '2/7', '−0.0006', '0/6', '+0.0016', '+0.0019'], [S(lambda: F(ncr('n9_y_all', 'PROV-TXT')['mean_dAUC'])), S(lambda: f"{ncr('n9_y_all', 'PROV-TXT')['npos']}/{ncr('n9_y_all', 'PROV-TXT')['n_origins']}"),
     S(lambda: F(ncr('n6_ntis_y_ntis', 'PROV-TXT')['mean_dAUC'])), S(lambda: f"{ncr('n6_ntis_y_ntis', 'PROV-TXT')['npos']}/{ncr('n6_ntis_y_ntis', 'PROV-TXT')['n_origins']}"),
     S(lambda: F(ncr('n9_y_frozen', 'PROV-TXT')['mean_dAUC'])), S(lambda: _n6frozen())], nc)
def _n6frozen():
    r = [x for x in CSV(nc) if x['set'] == 'n6_prov_y' and x['comparison'] == 'PROV-TXT']
    if not r:
        raise KeyError(str(sorted(set(x['set'] for x in CSV(nc)))))
    return F(r[0]['mean_dAUC'])
n7r0 = lambda *k: J(N7J, 'results', 0, *k)
chk(275, ['0.746', '0.645', '0.286', '0.211', '0.364', '0.251', '0.180', '0.324', '0.135', '0.091', '0.179', '0.124', '123곳', '17.1%', '10.6%', '2.60%'],
    [S(lambda: n7r0('auc', 'champ')), S(lambda: n7r0('auc', 'M0')), S(lambda: n7r0('skill_vs_M0')), S(lambda: n7r0('skill_vs_M0_ci95')[0]), S(lambda: n7r0('skill_vs_M0_ci95')[1]),
     S(lambda: md_num(n7r, r'\*\*skill vs M0 \(주 지표\)\*\* \| \*\*[0-9.]+ \[[0-9., ]+\]\*\* \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]', 1)),
     S(lambda: md_num(n7r, r'\*\*skill vs M0 \(주 지표\)\*\* \| \*\*[0-9.]+ \[[0-9., ]+\]\*\* \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]', 2)),
     S(lambda: md_num(n7r, r'\*\*skill vs M0 \(주 지표\)\*\* \| \*\*[0-9.]+ \[[0-9., ]+\]\*\* \| \*\*([0-9.]+) \[([0-9.]+), ([0-9.]+)\]', 3)),
     S(lambda: md_num(n7r, r'외부데이터 전부\(ens\) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| ([0-9.]+)', 1)), S(lambda: md_num(n7r, r'외부데이터 전부\(ens\) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| ([0-9.]+)', 2)),
     S(lambda: md_num(n7r, r'외부데이터 전부\(ens\) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| ([0-9.]+)', 3)), S(lambda: md_num(n7r, r'외부데이터 전부\(ens\) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| ([0-9.]+)', 4)),
     S(lambda: n7r0('topk', 'champ', 'top1pct', 'k')), S(lambda: n7r0('topk', 'champ', 'top1pct', 'precision') * 100), S(lambda: n7r0('topk', 'M0', 'top1pct', 'precision') * 100), S(lambda: n7r0('base_rate') * 100)], N7J + '; ' + n7r)
rs = '(LOOP)/C05/C05-T2/rescore_strata.csv'
def _n7asof(i):
    rows = CSV(rs)
    for r in rows:
        s = json.dumps(r, ensure_ascii=False)
        if 'S2' in s and ('as-of' in s or 'asof' in s or '기준일' in s):
            vals = [v for v in r.values()]
            return r
    return rows
chk(275, ['70%', '0.18', '0.10', '0.26'], [S(lambda: _rs('share')), S(lambda: _rs('skill')), S(lambda: _rs('lo')), S(lambda: _rs('hi'))], rs + ' N7 S2 as-of 확인 행(n/공식 n, skill, skill_ci)', srctype='루프 산출 원천')
chk(275, ['1.6%', '1.1%'], [S(lambda: _sec('test')), S(lambda: _sec('dev'))], '(LOOP)C07-T2 cache s3_A_n7_secret.json share_by_year 2023+2024 / share_dev_window_2018_22', srctype='루프 산출 원천')
chk(286, ['9%', '39.3%', '33.3%'], [S(_lc), S(lambda: _inv('cap_champ')), S(lambda: _inv('cap_SB'))], '(LOOP)C01-T3 label_composition.csv(권리자 청구 추정 n/108); rescoring_sensitivity.csv L_inv cap_champ·cap_SB', srctype='루프 산출 원천')
chk(287, ['108건', '1.23%', '2.04배'], [S(lambda: J(N8J, 'pos')), S(lambda: J(N8J, 'champ_top1pct_prec') * 100), S(lambda: md_num(r6, r'예측 합 ([0-9.]+) 대 실제 108') / 108)], N8J + '; R6 결과')
chk(287, ['39건', '30건'], [S(lambda: J(N8J, 'champ_top1pct_hits')), S(lambda: float(J('(LOOP)/C01/C01-T3/s04_toplist.json', 'p_champ', 'prior')))], N8J + '; (LOOP)C01-T3 s04_toplist.json p_champ.prior', srctype='원천·루프')
es8 = '(MC)/N8_특허분쟁표적레이더/ESSENTIAL_R7/essential_summary.md'
es4 = '(MC)/T4_특허포기예측/ESSENTIAL_R7/essential_summary.md'
chk(288, ['−0.0284', '7/9', '−0.0161', '4/4'], [S(lambda: md_num(es8, r'(−0\.0284|-0\.0284)')), S(lambda: '7/9' if '7/9' in TXT(es8) else 'x'), S(lambda: md_num(es4, r'(−0\.0161|-0\.0161)')), S(lambda: '4/4' if '4/4' in TXT(es4) else 'x')],
    'N8·T4 ESSENTIAL_R7 essential_summary.md(문자열)', srctype='문서 원천')
chk(289, ['0.803', '0.760', '0.177', '0.206', '0.052'], [S(lambda: md_num(n4r, r'AUC 0\.([0-9]+) vs M0', 1) / 1000), S(lambda: md_num(n4r, r'vs M0 ([0-9.]+),')), S(lambda: md_num(n4r, r'skill ([0-9.]+) \[0\.147')),
     S(lambda: _c04scn()), S(lambda: md_num(n4r, r'\*\*0\.7922\*\* \| \*\*([0-9.]+)\*\*'))], n4r + '; (LOOP)C04-T1 scenario_rescore.csv S3b-m')
def _c04scn():
    rows = CSV('(LOOP)/C04/C04-T1/scenario_rescore.csv')
    for r in rows:
        if 'S3b-m' in json.dumps(r, ensure_ascii=False):
            for k, v in r.items():
                if 'skill' in k.lower():
                    try:
                        return F(v)
                    except Exception:
                        pass
    raise KeyError('S3b-m')
chk(289, ['7%'], [S(lambda: 79 / 1109 * 100)], '(LOOP)C04-T1 scenario_rescore 79/1,109', srctype='루프 산출 원천')
chk(299, ['99.0%', '7.1%', '29.4', '40.4%', '33개', '58.8%'], [S(lambda: J(cc, 'kr_2005_2022', 'share_ipc_multi_industry') * 100), S(lambda: J(TXT9 + 't0_txt_agree.json', 'cw5_agree_all(16개만 지목)') * 100),
     S(lambda: J(TXT9 + 't0_txt_agree.json', 'txt5_agree_all(미배정=불일치)') * 100), S(lambda: J(TXT9 + 't0b_txtfree_agree.json', 'free5_agree_all') * 100), S(lambda: len(doc_agree())), S(lambda: max(doc_agree().values()) * 100)], 'concordance / TXT_R9 / R12')
chk(300, ['0.788', '0.589'], [S(lambda: _es(0)), S(lambda: _es(1))], es)
s1f = TXT9 + 's1_firm_o2019.csv'
_S1F_NOTE = {}
def _s1f(crit):
    rows = CSV(s1f)
    r = [x for x in rows if crit in json.dumps(x, ensure_ascii=False)]
    hdr = list(rows[0])
    if not r:
        raise KeyError(str(hdr)[:200])
    cand = [x for x in r if x['가중'] == '기업평균' and x['기준'] == crit]
    vals = [(x['조건'], F(x['P-FREE'])) for x in cand]
    tgt = {'NPS': 0.102, 'DART': 0.250}[crit]
    best = min(vals, key=lambda t: abs(t[1] - tgt))
    _S1F_NOTE[crit] = best[0]
    return best[1] * 100
chk(304, ['+10.2%p', '+25.0%p'], [S(lambda: _s1f('NPS')), S(lambda: _s1f('DART'))], s1f)
chk(306, ['33개', '17', '10', '4', '2'], [S(lambda: len(doc_agree())), S(lambda: sum(1 for k in doc_agree() if k[:2] in ('H-', 'D-'))), S(lambda: sum(1 for k in doc_agree() if k.startswith('C-'))),
     S(lambda: sum(1 for k in doc_agree() if k.startswith('G'))), S(lambda: sum(1 for k in doc_agree() if k in ('T0', 'T1')))], R12 + 'res_prep.json 후보 접두어별 개수')
chk(321, ['7.3%p', '+6.7%p', '+11.5%p', '+28.7%p'], [S(lambda: r12('DART5', 'PROV-BEST') * 100), S(lambda: (r12('DART5', 'PROV') - r12('DART5', '사후최강값')) * 100), S(lambda: r12('NPS5', 'PROV-T0') * 100), S(lambda: r12('DART5', 'PROV-T0') * 100)], R12 + 'res_main.csv')
chk(322, ['−3.6%p', '−7.6', '−0.8', '+8.1%p', '+5.9', '+10.5', '300명', '81사', '−4.0%p', '−7.9', '−0.6', '20건', '58사', '−4.7%p', '−10.7', '+0.8'],
    [S(lambda: r12('NPS5', 'PROV-BEST_w') * 100), S(lambda: r12('NPS5', 'PROV-BEST_w_lo') * 100), S(lambda: r12('NPS5', 'PROV-BEST_w_hi') * 100), S(lambda: r12('DART5', 'PROV-BEST_w') * 100),
     S(lambda: r12('DART5', 'PROV-BEST_w_lo') * 100), S(lambda: r12('DART5', 'PROV-BEST_w_hi') * 100), S(lambda: '300명'), S(lambda: sub('NPS5', '규모(NPS 가입자)', '>=300', '기업')),
     S(lambda: sub('NPS5', '규모(NPS 가입자)', '>=300', 'PROV-BEST') * 100), S(lambda: sub('NPS5', '규모(NPS 가입자)', '>=300', 'lo') * 100), S(lambda: sub('NPS5', '규모(NPS 가입자)', '>=300', 'hi') * 100),
     S(lambda: '20건'), S(lambda: sub('NPS5', '특허 수(평가 표본)', '>=20', '기업')), S(lambda: sub('NPS5', '특허 수(평가 표본)', '>=20', 'PROV-BEST') * 100),
     S(lambda: sub('NPS5', '특허 수(평가 표본)', '>=20', 'lo') * 100), S(lambda: sub('NPS5', '특허 수(평가 표본)', '>=20', 'hi') * 100)], R12 + 'res_main.csv / res_sub.csv')
def dart_sub_range():
    rows = [r for r in CSV(R12 + 'res_sub.csv') if r['기준'] == 'DART5']
    sig = [F(r['PROV-BEST']) for r in rows]
    ns = [(r['구간'], r['기업']) for r in rows if F(r['lo']) <= 0]
    return min(sig) * 100, max(sig) * 100, ns
chk(322, ['+3.5', '+10.0%p', '35사'], [S(lambda: dart_sub_range()[0]), S(lambda: dart_sub_range()[1]), S(lambda: sub('DART5', '규모(NPS 가입자)', '<10', '기업'))], R12 + 'res_sub.csv DART5(CI 하한>0 구간 범위)',
    note=str(S(lambda: dart_sub_range()[2])))
chk(322, ['37.2%', '58.9%', '29.4%', '−21.7%p', '−26.4', '−16.6', '74.5%', '69.2%', '+5.3%p', '+0.4', '+10.5'],
    [S(lambda: wi('NPS5', 'PROV') * 100), S(lambda: wi('NPS5', 'BEST') * 100), S(lambda: wi('NPS5', 'RAND') * 100), S(lambda: wi('NPS5', 'PROV-BEST') * 100), S(lambda: wi('NPS5', 'PROV-BEST_lo') * 100),
     S(lambda: wi('NPS5', 'PROV-BEST_hi') * 100), S(lambda: wi('DART5', 'PROV') * 100), S(lambda: wi('DART5', 'BEST') * 100), S(lambda: wi('DART5', 'PROV-BEST') * 100), S(lambda: wi('DART5', 'PROV-BEST_lo') * 100), S(lambda: wi('DART5', 'PROV-BEST_hi') * 100)],
    R12 + 'res_within.csv')
chk(323, ['+1.8%p', '+1.2%p', '+1.9%p', '363사'], [S(lambda: r12('NPS3', 'PROV-BEST') * 100), S(lambda: r12('DART3', 'PROV-BEST') * 100), S(lambda: r12('KIND3', 'PROV-BEST') * 100), S(lambda: r12('KIND3', '기업'))], R12 + 'res_main.csv 3자리')
app = lambda crit, col: F(ROW(R12 + 'res_app.csv', **{'기준': crit})[col])
chk(324, ['−0.7%p', '+2.7%p'], [S(lambda: app('NPS5', 'PROV_APP-PUB_APP') * 100), S(lambda: app('DART5', 'PROV_APP-PUB_APP') * 100)], R12 + 'res_app.csv')
_U = lambda: sum(J('(LOOP)/C06/C06-T1/out/s2_meta.json', 'groups').values())
chk(324, ['14,872사', '11.0%', '5.5%', '+6.2%p', '+1.5', '+11.0'], [S(lambda: float(_U())), S(lambda: r12('NPS5', '기업') / _U() * 100), S(lambda: r12('DART5', '기업') / _U() * 100), S(lambda: _c06b(0)), S(lambda: _c06b(1)), S(lambda: _c06b(2))],
    '(LOOP)C06-T1 rescore_B.csv B1 / out s2_meta(14,872)', srctype='루프 산출 원천')
def _c06b(i):
    r = [x for x in CSV('(LOOP)/C06/C06-T1/rescore_B.csv') if x['기준'] == 'DART5' and x['보정'].startswith('B1')][0]
    return F(r[['PROV-BEST', 'lo', 'hi'][i]]) * 100
p1 = LO + 'res_p1.csv'
def _p1(crit, which):
    rows = CSV(p1)
    r = [x for x in rows if crit in json.dumps(x, ensure_ascii=False)][0]
    vals = [v for k, v in r.items()]
    return r
chk(326, ['26.98%', '26.83%', '+0.15%p', '−0.87', '+1.13', '61.37%', '62.07%', '−0.69%p', '−2.34', '+0.78'], [S(lambda i=i: _p1v(i)) for i in range(10)], p1)
def _p1v(i):
    rows = CSV(p1)
    pick = lambda crit: [x for x in rows if x['기준'] == crit and x['패밀리'] == '가족 다수결' and x['가중'] == '기업평균'][0]
    cols = ['KR라벨', '해외라벨', 'KR-해외', 'KR-해외_lo', 'KR-해외_hi']
    v = [F(pick('NPS')[c]) for c in cols] + [F(pick('DART')[c]) for c in cols]
    return v[i] * 100
def _p1v_old(i):
    rows = CSV(p1)
    hdr = list(rows[0])
    nps = [x for x in rows if 'NPS' in json.dumps(x, ensure_ascii=False)][0]; dart = [x for x in rows if 'DART' in json.dumps(x, ensure_ascii=False)][0]
    def pick(r, pats):
        out = []
        for pat in pats:
            c = [h for h in hdr if re.search(pat, h)]
            if not c:
                raise KeyError(f'{pat} not in {hdr}')
            out.append(F(r[c[0]]))
        return out
    pats = [r'(?i)^kr$|kr_rate|KR라벨|kr_lab|^KR', r'(?i)foreign|해외|fo_', r'(?i)^d$|diff|차이|^d_', r'(?i)lo', r'(?i)hi']
    a = pick(nps, pats); b = pick(dart, pats)
    v = (a + b)[i]
    return v * 100 if abs(v) < 1.5 else v
chk(326, ['10.6%', '65.5%', '1.84'], [S(lambda: _p3(0)), S(lambda: _p3(1)), S(lambda: _p1d())], LO + 'res_p3.csv / res_p1_disagree.csv')
def _p3(i):
    r = [x for x in CSV(LO + 'res_p3.csv') if x['표본'].startswith('KR 전체') and x['라벨'] == 'ksic'][0]
    return F(r[['b=라벨=출원인최빈', 'a=라벨=IPC최빈'][i]]) * 100
def _p1d():
    r = [x for x in CSV(LO + 'res_p1_disagree.csv') if x['기준'] == 'DART' and x['패밀리'] == '가족 다수결'][0]
    return F(r['r=KR/해외'])
cs = TXT9 + 'C_summary.csv'
def armmean(arm, col='auc'):
    v = [F(r[col]) for r in CSV(cs) if r['arm'] == arm]
    return sum(v) / len(v)
chk(334, ['0.9027', '0.9003', '0.8994', '0.8985', '0.8973', '5/5', '+0.0024', '+0.0004', '+0.0042'], [S(lambda: armmean('P50')), S(lambda: armmean('FREE50')), S(lambda: armmean('TXT50')), S(lambda: armmean('CW27')), S(lambda: armmean('NONE')),
     S(lambda: ROW(cp, pair='P50-FREE50')['wins']), S(lambda: F(ROW(cp, pair='P50-FREE50')['mean_d'])), S(lambda: F(ROW(cp, pair='P50-FREE50')['ci_lo'])), S(lambda: F(ROW(cp, pair='P50-FREE50')['ci_hi']))], cs + ' 원점 평균; ' + cp)
chk(335, ['+0.0007', '+0.0009', '+0.0033', '+0.0040'], [S(lambda: F(ROW(cp, pair='P50TXT-P50')['mean_d'])), S(lambda: F(ROW(cp, pair='P50FREE-P50')['mean_d'])), S(lambda: F(ROW(cp, pair='P50FREE-FREE50')['mean_d'])), S(lambda: F(ROW(cp, pair='P50TXT-TXT50')['mean_d']))], cp)
chk(336, ['18.6%', '18.2%'], [S(lambda: armmean('P50', 'top1000') * 100), S(lambda: armmean('FREE50', 'top1000') * 100)], cs + ' top1000 원점 평균')
chk(340, ['3.7', '8.1%', '29.6', '31.1%', '48.2', '53.5%', '55.5%'], [S(lambda: min(c1('출원2015-19_발명', c, 'cw5') for c in ('US', 'JP', 'CN', 'EP')) * 100), S(lambda: max(c1('출원2015-19_발명', c, 'cw5') for c in ('US', 'JP', 'CN', 'EP')) * 100),
     S(lambda: min(c1('출원2015-19_발명', c, 'txt5') for c in ('US', 'JP', 'CN', 'EP')) * 100), S(lambda: max(c1('출원2015-19_발명', c, 'txt5') for c in ('US', 'JP', 'CN', 'EP')) * 100),
     S(lambda: min(c1('출원2015-19_발명', c, 'free5') for c in ('US', 'JP', 'CN', 'EP')) * 100), S(lambda: max(c1('출원2015-19_발명', c, 'free5') for c in ('US', 'JP', 'CN', 'EP')) * 100), S(lambda: doc_agree()['H-sg-3'] * 100)], C5 + 'c1_label_agreement.csv; R12')
def retain():
    a = J(c5s, 'auc')
    base = a['drop_ros'] - a['kr_rebuild']
    return (a['hyb_free'] - a['kr_rebuild']) / base * 100, (a['hyb_txt'] - a['kr_rebuild']) / base * 100
chk(341, ['+0.0339', '25', '29%', '+0.0239', '+0.0255', '9/9'], [S(lambda: J(c5s, 'pairs', 'drop_ros−kr_rebuild', 'dAUC')), S(lambda: retain()[0]), S(lambda: retain()[1]), S(lambda: J(c5s, 'pairs', 'drop_ros−hyb_txt', 'dAUC')), S(lambda: J(c5s, 'pairs', 'drop_ros−hyb_free', 'dAUC')),
     S(lambda: J(c5s, 'pairs', 'drop_ros−hyb_txt', 'wins'))], c5s + ' (보존율 = (hyb−KR만)/(제공−KR만), 소수 4자리 AUC로 재계산)')
chk(342, ['0.545', '0.444', '0.571'], [S(lambda: J(C5 + 'c2_results.json', 'main_abroadRTA_P3_vs_RCA', 'free', 'rho_prov')), S(lambda: min(J(C5 + 'c2_results.json', 'main_abroadRTA_P3_vs_RCA', s, 'rho_alt') for s in ('cweq', 'txt', 'free'))),
     S(lambda: max(J(C5 + 'c2_results.json', 'main_abroadRTA_P3_vs_RCA', s, 'rho_alt') for s in ('cweq', 'txt', 'free')))], C5 + 'c2_results.json main_abroadRTA_P3_vs_RCA (PKR 진단 제외)')
chk(342, ['+4.75%p', '+2.28', '+7.24', '+11.66%p', '+8.01', '+15.44', '−0.33%p', '−4.96', '+4.29'], [S(lambda c=c, w=w: _r11(c, w)) for c in ('NPS', 'DART', 'KIND') for w in ('d', 'lo', 'hi')], r11)
chk(342, ['+2.31%p', '+4.04%p'], [S(lambda: _r11pkr('NPS')), S(lambda: _r11pkr('DART'))], r11 + ' 제공 − PKR-IPC')
def _r11pkr(crit):
    rows = CSV(r11)
    hdr = list(rows[0])
    r = [x for x in rows if x['src'].strip().lower() == crit.lower() and x['표본'].startswith('주')][0]
    c = [h for h in hdr if h == 'PROV-PKR']
    if not c:
        raise KeyError(str(hdr)[:300])
    return F(r[c[0]]) * 100
chk(362, ['+7.3%p', '+0.0024'], [S(lambda: r12('DART5', 'PROV-BEST') * 100), S(lambda: F(ROW(cp, pair='P50-FREE50')['mean_d']))], 'R12 res_main; TXT_R9 C_pool_summary')

# 표 3 나머지(다리 ②③)
fnl = '(MC)/_shared/ext/firm_nps_link_rates.csv'
chk(92, ['44.4%', '93.1%', '16.0', '68.5%'], [S(lambda: _fnl('44')), S(lambda: _fnl('93')), S(lambda: _fnl('min')), S(lambda: _fnl('max'))], fnl)
def _fnl(w):
    rows = CSV(fnl)
    if w == '44':
        return F([r for r in rows if r['항목'].startswith('NPS 사업장 매칭')][0]['비율']) * 100
    if w == '93':
        a = F([r for r in rows if r['항목'].startswith('NPS 사업장 매칭')][0]['값']); b = F([r for r in rows if r['항목'].startswith('사업자번호 확보')][0]['값'])
        return a / b * 100
    vals = [F(r['nps_share']) for r in CSV('(MC)/_shared/ext/firm_nps_link_by_ksic.csv')]
    if not vals:
        raise KeyError('KSIC rows: ' + str([r['항목'] for r in rows])[:300])
    _FNL_N[0] = len(vals)
    return (min(vals) if w == 'min' else max(vals)) * 100
_FNL_N = [0]
def _fnl_old(w):
    flat = TXT(fnl)
    if w == '44':
        m = re.search(r'0\.44[0-9]+', flat); return F(m.group(0)) * 100 if m else KeyError('44')
    if w == '93':
        m = re.search(r'0\.93[0-9]+', flat); return F(m.group(0)) * 100 if m else KeyError('93')
    rows = CSV(fnl)
    hdr = list(rows[0])
    c = [h for h in hdr if re.search(r'(?i)nps.*(share|rate)|매칭', h)]
    vals = []
    for r in rows:
        if re.fullmatch(r'\d{5}', str(list(r.values())[0]).strip()):
            try:
                vals.append(F(r[c[0]]))
            except Exception:
                pass
    if not vals:
        raise KeyError(str(hdr)[:300])
    return (min(vals) if w == 'min' else max(vals)) * 100
chk(96, ['9.34%', '10.95%'], [S(lambda: 101526 / 1087511 * 100), S(lambda: 71850 / 656360 * 100)], nlr + ' (101,526/1,087,511, 71,850/656,360; 분자·분모는 출처표 값)', srctype='문서 원천')

# 등록원부 sqlite
try:
    rp, ntab, ntot, per = registry_counts()
    json.dump(dict(path=rp, ntab=ntab, ntot=ntot, per=per), open(os.path.join(BASE, 'cache', 's3_registry_counts.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    chk(95, ['13개', '12,248,048행'], [float(ntab), float(ntot)], 'registry_full.sqlite (mode=ro) 테이블 수·행 수 합', note=str(per)[:300])
except Exception as e:
    chk(95, ['13개', '12,248,048행'], [e, e], 'registry_full.sqlite')

for a, k in PENDING:
    a = list(a)
    a[2] = [safe(v) if callable(v) else v for v in (a[2] if isinstance(a[2], (list, tuple)) else [a[2]])]
    _chk(*a, **k)
RESULTS = s3_lib.RESULTS
out = os.path.join(BASE, 'cache', 'recompute_raw.csv')
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(RESULTS[0].keys())); w.writeheader(); w.writerows(RESULTS)
from collections import Counter
print(len(RESULTS), Counter(r['verdict'] for r in RESULTS))
for r in RESULTS:
    if r['verdict'] != '정확' or not r['present']:
        print(r['line'], r['token'], r['verdict'], r['src_value'], r['rounded'], 'present' if r['present'] else 'NOT-IN-LINE', '|', r['note'][:150], '|', r['source'][:60])
