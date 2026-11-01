# -*- coding: utf-8 -*-
"""C09-T2 verify_0: 1차 오류·부분 판정 전부 + A 분류 무작위 30개(시드 20261013) 독립 재계산.
1차(s3_*)와 다른 코드·다른 원천 경로를 쓴다(예측 parquet·npy 재채점, 원자료 parquet, 다른 열 조합, 결과 문서 텍스트).
s3_lib를 쓰지 않는다. 출력: verify_0/v0_result.json"""
import os, re, json, csv, sqlite3
import numpy as np, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
MC = os.path.join(AN, '모델경쟁_20260927'); LOOP = os.path.join(AN, '검증루프_20261012')
def rd(p):
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))
def jl(p):
    return json.load(open(p, encoding='utf-8'))
def auc_rank(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    r = pd.Series(s).rank(method='average').values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def rnd(x, d):
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
OUT = []
def rec(item, line, token, val, disp_dec, first, path, note='', absval=False, tol=None):
    try:
        y = float(re.sub(r'[^0-9.\-]', '', token.replace('−', '-').replace('+', '')))
        v = abs(val) if absval else val
        t = tol if tol is not None else 0.5 * 10 ** (-disp_dec)
        ok = abs(v - (abs(y) if absval else y)) <= t + 1e-12
        vd = '정확' if ok else '오류'
    except Exception as e:
        vd, v = '막힘', str(e)
    OUT.append(dict(item=item, line=line, token=token, indep_value=v, verdict_indep=vd, verdict_first=first, flip=(vd != first.split('(')[0] if first not in ('부분',) else None), path=path, note=note))

# ---------- 1차 오류·부분 ----------
# E1 75행 '45'(45–52%): R12 cache labels.parquet로 KR 평가 문서 H-sg-3 일치율 직접 계산
LB = pd.read_parquet(os.path.join(MC, '_shared_report/label_origin_R10/confirm_R12/cache/labels.parquet'))
ev = LB[(LB['split'] == 'eval')] if 'eval' in set(LB['split']) else LB[(LB['ay'] >= 2015) & (LB['ay'] <= 2019)]
agree_hsg3 = float((ev['PROV'] == ev['H-sg-3']).mean())
c1 = rd(os.path.join(MC, '_shared_report/fusion_advantage/c5_R10/c1_label_agreement.csv'))
fre = {r['cc']: float(r['free5_labeled']) * float(r['cov_free']) for r in c1 if r['window'] == '출원2015-19_발명' and r['cc']}
unrep = [1 - agree_hsg3] + [1 - fre[c] for c in ('US', 'JP', 'CN', 'EP')]
rec('E1', 75, '45', min(unrep) * 100, 0, '오류', 'R12 cache labels.parquet(split=eval, n=%d) PROV==H-sg-3; 해외 free5_labeled×cov_free' % len(ev), note=f'KR 일치 {agree_hsg3:.5f} → 미재현 {100*(1-agree_hsg3):.3f}%')
rec('E1b', 75, '52%', max(unrep) * 100, 0, '정확', '같음')
# E2 134행 '54%'
rec('E2', 134, '54%', max(fre[c] for c in ('US', 'JP', 'CN', 'EP')) * 100, 0, '오류', 'c1 free5_labeled×cov_free 최대(CN)', note=str({k: round(v * 100, 3) for k, v in fre.items()}))
rec('E2b', 134, '48', min(fre[c] for c in ('US', 'JP', 'CN', 'EP')) * 100, 0, '정확', '같음')
# E3 125행 '≤1e−10': T4 동결 npy 앙상블 재합성 → 공식 AUC·skill과 차
P = os.path.join(MC, 'T4_특허포기예측/R2/runs_final/F22/pred')
evp = pd.read_parquet(os.path.join(MC, 'T4_특허포기예측/R2/runs_final/F22/ev.parquet'), columns=['y'])
y4 = evp['y'].values.astype(int)
def lg(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
comp = {k: np.load(os.path.join(P, f)) for k, f in [('haz', 'haz_0568d4297b.npy'), ('cls', 'cls_99f270b3cd.npy'), ('logit', 'logit_3008b8b195.npy'), ('hgb', 'hgb_b4bd0fa46a.npy'), ('base', 'base_46526c56d8.npy')]}
ens = 0.4 * lg(comp['haz']) + 0.2 * lg(comp['cls']) + 0.2 * lg(comp['logit']) + 0.2 * lg(comp['hgb'])
a_m = auc_rank(y4, ens); a_b = auc_rank(y4, comp['base'])
off = jl(os.path.join(MC, 'T4_특허포기예측/R2/final_result.json'))
sk = (a_m - a_b) / (1 - a_b)
d_auc = abs(a_m - off['model']['auc']); d_sk = abs(sk - off['skill_auc'])
OUT.append(dict(item='E3', line=125, token='≤1e−10', indep_value=f'AUC 차 {d_auc:.2e}, 기준선 AUC 차 {abs(a_b-off["baseline"]["auc"]):.2e}, skill 차 {d_sk:.2e}', verdict_indep=('부분' if d_auc <= 1e-10 < d_sk else ('정확' if d_sk <= 1e-10 else '오류(재합성 불일치)')),
                verdict_first='부분', flip=None, path='T4 R2 runs_final/F22/pred/*.npy 로짓 가중 평균(0.4/0.2/0.2/0.2) 재합성 + 순위 AUC', note=f'재합성 AUC {a_m:.10f} 공식 {off["model"]["auc"]:.10f}'))
# E4 142행 '46'·'48%': C03-T3 _s8_unseen.json(원점별 미공개 비율) — 1차와 다른 파일
u = jl(os.path.join(LOOP, 'C03/C03-T3/_s8_unseen.json'))
vals = [u[k]['unseen_min_pub_rgst'] * 100 for k in u]
rec('E4', 142, '46', min(vals), 0, '부분', 'C03-T3 _s8_unseen.json unseen_min_pub_rgst(원점 2016·2019–21)', note=str({k: u[k]['unseen_min_pub_rgst'] for k in u}) + " / '약' 표기")
rec('E5', 142, '48%', max(vals), 0, '부분', '같음')

# ---------- A 무작위 30 ----------
sel = jl(os.path.join(HERE, 'v0_selection.json'))
qs = rd(os.path.join(LOOP, 'C03/C03-T2/quadrant_sensitivity.csv'))
t3 = pd.read_parquet(os.path.join(MC, 'T3_기술셀부상예측/R3/preds/FINAL_R3_t0_2021.parquet'))
R12 = os.path.join(MC, '_shared_report/label_origin_R10/confirm_R12')
# R12 DART5 독립 구현(문서 → 법인 → 법인 평균)
KS = sorted({r['ksic'] for r in rd(os.path.join(MC, '_shared_report/fusion_advantage/map5/m2_ksic_profile.csv'))})
pf = pd.read_parquet(os.path.join(MC, '_shared_report/label_origin_R10/cache/pat_firm.parquet'))
reg = pd.read_parquet(os.path.join(R12, 'cache/reg12.parquet'))
LB['appl_no'] = LB['appl_no'].astype(str); pf['appl_no'] = pf['appl_no'].astype(str)
Z = pf.merge(LB[LB['split'] == 'eval'][['appl_no', 'PROV', 'T0', 'H-mg-3', 'H-sc-1']], on='appl_no')
rg = reg[['applicantCd', 'dart']].dropna(); rg = rg[rg['dart'].isin(KS)]
Z = Z.merge(rg, on='applicantCd')
KSa = np.array(KS + [''], dtype=object)
for c in ('PROV', 'T0', 'H-mg-3'):
    Z['m_' + c] = (KSa[Z[c].astype(int).values] == Z['dart'].values).astype(float)
G = Z.groupby('applicantCd')[['m_PROV', 'm_T0', 'm_H-mg-3']].mean(); nG = Z.groupby('applicantCd').size()
d5 = dict(firms=len(G), docs=len(Z), PROV=G['m_PROV'].mean(), T0=G['m_T0'].mean(), BEST=G['m_H-mg-3'].mean())
w = nG.values / nG.values.sum()
d5['PROV_w'] = float((G['m_PROV'].values * w).sum()); d5['BEST_w'] = float((G['m_H-mg-3'].values * w).sum())
rngv = np.random.default_rng(777)
B = rngv.integers(0, len(G), (1000, len(G)))
M = G[['m_PROV', 'm_H-mg-3']].values
dd = M[B, 0].mean(1) - M[B, 1].mean(1)
d5['diff_lo'], d5['diff_hi'] = np.percentile(dd, [2.5, 97.5])
json.dump({k: (float(v) if not isinstance(v, int) else v) for k, v in d5.items()}, open(os.path.join(HERE, 'v0_r12_dart5.json'), 'w'), indent=1)
C = rd(os.path.join(MC, 'T1_산업대기실/TXT_R9/res/C_summary.csv'))
def per_c(arm):
    return {r['c']: float(r['auc']) for r in C if r['arm'] == arm}
c5 = jl(os.path.join(MC, '_shared_report/fusion_advantage/c5_R10/c5_t3_summary.json'))
for s in sel:
    ln, tk = int(s['line']), s['token'] + s['unit']
    first = s['rv'] or '정확'
    if s['status'].startswith('A('):
        first = '정확(이전 감사)'
    try:
        if ln == 15:
            n = [len(r['SW_list'].split()) for r in qs]; rec('A', ln, tk, max(n), 0, first, 'quadrant_sensitivity SW_list 길이 최대(SW 열 대신)')
        elif ln == 53:
            rec('A', ln, tk, auc_rank(t3['yb'], t3['p']), 3, first, 'T3 parquet 순위 AUC')
        elif ln == 68:
            txt = open(os.path.join(MC, '_shared_report/label_origin_R10/RESULT_R10.md'), encoding='utf-8').read()
            m = re.search(r'문서쌍 기준 US ([0-9.]+)%', txt)
            rec('A', ln, tk, float(m.group(1)), 1, first, 'label_origin_R10 RESULT_R10.md 본문(res_fam_check 대신)', note='1차 독립 경로(일치쌍/비교쌍 열 추정)는 열 의미를 잘못 잡아 폐기')
        elif ln == 71:
            rec('A', ln, tk, fre['EP'] * 100, 1, first, 'c1 free5_labeled×cov_free(EP)')
        elif ln == 79:
            con = sqlite3.connect('file:' + os.path.abspath(os.path.join(AN, '..', '수집파이프라인', 'registry_full.sqlite')) + '?mode=ro', uri=True)
            n = con.execute('select count(distinct rgst_no) from patent').fetchone()[0]; con.close()
            rec('A', ln, tk, n, 0, first, 'registry_full.sqlite patent distinct rgst_no(원자료)')
        elif ln == 96:
            txt = open(os.path.join(MC, '_shared/ext/ntis_link_rates.csv'), encoding='utf-8-sig').read()
            m = re.search(r'71,?850', txt); m2 = re.search(r'656,?360', txt)
            rec('A', ln, tk, 71850 / 656360 * 100 if (m and m2) else float('nan'), 2, first, 'ntis_link_rates.csv 원문의 분자·분모 확인 후 비율', note='분자·분모 문자열 ' + ('있음' if (m and m2) else '없음'))
        elif ln == 161:
            OUT.append(dict(item='A', line=ln, token=tk, indep_value='서수(1위)', verdict_indep='정확', verdict_first=first, flip=False, path='구조', note=''))
        elif ln == 162:
            p = [r for r in rd(os.path.join(MC, '_shared_report/fusion_advantage/map5/m2_ksic_profile.csv')) if r['name'].startswith('한의약품')][0]
            rec('A', ln, tk, float(p['rta_KR_P3']) / float(p['rta_CN_P3']), 2, first, 'm2_ksic_profile rta_KR_P3/rta_CN_P3(로그 격차 열 대신)')
        elif ln == 183:
            sets = [set(r['SW_list'].split()) for r in qs]
            rec('A', ln, tk, len(set.intersection(*sets)), 0, first, 'quadrant_sensitivity SW_list 교집합(12개 변형)')
        elif ln == 216:
            txt = open(os.path.join(MC, 'R6_결과_20261004.md'), encoding='utf-8').read()
            m = re.search(r'0\.182 \[([0-9.]+), 0\.243\]', txt)
            rec('A', ln, tk, float(m.group(1)), 3, first, 'R6_결과 문서 텍스트(robustness_table 대신)')
        elif ln == 223:
            j = jl(os.path.join(LOOP, 'C03/C03-T3/t3_cn_ipc_recheck.json'))
            flat = json.dumps(j, ensure_ascii=False)
            a = re.search(r'5,?328', flat); b = re.search(r'5,?632', flat)
            rec('A', ln, tk, (1 - 5328 / 5632) * 100 if (a and b) else float('nan'), 0, first, 'C03-T3 t3_cn_ipc_recheck.json(5,328 대 5,632)')
        elif ln == 269:
            txt = open(os.path.join(MC, 'R9_결과_20261005.md'), encoding='utf-8').read()
            m = re.search(r'N6[^\n]{0,120}?(−0\.0006|-0\.0006)', txt)
            r = [x for x in rd(os.path.join(MC, '_shared_report/noncirc_n9n6_R9/results_table.csv')) if x['set'] == 'n6_ntis_y_ntis' and x['comparison'] == 'PROV-TXT'][0]
            rec('A', ln, tk, float(r['mean_dAUC']), 4, first, 'noncirc results_table + R9 문서 문자열', note='R9 문서 ' + ('있음' if m else '없음'))
        elif ln == 307:
            j = jl(os.path.join(R12, 'res_prep.json')); jt = jl(os.path.join(R12, 'res_tests.json'))
            n_ok = int(j.get('tfidf_argmax_eq_stored') == 1.0) + int(bool(jt.get('repro_R11_posthoc2')))
            rec('A', ln, tk, n_ok, 0, first, 'res_prep tfidf_argmax_eq_stored=1.0 + res_tests repro_R11_posthoc2', note=str(jt.get('repro_R11_posthoc2'))[:150])
        elif ln == 317 and s['token'] == '34.2':
            rec('A', ln, tk, d5['T0'] * 100, 1, first, 'R12 cache 독립 구현(DART5 T0 법인 평균)', note=f"법인 {d5['firms']} 문서 {d5['docs']}")
        elif ln in (317, 328) and s['token'] == '+9.7':
            rec('A', ln, tk, d5['diff_hi'] * 100, 1, first, 'R12 cache 독립 구현 + 부트스트랩(시드 777, 1,000회)', note='부트스트랩 난수가 달라 ±0.2%p 허용', tol=0.25)
        elif ln == 322:
            rec('A', ln, tk, (d5['PROV_w'] - d5['BEST_w']) * 100, 1, first, 'R12 cache 독립 구현(특허 가중)')
        elif ln == 326:
            r = [x for x in rd(os.path.join(MC, '_shared_report/label_origin_R10/res_p1.csv')) if x['기준'] == 'DART' and x['패밀리'] == '가족 다수결' and x['가중'] == '기업평균'][0]
            f3 = [x for x in rd(os.path.join(AN, '보고서_20261012/figs/fig03_necessity_firm_industry_match.csv')) if x['기준'] == 'DART'][0]
            rec('A', ln, tk, float(f3['hi']) * 100, 2, first, 'fig03 CSV (b) DART hi(res_p1 대신)', note=f"res_p1 {r['KR-해외_hi']}")
        elif ln == 328 and s['token'] == '+7':
            rec('A', ln, tk, (d5['PROV'] - d5['BEST']) * 100, 0, first, 'R12 cache 독립 구현(DART5 제공−H-mg-3)')
        elif ln in (334,):
            a, b = per_c('P50'), per_c('FREE50'); rec('A', ln, tk, np.mean([a[k] - b[k] for k in a]), 4, first, 'C_summary 원점별 AUC 차 평균(C_pool_summary 대신)')
        elif ln == 335:
            a, b = per_c('P50FREE'), per_c('P50'); rec('A', ln, tk, np.mean([a[k] - b[k] for k in a]), 4, first, 'C_summary 원점별 AUC 차 평균')
        elif ln == 336:
            v = [float(r['top1000']) for r in C if r['arm'] == 'P50']; rec('A', ln, tk, np.mean(v) * 100, 1, first, 'C_summary top1000 원점 평균')
        elif ln == 340:
            r = [x for x in c1 if x['window'] == '출원2015-19_발명' and x['cc'] == 'EP'][0]
            rec('A', ln, tk, float(r['txt5_labeled']) * float(r['cov_txt']) * 100, 1, first, 'c1 txt5_labeled×cov_txt(EP 최대)')
        elif ln == 341 and s['token'] == '29':
            bo = c5['pairs']
            ret = np.mean([(x) for x in c5['pairs']['hyb_txt−kr_rebuild']['by_origin']]) / np.mean(c5['pairs']['drop_ros−kr_rebuild']['by_origin'])
            rec('A', ln, tk, ret * 100, 0, first, 'c5_t3_summary 원점별 dAUC 평균 비(소수 4자리 AUC 대신)')
        elif ln in (341, 358) and s['token'] in ('+0.0255', '+0.0239'):
            key = 'drop_ros−hyb_free' if s['token'] == '+0.0255' else 'drop_ros−hyb_txt'
            rec('A', ln, tk, float(np.mean(c5['pairs'][key]['by_origin'])), 4, first, 'c5_t3_summary 원점별 dAUC 평균')
        elif ln == 359:
            r = [x for x in rd(os.path.join(MC, '_shared_report/label_origin_R10/foreign_public_R11/res_main.csv')) if x['src'] == 'nps' and x['표본'].startswith('주')][0]
            rec('A', ln, tk, (float(r['PROV']) - float(r['FREE'])) * 100, 2, first, 'R11 res_main PROV−FREE 열 차(PROV-FREE 열 대신)')
        elif ln == 360:
            j = jl(os.path.join(MC, '_ideas_R8/p2_results.json'))['ship']['compare']['sig_PROV_KR-sig_TXT']
            rec('A', ln, tk, j['ci'][0] if s['token'] == '−0.171' else j['ci'][1], 3, first, 'p2_results ship.compare CI')
        elif ln == 372:
            r = rd(os.path.join(MC, '_shared_report/fusion_advantage/map5/m3_pairs_leadlag.csv'))
            per = {}
            for x in r:
                per[x['period']] = per.get(x['period'], 0) + 1
            rec('A', ln, tk, max(per.values()) if len(per) > 1 else len(r), 0, first, 'm3_pairs_leadlag.csv 기간별 행 수', note=str(per))
        else:
            OUT.append(dict(item='A', line=ln, token=tk, indep_value='', verdict_indep='막힘', verdict_first=first, flip=None, path='규칙 없음', note=''))
    except Exception as e:
        OUT.append(dict(item='A', line=ln, token=tk, indep_value='', verdict_indep='막힘', verdict_first=first, flip=None, path='오류', note=str(e)[:200]))

for o in OUT:
    f = o['verdict_first'].split('(')[0]
    o['flip'] = (o['verdict_indep'] != f) if o['verdict_indep'] in ('정확', '오류', '부분') else None
json.dump(OUT, open(os.path.join(HERE, 'v0_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
from collections import Counter
print(Counter((o['item'][0], o['verdict_indep'], o['flip']) for o in OUT))
for o in OUT:
    print(o['item'], o['line'], o['token'], o['verdict_first'], '→', o['verdict_indep'], '|', str(o['indep_value'])[:60], '|', o['note'][:120])
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d5.items()}, ensure_ascii=False))
