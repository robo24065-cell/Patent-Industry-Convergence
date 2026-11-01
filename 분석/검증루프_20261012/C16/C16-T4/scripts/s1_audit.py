# C16-T4 s1: v15(d3727aa0) -> 현재 원고(manuscript_v3.md, 4cbf3d55) 변경 25행 감사
# 본문: ERRATA 원안 문구 포함 여부 + C15-T4 시험 빌드(권장 묶음)·v16 N0와 동일성
# 부록: 원천 파일에서 숫자를 직접 다시 읽어 원고 표시 자릿수로 반올림 대조
import csv, json, re, sys, hashlib, sqlite3, difflib
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding='utf-8')
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
AN = BASE + r"\분석"
R = AN + r"\보고서_20261012"
MC = AN + r"\모델경쟁_20260927"
L = AN + r"\검증루프_20261012"
OUT = L + r"\C16\C16-T4"

def rd(p):
    return open(p, encoding='utf-8').read().replace('\r\n', '\n').split('\n')
def sha(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()[:8]
V15p = R + r"\drafts\manuscript_for_v15.md"; V16p = R + r"\drafts\manuscript_for_v16.md"; V17p = R + r"\manuscript_v3.md"
RECp = L + r"\C15\C15-T4\build_test\ms_F1b1_F1b2_F2z_F3_F4a_F4b_F4c_F5b_F6_F7.md"
N0p = L + r"\C15\C15-T4\build_test\ms_N0.md"
V15, V16, V17, REC, N0 = rd(V15p), rd(V16p), rd(V17p), rd(RECp), rd(N0p)
hashes = {k: sha(p) for k, p in dict(v15=V15p, v16=V16p, v17=V17p, rec=RECp, n0=N0p).items()}
assert hashes['v15'] == 'd3727aa0' and hashes['v17'] == '4cbf3d55', hashes
changed = [i + 1 for i in range(len(V17)) if V15[i] != V17[i]]
APPX_START = next(i + 1 for i, s in enumerate(V17) if s.startswith('# [부록'))

def r2(x, nd):
    q = Decimal(1).scaleb(-nd)
    return Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
def fmt(x, nd, pct=False, sign=False):
    v = r2(x * 100 if pct else x, nd)
    s = f"{v:.{nd}f}" if nd > 0 else f"{v:.0f}"
    if sign and v > 0: s = '+' + s
    return s.replace('-', '−')
def num_in(line, s):
    # 숫자 토큰 경계 확인
    pat = r'(?<![0-9.])' + re.escape(s) + r'(?![0-9])'
    return re.search(pat, line) is not None

# ---------- 원천 읽기 ----------
def jload(p): return json.load(open(p, encoding='utf-8'))
def cload(p): return list(csv.DictReader(open(p, encoding='utf-8-sig')))
sb15 = cload(L + r"\C15\C15-T2\seed_bundles.csv")
vd15 = {r['track']: r for r in cload(L + r"\C15\C15-T2\variance_decomp.csv")}
sb13 = cload(L + r"\C13\C13-T2\seed_bundles.csv")
vd13 = {r['track']: r for r in cload(L + r"\C13\C13-T2\variance_decomp.csv")}
t3s = jload(L + r"\C15\C15-T3\res\s5_t3_summary.json")['auc_raw']
v5 = jload(L + r"\C11\C11-T2\verify_5\v5_frac.json")
v5out = jload(L + r"\C11\C11-T2\verify_5\v5_out.json")
pkr = {r['cc']: r for r in cload(L + r"\C14\C14-T2\pkr_ipc_altA.csv")}
agree = {(r['window'], r['cc']): r for r in cload(L + r"\C10\C10-T3\res\s2_agree.csv")}
c1 = {(r['window'], r['cc']): r for r in cload(MC + r"\_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv")}
s04 = jload(L + r"\C14\C14-T3\cache\s04_out.json")
hor = cload(L + r"\C15\C15-T1\cache\s4_horizon.csv")
ntis = {r['group']: r for r in cload(MC + r"\_shared_report\ntis_t3_cohort_lapse_transfer.csv")}
f3 = jload(MC + r"\T3_기술셀부상예측\R3\final_result_r3.json")
f1 = jload(MC + r"\T1_산업대기실\R3\final_result_r3.json")
n9 = jload(MC + r"\N9_정부RnD과제_특허성과예측\final_result_r6.json")
ovp = cload(L + r"\C02\C02-T2\official_vs_provided.csv")
tvd = {r['지표']: r for r in cload(MC + r"\_shared_report\fusion_advantage\split34\t1_variance_decomposition.csv")}
c2 = jload(MC + r"\_shared_report\fusion_advantage\c5_R10\c2_results.json")
r2md = open(MC + r"\R2_결과_20260928.md", encoding='utf-8').read()
con = sqlite3.connect('file:' + BASE + r"\수집파이프라인\registry_full.sqlite" + '?mode=ro', uri=True)
reg = dict((p, n) for p, n in con.execute("select substr(rgst_no,1,2), count(*) from patent group by 1"))
con.close()

def T(track): return [r for r in sb15 if r['track'] == track]
def fl(rows, k): return [float(r[k]) for r in rows if r[k] not in ('', None)]
def hz(h, grp, col):
    return float(next(r for r in hor if r['h'] == str(h) and r['group'] == grp)[col])

checks = []  # (line, label, ms_string, src_value_float_or_str, nd, pct, sign, source)
def add(line, label, ms, val, nd, src, pct=False, sign=False, exact=False):
    checks.append(dict(line=line, label=label, ms=ms, val=val, nd=nd, pct=pct, sign=sign, src=src, exact=exact))

# 482
add(482, 'FREE-IPC F ρ(사전 규칙)', '0.373', v5['asc_free_F']['rho'], 3, r'C11\C11-T2\verify_5\v5_frac.json asc_free_F.rho')
add(482, '한국 RTA>1', '31', v5['asc_free_F']['kr_rta_gt1'], 0, r'C11\C11-T2\verify_5\v5_frac.json asc_free_F.kr_rta_gt1')
add(482, '표 동률만 반대 ρ', '0.473', v5['desc_free_F']['rho'], 3, r'C11\C11-T2\verify_5\v5_frac.json desc_free_F.rho')
def find_num(o, keypat):
    out = []
    def w(x, p):
        if isinstance(x, dict):
            for k, v in x.items(): w(v, p + '/' + k)
        elif isinstance(x, list):
            for i, v in enumerate(x): w(v, p + f'[{i}]')
        elif isinstance(x, (int, float)) and re.search(keypat, p): out.append((p, x))
    w(o, ''); return out
full = [('range_168/REG_tie_desc_full/max_raw', v5out['range_168']['REG_tie_desc_full']['max_raw'])]
add(482, '배정까지 반대 ρ(tie_desc_full)', '0.476', full[0][1] if full else float('nan'), 3, r'C11\C11-T2\verify_5\v5_out.json ' + (full[0][0] if full else '없음'))
add(482, '본문 168행 최대', '0.48', full[0][1] if full else float('nan'), 2, r'같은 값 2자리')
# 483
add(483, 'T3 dev2final 공식', '−0.005', float(vd15['T3']['dev2final_official']), 3, r'C15\C15-T2\variance_decomp.csv T3 dev2final_official')
add(483, 'T3 dev2final 시드 최소', '−0.005', float(vd15['T3']['dev2final_min']), 3, r'C15\C15-T2\variance_decomp.csv T3 dev2final_min')
add(483, 'T3 dev2final 시드 최대', '+0.003', float(vd15['T3']['dev2final_max']), 3, r'C15\C15-T2\variance_decomp.csv T3 dev2final_max', sign=True)
add(483, 'T1 dev2final 공식', '+0.126', float(vd15['T1']['dev2final_official']), 3, r'C15\C15-T2\variance_decomp.csv T1 dev2final_official', sign=True)
# 485
add(485, 'T3 공식 skill', '0.423', f3['summary']['mean_skill'], 3, r'(MC)\T3\R3\final_result_r3.json summary.mean_skill')
add(485, 'T3 CI 하', '0.366', f3['ci95']['skill'][0], 3, r'final_result_r3.json ci95.skill[0]')
add(485, 'T3 CI 상', '0.473', f3['ci95']['skill'][1], 3, r'final_result_r3.json ci95.skill[1]')
t3sk = fl(T('T3'), 'skill')
add(485, 'T3 시드 skill 최소', '0.423', min(t3sk), 3, r'C15\C15-T2\seed_bundles.csv T3 skill min')
add(485, 'T3 시드 skill 최대', '0.432', max(t3sk), 3, r'C15\C15-T2\seed_bundles.csv T3 skill max')
off3 = float(next(r for r in T('T3') if r['official_bundle'] == 'True')['skill'])
checks.append(dict(line=485, label='공식이 최솟값', ms='공식이 최솟값', val='예' if off3 == min(t3sk) else '아니오', nd=None, pct=False, sign=False, src='seed_bundles T3 official skill == min', exact=True))
# 488
add(488, 'T1 공식 skill', '0.473', f1['skill']['auc_all'], 3, r'(MC)\T1\R3\final_result_r3.json skill.auc_all')
t1sk = fl(T('T1'), 'skill'); t1lo = fl(T('T1'), 'ci_lo')
add(488, 'T1 시드 skill 최소', '0.472', min(t1sk), 3, r'C15\C15-T2\seed_bundles.csv T1 skill min')
add(488, 'T1 시드 skill 최대', '0.475', max(t1sk), 3, r'C15\C15-T2\seed_bundles.csv T1 skill max')
add(488, 'T1 시드 하한 최소', '0.453', min(t1lo), 3, r'seed_bundles T1 ci_lo min')
add(488, 'T1 시드 하한 최대', '0.456', max(t1lo), 3, r'seed_bundles T1 ci_lo max')
add(488, 'T1 공식 CI 하', '0.454', float(vd15['T1']['ci_lo_official']), 3, r'variance_decomp T1 ci_lo_official')
# 498/499/501
for line, cc, ms in [(498, 'US', dict(n='556,311', g='38.2', c5='3.7', txt='30.0', free='50.5', pkr='71.9')),
                     (499, 'JP', dict(n='358,744', g='37.3', c5='5.8', txt='29.4', free='48.0', pkr='73.8')),
                     (501, 'EP', dict(n='187,919', g='37.0', c5='6.4', txt='30.7', free='53.1', pkr='72.8'))]:
    w = '출원2015-19_발명'
    add(line, f'{cc} n', ms['n'], float(c1[(w, cc)]['n']), 0, r'(MC)\c5_R10\c1_label_agreement.csv n', exact=True)
    add(line, f'{cc} 연계표 그룹', ms['g'], float(c1[(w, cc)]['cw_group']), 1, 'c1_label_agreement.csv cw_group', pct=True)
    add(line, f'{cc} 연계표 5자리', ms['c5'], float(c1[(w, cc)]['cw5']), 1, 'c1_label_agreement.csv cw5', pct=True)
    add(line, f'{cc} TXT(대안 A)', ms['txt'], float(agree[(w, cc)]['txt5_reg']), 1, r'C10\C10-T3\res\s2_agree.csv txt5_reg', pct=True)
    add(line, f'{cc} FREE-IPC(대안 A)', ms['free'], float(agree[(w, cc)]['free5_reg']), 1, r'C10\C10-T3\res\s2_agree.csv free5_reg', pct=True)
    add(line, f'{cc} PKR-IPC 대안 A', ms['pkr'], float(pkr[cc]['대안A_REG']), 1, r'C14\C14-T2\pkr_ipc_altA.csv 대안A_REG')
    add(line, f'{cc} PKR-IPC 독립재현', ms['pkr'], float(pkr[cc]['독립재현_REG']), 1, r'C14\C14-T2\pkr_ipc_altA.csv 독립재현_REG')
# 503
add(503, 'KR 등록 계', '656,360', reg['10'] + reg['20'], 0, r'수집파이프라인\registry_full.sqlite(mode=ro) patent count', exact=True)
add(503, '특허(10)', '615,042', reg['10'], 0, 'registry_full.sqlite rgst_no 10', exact=True)
add(503, '실용신안(20)', '41,318', reg['20'], 0, 'registry_full.sqlite rgst_no 20', exact=True)
# 519
add(519, '제공(drop_ros)', '0.7867', t3s['drop_ros'], 4, r'C15\C15-T3\res\s5_t3_summary.json auc_raw.drop_ros')
add(519, 'KR만', '0.7529', t3s['kr_rebuild'], 4, 'auc_raw.kr_rebuild')
add(519, 'hyb_txt(REG)', '0.7626', t3s['hyb_txt_reg@C10'], 4, 'auc_raw.hyb_txt_reg@C10')
add(519, 'hyb_free(REG)', '0.7605', t3s['hyb_free_reg@C10'], 4, 'auc_raw.hyb_free_reg@C10')
add(519, 'hyb_pkr(REG)', '0.7755', t3s['hyb_pkr_reg'], 4, 'auc_raw.hyb_pkr_reg')
# 534
n13 = sum(1 for r in sb13); lo13 = sum(1 for r in sb13 if float(r['ci_lo']) > 0)
n15 = len(sb15); lo15 = sum(1 for r in sb15 if float(r['ci_lo']) > 0)
checks.append(dict(line=534, label='M0 대비 하한>0 묶음', ms='76/76', val=f'{lo13 + lo15}/{n13 + n15}', nd=None, pct=False, sign=False, src=r'C13\C13-T2 + C15\C15-T2 seed_bundles.csv ci_lo>0 계수', exact=True))
t4 = [r for r in sb13 if r['track'] == 'T4']; t4s = [float(r['strong_ci_lo']) for r in t4]
add(534, 'T4 강화 하한 최소', '−0.016', min(t4s), 3, r'C13\C13-T2\seed_bundles.csv T4 strong_ci_lo min')
add(534, 'T4 강화 하한 최대', '−0.003', max(t4s), 3, 'C13 seed_bundles T4 strong_ci_lo max')
s1lo = fl(T('T1'), 'strong_ci_lo'); s3lo = fl(T('T3'), 'strong_ci_lo')
add(534, 'T1 SBB 하한 최소', '0.376', min(s1lo), 3, r'C15\C15-T2\seed_bundles.csv T1 strong_ci_lo min')
add(534, 'T1 SBB 하한 최대', '0.384', max(s1lo), 3, 'C15 seed_bundles T1 strong_ci_lo max')
add(534, 'T3 SB3 하한 최소', '0.124', min(s3lo), 3, 'C15 seed_bundles T3 strong_ci_lo min')
add(534, 'T3 SB3 하한 최대', '0.143', max(s3lo), 3, 'C15 seed_bundles T3 strong_ci_lo max')
ratios = [float(r['ratio_seed_sd_to_boot_se']) for r in list(vd13.values()) + list(vd15.values())]
add(534, 'sd/SE 배율 최소(8트랙)', '0.06', min(ratios), 2, r'C13·C15 variance_decomp.csv ratio_seed_sd_to_boot_se min')
add(534, 'sd/SE 배율 최대(8트랙)', '0.19', max(ratios), 2, 'ratio max')
add(534, 'T3 공식 skill', '0.423', off3, 3, 'C15 seed_bundles T3 official')
add(534, 'T3 시드 최대', '0.432', max(t3sk), 3, 'C15 seed_bundles T3 max')
# 542
add(542, 'right_list 행', '3,366,046', s04['rl_total_rows'], 0, r'C14\C14-T3\cache\s04_out.json rl_total_rows', exact=True)
add(542, '범위밖 고유 특허', '1,494,298', s04['rl_patent_unique_out_of_scope'], 0, 's04_out.json rl_patent_unique_out_of_scope', exact=True)
add(542, '목록 고유 특허 대비', '73.0', s04['rl_patent_unique_out_of_scope'] / s04['rl_patent_unique'], 1, 's04_out.json oos/unique', pct=True)
# 577
tk = [int(float(r['topk2'])) for r in T('T1')]; offk = int(float(next(r for r in T('T1') if r['official_bundle'] == 'True')['topk2']))
add(577, 'T1 상위1000 공식', '31.3', f1['topk']['model']['1000']['precision'], 1, r'(MC)\T1\R3\final_result_r3.json topk.model.1000', pct=True)
add(577, '기준선', '12.4', f1['topk']['strongest_auc_baseline']['1000']['precision'], 1, 'final_result_r3.json topk.strongest_auc_baseline.1000', pct=True)
add(577, '무작위(기저율)', '0.56', 7723 / 1369514, 2, 'final_result_r3.json n.pos/n.rows', pct=True)
add(577, '시드 최소', '29.8', min(tk) / 1000, 1, r'C15\C15-T2\seed_bundles.csv T1 topk2 min', pct=True)
add(577, '시드 최대', '31.6', max(tk) / 1000, 1, 'seed_bundles T1 topk2 max', pct=True)
rank = 1 + sum(1 for x in tk if x > offk)
checks.append(dict(line=577, label='공식 순위', ms='두 번째로 높음', val=f'{rank}위(값 {sorted(tk, reverse=True)})', nd=None, pct=False, sign=False, src='seed_bundles T1 topk2 순위', exact=True, ok=(rank == 2)))
# 638
base, kr = t3s['drop_ros'], t3s['kr_rebuild']
for lab, key, ms in [('hyb_txt 보존', 'hyb_txt_reg@C10', '29'), ('hyb_free 보존', 'hyb_free_reg@C10', '23'), ('hyb_pkr 보존', 'hyb_pkr_reg', '67')]:
    add(638, lab, ms, (t3s[key] - kr) / (base - kr), 0, f's5_t3_summary auc_raw ({key}−kr_rebuild)/(drop_ros−kr_rebuild)', pct=True)
add(638, '상실 하한(1−txt)', '71', 1 - (t3s['hyb_txt_reg@C10'] - kr) / (base - kr), 0, '1−보존(txt)', pct=True)
add(638, '상실 상한(1−free)', '77', 1 - (t3s['hyb_free_reg@C10'] - kr) / (base - kr), 0, '1−보존(free)', pct=True)
# 651
add(651, '5년 기원', '14.2', float(ntis['정부R&D 기원(전체)']['lapse5']), 1, r'(MC)\_shared_report\ntis_t3_cohort_lapse_transfer.csv lapse5', pct=True)
add(651, '5년 비기원', '21.0', float(ntis['정부R&D 기원 아님']['lapse5']), 1, 'ntis_t3 lapse5', pct=True)
add(651, '이전 기원', '10.5', float(ntis['정부R&D 기원(전체)']['transfer5']), 1, 'ntis_t3 transfer5', pct=True)
add(651, '이전 비기원', '11.5', float(ntis['정부R&D 기원 아님']['transfer5']), 1, 'ntis_t3 transfer5', pct=True)
add(651, '5년 기원(C15 소멸등록일)', '14.2', hz(5, '정부R&D 기원', 'regdate'), 1, r'C15\C15-T1\cache\s4_horizon.csv h5 regdate', pct=True)
add(651, '7년 기원', '37.7', hz(7, '정부R&D 기원', 'regdate'), 1, r'C15\C15-T1\cache\s4_horizon.csv h7 regdate', pct=True)
add(651, '7년 비기원', '35.2', hz(7, '비기원', 'regdate'), 1, 's4_horizon.csv h7 regdate', pct=True)
add(651, '10년 기원', '55.5', hz(10, '정부R&D 기원', 'regdate'), 1, 's4_horizon.csv h10 regdate', pct=True)
add(651, '10년 비기원', '52.2', hz(10, '비기원', 'regdate'), 1, 's4_horizon.csv h10 regdate', pct=True)
coh10 = next(r for r in hor if r['h'] == '10' and r['group'] == '비기원')['cohort']
checks.append(dict(line=651, label='10년 코호트', ms='등록 2007–15', val=coh10, nd=None, pct=False, sign=False, src='s4_horizon.csv h10 cohort', exact=True, ok=(coh10 == '2007–2015')))
# 본문 숫자(바뀐 구절만)
m1 = re.search(r'T1 skill (0\.\d+) \[', r2md); m3 = re.search(r'T3 (0\.\d+) \[', r2md)
add(109, 'T1 앞 라운드 최종', '0.418', float(m1.group(1)), 3, r'(MC)\R2_결과_20260928.md 11행')
add(109, 'T3 앞 라운드 최종', '0.392', float(m3.group(1)), 3, r'(MC)\R2_결과_20260928.md 12행')
kr_r = [float(r['ratio_pat']) for r in ovp if r['country'] == 'KR' and r['ratio_pat'] and 2012 <= int(r['year']) <= 2023]
cn_r = [float(r['ratio_t3year']) for r in ovp if r['country'] == 'CN' and r['ratio_t3year'] and 2018 <= int(r['year']) <= 2022]
add(143, 'KR 등록 비율 최소(2012–23)', '0.25', min(kr_r), 2, r'C02\C02-T2\official_vs_provided.csv KR ratio_pat')
add(143, 'KR 등록 비율 최대', '0.26', max(kr_r), 2, 'official_vs_provided.csv KR ratio_pat')
add(143, 'CN 비율 최소(2018–22)', '0.20', min(cn_r), 2, 'official_vs_provided.csv CN ratio_t3year')
add(143, 'CN 비율 최대', '0.21', max(cn_r), 2, 'official_vs_provided.csv CN ratio_t3year')
add(265, '원판 챔피언 AUC', '0.899', n9['secondary_orig_champion']['auc'], 3, r'(MC)\N9\final_result_r6.json secondary_orig_champion.auc')
add(265, '모형 AUC(비교)', '0.894', n9['auc'], 3, 'final_result_r6.json auc (원고에 없음, 우위 방향 확인용)')
for lab, key, ms in [('KR 그룹안 비중', 'g_KR', '0.561'), ('US', 'g_US', '0.551'), ('JP', 'g_JP', '0.596'), ('EP', 'g_EP', '0.563'), ('5국 합', 'g_5국', '29.1')]:
    v = float(tvd[key]['그룹안_분산비중_34'])
    if key == 'g_5국': add(191, lab, ms, v, 1, r'(MC)\fusion_advantage\split34\t1_variance_decomposition.csv', pct=True)
    else: add(191, lab + '(근거, 원고에 국가명만)', ms, v, 3, 't1_variance_decomposition.csv 그룹안_분산비중_34')

rows_chk = []
for c in checks:
    line = c['line']; txt = V17[line - 1]
    if c['nd'] is None:
        ok = c.get('ok', str(c['val']).startswith(c['ms']) or c['ms'] in str(c['val']))
        if c['label'] == '공식이 최솟값': ok = (c['val'] == '예')
        if c['label'] == 'M0 대비 하한>0 묶음': ok = (c['val'] == '76/76')
        present = c['ms'] in txt
        recomputed = str(c['val'])
    else:
        recomputed = fmt(c['val'], c['nd'], c['pct'], c['sign'])
        if c['exact']:
            recomputed = f"{int(round(c['val'])):,}"
        ms_cmp = c['ms'].replace('+', '') if not c['sign'] else c['ms']
        ok = (recomputed == c['ms']) or (recomputed == ms_cmp)
        present = c['ms'] in txt
        if '원고에 없음' in c['src'] or '근거, 원고에' in c['label']:
            present = True
    rows_chk.append(dict(행=line, 항목=c['label'], 원고값=c['ms'], 원천재계산=recomputed,
                         원천원값=(repr(c['val']) if c['nd'] is not None else ''), 원고에있음=present, 일치=bool(ok) and present, 출처=c['src']))
with open(OUT + r"\numeric_checks.csv", 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_chk[0].keys())); w.writeheader(); w.writerows(rows_chk)

# ---------- 본문·문구 대조 ----------
SPEC = {
 50: (['E-C15T4-01(관련, 표 1)'], ['라운드마다 새로 잠근 최종시험 1회'], []),
 79: (['E-C14T2-02 전체판', 'E-C14T3-02', 'E-C15T4-07'],
      ['① 모든 트랙은 모집단이나 결과를 제공 데이터로 정의한다(N6·N9 모집단은 NTIS, N4·N5·N7 결과는 외부 사건; 등록원부도',
       '(예외: 물가·산업연관표·생산지수의 현행 빈티지와 2026 권리목록의 보유 상태, 2.5)',
       '제공 범위 KR 등록특허·실용신안 656,360건을'], []),
 99: (['E-C15T4-03(b)'], ['KIPRIS 권리목록(범위밖 특허·디자인·실용신안)은 T1 피처로'], []),
 109: (['E-C15T4-01 01b'], ['가장 최근 구간은 라운드마다 새로 잠갔고,', 'T3는 누수를 고친 뒤 다시 채점해 수정 전 값(0.424)도 공개한다. T1·T3 공식 창은 앞 라운드의 다른 정의 최종시험(T1 0.418, T3 0.392)과 결과 기간이 겹친다.'],
       ['홀드아웃 재사용', '처음부터 잠가 정답을 열어 보지 않았고']),
 143: (['E-C14T2-01', 'E-C14T3-01', 'E-C15T4-02'],
       ['각각 빼도 T1·T3 skill 하락 0.0022 이하)', '시점을 맞추거나(T1 권리목록은 넘긴 권리가 빠짐, 영향 0.007 이하) 해석층으로만', '공식 통계 대비 건수 비율은 KR 등록 2012–23년 0.25–0.26, CN 2018–22년 0.20–0.21로 안정적이다.'],
       ['상한 0.007', '(KR 등록 0.25–0.26, CN 0.20–0.21; 예외 4.4)']),
 191: (['E-C15T4-05'], ['합계 추세는 연계표로 읽히지만 한국 지표는(US·JP·EP도) 그렇지 않다.'], []),
 229: (['E-C14T3-03', 'E-C15T4-03(a)'], ['2019년까지 출원한 특허 3건 이상이면서', 'KIPRIS 권리목록(범위밖 특허·디자인·실용신안), ECOS 물가'], ['원점 2019년에 특허 3건']),
 251: (['E-C15T4-03(c)'], ['KIPRIS Plus 권리목록·디자인·실용신안'], []),
 265: (['E-C15T4-04b'], ['두 원판 대비 우위는 순위 전반(AUC)에 있다(원판 챔피언 AUC 0.899는 더 높다).'], []),
 334: (['E-C14T1-01(선택 부분 미반영)'], ['법인×대상 업종 이원 군집 CI는 0 포함'], ['법인×서브클래스 이원 군집 CI는 0 포함']),
 362: (['E-C15T4-06'], ['예측 피처(T3 경계·N9·N6)'], []),
 482: (['E-C15T4-A1'], ['(최빈 동률 규칙 의존: 표 동률만 반대 0.473, 배정까지 반대 0.476 → 본문 168행 최대 0.48)'], []),
 483: (['E-C15T2-03'], ['T3 −0.005(학습 시드 10묶음 −0.005~+0.003)'], []),
 485: (['E-C15T2-02'], ['(다른 학습 시드 10묶음: 0.423–0.432, 공식이 최솟값, C15-T2)'], []),
 488: (['E-C15T2-02'], ['(다른 학습 시드 10묶음: 0.472–0.475, 하한 0.453–0.456, C15-T2)'], []),
 498: (['E-C14T2-03'], ['진단용 PKR-IPC 71.9%(C14-T2 대안 A)'], ['68.5%']),
 499: (['E-C14T2-03'], ['진단용 PKR-IPC 73.8%(C14-T2 대안 A)'], ['71.1%']),
 501: (['E-C14T2-03'], ['진단용 PKR-IPC 72.8%(C14-T2 대안 A)'], ['68.8%']),
 503: (['E-C15T4-07b'], ['제공 범위 KR 등록 656,360건(특허 615,042·실용신안 41,318)'], []),
 519: (['E-C14T2-04 → E-C15T3-01(대체)'], ['hyb_pkr(진단) 0.7755(등록번호 연결·C15-T3)'], ['0.7735', '미재계산']),
 534: (['E-C15T2-01'], [open(L + r"\C15\C15-T2\cache\l534_new.txt", encoding='utf-8').read().replace('\r\n', '\n').split('\n')[1].strip()], []),
 542: (['E-C14T3-05'], ['제공범위 밖 특허 1,494,298건(목록 고유 특허의 73.0%)'], []),
 577: (['E-C15T2-02'], ['다른 학습 시드 10묶음: 29.8–31.6%, 공식은 두 번째로 높음, C15-T2)'], []),
 638: (['E-C14T2-04 → E-C15T3-02(대체)'], ['hyb_pkr 67%(등록번호 연결·C15-T3)'], ['61%', '미재계산']),
 651: (['E-C15T1-02'], ['; 7년 내 37.7% 대 35.2%, 10년 내(등록 2007–15) 55.5% 대 52.2%(6년차 기한 집중, C15-T1)'], []),
}
SRC_COL = {  # ERRATA가 요구한 출처 칸 추가
 498: r'C14\C14-T2\pkr_ipc_altA.csv', 499: r'C14\C14-T2\pkr_ipc_altA.csv', 501: r'C14\C14-T2\pkr_ipc_altA.csv',
 534: r'C15\C15-T2\seed_bundles.csv', 651: r'C15\C15-T1\cache\s4_horizon.csv', 482: r'C11\C11-T2\verify_5\v5_frac.json',
}
audit = []
for ln in changed:
    v15, v16, v17 = V15[ln - 1], V16[ln - 1], V17[ln - 1]
    when = ('v16' if v15 != v16 else '') + ('+' if (v15 != v16 and v16 != v17) else '') + ('v17' if v16 != v17 else '')
    area = '부록' if ln >= APPX_START else '본문'
    ids, musts, forb = SPEC[ln]
    miss = [m for m in musts if m not in v17]
    forb_hit = [x for x in forb if x in v17]
    # 반박 반영 판단용 메모
    if area == '본문':
        same_rec = (v17 == REC[ln - 1]); same_n0 = (v16 == N0[ln - 1])
        if v16 != v17:
            build = '권장 묶음 시험 빌드와 동일' if same_rec else ('시험 빌드 없음(메인 판단)' if ln == 50 else '시험 빌드와 다름')
        else:
            build = 'v16=N0(document.xml 동일)·v17 불변' if same_n0 else 'N0와 다름'
    else:
        build = '제출본 밖(docx 미포함)'
    nchk = [r for r in rows_chk if r['행'] == ln]
    nok = sum(1 for r in nchk if r['일치']); nbad = len(nchk) - nok
    src_note = ''
    if ln in SRC_COL:
        src_note = '출처 칸 보충 있음' if SRC_COL[ln] in v17 else f'출처 칸 보충 없음({SRC_COL[ln]})'
    if miss:
        verdict = '원안 미반영(문구 차이)'
    elif forb_hit:
        verdict = '금지/옛 문구 잔존'
    else:
        verdict = '원안 반영'
    if nchk:
        verdict += f'; 숫자 {nok}/{len(nchk)} 일치' + (' — 불일치 있음' if nbad else '')
    if src_note and '없음' in src_note:
        verdict += '; ' + src_note
    audit.append(dict(행=ln, 영역=area, 바뀐판=when, ERRATA=';'.join(ids), 판정=verdict, 빌드대조=build,
                      문구누락=' | '.join(miss), 금지문구=' | '.join(forb_hit),
                      숫자근거=' ; '.join(f"{r['항목']} {r['원고값']}={r['원천재계산']}{'' if r['일치'] else '(X)'} [{r['출처']}]" for r in nchk)))
with open(OUT + r"\change_audit_raw.csv", 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(audit[0].keys())); w.writeheader(); w.writerows(audit)
json.dump(dict(hashes=hashes, changed=changed, n_changed=len(changed), appendix_start=APPX_START,
               n_checks=len(rows_chk), n_ok=sum(r['일치'] for r in rows_chk),
               bad=[r for r in rows_chk if not r['일치']]), open(OUT + r"\s1_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(hashes, len(changed), APPX_START, 'checks', len(rows_chk), 'ok', sum(r['일치'] for r in rows_chk))
for r in rows_chk:
    if not r['일치']: print('BAD', r)
for a in audit: print(a['행'], a['바뀐판'], a['판정'], '|', a['빌드대조'], '|', a['문구누락'][:80], a['금지문구'])
