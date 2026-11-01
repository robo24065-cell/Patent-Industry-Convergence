# C08-T2 verify_0: 감사 판정과 독립된 코드로 '부분'·'오류' 전부 + '정확' 무작위 10개를 다시 판정한다.
# 원칙: 현재 원고 줄은 줄 번호가 아니라 내용 앵커로 찾고, 숫자는 원천 파일(가능하면 원자료)에서 다시 계산하며,
#       필요한 조건은 원천 데이터에서 자동 도출(예: 끝 창에서 유의성이 바뀐 지표 → 원고에 그 단서가 있어야 함).
# 판정: 숫자 불일치 → 오류, 숫자 일치 + 도출된 조건 누락 → 부분, 모두 충족 → 정확.  (읽기 전용, 쓰기는 verify_0 안)
import sys, os, re, json, math, csv, hashlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
BASE = sys.argv[1]  # ...\분석
W = os.path.join(BASE, '검증루프_20261012', 'C08', 'C08-T2')
L = os.path.join(BASE, '검증루프_20261012')
MC = os.path.join(BASE, '모델경쟁_20260927')
RP = os.path.join(BASE, '보고서_20261012')
MS = os.path.join(RP, 'manuscript_v3.md')
ms_raw = open(MS, 'rb').read()
ms_sha = hashlib.sha1(ms_raw).hexdigest()
LINES = ms_raw.decode('utf-8').replace('\r\n', '\n').split('\n')


def line(anchor):
    hits = [(i + 1, l) for i, l in enumerate(LINES) if anchor in l]
    if len(hits) != 1:
        raise RuntimeError(f'앵커 {anchor!r} 일치 {len(hits)}건')
    return hits[0]


def num_ok(claim_text, value):
    """원고 표기(claim_text)의 자릿수 기준 ±0.5단위로 value와 일치하는지"""
    s = re.search(r'-?\d+(\.\d+)?', claim_text.replace('−', '-').replace(',', '')).group()
    dec = len(s.split('.')[1]) if '.' in s else 0
    return abs(float(s) - value) <= 0.5 * 10 ** (-dec) + 1e-12


out = []


def judge(block, src, anchor, nums, conds, extra=None):
    """nums: [(원고 표기, 재계산 값, 설명)], conds: [(필요 여부, [허용 정규식...], 설명)]"""
    ln, txt = line(anchor)
    num_res = [(c, round(v, 6), d, (c in txt) and num_ok(c, v)) for c, v, d in nums]
    cond_res = []
    for need, pats, d in conds:
        if not need:
            continue
        ok = any(re.search(p, txt) for p in pats)
        cond_res.append((d, ok))
    if not all(r[3] for r in num_res):
        v = '오류'
    elif not all(ok for _, ok in cond_res):
        v = '부분'
    else:
        v = '정확'
    out.append(dict(block=block, src=src, line=ln, verdict=v, nums=num_res, conds=cond_res, extra=extra))


# ---------- 원천 로드 ----------
# KOSIS 광업·제조업조사 10차(원자료 JSON) — C07-T1 내려받기
K = json.load(open(os.path.join(L, 'C07', 'C07-T1', 'download', 'kosis10', 'DT_1FS1001_2007_2020.json'), encoding='utf-8'))
kd = pd.DataFrame(K)
kd = kd[kd['C1'] == '00'].copy()
kd['code'] = kd['C2'].str.replace('C', '', regex=False)
kd['val'] = pd.to_numeric(kd['DT'], errors='coerce'); kd['yr'] = kd['PRD_DE'].astype(int)


def kgrow(code, itm, a=(2008, 2009), b=(2016, 2017)):
    s = kd[(kd.code == code) & (kd.ITM_ID == itm)].set_index('yr').val
    A = s.loc[list(a)].mean(); B = s.loc[list(b)].mean()
    return 100 * (np.log(B) - np.log(A)) / (b[0] - a[0])


def kval(code, itm, yr):
    s = kd[(kd.code == code) & (kd.ITM_ID == itm)].set_index('yr').val
    return s.get(yr, np.nan)


f4 = pd.read_csv(os.path.join(RP, 'figs', 'fig04_within_code_divergence.csv'), encoding='utf-8-sig')
sens = pd.read_csv(os.path.join(L, 'C07', 'C07-T1', 'sensitivity_313.csv'), encoding='utf-8-sig')
quad = pd.read_csv(os.path.join(L, 'C03', 'C03-T2', 'quadrant_sensitivity.csv'), encoding='utf-8-sig')
t1 = json.load(open(os.path.join(MC, 'T1_산업대기실', 'R3', 'final_result_r3.json'), encoding='utf-8'))['bootstrap_firm_cluster']
pros = json.load(open(os.path.join(L, 'C07', 'C07-T3', 'prospective_result.json'), encoding='utf-8'))['windows']['main']
cal = pd.read_csv(os.path.join(L, 'C07', 'C07-T3', 'calibration.csv'), encoding='utf-8-sig')
t3s = json.load(open(os.path.join(L, 'C04', 'C04-T3', 's3_results.json'), encoding='utf-8'))
lab = json.load(open(os.path.join(L, 'C03', 'C03-T1', 'label_reconstruct.json'), encoding='utf-8'))
ntis = json.load(open(os.path.join(L, 'C03', 'C03-T3', '_s7_ntis_totals.json'), encoding='utf-8'))
n9 = json.load(open(os.path.join(MC, 'N9_정부RnD과제_특허성과예측', 'final_result_r6.json'), encoding='utf-8'))
t3p = pd.read_parquet(os.path.join(MC, 'T3_기술셀부상예측', 'R3', 'preds', 'FINAL_R3_t0_2021.parquet'))
top100 = json.load(open(os.path.join(L, 'C03', 'C03-T3', 'verify_2', 'v2_t3_top100.json'), encoding='utf-8'))
v6 = open(os.path.join(RP, 'drafts', 'manuscript_for_v6.md'), encoding='utf-8').read().split('\n')

# KR 출원 성장률(그림 5 CSV)
def f4val(code, col_sub):
    cols = [c for c in f4.columns if c.startswith(col_sub)]
    r = f4[f4['ksic'].astype(str).str.replace('.0', '', regex=False) == code].dropna(subset=[cols[0]]).iloc[0]
    return float(r[cols[0]])

# ---------- B61 (192행) ----------
v1a = sens[sens.variant == 'V1a']
def sv(metric, col='orig'):
    return float(v1a[v1a.metric == metric][col].iloc[0])
flip = v1a[(v1a['p_cross_0.05'] == True)].metric.tolist()  # 끝 창에서 유의성이 바뀐 지표
need_emp = any('종사자 r' in m for m in flip); need_est_var = any('사업체 그룹 안 분산' in m for m in flip)
judge('B61', 'audit_v8_v9', 'KOSIS 5자리 통계',
      [('59.1%', 100 * sv('SP-VAR 출하 그룹 안 분산 비중'), '출하 분산 비중'), ('47.8%', 100 * sv('SP-VAR 사업체 그룹 안 분산 비중'), '사업체 분산 비중'),
       ('0.41', sv('SP-FE 사업체 r'), 'FE 사업체'), ('0.35', sv('SP-FE 출하 r'), 'FE 출하'), ('0.30', sv('SP-FE 종사자 r'), 'FE 종사자')],
      [(True, [r'10인 이상'], '10인 이상'), (True, [r'명목'], '명목'),
       (need_emp, [r'종사자[^.]*끝 창'], '종사자 끝 창 민감'),
       (need_est_var, [r'사업체[^.]*(분산|비중)[^.]*끝 창', r'끝 창[^.]*사업체[^.]*(분산|비중)', r'47\.8%[^)]*(민감|끝 창)'], '사업체 분산 비중 끝 창 비유의')],
      extra={'p_cross_flip_metrics': flip})

# ---------- B62 (194행) ----------
oled_est = [kval('26212', 'T01', y) for y in (2008, 2009, 2016, 2017)]
small = np.nanmax(oled_est) < 20
txt194 = line('같은 코드 C2620 안에서')[1]
judge('B62', 'audit_v8_v9', '같은 코드 C2620 안에서',
      [('−6.6%', f4val('26211', 'KR출원'), 'LCD KR'), ('+6.2%', f4val('26212', 'KR출원'), 'OLED KR'),
       ('−3.1%', kgrow('26211', 'T01'), '26211 사업체'), ('+7.2%', kgrow('26212', 'T01'), '26212 사업체'), ('+14.2%', kgrow('26111', 'T04'), '26111 출하')],
      [(True, [r'출하는[^.]*뺐'], '출하 대비 삭제'), (True, [r'10인 이상'], '10인 이상'),
       (small, [r'사업체 \d+[–-]?\d*개', r'안팎', r'소수 사업'], 'OLED 코드 사업체 소수(%s)' % oled_est)],
      extra={'oled_est_2008_09_16_17': oled_est, 'no_old_ship_numbers': ('+9.4' not in txt194 and '+0.3%' not in txt194)})

# ---------- B65 / B10 (242행) ----------
dv = {k.replace('diff_vs_main__', ''): v for k, v in t1.items() if k.startswith('diff_vs_main__')}
ci0 = [k for k, v in dv.items() if v['auc_all'][0] <= 0 <= v['auc_all'][2]]
lo = min(v['auc_all'][1] for v in dv.values()); hi = max(v['auc_all'][1] for v in dv.values())
for b, src in [('B65', 'audit_v8_v9'), ('B10', 'audit_v4_v7')]:
    judge(b, src, '폐업일 보정(사전 등록 보조 변형)',
          [('+0.0004', lo, '전체 AUC 최소'), ('+0.0011', hi, '전체 AUC 최대'), ('−0.0024', dv['model_nps']['auc_new'][1], '신규진입자 NPS')],
          [(len(ci0) > 0, [r'CI[가는]? ?0', r'0 포함'], f'전체 AUC CI 0 포함 변형 {ci0}')], extra={'ci_includes_0': ci0})

# ---------- B76 (395행) ----------
r_cal = cal[(cal.level == '전체') & (cal.window == 'main') & (cal.model == 'champ')].iloc[0]
raw_ratio = r_cal.sum_p_24m / r_cal.realized
judge('B76', 'audit_v8_v9', '「특허-산업 조기경보 연례 지수」를 제안한다.',
      [('90', pros['pos'], '양성'), ('0.314', pros['skill_vs_M0'], 'skill vs M0'), ('0.124', pros['skill_vs_M0_ci'][0], 'CI 하한'), ('0.516', pros['skill_vs_M0_ci'][1], 'CI 상한'),
       ('1.57', r_cal.ratio, '보정 배율(창 길이 보정)')],
      [(True, [r'공식 외'], '공식 외'), (pros['skill_vs_SB_ci'][0] < 0, [r'재현되지'], 'SB 대비 미재현'),
       (pros['days'] != 730, [r'18개월', r'24개월', r'창 길이'], f'창 {pros["days"]}일 ≠ 공식 730일'),
       (abs(raw_ratio - r_cal.ratio) > 0.1, [r'창 길이', r'보정'], f'1.57은 보정값(미보정 {raw_ratio:.2f})'),
       (True, [r'공란', r'결손', r'2\.1'], 'ERRATA 비고: 등록번호 공란 결손 2.1–2.9% 단서')])

# ---------- B07 (185행) ----------
hs = quad[quad.variant.str.startswith(('A1_', 'A1lo', 'A1hi', 'P_A1c_'))]
allv = quad[quad.variant != 'B_original']
rng_hs = (hs.rho_chg.min(), hs.rho_chg.max()); rng_all = (allv.rho_chg.min(), allv.rho_chg.max())
judge('B07', 'audit_v4_v7', 'RTA 변화와 RCA 변화의 관계',
      [('0.048', float(quad[quad.variant == 'B_original'].rho_chg.iloc[0]), '원판'), ('0.16', rng_hs[0], 'HS 정합 최소'), ('0.24', rng_hs[1], 'HS 정합 최대')],
      [(True, [r'0 포함'], 'CI 0 포함'),
       (abs(rng_all[0] - rng_hs[0]) > 0.05, [r'HS', r'정합'], f'범위는 HS 정합 변형만(전체 변형 {rng_all[0]:.3f}–{rng_all[1]:.3f})')])

# ---------- B09 / B30 (229행) ----------
rem = lab['official']['n']['removed_prior_pairs']
for b in ['B09', 'B30']:
    judge(b, 'audit_v4_v7', '어떤 산업에 아직 특허가 없는 기업 중',
          [('1,369,514', lab['official']['n']['rows'], '위험쌍'), ('7,723', lab['official']['n']['pos'], '진입'), ('0.564%', 100 * lab['official']['n']['pos'] / lab['official']['n']['rows'], '비율')],
          [(True, [r'2025-08', r'2025년 8월'], '진입 시한'), (True, [r'이름'], '이름 매칭'),
           (rem > 0, [r'선행 출원', r'공개·등록된 쌍', r'출원[^.]*없는 쌍'], f'선행 공개·등록 출원 쌍 제외({rem}쌍)')])

# ---------- B28 (219행) ----------
tp = t3s['top504']
txt219 = line('정확도의 큰 부분은 한국 이력과 IPC')[1]
small_share = tp['hits_D_le3'] / tp['hits']
attr_bad = bool(re.search(r'81%는[^,.]*소규모', txt219)) and abs(small_share - tp['hits_krW_le1'] / tp['hits']) > 0.1
judge('B28', 'audit_v4_v7', '정확도의 큰 부분은 한국 이력과 IPC',
      [('183', tp['hits'], '적중'), ('81%', 100 * tp['hits_krW_le1'] / tp['hits'], 'KR≤1'), ('41%', 100 * t3s['sab_k1']['rate'], '단일 출원인'),
       ('0.438', 0.422887490953899 + t3s['sab_k1']['dskill_A'], '제외 skill')],
      [(attr_bad, [r'45%'], f'"소규모"는 증가≤3 {small_share:.1%}에만 해당(81%에 귀속됨)')])

# ---------- B43 (323행) ----------
v6l = [l for l in v6 if '3자리 업종(판정 외)' in l][0]
cis = re.findall(r'\[([−\-+0-9.]+), ([−\-+0-9.]+)\]', v6l)
ci0_43 = [c for c in cis if float(c[0].replace('−', '-')) <= 0 <= float(c[1].replace('−', '-'))]
judge('B43', 'audit_v4_v7', '3자리 업종(판정 외)',
      [('+1.8%p', 1.8, 'NPS(v6 값)'), ('+1.2%p', 1.2, 'DART(v6 값)'), ('+1.9%p', 1.9, 'KIND(v6 값)')],
      [(len(ci0_43) > 0, [r'\[', r'0 포함', r'구별되지'], f'v6의 CI 중 0 포함 {ci0_43}')])

# ---------- 무작위 '정확' 10개 ----------
# B03 (99행)
po = quad[quad.variant == 'B_original'].iloc[0]; pc = quad[quad.variant == 'P_A1c_continuity(사후)'].iloc[0]
same = (po.SW == pc.SW) and (set(str(po.SW_list).split()) == set(str(pc.SW_list).split()))
judge('B03', 'audit_v4_v7', '¹ 2008년 값만 보면',
      [('49/50', 49 if pd.isna(kval('31312', 'T01', 2008)) else 50, '2008 동일코드')],
      [(True, [r'HS2017'], 'HS2017'), (True, [r'정합해도 표 5는 같다'] if same else [r'^$'], f'정합 시 표 5 동일={same}')])
# B12 (257행)
rat = [r['ratio_n'] for r in ntis['table'] if 2018 <= r['year'] <= 2024]
judge('B12', 'audit_v4_v7', 'NTIS Open API에서 과제',
      [('1,145,167', ntis['total_rows'], '전체 행')], [(True, [r'2018–24년 공식 통계와 같'] if min(rat) >= 0.9999 else [r'^$'], f'2018–24 비율 {min(rat)}')])
# B15 (374행)
judge('B15', 'audit_v4_v7', '③ 산업은 5자리 단위로 봐야 한다',
      [('−6.6%', f4val('26211', 'KR출원'), 'LCD KR'), ('+6.2%', f4val('26212', 'KR출원'), 'OLED KR')], [(True, [r'연평균\(로그\)'], '로그 정의')],
      extra='KR 값은 fig04 CSV 원천과 B62에서 대조')
# B16 (386행)
m_top1 = n9['top1']; comp = max(n9['top1_strong_orig'], n9['top1_pilot_orig'])
judge('B16', 'audit_v4_v7', '과학기술정보통신부·NTIS 운영기관',
      [], [(True, [r'원판 비교 모형과 같거나 낮음'] if m_top1 <= comp + 0.005 else [r'^$'], f'모형 top1 {m_top1:.3f} ≤ 원판 {comp:.3f}')])
# B22 (566행)
pb = top100['p_base']['100'] if '100' in top100['p_base'] else top100['p_base']
judge('B22', 'audit_v4_v7', '| OFF-T3-TOP100 |',
      [('50%', top100['p']['100']['pos_above'] + top100['p']['100']['tie_pos'], '모형 상위 100 적중'),
       ('23', 100 * pb['min'] / 100 if pb['min'] > 1 else pb['min'], '기준선 최소'), ('39', pb['max'], '기준선 최대')], [], extra={'p_base_100': pb})
# B25 (205행): 공식 레이블에서 직전 0·1건 셀의 최소 부상 문서 수를 직접 계산
t3p['fut'] = np.rint((t3p.kr_W + 1) * np.exp(t3p.y) - 1)
min0 = t3p[(t3p.kr_W == 0) & (t3p.yb == 1)].fut.min(); min1 = t3p[(t3p.kr_W == 1) & (t3p.yb == 1)].fut.min()
txt205 = line('기술셀은 KSIC 5자리 × IPC 메인그룹')[1]
m = re.search(r'직전 0건 셀은 (\d+)건, 1건 셀은 (\d+)건 이상 늘면', txt205)
claim_inc1 = int(m.group(2)) if m else None
judge('B25', 'audit_v4_v7', '기술셀은 KSIC 5자리 × IPC 메인그룹',
      [('2', min0 - 0, '0건 셀 최소 증가'), ('4' if claim_inc1 == 4 else str(claim_inc1), min1 - 1, '1건 셀 최소 증가(문구 "늘면" = 증가량)')],
      [(True, [r'로그 증가율'], '로그 증가율')], extra={'min_fut_krW0': min0, 'min_fut_krW1': min1, 'claimed_increase_krW1': claim_inc1})
# B33/B45 (402행)
for b in ['B33', 'B45']:
    judge(b, 'audit_v4_v7', '| 2단계 확장(2년차)', [],
          [(True, [r'4\.4-1·2 한계 해소'], '한계 해소 흡수'), (True, [r'초록·청구항 공개 배정 대비 필수성 재시험'], '공개 배정 비교'),
           (True, [r'.*'] if not any('2단계는' in l for l in LINES[:444]) else [r'^$'], '삭제 문단 잔존 없음')])
# B68 (253행)
judge('B68', 'audit_v8_v9', "산업용 로봇은 '기술 약·수출 강'",
      [('16.5%', kgrow('29280', 'T04'), '로봇 출하'), ('12.6%', kgrow('29280', 'T04', b=(2015, 2016)), '끝 창 2015–16')], [(True, [r'명목'], '명목')])
# B74 (343행)
judge('B74', 'audit_v8_v9', '- **정리.** 근거 C는', [],
      [(True, [r'단독 근거로 쓰지 않는다\.$'], '결함 정정 문장 삭제'),
       (True, [r'.*'] if any('JP만 50% 미만' in l for l in LINES) and any('C-AGREE-UB' in l for l in LINES) else [r'^$'], '3.6.4 JP·출처표 유지')])

# ---------- 비교 ----------
aud = {}
for fn in ['audit_v8_v9.csv', 'audit_v4_v7.csv']:
    for r in csv.DictReader(open(os.path.join(W, fn), encoding='utf-8-sig')):
        aud[(fn.replace('.csv', ''), r['block'])] = r['verdict']
flips = []
for o in out:
    a = aud.get((o['src'], o['block']))
    o['audit_verdict'] = a; o['flipped'] = (a != o['verdict'])
    if o['flipped']:
        flips.append((o['block'], a, o['verdict']))
res = dict(manuscript_sha1=ms_sha, n=len(out), n_flipped=len(flips), flips=flips, results=out)
json.dump(res, open(os.path.join(W, 'verify_0', 'v0_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for o in out:
    print(o['block'], o['line'], 'audit', o['audit_verdict'], '→ v0', o['verdict'], '| 숫자', [(c, v, ok) for c, v, d, ok in o['nums']], '| 조건', o['conds'])
print('sha1', ms_sha[:8], 'n', len(out), 'flips', flips)
