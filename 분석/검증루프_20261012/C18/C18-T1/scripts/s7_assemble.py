# -*- coding: utf-8 -*-
"""C18-T1 s7: 부트스트랩 취합(공식 재표본 CI 재현 + B=1000 짝지은 Δ 구간) → sens_results.csv. 사전 판정 적용."""
import sys, pathlib, json, glob
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; C = H/'cache'
BCOLS = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
s5 = json.load(open(C/'s5_out.json', encoding='utf-8'))
off = json.load(open(B/'분석/모델경쟁_20260927/T1_산업대기실/R3/final_result_r3.json', encoding='utf-8'))
q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
def sk(A, l):
    m = A[f'model|{l}']; b = A[[f'{c}|{l}' for c in BCOLS]].max(1); return m, (m - b) / (1 - b)
o = {}
Ao = pd.read_parquet(C/'s6_boot_off_0.parquet'); o['off_reps'] = len(Ao)
m0, s0 = sk(Ao, 'y0')
bo = off['bootstrap_firm_cluster']
o['official_ci_repro'] = dict(auc_here=q(m0), auc_official=bo['model']['auc_all'], skill_here=q(s0), skill_official=bo['skill_vs_strongest__model']['auc_all'])
A = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C/'s6_boot_c18_*.parquet')))]).drop_duplicates('rep').sort_values('rep')
o['c18_reps'] = len(A); o['c18_rep_range'] = [int(A.rep.min()), int(A.rep.max())]
mA0, sA0 = sk(A, 'y0'); o['c18_y0_skill_ci'] = q(sA0); o['c18_y0_auc_ci'] = q(mA0)
ciw_skill = bo['skill_vs_strongest__model']['auc_all'][2] - bo['skill_vs_strongest__model']['auc_all'][0]
ciw_auc = bo['model']['auc_all'][2] - bo['model']['auc_all'][0]
thr_s, thr_a = 0.25 * ciw_skill, 0.25 * ciw_auc
o['thresholds'] = dict(skill_ci_width=ciw_skill, auc_ci_width=ciw_auc, thr_skill=thr_s, thr_auc=thr_a, prereg_thr_skill_T1=0.0087)
rows = [dict(track='T1', scenario='P0 공식 재현', pos=s5['P0']['pos'], auc_all=s5['P0']['auc_all'], skill_all=s5['P0']['skill_all'], top1000=s5['P0']['top1000'],
             d_auc=0.0, d_skill=0.0, d_auc_ci='', d_skill_ci='', verdict='재현' if s5['P0']['pass'] else '불일치',
             note=f"공식 AUC {off['model']['auc_all']:.8f}, skill {off['skill']['auc_all']:.8f}, top1000 {off['topk']['model']['1000']['hits']}; 공식 재표본 CI 재현 skill {[round(x, 5) for x in q(s0)]} 대 공식 {[round(x, 5) for x in bo['skill_vs_strongest__model']['auc_all']]}")]
for nm, lab in [('S1_R_self_G', 'S1'), ('S2_R_any_G', 'S2')]:
    r = s5['scenarios'][nm]; m1, s1 = sk(A, lab); da = m1 - mA0; ds = s1 - sA0
    big = max(abs(r['d_skill_all']), abs(r['d_auc_all']))
    v = '영향 없음' if (abs(r['d_skill_all']) < thr_s and abs(r['d_auc_all']) < thr_a) else '경고'
    # 원고 보고 정밀도(skill 3자리, AUC 3자리, top1000 정수 31.3%)
    rep_change = (round(r['skill_all'], 3) != round(s5['P0']['skill_all'], 3)) or (round(r['auc_all'], 3) != round(s5['P0']['auc_all'], 3))
    if rep_change: v += ' / 보고 정밀도 변화(원고 숫자 아님: 공식 성적 불변, 민감도)'
    rows.append(dict(track='T1', scenario=f"{lab} 공백 달 누락 등록 보충({'원출원인 R_self' if lab == 'S1' else '모든 권리자 R_any(상한)'})", pos=r['pos'], auc_all=r['auc_all'], skill_all=r['skill_all'], top1000=r['top1000'],
                     d_auc=r['d_auc_all'], d_skill=r['d_skill_all'], d_auc_ci=str([round(x, 6) for x in q(da)]), d_skill_ci=str([round(x, 6) for x in q(ds)]), verdict=v,
                     note=f"새 양성 {r['new_pos']}(신규진입자 {r['new_pos_newc']}), 양성 +{r['pos_change_rate']*100:.2f}%, top1000 {s5['P0']['top1000']}→{r['top1000']}, 최강 기준선 AUC Δ{r['d_strongest']:+.5f}; 기준 |Δskill|<{thr_s:.4f}, |ΔAUC|<{thr_a:.5f}"))
    o[f'{lab}_boot'] = dict(d_auc=q(da), d_skill=q(ds), share_d_skill_lt0=float((ds < 0).mean()))
r = s5['scenarios']['S1b_R_self_all']
rows.append(dict(track='T1', scenario='S1b 모든 달 공란 보충(C03-T1 재현 점검)', pos=r['pos'], auc_all=r['auc_all'], skill_all=r['skill_all'], top1000=r['top1000'], d_auc=r['d_auc_all'], d_skill=r['d_skill_all'],
                 verdict='C03-T1 재현', note='C03-T1 주판 R_self: 양성 7,859, Δskill −0.0021523, ΔAUC +0.0000832 — 일치'))
rows.append(dict(track='T1', scenario='kr_registered 보충', d_auc=0.0, d_skill=0.0, verdict='추가 0', note='kr_registered 특허 등록번호 집합 = registry_full(대칭차 0) → 보충할 번호 없음'))
for t, why in [('T3', '공개연도=least(pub,rgst); 누락 문서 공개일 결측 0'), ('T4', '모집단 등록 <2022, 누락 최소 등록일 2023-11-15'), ('N4', 'G 노출 0(피처 창 끝 2023-11·12 경계 이하 364건, C_m ≥0.885)'),
               ('N5', '피처 rgd ≤2022-12-31'), ('N6', '출원·공개 기준, 공개일 결측 0'), ('N7', '모집단 등록 ≤2022-12-31'), ('N8', '모집단 등록 ≤2022-12-31, 결과는 모집단 특허 사건'), ('N9', 'NTIS 성과-출원 연결(출원연도), 등록번호 무관')]:
    rows.append(dict(track=t, scenario='구조적 노출 0', d_auc=0.0, d_skill=0.0, verdict='영향 없음(구조)', note=why))
pd.DataFrame(rows).to_csv(H/'sens_results.csv', index=False, encoding='utf-8-sig', float_format='%.8g')
json.dump(o, open(C/'s7_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(o, ensure_ascii=False, indent=0))
