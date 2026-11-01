# -*- coding: utf-8 -*-
"""s09: final_metrics.json + boot_summary.json + dev_metrics.csv + s5_out.json → sens_results.csv (사전 문턱으로 판정)"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
import numpy as np, pandas as pd
W = 0.034829; T1_, T2_ = 0.25 * W, 0.50 * W
def judge(d, auc=None):
    if d is None or (isinstance(d, float) and np.isnan(d)): return '미실행'
    if auc is not None and auc <= 0.81078: return '오류(AUC≤기준선)'
    a = abs(d); return '영향 없음' if a < T1_ else ('단서 필요' if a < T2_ else '오류')
F = json.load(open(HERE/'cache/final_metrics.json', encoding='utf-8'))
Bt = json.load(open(HERE/'cache/boot_summary.json', encoding='utf-8')) if (HERE/'cache/boot_summary.json').exists() else {}
D = pd.read_csv(HERE/'cache/dev_metrics.csv') if (HERE/'cache/dev_metrics.csv').exists() else pd.DataFrame(columns=['variant', 'origin', 'd_skill', 'skill'])
S5 = json.load(open(HERE/'cache/s5_out.json', encoding='utf-8'))
def dev(v):
    x = D[D.variant == v]; return x
def ci(k):
    b = Bt.get(k); return b['d_skill'] if b else None
rows = []
LBL = {'P0': 'P0 재현(무변형 재계산 주입)', 'S1': 'S1 ot_* 7열 제거(0)', 'S1v': 'S1v ot_* 7열 NaN(반박용)', 'S2': 'S2 원점 후 등록 + 이전일 미상(권리자 표기 불일치) 행 삭제',
       'S4': 'S4 역보정(셀 누락률만큼 행 복제)', 'AN_TI': 'AN_TI [보조] ti_* 4열을 생존(B) 행으로', 'AN_A': 'AN_A ot_ 슬롯 ← 범위안 원출원(A) 같은 정의', 'AN_B': 'AN_B ot_ 슬롯 ← 범위안 생존(B) 같은 정의'}
for v in ['P0', 'S1', 'S1v', 'S2', 'S3_1', 'S3_2', 'S3_3', 'S3_4', 'S3_5', 'S4', 'AN_A', 'AN_B', 'AN_TI']:
    f = F.get(v); x = dev(v)
    dm = float(x.d_skill.mean()) if len(x) else np.nan
    r = dict(variant=v, label=LBL.get(v, f'S3 누락 확대 섭동 시드 {v[-1]}'), final_skill=f['skill'] if f else np.nan, final_d_skill=f['d_skill'] if f else np.nan,
             final_auc=f['auc_all'] if f else np.nan, final_d_skill_new=f['d_skill_new'] if f else np.nan, final_top1000=f['top1000'] if f else np.nan,
             boot_d_skill_ci=json.dumps([round(q, 5) for q in ci(v)]) if ci(v) else '', dev_n_origins=int(len(x)), dev_mean_d_skill=dm,
             dev_d_skill_by_origin=json.dumps({int(o): round(float(d), 5) for o, d in zip(x.origin, x.d_skill)}),
             final_minus_dev=(f['d_skill'] - dm) if (f and len(x)) else np.nan)
    r['judge_final'] = judge(r['final_d_skill'], r['final_auc']) if f else '미실행'
    r['inflation'] = ('최종 부풀림 의심' if r['final_minus_dev'] < -T1_ else '부풀림 근거 없음') if np.isfinite(r['final_minus_dev']) else ''
    rows.append(r)
R = pd.DataFrame(rows)
# S3 평균
s3 = R[R.variant.str.startswith('S3_') & R.final_d_skill.notna()]
if len(s3):
    m = float(s3.final_d_skill.mean()); xs = D[D.variant.str.startswith('S3_')]
    dm = float(xs.groupby('origin').d_skill.mean().mean()) if len(xs) else np.nan
    rows.append(dict(variant='S3_mean', label=f'S3 시드 {len(s3)}개 평균(판정)', final_skill=float(s3.final_skill.mean()), final_d_skill=m,
                     final_auc=float(s3.final_auc.mean()), final_d_skill_new=float(s3.final_d_skill_new.mean()), final_top1000=float(s3.final_top1000.mean()),
                     boot_d_skill_ci=json.dumps([round(q, 5) for q in ci('S3_mean')]) if ci('S3_mean') else '', dev_n_origins=int(xs.origin.nunique()) if len(xs) else 0,
                     dev_mean_d_skill=dm, dev_d_skill_by_origin=json.dumps({int(o): round(float(d), 5) for o, d in xs.groupby('origin').d_skill.mean().items()}) if len(xs) else '',
                     final_minus_dev=m - dm if np.isfinite(dm) else np.nan, judge_final=judge(m),
                     inflation=('최종 부풀림 의심' if (m - dm) < -T1_ else '부풀림 근거 없음') if np.isfinite(dm) else '',
                     note=f'시드별 최종 Δ 최소 {s3.final_d_skill.min():.5f}, 최대 {s3.final_d_skill.max():.5f}'))
# 아날로그 차
if 'AN_A' in F and 'AN_B' in F:
    a, b = D[D.variant == 'AN_A'].set_index('origin'), D[D.variant == 'AN_B'].set_index('origin')
    oo = sorted(set(a.index) & set(b.index)); dd = {int(o): float(b.skill[o] - a.skill[o]) for o in oo}
    m = F['AN_B']['skill'] - F['AN_A']['skill']; dm = float(np.mean(list(dd.values()))) if dd else np.nan
    rows.append(dict(variant='AN_B−AN_A', label='아날로그 생존 편향 효과(같은 슬롯 B판 − A판, 판정)', final_skill=np.nan, final_d_skill=m, final_auc=np.nan,
                     final_d_skill_new=F['AN_B']['skill_new'] - F['AN_A']['skill_new'], final_top1000=F['AN_B']['top1000'] - F['AN_A']['top1000'],
                     boot_d_skill_ci=json.dumps([round(q, 5) for q in Bt['AN_BminusA']['d_skill']]) if 'AN_BminusA' in Bt else '', dev_n_origins=len(dd), dev_mean_d_skill=dm,
                     dev_d_skill_by_origin=json.dumps({k: round(v, 5) for k, v in dd.items()}), final_minus_dev=m - dm if np.isfinite(dm) else np.nan, judge_final=judge(m),
                     inflation=('최종 부풀림 의심' if (m - dm) < -T1_ else '부풀림 근거 없음') if np.isfinite(dm) else ''))
    sl = F['AN_A']['skill'] - F['S1']['skill'] if 'S1' in F else np.nan
    rows.append(dict(variant='AN_A−S1', label='참고: 아날로그 A판 슬롯 기여(AN_A − ot_* 제거)', final_d_skill=sl, judge_final='참고'))
rows.append(dict(variant='S5', label='S5 [보조] 원점 연말까지 등록분으로 ≥3건 법인만 평가(재학습 없음)', final_skill=S5['strict_subset']['skill'], final_d_skill=S5['d_skill_strict_minus_full'],
                 final_auc=S5['strict_subset']['auc'], judge_final=judge(S5['d_skill_strict_minus_full']),
                 note=f"제외 법인 {S5['eval_firms_need_post_origin']}/{S5['eval_firms_with_pairs']}, 제외 쌍 {S5['excluded_pairs']}·양성 {S5['excluded_pos']}; 부분집합 기준선 {S5['strict_subset']['base']:.4f}"))
R = pd.DataFrame(rows)
# 상한 추정(사전 규칙)
cand = {}
if 'S3_mean' in set(R.variant): cand['S3_mean'] = abs(float(R[R.variant == 'S3_mean'].final_d_skill.iloc[0]))
if 'S4' in F: cand['S4'] = abs(F['S4']['d_skill'])
if 'AN_B−AN_A' in set(R.variant): cand['AN_B−AN_A'] = abs(float(R[R.variant == 'AN_B−AN_A'].final_d_skill.iloc[0]))
if cand:
    ub = max(cand.values()); k = max(cand, key=cand.get)
    prop = None
    if 'AN_B−AN_A' in cand and 'S1' in F and 'AN_A' in F:
        prop = cand['AN_B−AN_A'] * abs(F['S1']['d_skill']) / max(abs(F['AN_A']['skill'] - F['S1']['skill']), 0.001)
    rows.append(dict(variant='UPPER', label=f'범위밖 생존 편향 영향 추정치 = max(|S3 평균|, |S4|, |AN_B−AN_A|) (최대: {k})', final_d_skill=ub, judge_final=judge(ub),
                     note=f'비례 환산 {prop:.5f}' if prop is not None else ''))
R = pd.DataFrame(rows)
R.to_csv(HERE/'sens_results.csv', index=False, encoding='utf-8-sig')
print(R[['variant', 'final_d_skill', 'judge_final', 'dev_n_origins', 'dev_mean_d_skill', 'inflation']].round(5).to_string(index=False))
