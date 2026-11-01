# -*- coding: utf-8 -*-
"""s09: 민감도 결과 모음 → sens_results.json, sens_results.csv (판정은 PREREG 기준 그대로 계산)"""
import json, pathlib
import numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parent.parent
F = json.load(open(H/'cache/final_metrics.json', encoding='utf-8')); Bt = json.load(open(H/'cache/boot_summary.json', encoding='utf-8'))
D = pd.read_csv(H/'cache/dev_metrics.csv')
CIW = 0.488452775 - 0.453623904; T25, T50 = 0.25 * CIW, 0.50 * CIW; BASE = F['S0']['base_auc_all']
def verdict(ds, auc):
    if abs(ds) >= T50 or auc <= BASE: return '수정 필요'
    if abs(ds) >= T25: return '단서 필요'
    return '영향 없음'
desc = {'S1': '비자사(보유자≠원출원인) 디자인·실용신안 행 제거', 'S2': '비자사 행은 추정 이전일 이후 원점에서만(추정 불가 제거)',
        'S3': '디자인·실용신안 10피처 0 대체(절제·재학습)', 'S4': '[보조] 물가 5피처 NaN', 'S5': '[보조] 산업연관표 5피처 NaN'}
rows = []; out = dict(official=F['official_json'], S0_repro=F['S0_repro'], S0_boot_repro_max_abs=Bt['S0_vs_official_boot_max_abs'], ci_width=CIW, thr25=T25, thr50=T50, variants={})
for v in ('S1', 'S2', 'S3', 'S4', 'S5'):
    f = F[v]; b = Bt[v]; dv = D[D.variant == v]
    dmean = float(dv.d_skill.mean()) if len(dv) else None
    r = dict(variant=v, desc=desc[v], auc=f['auc_all'], skill=f['skill'], d_skill=f['d_skill'], d_skill_ci_lo=b['d_skill'][0], d_skill_ci_hi=b['d_skill'][2],
             d_auc=f['d_auc'], auc_new=f['auc_new'], d_skill_new=f['d_skill_new'], top1000=f['top1000'], d_top1000=f['d_top1000'], capture_top1pct=f['capture_top1pct'],
             verdict=verdict(f['d_skill'], f['auc_all']), dev_origins=int(len(dv)), dev_mean_d_skill=dmean,
             dev_n_drop=int((dv.d_skill < 0).sum()) if len(dv) else None, final_minus_dev=(f['d_skill'] - dmean) if dmean is not None else None)
    if dmean is not None: r['inflation_verdict'] = '최종 부풀림 의심' if r['final_minus_dev'] < -T25 else '부풀림 근거 없음'
    rows.append(r); out['variants'][v] = r
pd.DataFrame(rows).to_csv(H/'sens_results.csv', index=False, encoding='utf-8-sig')
out['dev_per_origin'] = D.to_dict('records')
json.dump(out, open(H/'sens_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(pd.DataFrame(rows)[['variant', 'skill', 'd_skill', 'd_skill_ci_lo', 'd_skill_ci_hi', 'verdict', 'dev_mean_d_skill', 'final_minus_dev', 'top1000']].round(5).to_string())
