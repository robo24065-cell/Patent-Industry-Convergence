# -*- coding: utf-8 -*-
"""C15-T2 조립: cache/t1, cache/t3 묶음 JSON → seed_bundles.csv, variance_decomp.csv, verdicts.json, p0_result.json (PREREG §3·§4 판정)."""
import os, json, pathlib
import numpy as np, pandas as pd
W = pathlib.Path(__file__).resolve().parent.parent; C1 = W/'cache'/'t1'; C3 = W/'cache'/'t3'
K_ALL = list(range(10))
rows = []
# ---- T1 ----
for b in K_ALL:
    f = C1/f'b{b}_final.json'
    if not f.exists(): continue
    d = json.load(open(f, encoding='utf-8'))
    dv = C1/f'b{b}_dev.json'; dev = json.load(open(dv, encoding='utf-8'))['mean_skill'] if dv.exists() else np.nan
    rows.append(dict(track='T1', bundle=b, official_bundle=b == 0, seeds=json.dumps(d['seeds']), n=d['n'], pos=d['pos'], auc_champ=d['auc_champ'], auc_M0=d['auc_M0'],
                     M0_name=d['M0_arg'], skill=d['skill'], ci_lo=d['ci'][0], ci_hi=d['ci'][2], ci_lo_gt0=d['ci'][0] > 0,
                     strong_name='SBB', auc_strong=d['auc_strong'], skill_vs_strong=d['skill_vs_strong'], strong_ci_lo=d['strong_ci'][0], strong_ci_hi=d['strong_ci'][2],
                     strong_class='강화 우위' if d['strong_ci'][0] > 0 else '조건부', top100=d['top100'], topk2_name='top1000', topk2=d['top1000'],
                     capture_top1pct=d['capture_top1pct'], skill_vs_official_fixedM0=d['skill_vs_official_fixed'], ci_lo_official_fixedM0=d['ci_vs_official_fixed'][0],
                     ci_hi_official_fixedM0=d['ci_vs_official_fixed'][2], dev_mean_skill_fixedM0=dev,
                     dev2final=(d['skill_vs_official_fixed'] - dev) if np.isfinite(dev) else np.nan, auc_xgb_y=d['base_aucs']['xgb_y'], auc_xgb_y_u=d['base_aucs']['xgb_y_u']))
# ---- T3 ----
for b in K_ALL:
    f = C3/f'b{b}_final.json'
    if not f.exists(): continue
    d = json.load(open(f, encoding='utf-8')); o = d['by_origin']['2021']
    dv = C3/f'b{b}_dev.json'; dev = json.load(open(dv, encoding='utf-8'))['mean_skill'] if dv.exists() else np.nan
    rows.append(dict(track='T3', bundle=b, official_bundle=b == 0, seeds=json.dumps(dict(xgbc=d['seed'], nb=d['seed'], SB3=d['SB3']['seed'])), n=o['model']['n'], pos=o['model']['n_evt'],
                     auc_champ=o['model']['auc'], auc_M0=o['baseline']['auc'], M0_name='kr_mom 로지스틱', skill=o['skill'], ci_lo=d['ci95_M0']['skill'][0], ci_hi=d['ci95_M0']['skill'][1],
                     ci_lo_gt0=d['ci95_M0']['skill'][0] > 0, strong_name='SB3', auc_strong=d['SB3']['auc_sb'], skill_vs_strong=d['SB3']['skill_vs_sb'],
                     strong_ci_lo=d['SB3']['ci95'][0], strong_ci_hi=d['SB3']['ci95'][1], strong_class='강화 우위' if d['SB3']['ci95'][0] > 0 else '조건부',
                     top100=o['top100_hits'], topk2_name='top10pct', topk2=o['top10pct_hits'], p_at10=o['model']['p_at10'], dev_mean_skill=dev,
                     dev2final=(o['skill'] - dev) if np.isfinite(dev) else np.nan))
S = pd.DataFrame(rows); S.to_csv(W/'seed_bundles.csv', index=False, encoding='utf-8-sig')
OFFC = {'T1': '강화 우위', 'T3': '강화 우위'}
V, verd = [], {}
for trk, g in S.groupby('track'):
    g = g.sort_values('bundle'); off = g[g.bundle == 0].iloc[0]; K = len(g)
    se = (off.ci_hi - off.ci_lo) / 3.92; sse = (off.strong_ci_hi - off.strong_ci_lo) / 3.92
    share = float(g.ci_lo_gt0.mean()); nflip = int((g.strong_class != OFFC[trk]).sum())
    rank = int((g.skill > off.skill).sum() + 1); rank_lo = int((g.ci_lo > off.ci_lo).sum() + 1)
    others = g[g.bundle != 0]
    fav = bool(rank == 1 and (off.skill - others.skill.mean()) > 0.5 * se)
    vm = '판정 보류' if K < 6 else ('견고' if share == 1.0 else '시드 의존(N5형)')
    grade = '견고' if share >= 0.9 else ('경계' if share >= 0.5 else '취약')
    vs = '분류 유지' if nflip == 0 else f'분류 불안정({nflip}/{K})'
    d2f = g.dev2final.dropna()
    r = dict(track=trk, K=K, skill_official=off.skill, skill_mean=g.skill.mean(), skill_sd=g.skill.std(ddof=1), skill_min=g.skill.min(), skill_max=g.skill.max(),
             ci_lo_official=off.ci_lo, ci_lo_min=g.ci_lo.min(), ci_lo_max=g.ci_lo.max(), ci_lo_range=g.ci_lo.max() - g.ci_lo.min(), share_ci_lo_gt0=share,
             boot_se_official=se, ratio_seed_sd_to_boot_se=g.skill.std(ddof=1) / se,
             strong_name=off.strong_name, strong_official=off.skill_vs_strong, strong_mean=g.skill_vs_strong.mean(), strong_sd=g.skill_vs_strong.std(ddof=1),
             strong_min=g.skill_vs_strong.min(), strong_max=g.skill_vs_strong.max(), strong_ci_lo_official=off.strong_ci_lo, strong_ci_lo_min=g.strong_ci_lo.min(),
             strong_ci_lo_max=g.strong_ci_lo.max(), share_strong_ci_lo_gt0=float((g.strong_ci_lo > 0).mean()), strong_boot_se_official=sse,
             ratio_strong_sd_to_boot_se=g.skill_vs_strong.std(ddof=1) / sse, auc_champ_min=g.auc_champ.min(), auc_champ_max=g.auc_champ.max(),
             auc_M0_min=g.auc_M0.min(), auc_M0_max=g.auc_M0.max(), auc_strong_min=g.auc_strong.min(), auc_strong_max=g.auc_strong.max(),
             top100_official=int(off.top100), top100_min=int(g.top100.min()), top100_max=int(g.top100.max()),
             topk2_name=off.topk2_name, topk2_official=int(off.topk2), topk2_min=int(g.topk2.min()), topk2_max=int(g.topk2.max()),
             K_dev=int(len(d2f)), dev2final_official=off.dev2final, dev2final_min=d2f.min() if len(d2f) else np.nan, dev2final_max=d2f.max() if len(d2f) else np.nan,
             dev2final_mean=d2f.mean() if len(d2f) else np.nan, official_rank_skill=rank, official_rank_ci_lo=rank_lo,
             official_skill_percentile=float((g.skill < off.skill).mean()), favorable_selection_flag=fav, verdict_M0=vm, c13_grade=grade, verdict_strong=vs, n_strong_flip=nflip)
    if trk == 'T1':
        r.update(share_ci_lo_gt0_official_fixedM0=float((g.ci_lo_official_fixedM0 > 0).mean()), ci_lo_official_fixedM0_min=g.ci_lo_official_fixedM0.min(),
                 ci_lo_official_fixedM0_max=g.ci_lo_official_fixedM0.max(), skill_official_fixedM0_min=g.skill_vs_official_fixedM0.min(),
                 skill_official_fixedM0_max=g.skill_vs_official_fixedM0.max(), skill_official_fixedM0_sd=g.skill_vs_official_fixedM0.std(ddof=1))
    V.append(r); verd[trk] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in r.items() if k in ('K', 'verdict_M0', 'c13_grade', 'verdict_strong', 'n_strong_flip', 'share_ci_lo_gt0',
                                                                                         'share_strong_ci_lo_gt0', 'official_rank_skill', 'favorable_selection_flag', 'share_ci_lo_gt0_official_fixedM0', 'K_dev')}
pd.DataFrame(V).to_csv(W/'variance_decomp.csv', index=False, encoding='utf-8-sig')
allok = all(v['verdict_M0'] == '견고' and v['verdict_strong'] == '분류 유지' for v in verd.values()) and set(verd) == {'T1', 'T3'}
verd['manuscript'] = '확인(17·130·372행 수정 불요) + 534행 확장안' if allok else '단서 필요(17·372·534행 수정안)'
json.dump(verd, open(W/'verdicts.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
# ---- P0 ----
p0 = {}
d = json.load(open(C1/'b0_final.json', encoding='utf-8')); off = json.load(open(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실/R3/final_result_r3.json', encoding='utf-8'))
cis = json.load(open(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실/VERIFY_R5/ci_sb_t1.json', encoding='utf-8'))
sens = json.load(open(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실/VERIFY_R5/sens_t1.json', encoding='utf-8'))
sbb_off = [v for k, v in sens['strong_baselines'].items() if k.startswith('SBA + ')][0]
oc = off['bootstrap_firm_cluster']['skill_vs_strongest__model']['auc_all']
base_chk = json.load(open(C1/'base_b0_check.json', encoding='utf-8'))
p0['T1'] = dict(d_auc=d['auc_champ'] - off['model']['auc_all'], d_skill=d['skill'] - off['skill']['auc_all'], d_ci_lo=d['ci'][0] - oc[0], d_ci_hi=d['ci'][2] - oc[2],
                d_top100=d['top100'] - off['topk']['model']['100']['hits'], d_top1000=d['top1000'] - off['topk']['model']['1000']['hits'],
                d_auc_M0=d['auc_M0'] - off['baseline_strongest']['auc_all'], d_sbb_skill=d['skill_vs_strong'] - sbb_off['skill_model_vs_it'],
                d_sbb_ci_lo=d['strong_ci'][0] - cis['skill_vs_SBB'][0], d_sbb_ci_hi=d['strong_ci'][2] - cis['skill_vs_SBB'][1], **d['P0'], **base_chk)
pdv = C1/'p0_dev_o2008.json'
if pdv.exists(): p0['T1_dev_o2008'] = json.load(open(pdv, encoding='utf-8'))
dv0 = C1/'b0_dev.json'
if dv0.exists(): p0['T1_dev_b0'] = json.load(open(dv0, encoding='utf-8')).get('d_vs_official')
d3 = json.load(open(C3/'b0_final.json', encoding='utf-8')); o3 = json.load(open(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T3_기술셀부상예측/R3/final_result_r3.json', encoding='utf-8'))
s3 = json.load(open(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T3_기술셀부상예측/VERIFY_R5/ci_sb_t3.json', encoding='utf-8'))['R3pub']
s3k = [k for k in s3 if k.startswith('SB3')][0]
o = d3['by_origin']['2021']
p0['T3'] = dict(d_skill=o['skill'] - o3['summary']['mean_skill'], d_auc=o['model']['auc'] - o3['summary']['auc'], d_auc_M0=o['baseline']['auc'] - o3['summary']['auc_baseline'],
                d_ci_lo=d3['ci95_M0']['skill'][0] - o3['ci95']['skill'][0], d_ci_hi=d3['ci95_M0']['skill'][1] - o3['ci95']['skill'][1],
                d_SB3_skill=d3['SB3']['skill_vs_sb'] - 0.1822166752123721, d_SB3_ci_lo=d3['SB3']['ci95'][0] - s3[s3k][0], d_SB3_ci_hi=d3['SB3']['ci95'][1] - s3[s3k][1],
                top100=o['top100_hits'], top10pct=o['top10pct_hits'])
d3d = C3/'b0_dev.json'
if d3d.exists():
    sp = json.load(open(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T3_기술셀부상예측/R3/final_spec_r3.json', encoding='utf-8'))
    p0['T3_dev'] = json.load(open(d3d, encoding='utf-8'))['mean_skill'] - sp['dev']['mean_skill']
sc = C1/'p0_sbb_check.json'
if sc.exists():
    z = json.load(open(sc, encoding='utf-8')); p0['T1_sbb_vr5_formula'] = dict(d_ci=z['d_vs_ci_sb_t1'], n_order_diff=z['n_order_diff'], max_abs_score_diff=z['max_abs_score_diff'],
        note='VERIFY_R5는 rankdata 식 모형 점수를 써서 공식 blend와 근접 동점 순서가 달라 SBB skill·CI가 2e-8 다름. VR5 식 점수로는 CI 차 0(PREREG 변경 기록)')
num = [abs(v) for t in ('T1', 'T3') for k, v in p0[t].items() if isinstance(v, (int, float)) and k.startswith(('d_', 'px', 'pl', 'boot', 'xgb', 'sbb')) and not k.startswith('d_sbb')]
if sc.exists(): num += [abs(x) for x in p0['T1_sbb_vr5_formula']['d_ci']]
p0['max_abs_diff'] = max(num); p0['pass'] = p0['max_abs_diff'] <= 1e-12
json.dump(p0, open(W/'p0_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(verd, ensure_ascii=False, default=float)); print('P0', p0['pass'], p0['max_abs_diff'])
print(S[['track', 'bundle', 'skill', 'ci_lo', 'skill_vs_strong', 'strong_ci_lo', 'top100', 'topk2', 'dev2final']].round(4).to_string(index=False))
