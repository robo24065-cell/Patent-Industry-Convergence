# -*- coding: utf-8 -*-
"""P0 판정: 재적합 모형 예측·공식 점수 채점·부트스트랩 재표본 AUC가 공식과 비트 일치하는지 → p0_result.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16common import *
fr_ = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
off = json.load(open(CC/'var'/'OFF.json', encoding='utf-8')); A = np.load(CC/'var'/'OFF_boot.npy')
OB = np.concatenate([np.load(FC/f'r3boot_R200_{ci}.npy') for ci in range(20)])
p0a = json.load(open(CC/'p0'/'p0a.json', encoding='utf-8')); p0c = json.load(open(CC/'p0'/'p0c.json', encoding='utf-8'))
px = {sd: float(np.abs(np.load(CC/'p0'/f'px{sd}.npy') - np.load(RD/f'o2019_px{sd}.npy')).max()) for sd in (0, 1)}
pl = float(np.abs(np.load(CC/'p0'/'pl.npy') - np.load(RD/'o2019_pl.npy')).max())
cmpF = json.load(open(CC/'build'/'ctrlF_compare.json', encoding='utf-8'))
res = dict(
    retrain=dict(px_maxdiff=px, pl_maxdiff=pl, f_any_te_maxdiff=p0a['f_any_te_maxdiff_vs_C15A'], base6_maxdiff=p0c['base6_maxdiff_vs_official'], sbb_maxdiff_vs_VR5=p0c['sbb_keep_maxdiff_vs_VR5']),
    score=dict(skill=off['skill'], skill_official=fr_['skill']['auc_all'], skill_diff=off['skill'] - fr_['skill']['auc_all'],
               auc=off['auc'], auc_official=fr_['model']['auc_all'], auc_diff=off['auc'] - fr_['model']['auc_all'],
               top1000=off['top1000'], top1000_official=fr_['topk']['model']['1000']['hits'], top100=off['top100'],
               M0=off['auc_M0'], M0_official=fr_['baseline_strongest']['auc_all'], skill_new=off['skill_new'], skill_new_official=fr_['skill']['auc_new'],
               ci=off['ci'], ci_official=fr_['bootstrap_firm_cluster']['skill_vs_strongest__model']['auc_all']),
    boot=dict(model_auc_maxdiff_vs_r3boot=float(np.abs(A[:, 0] - OB[:, 0, 0]).max()), base6_auc_maxdiff_vs_r3boot=float(np.abs(A[:, 1:7] - OB[:, 4:, 0]).max())),
    full_rebuild_ctrl=dict(passed=cmpF['pass'], feat_maxdiff=cmpF['feat_maxdiff'], score_maxdiff=cmpF['score_maxdiff'], te_cols_maxdiff=cmpF['te_cols_maxdiff']))
res['P0_pass'] = bool(max(px.values()) == 0 and pl == 0 and res['score']['skill_diff'] == 0 and res['score']['auc_diff'] == 0 and res['score']['top1000'] == res['score']['top1000_official']
                      and res['boot']['model_auc_maxdiff_vs_r3boot'] < 1e-9 and cmpF['pass'])
jdump(res, HERE/'p0_result.json'); print(json.dumps(res, ensure_ascii=False, indent=1))
# 공식 채점 함수 경로(트랙 r2.mets·skill·final_run_r3.blend·topk, sklearn roc_auc_score)로 재적합 예측을 채점 — 비트 비교
import c16t1
import final_run_r3 as F
from r2 import mets, skill
o = pd.read_parquet(CC/'p0'/'s1_o.parquet'); o['px'] = np.mean([np.load(CC/'p0'/'px0.npy'), np.load(CC/'p0'/'px1.npy')], 0); o['pl'] = np.load(CC/'p0'/'pl.npy')
L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True); assert (L.firm.values == o.firm.values).all()
keep = ~L.prior_any.values; oo = o[keep].reset_index(drop=True); y = L.y_u.values[keep]; nc = o.newc.values[keep].astype(bool)
s = F.blend(oo); bs = pd.read_parquet(CC/'p0'/'base6.parquet').loc[keep].reset_index(drop=True)
mm = mets(y, s, nc); bm = {k: mets(y, bs[k].values, nc) for k in bs.columns}; st = max(v['auc_all'] for v in bm.values())
tk = F.topk(y, s)
res['score_trackpath'] = dict(skill=skill(mm['auc_all'], st), skill_diff=skill(mm['auc_all'], st) - fr_['skill']['auc_all'], auc_diff=mm['auc_all'] - fr_['model']['auc_all'],
                              top1000=tk['1000']['hits'], top100=tk['100']['hits'])
res['P0_pass'] = bool(max(px.values()) == 0 and pl == 0 and res['score_trackpath']['skill_diff'] == 0 and res['score_trackpath']['auc_diff'] == 0 and tk['1000']['hits'] == 313
                      and res['boot']['model_auc_maxdiff_vs_r3boot'] < 1e-9 and cmpF['pass'])
res['note'] = 'score(가중 AUC 구현)는 합산 순서 차로 1e-16 수준 차; score_trackpath(트랙 함수·sklearn)가 비트 비교 기준'
jdump(res, HERE/'p0_result.json'); print(json.dumps({k: res[k] for k in ('score_trackpath', 'P0_pass')}, ensure_ascii=False, indent=1))
