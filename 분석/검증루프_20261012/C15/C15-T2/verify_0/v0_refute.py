# -*- coding: utf-8 -*-
"""verify_0 반박 시도(PREREG §5): '견고는 묶음 안에서 모형·기준선 시드를 같이 바꾼 덕'이라는 가설.
가장 약한 모형 묶음(최종 AUC 최소)을 가장 강한 기준선 묶음(기준선 AUC 최대)과 교차 짝지어 CI 하한을 다시 잰다.
모든 묶음이 같은 부트스트랩 난수(재표본)를 쓰므로 재표본별 AUC 배열을 묶음 사이에서 그대로 짝지을 수 있다.
 T1 M0: cache/t1/boot/b{b}_m0_*.npz(모형·xgb_y·xgb_y_u) + 공식 r3boot(로지스틱 4개). 추가로 재표본마다 10묶음 기준선 중 최댓값(극단 짝)도 잰다.
 T1 SBB: b{b}_sbb_*.npz(모형·SBB). T3 SB3: preds_b*_final.parquet 점수로 ci_sb_t3 규칙(default_rng(11)) 재계산.
 → verify_0/refute_result.json"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = pathlib.Path(__file__).resolve().parent.parent; C1 = W/'cache'/'t1'; C3 = W/'cache'/'t3'
FC = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실/R3/cache_final/o2019")
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
S = pd.read_csv(W/'seed_bundles.csv'); out = {}
OFF = np.concatenate([np.load(FC/f'r3boot_R200_{ci}.npy') for ci in range(20)])
t1 = S[S.track == 'T1']; Bs = sorted(t1.bundle)
M = {b: np.concatenate([np.load(C1/'boot'/f'b{b}_m0_{ci}.npz')['auc'] for ci in range(20)]) for b in Bs}
SB = {b: np.concatenate([np.load(C1/'boot'/f'b{b}_sbb_{j}.npz')['auc'] for j in range(10)]) for b in Bs}
bm_weak = int(t1.loc[t1.auc_champ.idxmin(), 'bundle']); bb_strong = int(t1.loc[t1.auc_M0.idxmax(), 'bundle']); bs_strong = int(t1.loc[t1.auc_strong.idxmax(), 'bundle'])
def m0_skill(bm, bb):
    A = M[bm][:, 0]; B = np.column_stack([OFF[:, 4, 0], OFF[:, 5, 0], M[bb][:, 1], OFF[:, 7, 0], OFF[:, 8, 0], M[bb][:, 2]]).max(1)
    return (A - B) / (1 - B)
allB = np.column_stack([OFF[:, 4, 0], OFF[:, 5, 0], OFF[:, 7, 0], OFF[:, 8, 0]] + [M[b][:, 1] for b in Bs] + [M[b][:, 2] for b in Bs]).max(1)
A = M[bm_weak][:, 0]; ext = (A - allB) / (1 - allB)
allS = np.column_stack([SB[b][:, 1] for b in Bs]).max(1)
# 같은 재표본에서 SBB 열의 모형 AUC는 묶음마다 다르다 → 약한 모형 묶음의 모형 AUC 사용
As = SB[bm_weak][:, 0]
out['T1'] = dict(weak_model_bundle=bm_weak, strong_M0_bundle=bb_strong, strong_SBB_bundle=bs_strong,
                 cross_M0_ci=q(m0_skill(bm_weak, bb_strong)), cross_M0_point_skill=None,
                 extreme_M0_ci_max_over_all_baseline_seeds=q(ext),
                 cross_SBB_ci=q((As - SB[bs_strong][:, 1]) / (1 - SB[bs_strong][:, 1])),
                 extreme_SBB_ci_max_over_all_SBB_seeds=q((As - allS) / (1 - allS)))
aw = float(t1.auc_champ.min()); ab = float(t1.auc_M0.max()); asb = float(t1.auc_strong.max())
out['T1']['cross_M0_point_skill'] = (aw - ab) / (1 - ab); out['T1']['cross_SBB_point_skill'] = (aw - asb) / (1 - asb)
# T3
t3 = S[S.track == 'T3']; bw = int(t3.loc[t3.auc_champ.idxmin(), 'bundle']); bs = int(t3.loc[t3.auc_strong.idxmax(), 'bundle'])
Pw = pd.read_parquet(C3/f'preds_b{bw}_final.parquet'); Ps = pd.read_parquet(C3/f'preds_b{bs}_final.parquet')
assert (Pw.cell.values == Ps.cell.values).all()
sb_all = np.column_stack([pd.read_parquet(C3/f'preds_b{b}_final.parquet').SB3.values for b in sorted(t3.bundle)])
yb = Pw.yb.values; mg = Pw.mg.values; ums = np.unique(mg); gi = pd.Series(np.arange(len(mg))).groupby(mg).apply(lambda s: s.values).to_dict()
rng = np.random.default_rng(11); K, KX = [], []
for _ in range(1000):
    smp = rng.choice(ums, len(ums)); ii = np.concatenate([gi[u] for u in smp if u in gi]); y = yb[ii]
    am = roc_auc_score(y, Pw.p.values[ii]); a0 = roc_auc_score(y, Ps.SB3.values[ii]); K.append((am - a0) / (1 - a0))
    ax = max(roc_auc_score(y, sb_all[ii, j]) for j in range(sb_all.shape[1])); KX.append((am - ax) / (1 - ax))
out['T3'] = dict(weak_model_bundle=bw, strong_SB3_bundle=bs, cross_SB3_ci=q(K), extreme_SB3_ci_max_over_all_seeds=q(KX),
                 cross_SB3_point_skill=(float(t3.auc_champ.min()) - float(t3.auc_strong.max())) / (1 - float(t3.auc_strong.max())),
                 note='T3 M0(kr_mom 로지스틱)는 결정적이라 교차 짝이 없다. M0 대비 최악 = 묶음 중 최소 하한.')
lo = [out['T1']['cross_M0_ci'][0], out['T1']['cross_SBB_ci'][0], out['T3']['cross_SB3_ci'][0]]
out['verdict'] = '반박 실패(교차 최악 짝에서도 하한 > 0, 결론 유지)' if min(lo) > 0 else '반박 성공(교차 짝에서 하한 ≤ 0, 결론에 단서)'
out['extreme_all_gt0'] = bool(min(out['T1']['extreme_M0_ci_max_over_all_baseline_seeds'][0], out['T1']['extreme_SBB_ci_max_over_all_SBB_seeds'][0], out['T3']['extreme_SB3_ci_max_over_all_seeds'][0]) > 0)
json.dump(out, open(W/'verify_0'/'refute_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
