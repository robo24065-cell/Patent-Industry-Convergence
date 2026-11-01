# -*- coding: utf-8 -*-
"""T1 묶음 채점: 점추정(동결 score_final과 같은 mets·skill·topk) + M0 대비 법인 군집 CI(공식 boot_ci 규약) + SBB 대비 CI(ci_sb_t1 규약).
 python t1_score.py <b>   (시간 예산 480초, 남으면 종료코드 3 → run.sh 재실행, chunk 캐시에서 이어짐)
 M0 CI: chunk ci(0..19)마다 RandomState(3000+ci) 10회. 재표본 AUC: 모형·묶음 xgb_y·xgb_y_u는 새로 계산, 로지스틱 4개는 공식 r3boot_R200_{ci}.npy[:, 4,5,7,8, 0]
        (같은 재표본·같은 점수라 동일). 보조: 공식 고정 기준선(r3boot[:, 4:, 0]) 대비.
 SBB CI: RandomState(4242) 200회(20회 단위로 같은 난수열을 재생해 이어 계산)."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import final_run_r3 as F
from r3 import mets, skill

T0 = time.time(); BUDGET = 480
b = int(sys.argv[1]); FC = fcp(2019); RD = rd(b, 2019); BD = CT1/'boot'; BD.mkdir(parents=True, exist_ok=True)
fo = CT1/f'b{b}_final.json'
if fo.exists(): print('exists', b); sys.exit(0)
o = pd.read_parquet(RD/'o2019.parquet'); L = pd.read_parquet(CT1/'final_labels.parquet')
assert (o.firm.values == L.firm.values).all() and (o.k.values == L.k.values).all()
keep = ~L.prior_any.values
oo = o[keep].reset_index(drop=True); t = L[keep].reset_index(drop=True)
s = F.blend(oo); y = t.y_u.values; nc = t.newc.values.astype(bool); firm = t.firm.values
bs = pd.read_parquet(CT1/f'base_b{b}.parquet').loc[keep].reset_index(drop=True)
bso = pd.read_parquet(FC/'baseline_scores.parquet').loc[keep].reset_index(drop=True)
sbb = np.load(CT1/f'sbb_b{b}.npy'); assert len(sbb) == len(y)
KEYS = list(bso.columns); assert KEYS == ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u'], KEYS
fr_, inv = np.unique(firm, return_inverse=True)
order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')


def left(): return BUDGET - (time.time() - T0)


# ---- M0 CI ----
for ci in range(20):
    fp = BD/f'b{b}_m0_{ci}.npz'
    if fp.exists(): continue
    if left() < 75: print('budget stop m0', ci, flush=True); sys.exit(3)
    rng = np.random.RandomState(3000 + ci); arr = np.zeros((10, 3))
    for r in range(10):
        idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]
        arr[r] = [roc_auc_score(yy, s[idx]), roc_auc_score(yy, bs.xgb_y.values[idx]), roc_auc_score(yy, bs.xgb_y_u.values[idx])]
    np.savez(fp, auc=arr); print('m0 chunk', ci, round(time.time() - T0), flush=True)
# ---- SBB CI ----
for j in range(10):
    fp = BD/f'b{b}_sbb_{j}.npz'
    if fp.exists(): continue
    if left() < 75: print('budget stop sbb', j, flush=True); sys.exit(3)
    rng = np.random.RandomState(4242); arr = np.zeros((20, 2))
    for r in range(200):
        draw = rng.randint(0, len(fr_), len(fr_))
        if r // 20 != j: continue
        idx = np.concatenate([order[st[i]:en[i]] for i in draw]); yy = y[idx]
        arr[r % 20] = [roc_auc_score(yy, s[idx]), roc_auc_score(yy, sbb[idx])]
    np.savez(fp, auc=arr); print('sbb chunk', j, round(time.time() - T0), flush=True)
# ---- 조립 ----
q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
A = np.concatenate([np.load(BD/f'b{b}_m0_{ci}.npz')['auc'] for ci in range(20)])
OFF = np.concatenate([np.load(FC/f'r3boot_R200_{ci}.npy') for ci in range(20)])      # (200, 10 keys, 2)
Bb = np.column_stack([OFF[:, 4, 0], OFF[:, 5, 0], A[:, 1], OFF[:, 7, 0], OFF[:, 8, 0], A[:, 2]])   # 묶음 기준선 6변형
bmax = Bb.max(1); bmax_off = OFF[:, 4:, 0].max(1)
sk_b = (A[:, 0] - bmax) / (1 - bmax); sk_off = (A[:, 0] - bmax_off) / (1 - bmax_off)
SB = np.concatenate([np.load(BD/f'b{b}_sbb_{j}.npz')['auc'] for j in range(10)])
sk_sbb = (SB[:, 0] - SB[:, 1]) / (1 - SB[:, 1])
m = mets(y, s, nc); bm = {k: mets(y, bs[k].values, nc) for k in KEYS}; bmo = {k: mets(y, bso[k].values, nc) for k in KEYS}
strongest = max(v['auc_all'] for v in bm.values()); strongest_off = max(v['auc_all'] for v in bmo.values())
a_sbb = float(roc_auc_score(y, sbb))
tk = F.topk(y, s)
res = dict(track='T1', bundle=b, seeds=seeds(b), n=int(len(y)), pos=int(y.sum()), auc_champ=m['auc_all'], auc_new=m['auc_new'],
           auc_M0=strongest, M0_arg=max(bm, key=lambda k: bm[k]['auc_all']), base_aucs={k: v['auc_all'] for k, v in bm.items()},
           skill=skill(m['auc_all'], strongest), ci=q(sk_b), skill_new=skill(m['auc_new'], max(v['auc_new'] for v in bm.values())),
           auc_M0_official_fixed=strongest_off, skill_vs_official_fixed=skill(m['auc_all'], strongest_off), ci_vs_official_fixed=q(sk_off),
           auc_strong=a_sbb, skill_vs_strong=skill(m['auc_all'], a_sbb), strong_ci=q(sk_sbb), strong_name='SBB',
           top100=tk['100']['hits'], top1000=tk['1000']['hits'], top10000=tk['10000']['hits'],
           capture_top1pct=float(y[np.argsort(-s)[:len(y) // 100]].sum() / y.sum()), n_boot=int(len(A)), n_boot_sbb=int(len(SB)))
if b == 0:
    res['P0'] = dict(boot_model_maxdiff_vs_official=float(np.abs(A[:, 0] - OFF[:, 0, 0]).max()),
                     boot_xgb_y_maxdiff=float(np.abs(A[:, 1] - OFF[:, 6, 0]).max()), boot_xgb_y_u_maxdiff=float(np.abs(A[:, 2] - OFF[:, 9, 0]).max()))
    fz = pd.read_parquet(R3/'runs'/'FROZEN_final'/'o2019.parquet')
    res['P0'].update(px_maxdiff=float(np.abs(o.px.values - fz.px.values).max()), pl_maxdiff=float(np.abs(o.pl.values - fz.pl.values).max()))
json.dump(res, open(fo, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.DataFrame(dict(firm=firm, k=t.k.values, y=y, newc=nc, s=s, xgb_y=bs.xgb_y.values, xgb_y_u=bs.xgb_y_u.values, sbb=sbb)).to_parquet(CT1/f'scores_b{b}.parquet', index=False)
print('T1 b', b, 'skill', round(res['skill'], 6), [round(v, 4) for v in res['ci']], 'SBB', round(res['skill_vs_strong'], 4), [round(v, 4) for v in res['strong_ci']],
      'top1000', res['top1000'], round(time.time() - T0), 'peak', G.rss_peak(), flush=True)
