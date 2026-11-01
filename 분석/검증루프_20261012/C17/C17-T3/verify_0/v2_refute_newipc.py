# C17-T3 verify_0 (재시도) 반박: 'B2 개발 대 최종 비교 혼합' 발견
# (A) t3_newipc_sens.csv V1/V1r/V48/V48r 최종·개발 Δ를 동결 preds만으로 독립 재계산
# (B) 비교 공정성: 개발 원점에서 제외되는 U셀 수·양성 수는 최종(90셀·34양성)보다 훨씬 적다.
#     같은 수·같은 양성 수의 비U 셀을 무작위로 뺀 귀무분포 대비 관측 Δ의 백분위를 최종·개발 원점별로 비교.
import os, sys, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, "검증루프_20261012", "C17", "C17-T3")
R3 = os.path.join(ROOT, "모델경쟁_20260927", "T3_기술셀부상예측", "R3", "preds")
rm = pd.read_csv(os.path.join(W, 'cache', 'v2_remap.csv'), encoding='utf-8-sig')
UMG = set(rm.new_mg)
H10B64 = lambda mg: mg.str.startswith('H10') | mg.str.startswith('B64U')
F = pd.read_parquet(os.path.join(R3, 'FINAL_R3_t0_2021.parquet'))
D = pd.read_parquet(os.path.join(R3, 'dev_preds_r3.parquet'))
def sk(y, p, pb):
    a, ab = roc_auc_score(y, p), roc_auc_score(y, pb); return (a - ab) / (1 - ab)
hk = lambda cs: np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in cs], dtype=np.int64)
def relabel(d):
    kr_fut = np.round(np.exp(d.y.values.astype(float)) * (d.kr_W.values + 1) - 1)
    tie = kr_fut - d.kr_W.values; k = int(round(0.10 * len(d)))
    o = np.lexsort((hk(d.cell.values), -tie, -d.y.values)); yb = np.zeros(len(d), int); yb[o[:k]] = 1; return yb
out = {}
rng = np.random.default_rng(20261017); B = 2000
def analyze(d, mask, tag):
    d = d.reset_index(drop=True); pb = d.p_base.values.astype(float)
    s0 = sk(d.yb.values, d.p.values, pb)
    e = d[~mask].reset_index(drop=True)
    dv = sk(e.yb.values, e.p.values, e.p_base.values.astype(float)) - s0
    dvr = sk(relabel(e), e.p.values, e.p_base.values.astype(float)) - s0
    nU, pU = int(mask.sum()), int(d.yb[mask].sum())
    # 귀무: 비U에서 양성 pU개 + 음성 nU-pU개 무작위 제외
    pos_idx = np.where((~mask) & (d.yb.values == 1))[0]; neg_idx = np.where((~mask) & (d.yb.values == 0))[0]
    null = []
    for _ in range(B):
        drop = np.r_[rng.choice(pos_idx, pU, replace=False), rng.choice(neg_idx, nU - pU, replace=False)]
        keep = np.ones(len(d), bool); keep[drop] = False
        null.append(sk(d.yb.values[keep], d.p.values[keep], pb[keep]) - s0)
    null = np.array(null)
    # U셀 내부 AUC(모형 대 기준) - 양성/음성 둘 다 있을 때
    sub = d[mask]
    inner = None
    if 0 < pU < nU:
        inner = dict(auc_p=float(roc_auc_score(sub.yb, sub.p)), auc_base=float(roc_auc_score(sub.yb, sub.p_base.astype(float))))
    r = dict(tag=tag, n=int(len(d)), nU=nU, posU=pU, pos_rate_U=round(pU / max(nU, 1), 3), base_skill=round(s0, 4),
             d_excl=round(dv, 4), d_excl_relabel=round(dvr, 4),
             null_mean=round(float(null.mean()), 4), null_sd=round(float(null.std()), 4),
             pct_obs_in_null=round(float((null <= dv).mean()), 4), z=round(float((dv - null.mean()) / null.std()), 2) if null.std() > 0 else None,
             d_per_cell_x100=round(100 * dv / max(nU, 1), 4), inner=inner)
    return r
for name, fn in [('U90', lambda d: d.mg.isin(UMG).values), ('T48', lambda d: H10B64(d.mg).values)]:
    res = [analyze(F, fn(F), 'final2021')]
    for t0, g in D.groupby('t0'): res.append(analyze(g, fn(g), f'dev{t0}'))
    out[name] = res
    print(name)
    print(pd.DataFrame([{k: v for k, v in r.items() if k != 'inner'} for r in res]).to_string())
json.dump(out, open(os.path.join(W, 'verify_0', 'v2_refute_newipc_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
