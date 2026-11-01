# C03-T1 s4: 법인 군집 부트스트랩(공식과 같은 재표본: RandomState(3000+ci), chunk 10, R=200)으로
#            공식 CI 재현 확인 + 복원판−공식 차이(ΔAUC·Δskill)의 95% 구간. 가중 AUC(재표본 중복 = 가중치)로 계산.
import sys, json, time, pathlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; WC = W/'cache'
BCOLS = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
ev = pd.read_parquet(WC/'eval_o2019.parquet')
nc = ev.newc.values.astype(bool)
Y = {'official': np.maximum(ev.yA_reg.values, ev.yA_pub.values).astype(np.float64), 'R_self': np.load(WC/'lab_R_self.npy').astype(np.float64), 'R_any': np.load(WC/'lab_R_any.npy').astype(np.float64)}
S = {'model': ev.s.values}; S.update({c: ev['b_' + c].values for c in BCOLS})
prep = {}
for kk, sc in S.items():
    o = np.argsort(sc, kind='mergesort'); ss = sc[o]; st = np.r_[0, np.nonzero(np.diff(ss))[0] + 1]
    prep[kk] = (o, st, {l: y[o] for l, y in Y.items()}, nc[o])
def wauc(w_sorted, y_sorted, st):
    Gp = np.add.reduceat(w_sorted * y_sorted, st); Gn = np.add.reduceat(w_sorted * (1 - y_sorted), st)
    below = np.cumsum(Gn) - Gn
    return float((Gp * (below + 0.5 * Gn)).sum() / (Gp.sum() * Gn.sum()))
def all_aucs(w):
    out = {}
    for kk, (o, st, ys, ncs) in prep.items():
        ws = w[o]; wn = ws * ncs
        for l, ys_ in ys.items(): out[(kk, l, 'all')] = wauc(ws, ys_, st); out[(kk, l, 'new')] = wauc(wn, ys_, st)
    return out
# 점추정 확인(가중치 1)
pt = all_aucs(np.ones(len(ev)))
print('point model official', pt[('model', 'official', 'all')], pt[('model', 'official', 'new')], round(time.time() - t0), flush=True)
fr_, inv = np.unique(ev.firm.values, return_inverse=True)
rows = []
for ci in range(20):
    rng = np.random.RandomState(3000 + ci)
    for r in range(10):
        cnt = np.bincount(rng.randint(0, len(fr_), len(fr_)), minlength=len(fr_)); w = cnt[inv].astype(np.float64)
        a = all_aucs(w); rows.append({f'{k[0]}|{k[1]}|{k[2]}': v for k, v in a.items()})
    print('chunk', ci, round(time.time() - t0), flush=True)
A = pd.DataFrame(rows); A.to_parquet(WC/'boot_aucs.parquet', index=False)
q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
res = {}
for l in Y:
    for sub in ('all', 'new'):
        m = A[f'model|{l}|{sub}']; b = A[[f'{c}|{l}|{sub}' for c in BCOLS]].max(1); sk = (m - b) / (1 - b)
        res[f'{l}_{sub}'] = dict(auc=q(m), strongest=q(b), skill=q(sk), skill_lo_gt0=bool(np.percentile(sk, 2.5) > 0), model_gt_strongest_share=float((m > b).mean()))
        A[f'skill|{l}|{sub}'] = sk
for l in ('R_self', 'R_any'):
    for sub in ('all', 'new'):
        res[f'delta_{l}_{sub}'] = dict(d_auc=q(A[f'model|{l}|{sub}'] - A[f'model|official|{sub}']), d_skill=q(A[f'skill|{l}|{sub}'] - A[f'skill|official|{sub}']))
off = json.load(open(ROOT/'분석/모델경쟁_20260927/T1_산업대기실/R3/final_result_r3.json', encoding='utf-8'))['bootstrap_firm_cluster']
res['official_ci_reproduction'] = dict(auc_all_official=off['model']['auc_all'], auc_all_here=res['official_all']['auc'],
                                       skill_all_official=off['skill_vs_strongest__model']['auc_all'], skill_all_here=res['official_all']['skill'],
                                       auc_new_official=off['model']['auc_new'], auc_new_here=res['official_new']['auc'],
                                       skill_new_official=off['skill_vs_strongest__model']['auc_new'], skill_new_here=res['official_new']['skill'])
json.dump(res, open(WC/'s4_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=0)[:4000]); print('done', round(time.time() - t0))
