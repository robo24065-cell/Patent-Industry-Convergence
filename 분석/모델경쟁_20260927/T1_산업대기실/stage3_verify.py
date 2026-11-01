# 3단계 독립 대조: final_run --mode holdout 이 남긴 eval_scores.parquet 로 지표·기준선·법인 군집 부트스트랩 CI(skill 포함)를 따로 다시 계산.
# 기준선도 base_app 패널에서 독립적으로 다시 적합(로지스틱, 스태킹/단일)해 final_run 값과 대조한다. 모형은 건드리지 않는다.
import sys, json, pathlib, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.stdout.reconfigure(encoding='utf-8')
HERE = pathlib.Path(__file__).resolve().parent; FC = HERE/'final_cache'/'holdout'
t0 = time.time()
e = pd.read_parquet(FC/'eval_scores.parquet')
res = json.load(open(HERE/'holdout_result.json', encoding='utf-8'))
y = e.y.values.astype(int); nc = e.newc.values.astype(bool); firm = e.firm.values
out = {'n_rows': int(len(e)), 'n_firms': int(pd.Series(firm).nunique()), 'n_pos': int(y.sum()), 'n_new_rows': int(nc.sum()), 'n_new_pos': int(y[nc].sum()),
       'n_new_firms': int(pd.Series(firm[nc]).nunique()), 'dup_pairs': int(e.duplicated(['firm', 'k']).sum())}

def m4(s):
    return dict(auc_all=roc_auc_score(y, s), pr_all=average_precision_score(y, s), auc_new=roc_auc_score(y[nc], s[nc]), pr_new=average_precision_score(y[nc], s[nc]))

# 1) 독립 기준선 재적합(트랙 고정 기준선: 규모+density+kpop+DART, 로지스틱)
BASE = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'dm2', 'dm3', 'dm4', 'has_dart']
cuts = res['config']['train_cuts']
btr = pd.concat([pd.read_parquet(FC/f'base_app_c{c}.parquet') for c in cuts], ignore_index=True)
bte = pd.read_parquet(FC/f'base_app_c{res["config"]["eval_cut"]}.parquet')
assert (bte.firm.values == e.firm.values).all() and (bte.k.values == e.k.values).all() and (bte.y.values == e.y.values).all()
def lfit(tr):
    X = tr[BASE].fillna(0).values; sc = StandardScaler().fit(X); lr = LogisticRegression(max_iter=3000).fit(sc.transform(X), tr.y.values)
    return lr.predict_proba(sc.transform(bte[BASE].fillna(0).values))[:, 1]
b_stack = lfit(btr); b_single = lfit(btr[btr.cut == max(cuts)])
# 기준선 변형(강화 여부 점검): 규모만 / 고정기준선 XGB(같은 모형족)
import xgboost as xgb
xb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
xb.fit(btr[BASE + ['ind2']].values.astype(np.float32), btr.y.values); b_xgb = xb.predict_proba(bte[BASE + ['ind2']].values.astype(np.float32))[:, 1]
S = {'model': e.score.values, 'model_xgb_only': e.px.values, 'base_stack_indep': b_stack, 'base_single_indep': b_single, 'base_xgb_indep': b_xgb,
     'base_stack_final_run': e.sb_fixed.values, 'base_same_info_strict': e.sb_strict.values}
out['point'] = {k: {m: float(v) for m, v in m4(s).items()} for k, s in S.items()}
out['max_abs_diff_stack_indep_vs_final_run'] = float(np.max(np.abs(b_stack - e.sb_fixed.values)))
# 2) 법인 군집 부트스트랩(R=150): 모형·기준선(강화=지표별 max of stack/single/xgb 는 점추정에만) — skill CI는 stack 기준선과 '재표본마다 최강 기준선' 둘 다
fr, inv = np.unique(firm, return_inverse=True); order = np.argsort(inv, kind='stable')
st = np.searchsorted(inv[order], np.arange(len(fr))); en = np.searchsorted(inv[order], np.arange(len(fr)), side='right')
R = 150; rng = np.random.RandomState(7); keys = ['model', 'base_stack_indep', 'base_xgb_indep', 'base_same_info_strict']
A = np.zeros((R, len(keys), 4))
for r in range(R):
    idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr), len(fr))]); yy = y[idx]; nn = nc[idx]
    for j, kk in enumerate(keys):
        ss = S[kk][idx]
        A[r, j] = [roc_auc_score(yy, ss), average_precision_score(yy, ss), roc_auc_score(yy[nn], ss[nn]), average_precision_score(yy[nn], ss[nn])]
q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
M = ['auc_all', 'pr_all', 'auc_new', 'pr_new']
out['boot_R'] = R
out['ci'] = {kk: {m: q(A[:, j, i]) for i, m in enumerate(M)} for j, kk in enumerate(keys)}
sk = lambda a, b: (a - b) / (1 - b)
bmax = A[:, 1:3, :].max(1)   # 재표본마다 두 기준선(로지스틱 스태킹·XGB) 중 최강
out['ci_skill'] = {
    'auc_all_vs_stack': q(sk(A[:, 0, 0], A[:, 1, 0])), 'auc_new_vs_stack': q(sk(A[:, 0, 2], A[:, 1, 2])),
    'auc_all_vs_strongest': q(sk(A[:, 0, 0], bmax[:, 0])), 'auc_new_vs_strongest': q(sk(A[:, 0, 2], bmax[:, 2])),
    'pr_new_ratio_vs_stack': q(A[:, 0, 3] / A[:, 1, 3]), 'pr_all_ratio_vs_stack': q(A[:, 0, 1] / A[:, 1, 1])}
# 3) 비전문가용: 신규진입자 상위 목록 적중
def topk(s, mask, n):
    o = np.argsort(-s[mask])[:n]; return float(y[mask][o].mean())
out['plain'] = {'base_rate_all': float(y.mean()), 'base_rate_new': float(y[nc].mean()),
                'hit_top1pct_all': topk(e.score.values, np.ones(len(y), bool), int(len(y) * 0.01)),
                'hit_top1pct_new': topk(e.score.values, nc, int(nc.sum() * 0.01)),
                'hit_top1000_new': topk(e.score.values, nc, 1000), 'hit_top1000_new_baseline': topk(b_stack, nc, 1000),
                'hit_top1pct_all_baseline': topk(b_stack, np.ones(len(y), bool), int(len(y) * 0.01)),
                # 실제 진입의 몇 %를 상위 10% 안에서 잡았나(재현율)
                'recall_top10pct_all': float(y[np.argsort(-e.score.values)[:int(len(y) * 0.1)]].sum() / y.sum()),
                'recall_top10pct_all_baseline': float(y[np.argsort(-b_stack)[:int(len(y) * 0.1)]].sum() / y.sum()),
                'recall_top10pct_new': float(y[nc][np.argsort(-e.score.values[nc])[:int(nc.sum() * 0.1)]].sum() / y[nc].sum()),
                'recall_top10pct_new_baseline': float(y[nc][np.argsort(-b_stack[nc])[:int(nc.sum() * 0.1)]].sum() / y[nc].sum())}
out['sec'] = round(time.time() - t0)
json.dump(out, open(HERE/'holdout_verify.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=1))
