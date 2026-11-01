# 평가 원점 v 하나의 모형 학습·예측. 사용: python -I s2_models.py v grp
#  grp 'sb' = SB, PI(로지스틱) / 'base' = pub, prov / 'pubarms' = free, hsc / 'ref' = all, pubctx, scagg
# 학습 = 원점 v-3..v-1 저장 행(음성 1/3 재추출, 가중 x3), 평가 = 원점 v 저장 행 전체(가중).
# 출력 cache/pred_{v}_{grp}.parquet (행 순서 = u_CN_v 다음 u_EP_v), cache/predlog_{v}_{grp}.json
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib12 import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

v = int(sys.argv[1]); grp = sys.argv[2]
assert v in EVAL_Y
t = time.time()
TR = [v - 3, v - 2, v - 1]
for y in TR:
    assert ymd(y, 6, 30) < ymd(v), '학습 원점 결과 창은 평가 원점 t0 전에 끝나야 한다'

ARMS = {'pub': PUBF, 'prov': PUBF + ctxcols('prov'), 'free': PUBF + ctxcols('free'), 'hsc': PUBF + ctxcols('hsc'),
        'all': PUBF + ctxcols('prov') + ctxcols('free') + ctxcols('hsc'),
        'pubctx': PUBF + ctxcols('free') + ctxcols('hsc'), 'scagg': PUBF + ctxcols('scagg')}
GRP = {'sb': ['SB', 'PI'], 'base': ['pub', 'prov'], 'pubarms': ['free', 'hsc'], 'ref': ['all', 'pubctx', 'scagg']}[grp]
need = sorted(set(sum([ARMS[m] for m in GRP if m in ARMS], [])) | set(PUBF))
cols = ['cc', 'aid', 's', 'Y', 'y', 'w'] + need


def load(y, sub):
    fs = []
    for cc in CCS:
        f = pd.read_parquet(C + rf'\u_{cc}_{y}.parquet', columns=cols)
        fs.append(f)
    f = pd.concat(fs, ignore_index=True)
    if sub:
        rng = np.random.default_rng(SEED + y + 7)
        keep = (f.y.values == 1) | (rng.random(len(f)) < 1 / 3)
        f = f[keep].copy()
        f.loc[f.y == 0, 'w'] *= 3
    return f


tr = pd.concat([load(y, True) for y in TR], ignore_index=True)
ev = load(v, False)
assert (tr.Y < v).all() and (ev.Y == v).all()
# 결측 대치(학습 평균)
fill = {c: float(np.nanmean(tr[c].values)) for c in need}
for c in need:
    tr[c] = tr[c].fillna(fill[c]); ev[c] = ev[c].fillna(fill[c])
out('loaded', len(tr), len(ev), round(time.time() - t, 1))

P = pd.DataFrame({'cc': ev.cc.values, 'aid': ev.aid.values, 's': ev.s.values, 'y': ev.y.values, 'w': ev.w.values})
log = {'v': v, 'grp': grp, 'n_tr': len(tr), 'n_ev': len(ev), 'pos_tr': int(tr.y.sum()), 'pos_ev': int(ev.y.sum()), 'auc': {}}


def wauc(y, s, w):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s, sample_weight=w))


for m in GRP:
    if m in ('SB', 'PI'):
        fc = ['f_log3', 's_n5', 'p_relmean'] if m == 'SB' else ['pi_rfirm', 'pi_rsc']
        sc = StandardScaler().fit(tr[fc].values)
        lr = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(tr[fc].values), tr.y.values, sample_weight=tr.w.values)
        P[m] = lr.decision_function(sc.transform(ev[fc].values)).astype(np.float32)
    else:
        fc = ARMS[m]
        dtr = xgb.DMatrix(tr[fc].values, label=tr.y.values, weight=tr.w.values)
        dev = xgb.DMatrix(ev[fc].values)
        ps = []
        for sd in [0, 1]:
            prm = dict(objective='binary:logistic', tree_method='hist', eta=0.05, max_depth=5, min_child_weight=20,
                       subsample=0.8, colsample_bytree=0.7, reg_lambda=5, nthread=2, seed=sd, eval_metric='auc')
            bst = xgb.train(prm, dtr, num_boost_round=300)
            ps.append(bst.predict(dev))
            log.setdefault('seed_auc', {})[f'{m}_{sd}'] = wauc(ev.y.values, ps[-1], ev.w.values)
        P[m] = np.mean(ps, 0).astype(np.float32)
    log['auc'][m] = wauc(P.y.values, P[m].values, P.w.values)
    out(m, round(log['auc'][m], 5), round(time.time() - t, 1))
P.to_parquet(C + rf'\pred_{v}_{grp}.parquet', index=False)
log['secs'] = round(time.time() - t, 1)
json.dump(log, open(C + rf'\predlog_{v}_{grp}.json', 'w'), indent=1)
out('done', log['secs'])
