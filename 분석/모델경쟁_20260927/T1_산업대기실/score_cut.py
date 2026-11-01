# 동결 v31 모형으로 임의 기준시점(≤2009) 패널 채점 — 결과변수 창이 2014를 넘는 기준시점은 build_panel이 거부
import os; assert os.environ.get('T1_STRICT') == '1'
from data8 import *
import xgboost as xgb, pickle
spec = json.load(open(T1/'models/v31_spec.json')); COLS = spec['cols']
def load_cut(c):
    d = comb(c, 'None'); d = add_ep(d)
    for fn, cols in ((sib_panel, SIBF), (ins_panel, INSF)):
        x = fn(c, 'None'); d[cols] = x[cols].values
    x = tim_panel(c, 'None'); d['gap_last'] = x.gap_last.values; d['n_last1'] = x.n_last1.values; d['foreign'] = x.foreign.values
    d = add_pa(d)
    x = rr_panel(c, 'None'); d[RRF] = x[RRF].values
    x = now_panel(c, 'None'); d[NOWF] = x[NOWF].values
    return d
def score(d):
    X = d[COLS].values.astype(np.float32); ps = []
    for sd in range(3):
        m = xgb.XGBClassifier(); m.load_model(T1/f'models/v31_xgb_seed{sd}.json'); ps.append(m.predict_proba(X)[:, 1])
    px = np.mean(ps, 0); L = pickle.load(open(T1/'models/v31_logit.pkl', 'rb'))
    pl = L['lr'].predict_proba(L['sc'].transform(d[L['cols']].fillna(0).values))[:, 1]
    return 0.9 * rank(px) + 0.1 * rank(pl), px
if __name__ == '__main__':
    d = load_cut(2009); s, px = score(d); m = metrics(d, s); print('frozen v31 재현 (c=2009)', fmt(m))
