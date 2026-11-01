# C18-T4 s5: K1 경량 LR 재학습(PREREG §5 G1 '하나만 있는 후보는 통과하면 LR을 새로 학습'). 사양 = C12 s6_p5null / C14 c1_null의 G_of와 같음
#  (학습 = 원점 v-1 저장 행, 나라별 음성 상한 150,000(같은 rng), 표준화 + LogisticRegression(C=1, lbfgs, max_iter=300), 결정함수 점수).
#  → P0(b)와 P0(c)가 같은 경량 LR을 쓴다(C14 교훈). 팔: base(PUBF), prov, free, hsc, scagg.
# C12 캐시는 읽기만. 출력 cache/fr_K1_{v}_lr.parquet (행 순서 = fr_K1_{v} = C12 pred = u_CN_v 다음 u_EP_v)
# 사용: python -I s5_k1_lr.py v1 v2 ...
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')
t = time.time()
SEED12 = 20261013
CCS = ['CN', 'EP']
PUBF = ['f_log3', 'f_log1', 'f_gr', 'f_nsc', 'f_age', 'f_missc', 'is_cn', 's_n5', 's_grow', 's_nfirm', 's_newrate',
        'p_sec', 'p_relmean', 'p_relmax', 'p_cls', 'p_sect', 'p_cf', 'pi_rfirm', 'pi_rsc']
CTX = ['dot', 'cos', 'adopt', 'entry', 'qmain', 'mom_c', 'sind_c', 'mom_kr', 'sind_kr', 'mom_5', 'sind_5']
ARMS = ['prov', 'free', 'hsc', 'scagg']
ctxc = lambda a: [f'{a}_{x}' for x in CTX]
cols = ['cc', 'aid', 's', 'Y', 'y', 'w'] + PUBF + sum([ctxc(a) for a in ARMS], [])
# PUBF 목록이 C12 lib12와 같은지 확인(읽기만)
src = open(LOOP + r'\C12\C12-T4\scripts\lib12.py', encoding='utf-8').read()
assert "PUBF = ['f_log3', 'f_log1', 'f_gr', 'f_nsc', 'f_age', 'f_missc', 'is_cn', 's_n5', 's_grow', 's_nfirm', 's_newrate'," in src


def load(Y, train):
    fs = []
    for cc in CCS:
        u = pd.read_parquet(C12C + rf'\u_{cc}_{Y}.parquet', columns=cols)
        if train:
            rng = np.random.default_rng(SEED12 + Y + (0 if cc == 'CN' else 100) + 11)
            neg = np.flatnonzero(u.y.values == 0)
            cap = 150000
            keep = np.ones(len(u), bool)
            if len(neg) > cap:
                keep[rng.choice(neg, len(neg) - cap, replace=False)] = False
                fac = len(neg) / cap
            else:
                fac = 1.0
            u['w_tr'] = np.where(u.y.values == 0, u.w.values * fac, u.w.values)
            u = u[keep]
        fs.append(u)
    return pd.concat(fs, ignore_index=True)


log = {}
for v in [int(a) for a in sys.argv[1:]]:
    tr = load(v - 1, True); ev = load(v, False)
    assert (tr.Y == v - 1).all() and (ev.Y == v).all()
    F = load_frame('K1', v)
    assert len(F) == len(ev) and (F.y.values == ev.y.values).all() and np.allclose(F.w.values, ev.w.values)
    assert ((F.f.values % 10_000_000) == ev.aid.values).all()
    G = F[['f', 'k', 's', 'cc', 'y', 'w']].copy()
    lg = {'n_tr': len(tr), 'pos_tr': int(tr.y.sum()), 'n_ev': len(ev), 'auc': {}}
    for arm in ['base'] + ARMS:
        cs = PUBF + (ctxc(arm) if arm != 'base' else [])
        A = tr[cs].values.astype(np.float64); Bm = ev[cs].values.astype(np.float64)
        mu = np.nanmean(A, 0)
        A = np.where(np.isnan(A), mu, A); Bm = np.where(np.isnan(Bm), mu, Bm)
        sc = StandardScaler().fit(A)
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=300).fit(sc.transform(A), tr.y.values, sample_weight=tr.w_tr.values)
        G[arm] = lr.decision_function(sc.transform(Bm))
        lg['auc'][arm] = AUCPrep(G[arm].values, G.y.values).auc(G.w.values.astype(float))
        out(v, arm, round(lg['auc'][arm], 5), round(time.time() - t, 1))
    G.to_parquet(C + rf'\fr_K1_{v}_lr.parquet', index=False)
    log[v] = lg
    json.dump(log, open(C + rf'\s5_k1_lr_{"_".join(sys.argv[1:])}.json', 'w'), indent=1)
    assert time.time() - t < 530, 'time guard'
out('done', round(time.time() - t, 1))
