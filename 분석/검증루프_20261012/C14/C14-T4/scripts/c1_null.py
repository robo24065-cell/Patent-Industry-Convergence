# C14-T4 c1: P0(c-ii) IPC 조건부 무작위 라벨 귀무 — C12 s6_p5null.py 설계를 그대로 옮기고 시드만 새로(SEED14 + b), 출력은 C14 cache.
# 사용: python -I c1_null.py b0 b1   (b0 = -1 이면 실제 G_real 재계산 + 공개 팔 G_free·G_hsc 포함)
# 귀무: 그 나라 문서(공개 < 2025) 제공 라벨을 (나라, 공개연도, 메인 서브클래스) 안 경험 분포에서 복원추출(서브클래스 없으면 (나라, 연도)).
# C12 캐시(u_*, docs_*, app_*, agg5, sc_list, ksic_list)는 읽기만.
import sys, os, time, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location('lib14', os.path.join(HERE, 'lib14.py')); L14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(L14)
sys.path.insert(0, L14.C12 + r'\scripts')
from feat12 import *          # C = C12 cache(읽기 전용)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
OUTC = L14.C
SEED14 = L14.SEED
assert C != OUTC

b0, b1 = int(sys.argv[1]), int(sys.argv[2])
t = time.time()
ORIG = [2019] + EVAL_Y
CT = ctxcols('prov')


class Lite:
    def __init__(self, cc):
        self.cc = cc
        SCL = json.load(open(C + r'\sc_list.json')); self.S = len(SCL); si = {s: i for i, s in enumerate(SCL)}
        d = pd.read_parquet(C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'k', 'sc'])
        assert d.pub.max() < CUT and (d.did.values == np.arange(len(d))).all()
        self.d_pub = d.pub.values.astype(np.int64); self.d_sc = d.sc.map(si).fillna(-1).astype(np.int32).values
        self.d_k = d.k.values.astype(np.int32); self.d_year = (self.d_pub // 10000).astype(np.int32)
        a = pd.read_parquet(C + rf'\app_{cc}.parquet', columns=['did', 'key'])
        keys = pd.Index(a.key.unique()); aid = keys.get_indexer(a.key).astype(np.int64)
        o = np.lexsort((a.did.values, aid))
        self.m_aid = aid[o]; self.m_did = a.did.values[o].astype(np.int64)
        self.m_pub = self.d_pub[self.m_did]; self.m_sc = self.d_sc[self.m_did]; self.nF = len(keys)
        self.kr = pd.read_parquet(C + r'\docs_KR.parquet', columns=['pub', 'k'])
        assert self.kr.pub.max() < CUT
        self.agg5 = pd.read_parquet(C + r'\agg5.parquet', columns=['cc', 'y', 'n', 'k'])
        assert self.agg5.y.max() < 2025
        self.kr_lab = {'prov': self.kr.k.values.astype(np.int32)}
        self.ag_lab = {'prov': self.agg5.k.values.astype(np.int32)}
        g = pd.DataFrame({'y': self.d_year, 'sc': self.d_sc})
        self.gid = g.groupby(['y', 'sc']).ngroup().values
        order = np.argsort(self.gid, kind='stable')
        gs = self.gid[order]
        self.gstart = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        self.gsize = np.diff(np.r_[self.gstart, len(gs)])
        self.k_sorted = self.d_k[order]
        self.yid = pd.factorize(self.d_year)[0]
        oy = np.argsort(self.yid, kind='stable'); ys = self.yid[oy]
        self.ystart = np.r_[0, np.flatnonzero(np.diff(ys)) + 1]; self.ysize = np.diff(np.r_[self.ystart, len(ys)])
        self.k_ysorted = self.d_k[oy]
        self.nosc = self.d_sc < 0
        self.H = {}

    mwin = Country.mwin
    dwin = Country.dwin
    context = Country.context
    trends = Country.trends
    kg_from = staticmethod(Country.kg_from)

    def null_labels(self, rng):
        g = self.gid
        start = self.gstart[g]; size = self.gsize[g]
        r = rng.random(len(g))
        pick = start + (r * size).astype(np.int64)
        lab = self.k_sorted[pick]
        yi = self.yid[self.nosc]
        lab[self.nosc] = self.k_ysorted[self.ystart[yi] + (r[self.nosc] * self.ysize[yi]).astype(np.int64)]
        return lab

    def hmats(self, Y):
        if Y not in self.H:
            t0 = ymd(Y)
            w = self.mwin(addy(t0, -10), t0) & (self.m_sc >= 0)
            Hd = sp.csr_matrix((np.ones(w.sum(), np.float32), (self.m_aid[w], self.m_sc[w])), shape=(self.nF, self.S))
            Hd.sum_duplicates(); Hd.data[:] = 1
            fa = pd.DataFrame({'a': self.m_aid[w], 's': self.m_sc[w], 'p': self.m_pub[w]}).groupby(['a', 's']).p.min().reset_index()
            fa = fa[fa.p >= addy(t0, -3)]
            Hn = sp.csr_matrix((np.ones(len(fa), np.float32), (fa.a.values, fa.s.values)), shape=(self.nF, self.S))
            self.H[Y] = (Hd, Hn)
        return self.H[Y]

    def ctx(self, Y, lab, aid, s):
        Hd, Hn = self.hmats(Y)
        kgs = self.trends(Y, lab, 'prov')
        F, _, _ = self.context(ymd(Y), lab, aid, s, Hd, Hn, kgs)
        return np.column_stack([F[x] for x in CTX]).astype(np.float32)


ROWS = {}
for cc in CCS:
    for Y in ORIG:
        u = pd.read_parquet(C + rf'\u_{cc}_{Y}.parquet', columns=['aid', 's', 'y', 'w'] + PUBF + CT + ctxcols('free') + ctxcols('hsc'))
        if Y < 2024:
            rng = np.random.default_rng(SEED + Y + (0 if cc == 'CN' else 100) + 11)   # C12와 같은 학습 음성 상한 추출(설계 동일)
            neg = np.flatnonzero(u.y.values == 0)
            cap = 150000
            keep_tr = np.ones(len(u), bool)
            if len(neg) > cap:
                drop = rng.choice(neg, len(neg) - cap, replace=False)
                keep_tr[drop] = False
                fac = len(neg) / cap
            else:
                fac = 1.0
            u['w_tr'] = np.where(u.y.values == 0, u.w.values * fac, u.w.values)
            u['tr'] = keep_tr
        ROWS[(cc, Y)] = u
out('rows loaded', round(time.time() - t, 1))
CACHE = {}


def G_of(ctxd):
    per = []
    for v in EVAL_Y:
        tr = [(ROWS[(cc, v - 1)], ctxd[(cc, v - 1)]) for cc in CCS]
        ev = [(ROWS[(cc, v)], ctxd[(cc, v)]) for cc in CCS]
        Xp_tr = np.vstack([u[PUBF].values[u.tr.values] for u, _ in tr]); Xc_tr = np.vstack([c[u.tr.values] for u, c in tr])
        y_tr = np.concatenate([u.y.values[u.tr.values] for u, _ in tr]); w_tr = np.concatenate([u.w_tr.values[u.tr.values] for u, _ in tr])
        Xp_ev = np.vstack([u[PUBF].values for u, _ in ev]); Xc_ev = np.vstack([c for _, c in ev])
        y_ev = np.concatenate([u.y.values for u, _ in ev]); w_ev = np.concatenate([u.w.values for u, _ in ev])
        mu = np.nanmean(Xp_tr, 0)
        Xp_tr = np.where(np.isnan(Xp_tr), mu, Xp_tr); Xp_ev = np.where(np.isnan(Xp_ev), mu, Xp_ev)
        res = {}
        for nm, (A, Bm) in {'pub': (Xp_tr, Xp_ev), 'ctx': (np.hstack([Xp_tr, Xc_tr]), np.hstack([Xp_ev, Xc_ev]))}.items():
            if nm == 'pub' and 'pub' in CACHE.get(v, {}):
                res[nm] = CACHE[v]['pub']; continue
            sc = StandardScaler().fit(A)
            lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=300).fit(sc.transform(A), y_tr, sample_weight=w_tr)
            res[nm] = float(roc_auc_score(y_ev, lr.decision_function(sc.transform(Bm)), sample_weight=w_ev))
        CACHE.setdefault(v, {})['pub'] = res['pub']
        per.append((res['ctx'] - res['pub']) / (1 - res['pub']))
    return float(np.mean(per)), per


out_ = {'b0': b0, 'b1': b1, 'seed_rule': 'SEED14 + b', 'SEED14': SEED14}
fn = OUTC + rf'\c1_null_{sys.argv[1]}_{b1}.json'
if b0 < 0:
    out_['G_real'] = G_of({k: u[CT].values.astype(np.float32) for k, u in ROWS.items()})
    out_['G_free'] = G_of({k: u[ctxcols('free')].values.astype(np.float32) for k, u in ROWS.items()})
    out_['G_hsc'] = G_of({k: u[ctxcols('hsc')].values.astype(np.float32) for k, u in ROWS.items()})
    out(json.dumps(out_), round(time.time() - t, 1))
    json.dump(out_, open(fn, 'w'), indent=1)
    b0 = 0
LT = {cc: Lite(cc) for cc in CCS}
out('lite loaded', round(time.time() - t, 1))
nulls = []
for b in range(b0, b1):
    rng = np.random.default_rng(SEED14 + b)
    cd = {}
    for cc in CCS:
        lab = LT[cc].null_labels(rng)
        for Y in ORIG:
            u = ROWS[(cc, Y)]
            cd[(cc, Y)] = LT[cc].ctx(Y, lab, u.aid.values, u.s.values)
    g, per = G_of(cd)
    nulls.append({'b': b, 'G': g, 'per': per})
    out('null', b, round(g, 5), round(time.time() - t, 1))
    out_['nulls'] = nulls
    json.dump(out_, open(fn, 'w'), indent=1)
    if time.time() - t > 470:
        out('time guard stop at', b)
        break
out_['secs'] = round(time.time() - t, 1)
json.dump(out_, open(fn, 'w'), indent=1)
out('done', out_['secs'])
