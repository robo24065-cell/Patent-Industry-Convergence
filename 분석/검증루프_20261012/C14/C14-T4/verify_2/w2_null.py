# verify_2 w2: P0(c-ii) 귀무 독립 재실행. 사용: python -I w2_null.py b0 b1 [real]
# - 시드: 777000000 + b (C12 20261013..20261212, C14 20261017..20261056 과 겹치지 않음 → 진짜 새 추출)
# - 귀무 라벨 생성은 독립 구현(pandas groupby + rng.integers), 피처 식(context/trends)은 사전 등록 설계 그대로 feat12 사용.
# - 'real' 인자: 실제 라벨을 이 파이프라인(재계산 ctx)에 넣은 G_real도 계산.
# 쓰기: verify_2 안에만. C12 캐시는 읽기만. 공개일 < 2025 assert.
import sys, os, time, json
sys.dont_write_bytecode = True
C12S = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T4\scripts'
OUTD = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T4\verify_2'
sys.path.insert(0, C12S)
from feat12 import *   # C = C12 cache (읽기)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
assert os.path.abspath(C) != os.path.abspath(OUTD)
b0, b1 = int(sys.argv[1]), int(sys.argv[2]); do_real = len(sys.argv) > 3
t = time.time()
ORIG = [2019] + EVAL_Y
CT = ctxcols('prov')
NSEED = 777000000


class V:
    def __init__(self, cc):
        self.cc = cc
        SCL_ = json.load(open(C + r'\sc_list.json')); self.S = len(SCL_); si = {s: i for i, s in enumerate(SCL_)}
        d = pd.read_parquet(C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'k', 'sc'])
        assert d.pub.max() < CUT and (d.did.values == np.arange(len(d))).all()
        self.d_pub = d.pub.values.astype(np.int64); self.d_sc = d.sc.map(si).fillna(-1).astype(np.int32).values
        self.d_k = d.k.values.astype(np.int32); self.d_year = (self.d_pub // 10000).astype(np.int32)
        a = pd.read_parquet(C + rf'\app_{cc}.parquet', columns=['did', 'key'])
        keys = pd.Index(a.key.unique()); aid = keys.get_indexer(a.key).astype(np.int64)
        o = np.lexsort((a.did.values, aid))
        self.m_aid = aid[o]; self.m_did = a.did.values[o].astype(np.int64)
        self.m_pub = self.d_pub[self.m_did]; self.m_sc = self.d_sc[self.m_did]; self.nF = len(keys)
        self.kr = pd.read_parquet(C + r'\docs_KR.parquet', columns=['pub', 'k']); assert self.kr.pub.max() < CUT
        self.agg5 = pd.read_parquet(C + r'\agg5.parquet', columns=['cc', 'y', 'n', 'k']); assert self.agg5.y.max() < 2025
        self.kr_lab = {'prov': self.kr.k.values.astype(np.int32)}
        self.ag_lab = {'prov': self.agg5.k.values.astype(np.int32)}
        # 독립 구현: 그룹별 인덱스 목록
        df = pd.DataFrame({'y': self.d_year, 'sc': self.d_sc, 'i': np.arange(len(d))})
        self.grp_sc = [(g.i.values) for _, g in df[df.sc >= 0].groupby(['y', 'sc'])]
        self.grp_y = {y: g.i.values for y, g in df.groupby('y')}
        self.nosc_by_y = {y: g.i.values for y, g in df[df.sc < 0].groupby('y')}
        self.H = {}

    mwin = Country.mwin
    dwin = Country.dwin
    context = Country.context
    trends = Country.trends
    kg_from = staticmethod(Country.kg_from)

    def null_labels(self, rng):
        lab = self.d_k.copy()
        for idx in self.grp_sc:
            lab[idx] = self.d_k[idx[rng.integers(0, len(idx), len(idx))]]
        for y, idx in self.nosc_by_y.items():
            pool = self.grp_y[y]
            lab[idx] = self.d_k[pool[rng.integers(0, len(pool), len(idx))]]
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
        u = pd.read_parquet(C + rf'\u_{cc}_{Y}.parquet', columns=['aid', 's', 'y', 'w'] + PUBF + CT)
        if Y < 2024:
            rng = np.random.default_rng(SEED + Y + (0 if cc == 'CN' else 100) + 11)   # 사전 등록 설계와 같은 학습 음성 상한
            neg = np.flatnonzero(u.y.values == 0); cap = 150000
            keep = np.ones(len(u), bool); fac = 1.0
            if len(neg) > cap:
                keep[rng.choice(neg, len(neg) - cap, replace=False)] = False; fac = len(neg) / cap
            u['w_tr'] = np.where(u.y.values == 0, u.w.values * fac, u.w.values); u['tr'] = keep
        ROWS[(cc, Y)] = u
PUBAUC = {}


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

        def fit(A, Bm):
            sc = StandardScaler().fit(A)
            lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=300).fit(sc.transform(A), y_tr, sample_weight=w_tr)
            return float(roc_auc_score(y_ev, lr.decision_function(sc.transform(Bm)), sample_weight=w_ev))
        if v not in PUBAUC:
            PUBAUC[v] = fit(Xp_tr, Xp_ev)
        a_ctx = fit(np.hstack([Xp_tr, Xc_tr]), np.hstack([Xp_ev, Xc_ev]))
        per.append((a_ctx - PUBAUC[v]) / (1 - PUBAUC[v]))
    return float(np.mean(per)), per


LT = {cc: V(cc) for cc in CCS}
print('loaded', round(time.time() - t, 1), flush=True)
out = {'b0': b0, 'b1': b1, 'seed_rule': 'NSEED + b', 'NSEED': NSEED}
fn = OUTD + rf'\w2_null_{b0}_{b1}.json'
if do_real:
    out['G_real_stored'] = G_of({k: u[CT].values.astype(np.float32) for k, u in ROWS.items()})
    rec = {}
    mx = 0.0
    for cc in CCS:
        for Y in ORIG:
            u = ROWS[(cc, Y)]
            rec[(cc, Y)] = LT[cc].ctx(Y, LT[cc].d_k, u.aid.values, u.s.values)
            mx = max(mx, float(np.nanmax(np.abs(rec[(cc, Y)] - u[CT].values.astype(np.float32)))))
    out['ctx_recompute_maxabs'] = mx
    out['G_real_recomputed'] = G_of(rec)
    print('real', out['G_real_stored'][0], out['G_real_recomputed'][0], mx, round(time.time() - t, 1), flush=True)
    json.dump(out, open(fn, 'w'), indent=1)
nulls = []
for b in range(b0, b1):
    rng = np.random.default_rng(NSEED + b)
    cd = {}
    for cc in CCS:
        lab = LT[cc].null_labels(rng)
        # 귀무 라벨 sanity: (연도, 서브클래스) 그룹별 라벨 분포가 실제와 기대상 같음 → 전체 라벨 빈도 차 기록
        for Y in ORIG:
            u = ROWS[(cc, Y)]
            cd[(cc, Y)] = LT[cc].ctx(Y, lab, u.aid.values, u.s.values)
        if b == b0:
            out.setdefault('lab_check', {})[cc] = {'frac_changed': float((lab != LT[cc].d_k).mean()),
                                                     'max_abs_freq_diff': float(np.abs(np.bincount(lab[lab >= 0], minlength=NK) / (lab >= 0).sum() - np.bincount(LT[cc].d_k[LT[cc].d_k >= 0], minlength=NK) / (LT[cc].d_k >= 0).sum()).max())}
    g, per = G_of(cd)
    nulls.append({'b': b, 'G': g, 'per': per})
    out['nulls'] = nulls
    json.dump(out, open(fn, 'w'), indent=1)
    print('null', b, round(g, 5), round(time.time() - t, 1), flush=True)
    if time.time() - t > 470:
        print('time guard', b, flush=True); break
out['secs'] = round(time.time() - t, 1)
json.dump(out, open(fn, 'w'), indent=1)
print('done', out['secs'])
