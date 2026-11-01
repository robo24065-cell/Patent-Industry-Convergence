# verify_0 V3: '라벨 기계' 귀무 — 문서 라벨을 (나라, 공개연도, IPC 서브클래스)별 제공 라벨 분포에서 무작위로 다시 뽑아(기업 고유 정보 제거)
# 같은 진입 사건·간단 피처를 만들고, 실제 라벨과 예측 가능성(AUC)을 비교한다. 학습 원점 2023 → 평가 2024(로지스틱 4피처).
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

C = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C11/C11-T4/cache'
OUT = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C11/C11-T4/verify_0'
NK = 50
t = time.time()


def synth(d, seed):
    """d: did, pub, k, sc. (연도, sc)별 k 분포에서 재추출."""
    rng = np.random.default_rng(seed)
    d = d.copy(); d['yr'] = d.pub // 10000
    d['sc2'] = d.sc.fillna('NA')
    out = np.empty(len(d), np.int64)
    for (yr, sc), idx in d.groupby(['yr', 'sc2']).indices.items():
        ks = d.k.values[idx]
        out[idx] = rng.choice(ks, size=len(idx), replace=True)       # 같은 셀 경험 분포에서 복원 추출
    return out


def build(m, lab, Y):
    t0 = Y * 10000 + 101; wlo, whi = Y * 10000 + 101, Y * 10000 + 630
    s3 = m[(m.pub >= t0 - 30000) & (m.pub < t0)]
    n3 = s3.drop_duplicates(['aid', 'did']).groupby('aid').size()
    n1 = s3[s3.pub >= t0 - 10000].drop_duplicates(['aid', 'did']).groupby('aid').size()
    ids = pd.Index(np.sort(n3[n3 >= 10].index.intersection(n1[n1 >= 1].index).values))
    def M(lo, hi, col):
        s = m[(m.pub >= lo) & (m.pub < hi)]; r = ids.get_indexer(s.aid); ok = r >= 0
        A = np.zeros((len(ids), NK), np.float32); np.add.at(A, (r[ok], s[col].values[ok]), 1); return A
    n5 = M(t0 - 50000, t0, lab); nw = M(wlo, whi + 1, lab); h3 = M(t0 - 30000, t0, lab)
    risk = n5 == 0; y = (nw >= 1) & risk
    H = (h3 > 0).astype(np.float32); Co = H.T @ H; dg = np.diag(Co).copy(); dg[dg == 0] = 1
    P = Co / dg[None, :]; phi = np.minimum(P, P.T); np.fill_diagonal(phi, 0); den = phi.sum(1); den[den == 0] = 1
    dens = (H @ phi.T) / den[None, :]
    s = s3[s3.sc.notna()]; scs = pd.Index(s.sc.unique())
    Q = np.zeros((len(scs), NK), np.float32); np.add.at(Q, (scs.get_indexer(s.sc), s[lab].values), 1)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1)
    r = ids.get_indexer(s.aid); ok = r >= 0
    import scipy.sparse as sp
    F = sp.csr_matrix((np.ones(ok.sum(), np.float32), (r[ok], scs.get_indexer(s.sc)[ok])), shape=(len(ids), len(scs)))
    nd = np.asarray(F.sum(1)).ravel(); nd[nd == 0] = 1
    meanq = np.asarray(F @ Q) / nd[:, None]
    flog1 = np.repeat(np.log1p(n1.reindex(ids).values.astype(float))[:, None], NK, 1)
    nk3 = np.repeat(np.log1p(H.sum(1, keepdims=True)), NK, 1)
    X = np.stack([meanq[risk], dens[risk], flog1[risk], nk3[risk]], 1)
    return X, y[risk].astype(int)


res = {}
for cc in ['CN', 'EP']:
    d = pd.read_parquet(f'{C}/docs_{cc}.parquet', columns=['did', 'pub', 'k', 'sc'])
    a = pd.read_parquet(f'{C}/app_{cc}.parquet', columns=['did', 'key'])
    d['k_syn'] = synth(d, 2026)
    d['k'] = d.k.astype(np.int64)
    m = a.merge(d, on='did'); m['aid'] = pd.Index(m.key.unique()).get_indexer(m.key)
    for lab in ['k', 'k_syn']:
        Xtr, ytr = build(m, lab, 2023); Xte, yte = build(m, lab, 2024)
        sc = StandardScaler().fit(Xtr); lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xtr), ytr)
        s = lr.decision_function(sc.transform(Xte))
        r = dict(n=int(len(yte)), pos=int(yte.sum()), base=float(yte.mean()), auc_logit=float(roc_auc_score(yte, s)),
                 auc_meanq=float(roc_auc_score(yte, Xte[:, 0])), auc_dens=float(roc_auc_score(yte, Xte[:, 1])))
        res[f'{cc}_{lab}'] = r
        print(cc, lab, r, round(time.time() - t, 1), flush=True)
json.dump(res, open(f'{OUT}/v3_out.json', 'w'), indent=1)
