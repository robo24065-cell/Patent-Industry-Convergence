# C14-T4 a1: 구조 확인 SE용 부트스트랩(PREREG §1.5). 인자: kind ∈ firm | sc | cell | prod
# 원점 2020–24 저장 예측(C12 캐시, 읽기만). 군집 색인은 원점 공통(같은 기업·서브클래스·칸 가중을 모든 원점에 공유).
# 시드: 기업 SEED+1, 서브클래스 SEED+2, 칸 SEED+3 (prod = 기업 × 서브클래스, 같은 추출). B=400.
# 출력 cache/a1_{kind}.npz : auc[b, v, m] (m = prov, free, hsc, all, pubctx)
import sys, os, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
kind = sys.argv[1]
Bn = 400
MODELS = ['prov', 'free', 'hsc', 'all', 'pubctx']
t = time.time()
fr = load_c12_frames(models=tuple(MODELS))
fk = [f.cc.astype(str) + ':' + f.aid.astype(str) for f in fr]
fidx = pd.Index(pd.concat(fk).unique())
sidx = pd.Index(np.unique(np.concatenate([f.s.values for f in fr])))
ck = [k + ':' + f.s.astype(str) for k, f in zip(fk, fr)]
cidx = pd.Index(pd.concat(ck).unique())
maps = [(fidx.get_indexer(a), sidx.get_indexer(f.s.values), cidx.get_indexer(c)) for a, f, c in zip(fk, fr, ck)]
WF = np.random.default_rng(SEED + 1).poisson(1.0, (Bn, len(fidx))).astype(np.float32) if kind in ('firm', 'prod') else None
WS = np.random.default_rng(SEED + 2).poisson(1.0, (Bn, len(sidx))).astype(np.float32) if kind in ('sc', 'prod') else None
WC = np.random.default_rng(SEED + 3).poisson(1.0, (Bn, len(cidx))).astype(np.float32) if kind == 'cell' else None
preps = [[AUCPrep(f[m].values, f.y.values) for m in MODELS] for f in fr]
A = np.zeros((Bn, len(fr), len(MODELS)))
for b in range(Bn):
    for k, f in enumerate(fr):
        fi, si, ci = maps[k]
        if kind == 'firm':
            g = WF[b, fi]
        elif kind == 'sc':
            g = WS[b, si]
        elif kind == 'cell':
            g = WC[b, ci]
        else:
            g = WF[b, fi] * WS[b, si]
        w = f.w.values * g
        for j in range(len(MODELS)):
            A[b, k, j] = preps[k][j].auc(w)
    if b % 50 == 0:
        out(kind, b, round(time.time() - t, 1))
    assert time.time() - t < 530, 'time guard'
pt = np.array([[preps[k][j].auc(f.w.values.astype(np.float64)) for j in range(len(MODELS))] for k, f in enumerate(fr)])
np.savez(C + rf'\a1_{kind}.npz', auc=A, pt=pt, models=np.array(MODELS), npos=np.array([int(f.y.sum()) for f in fr]),
         nf=len(fidx), ns=len(sidx), nc=len(cidx))
out('done', kind, round(time.time() - t, 1), 'clusters', len(fidx), len(sidx), len(cidx))
