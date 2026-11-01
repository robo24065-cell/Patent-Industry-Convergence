# verify_0 v3: X1' P0(b) 다른 코드로 재계산·반박 시도.
#  (1) 저장 예측으로 이원 군집 SE를 다항(복원추출) 방식·다른 시드로 다시 계산
#  (2) 다른 학습기(로지스틱 표준화)와 XGB 시드 3개로 ΔAUC(prov−pub)와 SE 재계산
#  (3) 누수 반박: 평가 원점 단위 표본 300개의 피처(b_n3, b_npart, x_msh)를 원자료(meta)에서 독립 재계산해 대조
import sys, json
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
sys.stdout.reconfigure(encoding='utf-8')
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
W = B + '/분석/검증루프_20261012/C13/C13-T4'
C = W + '/cache'
LOCK = 1981.5073339898618
res = {}
# (1)
pr = pd.read_parquet(C + '/s5_pred_2023.parquet')
y = pr.y.values; ks = pr.modal_prov.values
uk, kinv = np.unique(ks, return_inverse=True)
rng = np.random.default_rng(777)


def two_way_se(p1, p2, B_=500):
    d = []
    n = len(y)
    for _ in range(B_):
        wu = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        wk = np.bincount(rng.integers(0, len(uk), len(uk)), minlength=len(uk)).astype(float)[kinv]
        w = wu * wk
        if w[y == 1].sum() == 0:
            continue
        d.append(roc_auc_score(y, p1, sample_weight=w) - roc_auc_score(y, p2, sample_weight=w))
    return float(np.std(d, ddof=1)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


se, ci = two_way_se(pr.p_prov.values, pr.p_hsc.values)
scale = np.sqrt(y.sum() / LOCK)
res['v3_1_multinomial'] = {'SE_dev_two': se, 'ci': ci, 'power_two': float(norm.cdf(0.002 / (se * scale) - 1.96)),
                           'dAUC': float(roc_auc_score(y, pr.p_prov) - roc_auc_score(y, pr.p_hsc))}
print(res['v3_1_multinomial'], flush=True)
# (2)
BASE = ['b_n3', 'b_n1', 'b_gr', 'b_npart', 'b_new3', 'b_new1', 'b_cosh', 'b_nsc', 'b_iscn']
SX = ['x_nk', 'x_ent', 'x_msh', 'x_xpart', 'x_xpart_na', 'x_xnew3', 'x_indr', 'x_indn', 'x_imp', 'x_imp_na', 'x_modal_na']
T = list(range(2016, 2024))
L = {s: pd.concat([pd.read_parquet(f'{C}/u_{s}_{t}.parquet').assign(t0=t) for t in T], ignore_index=True) for s in ['prov', 'free', 'hsc']}
P = L['prov']; yy = P.y.values; tr = (P.t0 < 2023).values; ev = (P.t0 == 2023).values
assert (P.u[ev].values == pr.u.values).all()
out = {}
for name, mk in [('logit', lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
                 ('xgb_s1', lambda s: xgb.XGBClassifier(tree_method='hist', max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=1)),
                 ('xgb_s2', lambda s: xgb.XGBClassifier(tree_method='hist', max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=2)),
                 ('xgb_s3', lambda s: xgb.XGBClassifier(tree_method='hist', max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=3))]:
    pp = {}
    for s in ['prov', 'free', 'hsc']:
        X = np.c_[P[BASE].values, L[s][SX].values]
        m = mk(s); m.fit(X[tr], yy[tr]); pp[s] = m.predict_proba(X[ev])[:, 1]
    auc = {s: float(roc_auc_score(yy[ev], v)) for s, v in pp.items()}
    pub = 'free' if auc['free'] >= auc['hsc'] else 'hsc'
    se2, ci2 = two_way_se(pp['prov'], pp[pub], 300)
    out[name] = {'auc': auc, 'pub': pub, 'dAUC': auc['prov'] - auc[pub], 'SE_dev_two': se2, 'ci': ci2,
                 'power_two': float(norm.cdf(0.002 / (se2 * scale) - 1.96))}
    print(name, out[name], flush=True)
res['v3_2_models'] = out
# (3) 누수 반박: 표본 300 단위 피처 독립 재계산(원자료)
import duckdb, re
sys.path.insert(0, B + '/분석/검증루프_20261012/C12/C12-T4/scripts')
from lib12 import norm_key
smp = P[ev].sample(300, random_state=5)
con = duckdb.connect(); con.execute('set threads=2')
mism = {'b_n3': 0, 'x_msh': 0}
detail = []
for cc in ['CN', 'EP']:
    s = smp[smp.cc == cc]
    if len(s) == 0:
        continue
    keys = set(s.u.str[3:])
    d = con.execute(f"""select appl_no, min(cast(pub_date as integer)) pub, any_value(ksic) k, any_value(applicant) ap
                        from read_parquet('{B}/분석/모델경쟁_20260927/_shared/meta/*.parquet')
                        where cc='{cc}' and pub_date > '20181231' and pub_date <= '20231231' and length(pub_date)=8 group by appl_no""").fetchdf()
    assert d.pub.max() <= 20231231
    rec = []
    for an, pub, k, ap in zip(d.appl_no, d.pub, d.k, d.ap):
        if ap is None:
            continue
        for nm in re.split(r'[|;]', ap):
            nm = nm.strip()
            if nm:
                kk = norm_key(nm)
                if kk in keys:
                    rec.append((kk, an, pub, k))
    r = pd.DataFrame(rec, columns=['key', 'an', 'pub', 'k']).drop_duplicates(['key', 'an'])
    for _, row in s.iterrows():
        g = r[r.key == row.u[3:]]
        n3 = g[g.pub > 20201231].an.nunique()
        vc = g.k.value_counts()
        msh = vc.max() / vc.sum() if len(vc) else 0
        ok1 = abs(np.log1p(n3) - row.b_n3) < 1e-9; ok2 = abs(msh - row.x_msh) < 1e-9
        mism['b_n3'] += (not ok1); mism['x_msh'] += (not ok2)
        if not (ok1 and ok2) and len(detail) < 10:
            detail.append({'u': row.u, 'n3_raw': int(n3), 'b_n3': float(np.expm1(row.b_n3)), 'msh_raw': float(msh), 'x_msh': float(row.x_msh)})
res['v3_3_feature_recalc'] = {'n_sample': 300, 'mismatch': mism, 'detail': detail,
                              'note': '원자료는 공개일 ≤ 2023-12-31만 읽음(평가 원점 t0=2023). 조직 판정 없이 키 일치만 보므로 같은 키의 비조직 이름이 섞이면 차이가 날 수 있음.'}
print(res['v3_3_feature_recalc'], flush=True)
json.dump(res, open(W + '/verify_0/v3_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
