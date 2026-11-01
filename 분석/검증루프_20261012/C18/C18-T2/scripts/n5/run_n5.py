# C18-T2 N5 변형 재학습·채점 (final_run_r4.mode_final 사본; 공식 파일에는 쓰지 않음). 사용: python run_n5.py P0|V1|V1E
import os, sys, json, time
V = sys.argv[1]
W = os.environ['C18_W']
os.environ['C18_N5CACHE'] = os.path.join(W, 'cache', f'n5_{V}')
os.environ['C18_META'] = (r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet' if V == 'P0'
                          else os.path.join(W, 'cache', f'meta_{V}.parquet'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c18n5lib import *
SPEC = json.load(open(os.path.join(OUT, "final_spec_r4.json"), encoding="utf-8"))
FE = SPEC["champion_features"]; F0 = SPEC["baseline_M0"]
SEEDS = tuple(SPEC["seeds"])
t = time.time()


def panels(t0s, mode):
    k = load_kr(KS_MAIN); ex = load_family(k, KS_MAIN); E = load_ext(KS_MAIN)
    return pd.concat([build_panel(t0, KS_MAIN, mode, k, ex, E) for t0 in t0s], ignore_index=True)


def fp(tr, te, feats):
    return fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), model="xgb", seeds=SEEDS)


tr = panels(range(2012, 2021), "dev"); assert tr.t0.max() + 2 < F_YEAR
te = panels([T0_FINAL], "final")
pc, p0 = fp(tr, te, FE), fp(tr, te, F0)
y = te.y.values
A = dict(champ=float(auc(y, pc)), M0=float(auc(y, p0)))
(lo, hi), bm = cluster_boot_ci(y, pc, p0, te.app.values, n=1000, seed=2022)
R = dict(var=V, n=int(len(te)), pos=int(y.sum()), auc=A, skill=skill(A['champ'], A['M0']), ci95=[float(lo), float(hi)],
         n_train=int(len(tr)), fam_n_train_mean=float(tr.fam_n.mean()), fam_n_test_mean=float(te.fam_n.mean()), secs=round(time.time() - t))
D = os.path.join(W, 'cache', 'n5res'); os.makedirs(D, exist_ok=True)
pd.DataFrame({'appl_no': te.appl_no.values, 'app': te.app.values, 'y': y, 'p_champ': pc, 'p_M0': p0, 'fam_n': te.fam_n.values}).to_parquet(os.path.join(D, f'preds_{V}.parquet'), index=False)
tr[['appl_no', 't0', 'fam_n', 'y']].to_parquet(os.path.join(D, f'train_fam_{V}.parquet'), index=False)
json.dump(R, open(os.path.join(D, f'res_{V}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False))
