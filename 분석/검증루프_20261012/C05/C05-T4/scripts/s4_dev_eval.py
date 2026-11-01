# s4: 개발 전진 평가 집계 + 기업키 군집 부트스트랩(1,000회, 시드 20261012; 6개 평가 원점에 같은 기업키 가중) + 챔피언 기계 선택.
# 사용법: python s4_dev_eval.py boot <시작> <끝>  (부트스트랩 조각 저장) / python s4_dev_eval.py final (집계)
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
from libB import Prep, boot_weights
MODELS = ['M0', 'M1', 'M1_pub_TXT', 'M1_pub_IPC']
NB = 1000; SEED = 20261012
T0 = time.time()
P = {}
for v in EVAL:
    base = None
    for m in MODELS:
        d = pd.read_parquet(os.path.join(C, f'pred_{v}_{m}.parquet'))
        if base is None:
            base = d[['key', 'y']]
        else:
            assert (base.key.values == d.key.values).all() and (base.y.values == d.y.values).all()
        P[(v, m)] = d.p.values
    P[(v, 'key')] = base.key.values; P[(v, 'y')] = base.y.values
allkeys = pd.Index(sorted(set(np.concatenate([P[(v, 'key')] for v in EVAL]))))
codes = {v: allkeys.get_indexer(P[(v, 'key')]) for v in EVAL}
preps = {(v, m): Prep(P[(v, 'y')], P[(v, m)]) for v in EVAL for m in MODELS}


def metrics(wfun):
    r = {}
    for v in EVAL:
        w = wfun(v)
        for m in MODELS:
            pr = preps[(v, m)]
            a = pr.auc(w); ap, top = pr.ap_top(w)
            r[(v, m, 'auc')] = a; r[(v, m, 'ap')] = ap; r[(v, m, 'top1000')] = top
    return r


mode = sys.argv[1]
if mode == 'boot':
    b0, b1 = int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(b1):
        draw = rng.integers(0, len(allkeys), len(allkeys))  # 시드 고정 순서를 지키기 위해 앞 조각도 난수는 뽑는다
        if b < b0:
            continue
        cnt = np.bincount(draw, minlength=len(allkeys)).astype(float)
        r = metrics(lambda v: cnt[codes[v]])
        rows.append({'b': b, **{f'{k[0]}|{k[1]}|{k[2]}': val for k, val in r.items()}})
        if b % 100 == 0:
            print(b, round(time.time() - T0), flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(C, f'dev_boot_{b0}_{b1}.parquet'), index=False)
    print('saved', b0, b1, round(time.time() - T0))
    sys.exit(0)

# ---- final
pt = metrics(lambda v: np.ones(len(codes[v])))
BS = pd.concat([pd.read_parquet(os.path.join(C, f)) for f in sorted(os.listdir(C)) if f.startswith('dev_boot_')], ignore_index=True)
assert BS.b.nunique() == NB and len(BS) == NB, (BS.b.nunique(), len(BS))


def col(v, m, k):
    return BS[f'{v}|{m}|{k}'].values


def ci(x):
    return [round(float(np.quantile(x, 0.025)), 5), round(float(np.quantile(x, 0.975)), 5)]


def skill(a, a0):
    return (a - a0) / (1 - a0)


rows = []
for v in EVAL:
    for m in MODELS:
        a = pt[(v, m, 'auc')]; a0 = pt[(v, 'M0', 'auc')]
        rows.append(dict(origin=v, model=m, n=len(codes[v]), pos=int(P[(v, 'y')].sum()), auc=round(a, 5), auc_ci=ci(col(v, m, 'auc')),
                         skill_vs_M0=round(skill(a, a0), 5), skill_ci=ci(skill(col(v, m, 'auc'), col(v, 'M0', 'auc'))),
                         pr_auc=round(pt[(v, m, 'ap')], 5), pr_auc_ci=ci(col(v, m, 'ap')), top1000=round(pt[(v, m, 'top1000')], 5),
                         top1000_ci=ci(col(v, m, 'top1000'))))
# 원점 평균
summ = {}
for m in MODELS:
    ma = np.mean([pt[(v, m, 'auc')] for v in EVAL]); mab = np.mean([col(v, m, 'auc') for v in EVAL], axis=0)
    ds = [skill(pt[(v, m, 'auc')], pt[(v, 'M0', 'auc')]) for v in EVAL]
    dsb = np.mean([skill(col(v, m, 'auc'), col(v, 'M0', 'auc')) for v in EVAL], axis=0)
    rows.append(dict(origin='MEAN6', model=m, auc=round(ma, 5), auc_ci=ci(mab), skill_vs_M0=round(float(np.mean(ds)), 5), skill_ci=ci(dsb),
                     n_pos_origins=int(sum(d > 0 for d in ds)),
                     pr_auc=round(float(np.mean([pt[(v, m, 'ap')] for v in EVAL])), 5), pr_auc_ci=ci(np.mean([col(v, m, 'ap') for v in EVAL], axis=0)),
                     top1000=round(float(np.mean([pt[(v, m, 'top1000')] for v in EVAL])), 5),
                     top1000_ci=ci(np.mean([col(v, m, 'top1000') for v in EVAL], axis=0))))
    summ[m] = dict(mean_dskill=float(np.mean(ds)), per_origin=[round(d, 5) for d in ds], n_pos=int(sum(d > 0 for d in ds)), ci=ci(dsb))
# 필수성: M1 대비 M1_pub
ess = {}
for pm in ['M1_pub_TXT', 'M1_pub_IPC']:
    ds = [skill(pt[(v, 'M1', 'auc')], pt[(v, pm, 'auc')]) for v in EVAL]
    dsb = np.mean([skill(col(v, 'M1', 'auc'), col(v, pm, 'auc')) for v in EVAL], axis=0)
    ess[pm] = dict(mean_dskill_M1_minus_pub=round(float(np.mean(ds)), 5), ci=ci(dsb), per_origin=[round(d, 5) for d in ds], n_pos=int(sum(d > 0 for d in ds)))
    rows.append(dict(origin='MEAN6', model=f'M1_vs_{pm}', skill_vs_M0=None, skill_ci=None, dskill_M1_vs_pub=ess[pm]['mean_dskill_M1_minus_pub'],
                     dskill_ci=ess[pm]['ci'], n_pos_origins=ess[pm]['n_pos']))
s1 = summ['M1']
adopt = (s1['mean_dskill'] >= 0.003) and (s1['n_pos'] >= int(np.ceil(0.6 * len(EVAL))))
champ = 'M1' if adopt else 'M0'
rule = dict(rule='M1 채택 = 평균 Δskill ≥ +0.003 이고 Δ>0 원점 ≥ 60%(6개 중 4개)', M1_mean_dskill=round(s1['mean_dskill'], 5), M1_n_pos=s1['n_pos'],
            M1_dskill_ci=s1['ci'], adopt_M1=bool(adopt), champion=champ, essentiality=ess)
df = pd.DataFrame(rows)
for c in df.columns:
    if df[c].map(lambda x: isinstance(x, list)).any():
        df[c] = df[c].map(lambda x: json.dumps(x) if isinstance(x, list) else x)
df.to_csv(os.path.join(W, 'dev_eval.csv'), index=False, encoding='utf-8-sig')
json.dump(rule, open(os.path.join(C, 'champion_rule.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(rule, ensure_ascii=False, indent=1))
print(df[['origin', 'model', 'pos', 'auc', 'skill_vs_M0', 'pr_auc', 'top1000']].to_string())
