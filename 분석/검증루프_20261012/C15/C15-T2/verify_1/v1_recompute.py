# verify_1: C15-T2 발견 반박 시도 - 캐시 예측에서 AUC/skill/순위/sd/SE를 독립 재계산 (읽기 전용)
import sys, json, glob, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
B = sys.argv[1]; OUT = sys.argv[2]
def auc(y, s):
    y = np.asarray(y).astype(bool); r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1*(n1+1)/2) / (n1*n0)
sk = lambda a, a0: (a - a0) / (1 - a0)
res = {}
# ---------- T1 ----------
lab = pd.read_parquet(f'{B}/cache/t1/final_labels.parquet')
t1 = []
for b in range(10):
    d = pd.read_parquet(f'{B}/cache/t1/scores_b{b}.parquet')
    y = d['y'].values
    a = {c: auc(y, d[c].values) for c in ['s', 'xgb_y', 'xgb_y_u', 'sbb']}
    # logit 기준선(결정적) 포함 6변형 최강 확인
    base = pd.read_parquet(f'{B}/cache/t1/base_b{b}.parquet')
    base = pd.concat([lab[['firm', 'k']], base], axis=1)
    m = d[['firm', 'k']].merge(base, on=['firm', 'k'], how='left')
    assert len(m) == len(d) and m.isna().sum().sum() == 0
    ab = {c: auc(y, m[c].values) for c in base.columns if c not in ('firm', 'k')}
    m0n = max(ab, key=ab.get); a0 = ab[m0n]
    # boot
    bm = np.vstack([np.load(f)['auc'] for f in sorted(glob.glob(f'{B}/cache/t1/boot/b{b}_m0_*.npz'))])
    bs = np.vstack([np.load(f)['auc'] for f in sorted(glob.glob(f'{B}/cache/t1/boot/b{b}_sbb_*.npz'))])
    m0col = 1 if m0n == 'xgb_y' else 2
    bsk = (bm[:, 0] - bm[:, m0col]) / (1 - bm[:, m0col])
    bss = (bs[:, 0] - bs[:, 1]) / (1 - bs[:, 1])
    o = np.argsort(-d['s'].values, kind='stable')
    t1.append(dict(b=b, n=len(y), pos=int(y.sum()), auc=a['s'], aucM0=a0, M0=m0n, auc_xgb_y=a['xgb_y'], auc_xgb_y_u=a['xgb_y_u'],
                   skill=sk(a['s'], a0), auc_sbb=a['sbb'], skill_sbb=sk(a['s'], a['sbb']),
                   nboot=len(bm), se=bsk.std(ddof=1), lo=np.percentile(bsk, 2.5), nboot_sbb=len(bs), se_sbb=bss.std(ddof=1), lo_sbb=np.percentile(bss, 2.5),
                   top100=int(y[o[:100]].sum()), top1000=int(y[o[:1000]].sum()), logit_max=max(v for k, v in ab.items() if 'logit' in k)))
T1 = pd.DataFrame(t1)
# ---------- T3 ----------
t3 = []; rng = np.random.default_rng(20261017)
for b in range(10):
    d = pd.read_parquet(f'{B}/cache/t3/preds_b{b}_final.parquet')
    y = d['yb'].values
    a = auc(y, d['p']); a0 = auc(y, d['p_base']); asb = auc(y, d['SB3'])
    o = np.argsort(-d['p'].values, kind='stable')
    t3.append(dict(b=b, n=len(y), pos=int(y.sum()), y_eq_yb=bool((d['y'].values == y).all()), auc=a, aucM0=a0, skill=sk(a, a0), auc_sb3=asb, skill_sb3=sk(a, asb),
                   top100=int(y[o[:100]].sum()), top10=int(y[o[:int(round(0.1*len(y)))]].sum())))
T3 = pd.DataFrame(t3)
# T3 군집(mg) 부트스트랩 SE, 공식 묶음(b0), 자체 난수 1000회
d = pd.read_parquet(f'{B}/cache/t3/preds_b0_final.parquet'); y = d['yb'].values
g = d['mg'].values; ug, inv = np.unique(g, return_inverse=True)
idx_by = [np.where(inv == i)[0] for i in range(len(ug))]
bk, bk2 = [], []
for _ in range(1000):
    pick = rng.integers(0, len(ug), len(ug)); ii = np.concatenate([idx_by[i] for i in pick])
    yy = y[ii]
    if yy.sum() in (0, len(yy)): continue
    a = auc(yy, d['p'].values[ii]); bk.append(sk(a, auc(yy, d['p_base'].values[ii]))); bk2.append(sk(a, auc(yy, d['SB3'].values[ii])))
res['T3_boot_own'] = dict(n_clusters=len(ug), se_skill=float(np.std(bk, ddof=1)), se_sb3=float(np.std(bk2, ddof=1)), lo=float(np.percentile(bk, 2.5)), lo_sb3=float(np.percentile(bk2, 2.5)))
def summ(df, cols):
    out = {}
    for c in cols:
        v = df[c].values; off = v[0]
        out[c] = dict(official=float(off), mean=float(v.mean()), sd1=float(v.std(ddof=1)), sd0=float(v.std(ddof=0)), min=float(v.min()), max=float(v.max()),
                      rank_desc=int((v > off).sum() + 1), n_eq=int((v == off).sum()), range=float(v.max() - v.min()))
    return out
res['T1'] = summ(T1, ['auc', 'aucM0', 'skill', 'lo', 'skill_sbb', 'lo_sbb', 'top100', 'top1000', 'auc_sbb'])
res['T3'] = summ(T3, ['auc', 'aucM0', 'skill', 'skill_sb3', 'auc_sb3', 'top100', 'top10'])
res['T1']['ratio_sd_se'] = res['T1']['skill']['sd1'] / T1['se'][0]
res['T1']['ratio_sbb_sd_se'] = res['T1']['skill_sbb']['sd1'] / T1['se_sbb'][0]
res['T1']['se_official'] = float(T1['se'][0]); res['T1']['se_sbb_official'] = float(T1['se_sbb'][0])
res['T3']['ratio_sd_se_own'] = res['T3']['skill']['sd1'] / res['T3_boot_own']['se_skill']
res['T3']['ratio_sb3_sd_se_own'] = res['T3']['skill_sb3']['sd1'] / res['T3_boot_own']['se_sb3']
# 작업 산출과의 대조
sbd = pd.read_csv(f'{B}/seed_bundles.csv')
s1 = sbd[sbd.track == 'T1'].reset_index(drop=True); s3 = sbd[sbd.track == 'T3'].reset_index(drop=True)
res['maxdiff_vs_seed_bundles'] = dict(
    T1_auc=float(np.abs(T1.auc - s1.auc_champ).max()), T1_aucM0=float(np.abs(T1.aucM0 - s1.auc_M0).max()), T1_skill=float(np.abs(T1.skill - s1.skill).max()),
    T1_M0name_eq=bool((T1.M0 == s1.M0_name).all()), T1_lo=float(np.abs(T1.lo - s1.ci_lo).max()), T1_sbb=float(np.abs(T1.skill_sbb - s1.skill_vs_strong).max()),
    T1_lo_sbb=float(np.abs(T1.lo_sbb - s1.strong_ci_lo).max()), T1_top100=int(np.abs(T1.top100 - s1.top100).max()), T1_top1000=int(np.abs(T1.top1000 - s1.topk2).max()),
    T3_auc=float(np.abs(T3.auc - s3.auc_champ).max()), T3_skill=float(np.abs(T3.skill - s3.skill).max()), T3_sb3=float(np.abs(T3.skill_sb3 - s3.skill_vs_strong).max()),
    T3_lo_csv_rank_official=int((s3.ci_lo.values < s3.ci_lo.values[0]).sum()), T3_strong_lo_csv_rank_official=int((s3.strong_ci_lo.values < s3.strong_ci_lo.values[0]).sum()),
    T3_lo_range_csv=float(s3.ci_lo.max() - s3.ci_lo.min()), T1_lo_range_recalc=float(T1.lo.max() - T1.lo.min()),
    T3_top100=int(np.abs(T3.top100 - s3.top100).max()), T3_top10=int(np.abs(T3.top10 - s3.topk2).max()))
T1.to_csv(f'{OUT}/v1_T1_recalc.csv', index=False); T3.to_csv(f'{OUT}/v1_T3_recalc.csv', index=False)
json.dump(res, open(f'{OUT}/v1_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print(T1.to_string()); print(T3.to_string())
