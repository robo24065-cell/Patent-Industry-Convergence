# T3 레드팀 재검증(PROTOCOL_R5 §3) — 위약·민감도·상위 k·보정. 동결 예측/캐시는 읽기만, 산출은 VERIFY_R5/ 아래.
# 사용: python sens_t3.py pub   (R3 공개연도 E40-auditD, t0=2021)
#       python sens_t3.py app   (R2 출원연도 E57-audit, t0=2019·2020)
import os, sys, json, hashlib, time
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
VAR = sys.argv[1]
if VAR == 'pub':
    PRED = os.path.join(T3, 'R3', 'preds', 'FINAL_R3_t0_2021.parquet'); DEV = os.path.join(T3, 'R3', 'preds', 'dev_preds_r3.parquet')
    CACHE = os.path.join(T3, 'R3', 'cache_pub_final'); TEST = [2021]; TRAIN = list(range(2005, 2019)); TAG = 'R3pub'
    RERUN = {'새 패널 재빌드': os.path.join(HERE, 'preds', 'RERUN_FINAL_t0_2021.parquet'), '동결 패널+새 적합': os.path.join(HERE, 'preds', 'RERUN_ORIGCACHE_FINAL_t0_2021.parquet')}
else:
    PRED = os.path.join(T3, 'R2', 'preds', 'FINAL_t0_2019_2020.parquet'); DEV = os.path.join(T3, 'R2', 'preds', 'dev_preds.parquet')
    CACHE = os.path.join(T3, 'R2', 'cache_2023'); TEST = [2019, 2020]; TRAIN = list(range(2005, 2017)); TAG = 'R2app'
    RERUN = {'새 패널 재빌드': os.path.join(HERE, 'preds', 'R2_RERUN_REBUILT_FINAL.parquet'), '동결 패널+새 적합': os.path.join(HERE, 'preds', 'R2_RERUN_ORIGCACHE_FINAL.parquet')}
OUT = {}; t_start = time.time()
P = pd.read_parquet(PRED)
sk = lambda a, b: (a - b) / (1 - b)

def auc(y, s): return float(roc_auc_score(y, s))
def block_metrics(d, col):
    """원점별 AUC 평균과 skill(공식과 같은 '원점별 skill 평균')"""
    out = []
    for t, g in d.groupby('t0'):
        if g.yb.nunique() < 2: continue
        out.append((auc(g.yb, g[col]), auc(g.yb, g.p_base)))
    am = np.mean([a for a, _ in out]); ab = np.mean([b for _, b in out])
    return dict(auc=am, auc_base=ab, skill=float(np.mean([sk(a, b) for a, b in out])), n=int(len(d)), n_evt=int(d.yb.sum()))

# ---------------- 0) 재현: 재실행 예측 대조 ----------------
rep = {}
for nm, f in RERUN.items():
    if not os.path.exists(f): continue
    R = pd.read_parquet(f); m = P.merge(R, on=['cell', 't0'], suffixes=('', '_r'))
    rep[nm] = dict(n=int(len(R)), n_merge=int(len(m)), yb_mismatch=int((m.yb != m.yb_r).sum()), y_maxdiff=float(np.abs(m.y - m.y_r).max()),
                   p_base_maxdiff=float(np.abs(m.p_base - m.p_base_r).max()), p_maxdiff=float(np.abs(m.p - m.p_r).max()),
                   p_rank_corr=float(pd.Series(m.p).corr(pd.Series(m.p_r), method='spearman')),
                   skill_rerun=block_metrics(m.assign(p=m.p_r), 'p')['skill'])
OUT['reproduction_preds'] = rep
OUT['official_recomputed'] = block_metrics(P, 'p')
print('공식 재계산', OUT['official_recomputed']); print('재실행 대조', rep, flush=True)

# ---------------- 1) 위약: 층 안에서 레이블 섞기 ----------------
def kbin(x): return np.where(x == 0, 0, np.where(x <= 4, 1, 2))
need_kr = 'kr_W' not in P
F = {}
for t in sorted(set(TRAIN + TEST)):
    F[t] = pd.read_parquet(os.path.join(CACHE, f'feat_{t}.parquet'))
if need_kr:
    kw = pd.concat([F[t][['cell', 't0', 'kr_W', 'tot_W']] for t in TEST]); P = P.merge(kw, on=['cell', 't0'], how='left')
else:
    kw = pd.concat([F[t][['cell', 't0', 'tot_W']] for t in TEST]); P = P.merge(kw, on=['cell', 't0'], how='left')
assert P.kr_W.notna().all()
P['kbin'] = kbin(P.kr_W.values); P['stratum'] = P.t0.astype(str) + '|' + P.ksic + '|' + P.kbin.astype(str)
pl = {'층내(원점×KSIC×KR규모3구간)': [], '전체 무작위': []}
for s in range(200):
    rng = np.random.default_rng(1000 + s)
    d = P.copy(); d['yb'] = d.groupby('stratum').yb.transform(lambda v: rng.permutation(v.values)); pl['층내(원점×KSIC×KR규모3구간)'].append(block_metrics(d, 'p'))
    d = P.copy(); d['yb'] = d.groupby('t0').yb.transform(lambda v: rng.permutation(v.values)); pl['전체 무작위'].append(block_metrics(d, 'p'))
OUT['placebo'] = {k: dict(n_seeds=len(v), skill_mean=float(np.mean([x['skill'] for x in v])), skill_sd=float(np.std([x['skill'] for x in v])),
                          skill_q025=float(np.percentile([x['skill'] for x in v], 2.5)), skill_q975=float(np.percentile([x['skill'] for x in v], 97.5)),
                          auc_mean=float(np.mean([x['auc'] for x in v])), auc_base_mean=float(np.mean([x['auc_base'] for x in v])),
                          frac_ge_official=float(np.mean([x['skill'] >= OUT['official_recomputed']['skill'] for x in v]))) for k, v in pl.items()}
print('위약', json.dumps(OUT['placebo'], ensure_ascii=False), flush=True)

# ---------------- 2) 강화 기준선(같은 학습 원점, 같은 레이블 정의) ----------------
def hkey(cells): return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in cells], dtype=np.int64)
def label_block(g):
    k = max(1, int(round(0.10 * len(g)))); o = np.lexsort((hkey(g.cell.values), -(g.kr_fut - g.kr_W).values, -g.y.values))
    yb = np.zeros(len(g), int); yb[o[:k]] = 1; return yb
TR = []
for t in TRAIN:
    g = F[t][F[t].tot_W >= 50].reset_index(drop=True); g['yb'] = label_block(g); TR.append(g)
TR = pd.concat(TR, ignore_index=True)
TE = pd.concat([F[t][F[t].tot_W >= 50] for t in TEST], ignore_index=True)
TE = TE.merge(P[['cell', 't0', 'p', 'p_base', 'yb']], on=['cell', 't0'], how='inner')
assert len(TE) == len(P), (len(TE), len(P))
SB = {'SB1 로지스틱 5피처(KR모멘텀·KR규모·해외모멘텀·전체규모·KR점유)': ('logit', ['kr_mom', 'kr_lvl', 'fx_mom', 'tot_lvl', 'kr_share']),
      'SB2 로지스틱 18피처(국가별 모멘텀·규모·KR 이력)': ('logit', ['kr_mom', 'kr_mom_l1', 'kr_mom_l2', 'kr_g1', 'kr_lvl', 'kr_last', 'kr_slope5', 'kr_zero_yrs', 'fx_mom', 'fx_mom_l1',
                                                       'US_mom', 'CN_mom', 'JP_mom', 'EP_mom', 'tot_lvl', 'tot_mom', 'kr_share', 'gap_fx_kr']),
      'SB3 XGB KR 이력 12피처(해외 신호 없음)': ('xgb', ['kr_mom', 'kr_mom_l1', 'kr_mom_l2', 'kr_mom5', 'kr_g1', 'kr_acc', 'kr_lvl', 'kr_last', 'kr_slope5', 'kr_zero_yrs', 'kr_share', 'kr_share_chg'])}
for nm, (kind, fs) in SB.items():
    if kind == 'logit':
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(TR[fs].values, TR.yb.values)
        TE[nm] = m.predict_proba(TE[fs].values)[:, 1]
    else:
        bst = xgb.train(dict(objective='binary:logistic', max_depth=2, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, nthread=2, verbosity=0, seed=0),
                        xgb.DMatrix(TR[fs].values, label=TR.yb.values), 450)
        TE[nm] = bst.predict(xgb.DMatrix(TE[fs].values))
# SB0: 층 평균 부상률(학습 원점의 KSIC×KR규모3구간 부상률, 표본 적으면 KR규모 구간 평균으로 축소)
TR['kbin'] = kbin(TR.kr_W.values); TE['kbin'] = kbin(TE.kr_W.values)
g1 = TR.groupby('kbin').yb.mean(); g2 = TR.groupby(['ksic', 'kbin']).yb.agg(['sum', 'count'])
pri = 20.0
r = (g2['sum'] + pri * g1.reindex(g2.index.get_level_values(1)).values) / (g2['count'] + pri)
SB0 = 'SB0 층 평균 부상률(KSIC×KR규모, 학습 원점)'
TE[SB0] = [r.get((k, b), g1[b]) for k, b in zip(TE.ksic, TE.kbin)]
SB = {SB0: ('const', []), **SB}
sbres = {}
for nm in SB:
    rows = []
    for t, g in TE.groupby('t0'):
        a_m, a_b, a_s = auc(g.yb, g.p), auc(g.yb, g.p_base), auc(g.yb, g[nm]); rows.append((a_m, a_b, a_s))
    am, ab, as_ = [np.mean([r[i] for r in rows]) for i in range(3)]
    sbres[nm] = dict(auc_sb=float(as_), auc_model=float(am), skill_vs_sb=float(np.mean([sk(r[0], r[2]) for r in rows])), sb_skill_vs_official_base=float(np.mean([sk(r[2], r[1]) for r in rows])))
strongest = TE[list(SB)].copy()
OUT['strong_baselines'] = sbres
pl2 = []
for s_ in range(200):
    rng = np.random.default_rng(5000 + s_); d = TE.copy(); d['stratum'] = d.t0.astype(str) + '|' + d.ksic + '|' + d.kbin.astype(str)
    d['yb'] = d.groupby('stratum').yb.transform(lambda v: rng.permutation(v.values))
    pl2.append(np.mean([sk(auc(g.yb, g.p), auc(g.yb, g[SB0])) for _, g in d.groupby('t0')]))
OUT['placebo']['층내 섞기 — SB0(층 평균) 대비 skill'] = dict(n_seeds=200, skill_mean=float(np.mean(pl2)), skill_sd=float(np.std(pl2)),
    skill_q025=float(np.percentile(pl2, 2.5)), skill_q975=float(np.percentile(pl2, 97.5)))
print('위약2(SB0 대비)', OUT['placebo']['층내 섞기 — SB0(층 평균) 대비 skill'], flush=True)
best = max(sbres, key=lambda k: sbres[k]['auc_sb'])
OUT['strong_baseline_best'] = best
print('강화 기준선', json.dumps(sbres, ensure_ascii=False, indent=0), flush=True)

# ---------------- 3) 하위집단 ----------------
sub = {}
def sub_eval(mask, name):
    d = TE[mask]
    if d.groupby('t0').yb.nunique().min() < 2 or d.yb.sum() < 10: sub[name] = dict(n=int(len(d)), n_evt=int(d.yb.sum()), note='사건 부족'); return
    r = block_metrics(d, 'p'); r['auc_best_sb'] = float(np.mean([auc(g.yb, g[best]) for _, g in d.groupby('t0')]))
    r['skill_vs_best_sb'] = float(np.mean([sk(auc(g.yb, g.p), auc(g.yb, g[best])) for _, g in d.groupby('t0')])); r['evt_rate'] = float(d.yb.mean()); sub[name] = r
TE['kbin'] = kbin(TE.kr_W.values)
for b, nm in [(0, 'KR 과거3년 0건'), (1, 'KR 1–4건'), (2, 'KR ≥5건')]: sub_eval(TE.kbin.values == b, nm)
q = TE.groupby('t0').tot_W.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False))
for b, nm in [(0, '5개국 규모 하위 1/3'), (1, '규모 중간 1/3'), (2, '규모 상위 1/3')]: sub_eval(q.values == b, nm)
NEW_SUB = ('G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H')
NEW_MG = ('H04N 23', 'H04N 25', 'G06F 18')
newc = TE.mg.str[:4].isin(NEW_SUB) | TE.mg.isin(NEW_MG)
sub_eval(~newc.values, f'IPC 2022년 이후 신설 서브클래스·메인그룹 셀 제외(신설 {int(newc.sum())}셀)')
sub_eval(TE.ksic.str[:2].values == '26', 'KSIC 26(전자부품·통신)'); sub_eval(TE.ksic.str[:2].values != '26', 'KSIC 26 이외')
# KSIC별
kk = []
for (t, k), g in TE.groupby(['t0', 'ksic']):
    if g.yb.sum() >= 5 and g.yb.sum() < len(g): kk.append(dict(t0=t, ksic=k, n=len(g), evt=int(g.yb.sum()), auc=auc(g.yb, g.p), auc_base=auc(g.yb, g.p_base)))
kk = pd.DataFrame(kk); kk.to_csv(os.path.join(HERE, f'sens_{TAG}_by_ksic.csv'), index=False, encoding='utf-8-sig')
sub['KSIC별(부상≥5)'] = dict(n_ksic_blocks=int(len(kk)), frac_model_gt_base=float((kk.auc > kk.auc_base).mean()), median_auc=float(kk.auc.median()), median_auc_base=float(kk.auc_base.median()),
                          min_auc=float(kk.auc.min()))
OUT['subgroups'] = sub
print('하위집단', json.dumps(sub, ensure_ascii=False, indent=0), flush=True)

# ---------------- 4) 상위 k 적중률 ----------------
def topk(d, col, k):
    out = []
    for t, g in d.groupby('t0'):
        kk_ = k if k >= 1 else int(round(k * len(g))); o = np.argsort(-g[col].values, kind='stable')[:kk_]
        out.append((g.yb.values[o].sum(), kk_, g.yb.sum()))
    h = sum(x[0] for x in out); n = sum(x[1] for x in out); e = sum(x[2] for x in out)
    return dict(k_total=int(n), hits=int(h), precision=float(h / n), recall=float(h / e))
tk = {}
for k in (50, 100, 0.05, 0.10, 0.20):
    lab = f'상위 {k}개(원점별)' if k >= 1 else f'상위 {int(k*100)}%'
    tk[lab] = {'모형': topk(TE, 'p', k), '공식 기준선': topk(TE, 'p_base', k), best.split(' ')[0]: topk(TE, best, k), '무작위 기대 정밀도': float(TE.yb.mean())}
OUT['topk'] = tk
print('상위 k', json.dumps(tk, ensure_ascii=False), flush=True)

# ---------------- 5) 보정(개발 예측으로 Platt 적합 → 최종에 적용) ----------------
Dv = pd.read_parquet(DEV)
cal = {}
for col in ('p', 'p_base'):
    x = Dv[[col]].values
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x if col == 'p' else np.log(np.clip(x, 1e-6, 1) / np.clip(1 - x, 1e-6, 1)), Dv.yb.values)
    xt = TE[[col]].values if col == 'p' else np.log(np.clip(TE[[col]].values, 1e-6, 1) / np.clip(1 - TE[[col]].values, 1e-6, 1))
    ph = lr.predict_proba(xt)[:, 1]; y = TE.yb.values
    dec = pd.qcut(pd.Series(ph).rank(method='first'), 10, labels=False)
    rel = pd.DataFrame(dict(dec=dec, ph=ph, y=y)).groupby('dec').agg(pred=('ph', 'mean'), obs=('y', 'mean'), n=('y', 'size'))
    ece = float((rel.n * (rel.pred - rel.obs).abs()).sum() / rel.n.sum())
    brier = float(np.mean((ph - y) ** 2)); b0 = float(np.mean((y.mean() - y) ** 2))
    cal['모형(순위점수→Platt, 개발 원점 적합)' if col == 'p' else '공식 기준선(로지스틱 확률→Platt)'] = dict(
        brier=brier, brier_const=b0, brier_skill=1 - brier / b0, ece=ece, reliability=[dict(decile=int(i), pred=round(float(r.pred), 4), obs=round(float(r.obs), 4), n=int(r.n)) for i, r in rel.iterrows()])
# 공식 기준선 원확률의 보정(재적합 없이)
y = TE.yb.values; ph = TE.p_base.values; dec = pd.qcut(pd.Series(ph).rank(method='first'), 10, labels=False)
rel = pd.DataFrame(dict(dec=dec, ph=ph, y=y)).groupby('dec').agg(pred=('ph', 'mean'), obs=('y', 'mean'), n=('y', 'size'))
cal['공식 기준선 원확률(재적합 없음)'] = dict(ece=float((rel.n * (rel.pred - rel.obs).abs()).sum() / rel.n.sum()), mean_pred=float(ph.mean()), obs_rate=float(y.mean()))
OUT['calibration'] = cal
print('보정', json.dumps({k: {kk_: v[kk_] for kk_ in v if kk_ != 'reliability'} for k, v in cal.items()}, ensure_ascii=False), flush=True)

# ---------------- 6) 대형 출원인 제외 레이블 재계산 ----------------
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
BR = os.path.join(CACHE, 'base_rows.parquet').replace(os.sep, '/')
con.execute(f"create temp table kr as select ksic, mgn(ipc_raw) mg, year, appl_no, any_value(app1) app1 from read_parquet('{BR}') where cc='KR' and mgn(ipc_raw) is not null group by 1,2,3,4")
big = {}
for t in TEST:
    top = con.sql(f"select app1, count(distinct appl_no) n from kr where year between {t-2} and {t} and app1 is not null and app1<>'' group by 1 order by n desc limit 30").df()
    tot = con.sql(f"select count(distinct appl_no) n from kr where year between {t-2} and {t}").fetchone()[0]
    for N in (0, 10, 30):
        names = tuple(top.app1.head(N)) if N else ('__none__',)
        con.register('bn', pd.DataFrame(dict(app1=list(names))))
        c = con.sql(f"""select ksic||'|'||mg cell, sum(case when year between {t-2} and {t} then 1 else 0 end) past, sum(case when year between {t+1} and {t+3} then 1 else 0 end) fut
                       from kr where year between {t-2} and {t+3} and coalesce(app1,'') not in (select app1 from bn) group by 1""").df()
        g = TE[TE.t0 == t][['cell', 'kr_W', 'p', 'p_base', 'yb', best]].merge(c, on='cell', how='left').fillna({'past': 0, 'fut': 0})
        if N == 0:
            big[f't0={t} 확인(N=0)'] = dict(past_eq_krW=bool((g.past.values == g.kr_W.values).all()))
        g['y2'] = np.log((g.fut + 1) / (g.past + 1)); k = int(round(0.10 * len(g)))
        o = np.lexsort((hkey(g.cell.values), -(g.fut - g.past).values, -g.y2.values)); g['yb2'] = 0; g.loc[g.index[o[:k]], 'yb2'] = 1
        if N:
            a_m, a_b, a_s = auc(g.yb2, g.p), auc(g.yb2, g.p_base), auc(g.yb2, g[best])
            big[f't0={t} 상위 {N} 출원인 제외'] = dict(share_of_kr_docs=float(top.n.head(N).sum() / tot), label_overlap=float((g.yb2 & g.yb).sum() / g.yb.sum()),
                                                  auc=a_m, auc_base=a_b, skill=sk(a_m, a_b), auc_best_sb=a_s, skill_vs_best_sb=sk(a_m, a_s))
OUT['big_applicant_excluded'] = big
print('대형 출원인 제외', json.dumps(big, ensure_ascii=False, indent=0), flush=True)
TE[['cell', 'ksic', 'mg', 't0', 'kr_W', 'p', 'p_base', 'yb'] + list(SB)].to_parquet(os.path.join(HERE, f'sens_{TAG}_scores.parquet'))
OUT['created'] = time.strftime('%Y-%m-%d %H:%M:%S'); OUT['secs'] = round(time.time() - t_start)
json.dump(OUT, open(os.path.join(HERE, f'sens_{TAG}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('저장', f'sens_{TAG}.json', OUT['secs'], 's')
