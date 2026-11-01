# T1 레드팀 재검증(PROTOCOL_R5 §3-2·3) — 위약·강화 기준선·대형 출원인 제외·하위집단·상위 k·보정.
# 동결 예측(runs/FROZEN_final, FROZEN_dev)·동결 캐시는 읽기만. 산출은 VERIFY_R5/ 아래. 공식 정의 레이블은 verify_final_r3.py와 같은 원천 캐시로 만든다.
import os, sys, json, time, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
import xgboost as xgb
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
V = pathlib.Path(__file__).resolve().parent; T1 = V.parent; R3 = T1/'R3'; CACHE = T1/'cache'
RD = R3/'runs'/'FROZEN_final'; FC = R3/'cache_final'/'o2019'
C, H, END = 2019, 4, '20250822'
t_start = time.time(); OUT = {}
sk = lambda a, b: (a - b) / (1 - b)

# ---------- 레이블(공식 정의, 원천 캐시) ----------
o = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'newc', 'px', 'pl', 'prior_any'])
K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); kidx = {k: i for i, k in enumerate(K)}
a = pd.read_parquet(CACHE/'app_pub.parquet', columns=['applicant_cd', 'ay', 'pub_date', 'rgst_date', 'ksic'])
a = a[a.ay >= 2000].copy(); a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
pdt = a.pub_date.fillna('').astype(str); rdt = a.rgst_date.fillna('').astype(str)
vis_end = ((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))
key = pd.MultiIndex.from_arrays([o.firm.values, o.k.values])
pubset = pd.MultiIndex.from_frame(a[vis_end & (a.ay > C) & (a.ay <= C + H)][['applicant_cd', 'k']].drop_duplicates())
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['applicant_cd', 'ksic', 'ay']); ins = ins[(ins.ay > C) & (ins.ay <= C + H)].copy(); ins['k'] = ins.ksic.map(kidx)
regset = pd.MultiIndex.from_frame(ins[['applicant_cd', 'k']].dropna().astype({'k': int}).drop_duplicates())
y_all = np.maximum(key.isin(regset), key.isin(pubset)).astype(np.int8)
pd.DataFrame(dict(firm=o.firm, k=o.k, y_u=y_all)).to_parquet(V/'labels_c2019_official.parquet', index=False)
keep = ~o.prior_any.values
T = o[keep].reset_index(drop=True); y = y_all[keep]; n = len(T)
bs = pd.read_parquet(FC/'baseline_scores.parquet').loc[keep].reset_index(drop=True)
te = pd.read_parquet(FC/'prep_te.parquet', columns=['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']).loc[keep].reset_index(drop=True)
T['s'] = 0.9 * rankdata(T.px.values) / n + 0.1 * rankdata(T.pl.values) / n
for c in bs.columns: T[c] = bs[c].values
for c in te.columns: T[c] = te[c].values
BCOLS = list(bs.columns)

def fast_auc_factory(scores):
    r = rankdata(scores)
    def f(yy):
        n1 = yy.sum(); n0 = len(yy) - n1; return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return f
def auc(yy, s): return fast_auc_factory(s)(yy)
def eval_sub(mask, extra=()):
    d = T[mask]; yy = y[mask]
    if yy.sum() < 20: return dict(n=int(mask.sum()), pos=int(yy.sum()), note='양성 부족')
    am = auc(yy, d.s.values); bb = {c: auc(yy, d[c].values) for c in BCOLS}; b = max(bb.values())
    r = dict(n=int(mask.sum()), pos=int(yy.sum()), rate=float(yy.mean()), auc=am, auc_strongest=b, strongest=max(bb, key=bb.get), skill=sk(am, b))
    for e in extra: r[f'auc_{e}'] = auc(yy, d[e].values); r[f'skill_vs_{e}'] = sk(am, r[f'auc_{e}'])
    return r
OUT['official_recomputed'] = eval_sub(np.ones(n, bool))
print('공식 재계산', OUT['official_recomputed'], flush=True)

# ---------- 강화 기준선(학습 원점 2012–15, 학습 레이블 cap 2019 = 공식과 같은 학습 규칙) ----------
TRC = ['cut', 'k', 'newc', 'y_reg', 'y_pub', 'lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
tr = pd.read_parquet(FC/'prep_tr.parquet', columns=TRC); tr['y'] = np.maximum(tr.y_reg.values, tr.y_pub.values)
assert tr.cut.max() + H <= 2019
# SB0: 층 진입률(KSIC × 규모 10분위 × 신규진입자), 학습 원점 기준. 축소 사전값 = 규모분위×신규 평균
edges = np.unique(np.quantile(tr.lsize.values, np.linspace(0, 1, 11)))
qb = lambda x: np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
tr['qb'] = qb(tr.lsize.values); T['qb'] = qb(T.lsize.values)
g1 = tr.groupby(['qb', 'newc']).y.mean(); g2 = tr.groupby(['k', 'qb', 'newc']).y.agg(['sum', 'count'])
pri = 200.0; prior_v = g1.reindex(pd.MultiIndex.from_arrays([g2.index.get_level_values(1), g2.index.get_level_values(2)])).values
rate = pd.Series((g2['sum'].values + pri * prior_v) / (g2['count'].values + pri), index=g2.index)
idx = pd.MultiIndex.from_arrays([T.k.values, T.qb.values, T.newc.values])
T['SB0'] = rate.reindex(idx).values; T['SB0'] = T.SB0.fillna(pd.Series(g1.reindex(pd.MultiIndex.from_arrays([T.qb.values, T.newc.values])).values)).values
SBF = {'SBA': ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'k', 'newc'],
       'SBB': ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'k', 'newc', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']}
rs = np.random.RandomState(2019); kp = (tr.y.values == 1) | (rs.rand(len(tr)) < 0.25); w = np.where(tr.y.values == 1, 1.0, 4.0)[kp]
for nm, fs in SBF.items():
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
    m.fit(tr.loc[kp, fs].values.astype(np.float32), tr.y.values[kp], sample_weight=w)
    T[nm] = m.predict_proba(T[fs].values.astype(np.float32))[:, 1]
del tr
SBDESC = {'SB0': '층 진입률(KSIC×규모10분위×신규), 학습 원점', 'SBA': 'XGB 규모·근접도·인기도 + KSIC·신규 표시(엄격 가시 피처)',
          'SBB': 'SBA + 기업 최근 활동·산업 진입 동학(n_last1·gap_last·now1·now3·now_grow·k_grow·k_newent·vis_frac)'}
OUT['strong_baselines'] = {SBDESC[c]: dict(auc=auc(y, T[c].values), skill_model_vs_it=sk(OUT['official_recomputed']['auc'], auc(y, T[c].values)),
                                           its_skill_vs_official_strongest=sk(auc(y, T[c].values), OUT['official_recomputed']['auc_strongest'])) for c in ('SB0', 'SBA', 'SBB')}
print('강화 기준선', json.dumps(OUT['strong_baselines'], ensure_ascii=False), flush=True)

# ---------- 위약: 층 안에서 레이블 섞기 ----------
strat = pd.Series(T.k.astype(str) + '|' + T.qb.astype(str) + '|' + T.newc.astype(str)).values
codes, inv = np.unique(strat, return_inverse=True); order = np.argsort(inv, kind='stable'); bounds = np.r_[0, np.cumsum(np.bincount(inv))]
fm = fast_auc_factory(T.s.values); fb = {c: fast_auc_factory(T[c].values) for c in BCOLS}; f0 = fast_auc_factory(T.SB0.values)
pl = {'층내(KSIC×규모10분위×신규)': [], '전체 무작위': []}; pl0 = []
for sd in range(200):
    rng = np.random.default_rng(7000 + sd)
    ys = y.copy()
    for i in range(len(codes)):
        ii = order[bounds[i]:bounds[i + 1]]
        if len(ii) > 1: ys[ii] = rng.permutation(y[ii])
    am = fm(ys); b = max(f(ys) for f in fb.values()); pl['층내(KSIC×규모10분위×신규)'].append(sk(am, b)); pl0.append(sk(am, f0(ys)))
    yg = rng.permutation(y); pl['전체 무작위'].append(sk(fm(yg), max(f(yg) for f in fb.values())))
OUT['placebo'] = {k: dict(n_seeds=len(v), skill_mean=float(np.mean(v)), skill_sd=float(np.std(v)), q025=float(np.percentile(v, 2.5)), q975=float(np.percentile(v, 97.5))) for k, v in pl.items()}
OUT['placebo']['층내 섞기 — SB0(층 진입률) 대비'] = dict(n_seeds=200, skill_mean=float(np.mean(pl0)), skill_sd=float(np.std(pl0)), q025=float(np.percentile(pl0, 2.5)), q975=float(np.percentile(pl0, 97.5)))
print('위약', json.dumps(OUT['placebo'], ensure_ascii=False), flush=True)

# ---------- 대형 출원인 제외·하위집단 ----------
fs_ = T.groupby('firm').lsize.first().sort_values(ascending=False)
sub = {}
for N in (30, 100, int(round(0.01 * len(fs_)))):
    big = set(fs_.index[:N]); sub[f'규모 상위 {N}개 법인 제외'] = eval_sub(~T.firm.isin(big).values, extra=('SBB',))
sub['규모 상위 1% 법인만'] = eval_sub(T.firm.isin(set(fs_.index[:int(round(0.01 * len(fs_)))])).values)
sub['신규진입자(범위안 등록 0건 법인)'] = eval_sub(T.newc.values.astype(bool), extra=('SBB',))
sub['기존 보유 법인'] = eval_sub(~T.newc.values.astype(bool), extra=('SBB',))
fq = pd.qcut(fs_.rank(method='first'), 5, labels=False); T['fq'] = T.firm.map(fq).values
for qv in range(5): sub[f'법인 규모 5분위 {qv+1}(1=최소)'] = eval_sub(T.fq.values == qv)
fmeta = pd.read_parquet(CACHE/'firm_meta.parquet', columns=['applicant_cd', 'tp']).set_index('applicant_cd').tp
tp = T.firm.map(fmeta).fillna('미상').values
sub['국내법인'] = eval_sub(tp == '국내법인', extra=('SBB',)); sub['외국법인'] = eval_sub(tp == '외국법인', extra=('SBB',))
kk = []
for kv, g in T.groupby('k'):
    yy = y[g.index.values]
    if yy.sum() >= 20:
        am = auc(yy, g.s.values); b = max(auc(yy, g[c].values) for c in BCOLS); kk.append(dict(ksic=K[kv], n=len(g), pos=int(yy.sum()), auc=am, auc_strongest=b, skill=sk(am, b)))
kk = pd.DataFrame(kk); kk.to_csv(V/'sens_t1_by_ksic.csv', index=False, encoding='utf-8-sig')
sub['KSIC별(양성≥20)'] = dict(n_ksic=int(len(kk)), frac_model_gt_strongest=float((kk.auc > kk.auc_strongest).mean()), median_skill=float(kk.skill.median()), min_skill=float(kk.skill.min()), max_skill=float(kk.skill.max()))
OUT['subgroups'] = sub
print('하위집단', json.dumps(sub, ensure_ascii=False), flush=True)

# ---------- 상위 k 적중률 ----------
best_b = max(BCOLS, key=lambda c: auc(y, T[c].values))
def topk(sc, k):
    o_ = np.argsort(-sc, kind='stable')[:k]; return dict(k=int(k), hits=int(y[o_].sum()), precision=float(y[o_].mean()), recall=float(y[o_].sum() / y.sum()))
tk = {}
for kv in (100, 1000, int(round(0.01 * n)), int(round(0.05 * n)), int(round(0.10 * n))):
    tk[f'상위 {kv}'] = {'모형': topk(T.s.values, kv), f'공식 최강 기준선({best_b})': topk(T[best_b].values, kv), 'SBB': topk(T.SBB.values, kv), '무작위 기대 정밀도': float(y.mean())}
d = pd.DataFrame(dict(k=T.k.values, s=T.s.values, b=T[best_b].values, sb=T.SBB.values, y=y))
tk['산업별 상위 20 법인'] = {'모형': float(d.sort_values('s', ascending=False).groupby('k').head(20).y.mean()), '공식 최강 기준선': float(d.sort_values('b', ascending=False).groupby('k').head(20).y.mean()),
                         'SBB': float(d.sort_values('sb', ascending=False).groupby('k').head(20).y.mean())}
OUT['topk'] = tk
print('상위 k', json.dumps(tk, ensure_ascii=False), flush=True)

# ---------- 보정 ----------
from sklearn.linear_model import LogisticRegression
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
dv = pd.concat([pd.read_parquet(R3/'runs'/'FROZEN_dev'/f'o{c}.parquet', columns=['px', 'y_u', 'prior_any']) for c in (2008, 2009, 2010, 2011, 2012)])
dv = dv[~dv.prior_any.values]
pr = LogisticRegression(C=1e6, max_iter=1000).fit(lg(dv.px.values)[:, None], dv.y_u.values)
cal = {}
for nm, ph in (('원확률 px(음성추출 가중 XGB, 재보정 없음)', T.px.values), ('개발 원점 Platt 재보정', pr.predict_proba(lg(T.px.values)[:, None])[:, 1])):
    dec = pd.qcut(pd.Series(ph).rank(method='first'), 10, labels=False).values
    rel = pd.DataFrame(dict(dec=dec, ph=ph, y=y)).groupby('dec').agg(pred=('ph', 'mean'), obs=('y', 'mean'), n=('y', 'size'))
    top = np.argsort(-ph)[:int(0.01 * n)]
    br = float(np.mean((ph - y) ** 2)); b0 = float(np.mean((y.mean() - y) ** 2))
    cal[nm] = dict(mean_pred=float(ph.mean()), obs_rate=float(y.mean()), brier=br, brier_skill=1 - br / b0, ece=float((rel.n * (rel.pred - rel.obs).abs()).sum() / rel.n.sum()),
                   top1pct_pred=float(ph[top].mean()), top1pct_obs=float(y[top].mean()), reliability=[dict(decile=int(i), pred=float(r.pred), obs=float(r.obs)) for i, r in rel.iterrows()])
OUT['calibration'] = cal
print('보정', json.dumps({k: {a_: v[a_] for a_ in v if a_ != 'reliability'} for k, v in cal.items()}, ensure_ascii=False), flush=True)
OUT['secs'] = round(time.time() - t_start)
json.dump(OUT, open(V/'sens_t1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
T[['firm', 'k', 's', 'SB0', 'SBA', 'SBB']].to_parquet(V/'sens_t1_scores.parquet', index=False)
print('저장 sens_t1.json', OUT['secs'])
