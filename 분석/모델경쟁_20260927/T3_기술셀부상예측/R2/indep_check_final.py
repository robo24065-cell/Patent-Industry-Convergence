"""🔒 최종시험 독립 재계산 (final_run_r2.py --mode final 실행 뒤에만).
원본 메타(_shared/meta/*.parquet)에서 캐시·r2lib 를 거치지 않고 직접
  (1) 셀(KSIC×IPC 메인그룹) 연도별 국가별 출원수(셀 내 출원번호 중복 제거) → 적격(5개국 t0-2..t0 ≥50),
  (2) 레이블 y = log((KR t0+1..t0+3 +1)/(KR t0-2..t0 +1)), 상위 10% 부상(y↓, 절대증가↓, md5 앞 8자리↑),
  (3) 저장된 모형 점수 p·기준선 p_base 로 AUC·skill·PR-AUC·P@10% 를 다시 계산하고,
  (4) 기준선 로지스틱(kr_mom 단독, 학습 t0 2005..2016)을 원본 집계로 다시 적합해 p_base 와 대조,
  (5) 군집 부트스트랩(다른 시드)으로 CI 를 다시 계산한다.
"""
import os, sys, json, hashlib, time
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.abspath(os.path.join(HERE, '..', '..', '_shared', 'meta')).replace('\\', '/') + '/*.parquet'
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
N = con.sql(f"""
 with r as (select ksic, cc, appl_year::int y, appl_no,
      mgn(regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g')) mg
      from read_parquet('{META}') where appl_year between 2000 and 2023),
 d as (select distinct ksic, mg, cc, y, appl_no from r where mg is not null)
 select ksic, mg, cc, y, count(*) n from d group by all""").df()
print('집계', len(N), f'{time.time()-t:.0f}s', flush=True)
N['cell'] = N.ksic + '|' + N.mg
kr = N[N.cc == 'KR'].pivot_table(index='cell', columns='y', values='n', aggfunc='sum', fill_value=0)
tot = N.pivot_table(index='cell', columns='y', values='n', aggfunc='sum', fill_value=0)
yrs = list(range(2000, 2024)); kr = kr.reindex(index=tot.index, columns=yrs, fill_value=0); tot = tot.reindex(columns=yrs, fill_value=0)
S = lambda X, a, b: X.loc[:, a:b].sum(1)

def lab(t0):
    tw = S(tot, t0 - 2, t0); e = tw[tw >= 50].index
    past = S(kr, t0 - 2, t0)[e]; fut = S(kr, t0 + 1, t0 + 3)[e]
    y = np.log((fut + 1) / (past + 1)); tie = fut - past
    key = np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in e], dtype=np.int64)
    k = max(1, int(round(0.1 * len(e)))); o = np.lexsort((key, -tie.values, -y.values)); yb = np.zeros(len(e), int); yb[o[:k]] = 1
    mom = np.log((past + 1) / (S(kr, t0 - 5, t0 - 3)[e] + 1))
    return pd.DataFrame(dict(cell=e, y_ind=y.values, yb_ind=yb, kr_mom_ind=mom.values))

P = pd.read_parquet(os.path.join(HERE, 'preds', 'FINAL_t0_2019_2020.parquet'))
R = json.load(open(os.path.join(HERE, 'final_result.json'), encoding='utf-8'))
out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), by_origin={})
# 기준선 재적합용 학습 표본 (t0 2005..2016, 적격, 레이블 결과창 ≤2019)
TR = pd.concat([lab(t).assign(t0=t) for t in range(2005, 2017)], ignore_index=True)
bl = LogisticRegression(C=1e6, max_iter=1000).fit(TR[['kr_mom_ind']].values, TR.yb_ind.values)
blocks = []
for v in (2019, 2020):
    L_ = lab(v); p = P[P.t0 == v].merge(L_, on='cell', how='outer', indicator=True)
    both = p[p._merge == 'both']
    pb2 = bl.predict_proba(both[['kr_mom_ind']].values)[:, 1]
    a = roc_auc_score(both.yb_ind, both.p); a0 = roc_auc_score(both.yb_ind, both.p_base); a0r = roc_auc_score(both.yb_ind, pb2)
    k = max(1, int(round(0.1 * len(both)))); top = np.argsort(-both.p.values)[:k]
    o = dict(n_pred=int((P.t0 == v).sum()), n_indep=int(len(L_)), n_both=int(len(both)),
             only_pred=int((p._merge == 'left_only').sum()), only_indep=int((p._merge == 'right_only').sum()),
             y_maxabs_diff=float(np.abs(both.y - both.y_ind).max()), yb_agree=float((both.yb == both.yb_ind).mean()),
             n_evt_indep=int(both.yb_ind.sum()), auc=float(a), auc_base=float(a0), skill=float((a - a0) / (1 - a0)),
             auc_base_refit=float(a0r), base_rank_corr=float(pd.Series(pb2).corr(both.p_base.reset_index(drop=True), method='spearman')),
             prauc=float(average_precision_score(both.yb_ind, both.p)), p_at10=float(both.yb_ind.values[top].mean()),
             reported_auc=R['by_origin'][str(v)]['model']['auc'], reported_skill=R['by_origin'][str(v)]['skill'])
    out['by_origin'][str(v)] = o; print(v, json.dumps(o, ensure_ascii=False), flush=True)
    blocks.append((both.mg.values, both.yb_ind.values, both.p.values, both.p_base.values))
sk = [out['by_origin'][k]['skill'] for k in out['by_origin']]; out['mean_skill'] = float(np.mean(sk))
au = np.mean([out['by_origin'][k]['auc'] for k in out['by_origin']]); a0 = np.mean([out['by_origin'][k]['auc_base'] for k in out['by_origin']])
out['skill_of_mean_auc'] = float((au - a0) / (1 - a0)); out['auc'] = float(au); out['auc_base'] = float(a0)
# 군집 부트스트랩(시드 12345, 벡터화 방식은 final_run 과 독립 구현)
rng = np.random.default_rng(12345); ums = np.unique(np.concatenate([b[0] for b in blocks])); res = []
pos = [{u: np.where(b[0] == u)[0] for u in np.unique(b[0])} for b in blocks]
for _ in range(1000):
    s = rng.choice(ums, len(ums)); am = []; ab = []
    for b, g in zip(blocks, pos):
        ii = np.concatenate([g[u] for u in s if u in g]); yy = b[1][ii]
        am.append(roc_auc_score(yy, b[2][ii])); ab.append(roc_auc_score(yy, b[3][ii]))
    A, A0 = np.mean(am), np.mean(ab); res.append((A, A0, (A - A0) / (1 - A0), np.mean([(x - z) / (1 - z) for x, z in zip(am, ab)])))
res = np.array(res); q = lambda j: [float(x) for x in np.percentile(res[:, j], [2.5, 97.5])]
out['ci95_seed12345'] = dict(auc=q(0), auc_base=q(1), skill=q(2), skill_mean_of_origins=q(3), n_clusters=int(len(ums)))
out['reported'] = dict(mean_skill=R['summary']['mean_skill'], skill_of_mean_auc=R['summary']['skill_of_mean_auc'], ci95=R.get('ci95'))
json.dump(out, open(os.path.join(HERE, 'final_indep_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ['mean_skill', 'skill_of_mean_auc', 'auc', 'auc_base', 'ci95_seed12345']}, ensure_ascii=False, indent=1))
print(f'{time.time()-t:.0f}s')
