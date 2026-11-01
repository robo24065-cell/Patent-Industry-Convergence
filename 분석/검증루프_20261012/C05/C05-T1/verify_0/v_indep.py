# C05-T1 verify_0: 독립 재계산(다른 코드 경로). s2 의 캐시·함수 미사용.
#  - 셀 집계: duckdb 로 (ksic, cc, year, appl_no, ipc_raw) 고유행을 뽑고, ipc_raw → 메인그룹은 파이썬 정규식(자체 파서)으로 변환
#  - AUC: Mann–Whitney 순위식(가중판: 군집 재표집 횟수를 정수 가중치로) 직접 계산
#  - 부트스트랩: default_rng(0).choice(군집, n) 와 같은 난수열에서 군집별 뽑힌 횟수 = 가중치 (복제 대신 가중)
#  - 시나리오 S1, S1r, S2, S3(문서가중 평균) 재계산 + 반박 시도 r1–r3
import os, sys, re, json, hashlib, time
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
t0_ = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
V = os.path.join(ROOT, r"검증루프_20261012\C05\C05-T1\verify_0")
R3 = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3")
BR = os.path.join(R3, 'cache_pub_final', 'base_rows.parquet').replace('\\', '/')
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(f"create table r as select distinct ksic, cc, year, appl_no, ipc_raw from read_parquet('{BR}') where year between 2016 and 2024")
raw = con.sql("select distinct ipc_raw from r where ipc_raw is not null").df().ipc_raw
def mg_parse(s):
    t = re.sub(r'\s+', '', s)
    a = re.fullmatch(r'([A-H]\d{2}[A-Z])(\d{1,4})/(\d{1,6})', t)
    if a: return f"{a.group(1)} {int(a.group(2))}"
    b = re.fullmatch(r'([A-H]\d{2}[A-Z])(\d{3,6})', t)
    if b: return f"{b.group(1)} {int(t[4:-2])}"
    return None
mp = pd.DataFrame(dict(ipc_raw=raw, mg=[mg_parse(x) for x in raw])).dropna()
con.register('mp', mp)
cnt = con.sql("""select r.ksic, mp.mg, r.cc, r.year, count(distinct r.appl_no) n from r join mp using(ipc_raw) group by all""").df()
cnt['cell'] = cnt.ksic + '|' + cnt.mg
def wsum(df, a, b, cc=None):
    x = df[df.year.between(a, b)]
    if cc: x = x[x.cc == cc]
    return x.groupby('cell').n.sum()
tot_W = wsum(cnt, 2019, 2021); kr_W = wsum(cnt, 2019, 2021, 'KR'); kr_F = wsum(cnt, 2022, 2024, 'KR')
E = pd.DataFrame(dict(tot_W=tot_W)).join(kr_W.rename('kr_W')).join(kr_F.rename('kr_fut')).fillna(0)
E = E[E.tot_W >= 50].reset_index().rename(columns={'index': 'cell'})
def md5k(c): return int(hashlib.md5(c.encode()).hexdigest()[:8], 16)
def relabel(df):
    y = np.log((df.kr_fut + 1) / (df.kr_W + 1)); D = df.kr_fut - df.kr_W; k = int(round(0.1 * len(df)))
    key = pd.DataFrame(dict(y=-y.values, D=-D.values, h=[md5k(c) for c in df.cell])).sort_values(['y', 'D', 'h'], kind='mergesort')
    lab = np.zeros(len(df), int); lab[key.index.values[:k]] = 1; return lab
E['yb'] = relabel(E)
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
M = P[['cell', 'mg', 'p', 'p_base', 'yb']].rename(columns={'yb': 'yb_frozen'}).merge(E, on='cell', how='outer', indicator=True)
chk = dict(merge=M._merge.value_counts().to_dict(), yb_mismatch=int((M.yb_frozen != M.yb).sum()))
M = M[M._merge == 'both'].reset_index(drop=True); M['p_base'] = M.p_base.astype(float)
print('재현', chk)

def wauc(y, s, w=None):
    """가중 Mann–Whitney AUC: Σ_{i∈+,j∈-} w_i w_j [s_i>s_j + 0.5·(s_i=s_j)] / (W+·W-)"""
    y = np.asarray(y); s = np.asarray(s, float); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=w * (y == 1), minlength=len(u)); wn = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
def boot(cl, y, p, pb, nb=1000, seed=0):
    rng = np.random.default_rng(seed); um = np.unique(cl); idx = np.searchsorted(um, cl); res = []
    for _ in range(nb):
        dr = rng.choice(um, len(um)); w = np.bincount(np.searchsorted(um, dr), minlength=len(um))[idx]
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0: continue
        a, b = wauc(y, p, w), wauc(y, pb, w); res.append((a - b) / (1 - b))
    return [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]
def score(d, cl, y, name):
    a, b = wauc(y, d.p), wauc(y, d.p_base); s = (a - b) / (1 - b); ci = boot(np.asarray(cl), np.asarray(y), d.p.values, d.p_base.values)
    r = dict(scenario=name, n=int(len(d)), pos=int(np.sum(y)), auc=a, auc_base=b, skill=s, skill_lo=ci[0], skill_hi=ci[1]); print(r, flush=True); return r
out = dict(check=chk, scen=[])
out['scen'].append(score(M, M.mg, M.yb, 'S0'))
h10 = M.mg.str.match(r'^H10[A-Z] ')
d1 = M[~h10].reset_index(drop=True)
out['scen'].append(score(d1, d1.mg, d1.yb, 'S1'))
out['scen'].append(score(d1, d1.mg, relabel(d1), 'S1r'))
A2 = M.mg.str[:4].isin(['G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H', 'G06K', 'H01L'])
d2 = M[~A2].reset_index(drop=True)
out['scen'].append(score(d2, d2.mg, d2.yb, 'S2'))
# S3: 통합 셀(KSIC × H01L+H10), 문서 고유 재집계
hdoc = con.sql("""select distinct r.ksic, r.cc, r.year, r.appl_no from r join mp using(ipc_raw) where mp.mg like 'H01L %' or regexp_matches(mp.mg, '^H10[A-Z] ')""").df()
g = pd.DataFrame(dict(tot_W=hdoc[hdoc.year.between(2019, 2021)].groupby('ksic').size(),
                      kr_W=hdoc[hdoc.year.between(2019, 2021) & (hdoc.cc == 'KR')].groupby('ksic').size(),
                      kr_fut=hdoc[hdoc.year.between(2022, 2024) & (hdoc.cc == 'KR')].groupby('ksic').size())).fillna(0)
isH = M.mg.str.startswith('H01L') | h10
Hm = M[isH].copy(); Hm['ksic'] = Hm.cell.str.split('|').str[0]
agg = Hm.groupby('ksic').apply(lambda x: pd.Series(dict(p=(x.p * x.tot_W).sum() / x.tot_W.sum(), p_base=(x.p_base * x.tot_W).sum() / x.tot_W.sum())), include_groups=False)
g = g[g.tot_W >= 50].join(agg, how='inner').reset_index().rename(columns={'index': 'ksic'})
g['cell'] = g.ksic + '|H01L+H10'; g['mg'] = 'H01L+H10'
d3 = pd.concat([M[~isH][['cell', 'mg', 'p', 'p_base', 'tot_W', 'kr_W', 'kr_fut']], g[['cell', 'mg', 'p', 'p_base', 'tot_W', 'kr_W', 'kr_fut']]], ignore_index=True)
out['scen'].append(score(d3, d3.mg, relabel(d3), 'S3'))
out['S3_merged'] = g.assign(y=np.log((g.kr_fut + 1) / (g.kr_W + 1))).sort_values('kr_W', ascending=False).head(10).to_dict('records')

# ---------- 반박 시도 ----------
sc = cnt.assign(sc=cnt.mg.str[:4])
semi = sc[sc.sc.eq('H01L') | sc.sc.str.match(r'^H10[A-Z]$')]
# 문서 고유(국가별 H01L+H10 합): 같은 문서가 여러 KSIC에 있으므로 다시 고유화
sd = con.sql("""select distinct r.cc, r.year, r.appl_no, substr(mp.mg,1,4) sc from r join mp using(ipc_raw) where mp.mg like 'H01L %' or regexp_matches(mp.mg, '^H10[A-Z] ')""").df()
yr = sd.groupby(['cc', 'year', 'sc']).appl_no.nunique().unstack('sc', fill_value=0).sort_index()
yr['H10'] = yr[[c for c in yr.columns if c.startswith('H10')]].sum(1); yr['ALL'] = yr.H10 + yr.H01L; yr['H10_share'] = yr.H10 / yr.ALL
allc = con.sql("select cc, year, count(distinct appl_no) n from r group by all").df().set_index(["cc", "year"]).n.sort_index()
r1 = {}
for c in ['KR', 'US', 'EP', 'JP', 'CN']:
    a = yr.loc[c].loc[2019:2021].ALL.sum(); b = yr.loc[c].loc[2022:2024].ALL.sum()
    ta = allc.loc[c].loc[2019:2021].sum(); tb = allc.loc[c].loc[2022:2024].sum()
    r1[c] = dict(semi_2019_21=int(a), semi_2022_24=int(b), semi_log_growth=float(np.log(b / a)), all50ksic_log_growth=float(np.log(tb / ta)),
                 h10_share_2019_21=float(yr.loc[c].loc[2019:2021].H10.sum() / a), h10_share_2022_24=float(yr.loc[c].loc[2022:2024].H10.sum() / b))
out['r1_semis_growth'] = r1
out['r2_h10_share_by_year'] = {c: {int(y): round(float(v), 3) for y, v in yr.loc[c].H10_share.items()} for c in ['KR', 'US', 'EP', 'JP', 'CN']}
# r3: H10 양성 셀마다 같은 KSIC 원천 H01L 메인그룹(대응표 역매핑) 셀의 KR 변화
cy = pd.read_csv(os.path.join(V, '..', 'wipo', 'wipo_concordance_all.csv'), dtype=str)
cy = cy[(cy.default == 'Y') & cy.to_sc.str.startswith('H10', na=False) & (cy.from_sc == 'H01L')]
back = cy.groupby('to_mg').from_mg.agg(lambda s: s.value_counts().index[0]).to_dict()
pos = M[h10 & (M.yb == 1)].copy()
rows = []
for _, r in pos.iterrows():
    ks = r.cell.split('|')[0]; src = back.get(r.mg); sc_ = f"{ks}|{src}" if src else None
    kw = float(kr_W.get(sc_, 0)) if sc_ else np.nan; kf = float(kr_F.get(sc_, 0)) if sc_ else np.nan
    rows.append(dict(cell=r.cell, D=float(r.kr_fut - r.kr_W), src_cell=sc_, src_kr_W=kw, src_kr_fut=kf, src_D=kf - kw if sc_ else np.nan))
r3 = pd.DataFrame(rows); out['r3_source_change'] = r3.to_dict('records')
out['r3_summary'] = dict(n=len(r3), h10_D_sum=float(r3.D.sum()), src_D_sum=float(r3.drop_duplicates('src_cell').src_D.sum()),
                         n_src_declined=int((r3.drop_duplicates('src_cell').src_D < 0).sum()), n_src=int(r3.src_cell.nunique()))
json.dump(out, open(os.path.join(V, 'v_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: out[k] for k in ['r1_semis_growth', 'r2_h10_share_by_year', 'r3_summary']}, ensure_ascii=False, indent=1))
print(r3.to_string())
print('sec', round(time.time() - t0_, 1))
