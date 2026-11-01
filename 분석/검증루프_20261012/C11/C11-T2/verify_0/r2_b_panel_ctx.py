# -*- coding: utf-8 -*-
# 2차 반박 검증 B: (1) 내 reg 배정 + base_rows → 셀(KSIC×mg)×국가×연도 n을 파이썬 정규식 mg로 직접 집계해 작업자 txt_reg/panel_ccy.n과 전수 대조
# (2) 그 n으로 CTX 7피처(c_ind_kmom, c_ind_fmom, c_n_ind, c_entropy, c_cov, c_mix_shift, c_cell_kmax)를 독립 식으로 계산해 작업자 ctx_txt_reg와 대조.
import duckdb, json, time, re, numpy as np, pandas as pd
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
T3 = B + '/분석/모델경쟁_20260927/T3_기술셀부상예측'
W = B + '/분석/검증루프_20261012/C11/C11-T2'; V = W + '/verify_0'
C10 = B + '/분석/검증루프_20261012/C10/C10-T3'
t = time.time(); R = {}
con = duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{V}/_r2tmp'")
km = pd.read_csv(T3 + '/ESSENTIAL_R7/conc_ksic_map.csv', dtype=str)[['ksic', 'code']]; con.register('km', km)
con.execute("create table multi as select code from km group by code having count(*)>1")
con.execute(f"create table ta as select cc, appl_no, code, ksic_txt, \"method\" m from read_parquet('{T3}/TXT_R9/txt_assign.parquet')")
con.execute(f"create table rc as select cc, appl_no, sg from read_parquet('{T3}/ESSENTIAL_R7/rec_conc.parquet')")
con.execute("create table fx as select t.cc, t.appl_no, t.code, rc.sg from ta t left join rc using(cc, appl_no) where t.cc<>'KR' and t.code in (select code from multi)")
con.execute("create table kr as select appl_no, code, ksic_txt, m from ta where cc='KR'")
con.execute(f"""create table fam as select cc, appl_no, k from (select cc, appl_no, k, row_number() over (partition by cc, appl_no order by n desc, cast(k as int)) rn from (
    select fx.cc, fx.appl_no, kr.ksic_txt k, count(*) n from (select distinct cc, appl_no, kr_appl_no from read_parquet('{C10}/cache/links.parquet') where link='REG') l
    join fx on fx.cc=l.cc and fx.appl_no=l.appl_no join kr on kr.appl_no=l.kr_appl_no and kr.code=fx.code group by 1,2,3)) where rn=1""")
con.execute("create table krk as select kr.code, kr.ksic_txt k, rc.sg, split_part(rc.sg,'/',1) mg, left(rc.sg,4) sc from kr join rc on rc.cc='KR' and rc.appl_no=kr.appl_no where kr.m='text'")
for lvl in ('sg', 'mg', 'sc'):
    con.execute(f"""create table t_{lvl} as select code, key_, k from (select code, key_, k, sum(n) over (partition by code, key_) tot,
      row_number() over (partition by code, key_ order by n desc, cast(k as int)) rn from (select code, {lvl} key_, k, count(*) n from krk where {lvl} is not null and {lvl}<>'' group by 1,2,3)) where rn=1 and tot>=3""")
con.execute("create table t_code as select code, k from (select code, k, row_number() over (partition by code order by n desc, cast(k as int)) rn from (select code, k, count(*) n from krk group by 1,2)) where rn=1")
con.execute("""create table fxa as select fx.cc, fx.appl_no, coalesce(f.k, a.k, b.k, c.k, d.k) k from fx left join fam f using(cc, appl_no)
  left join t_sg a on a.code=fx.code and a.key_=fx.sg left join t_mg b on b.code=fx.code and b.key_=split_part(fx.sg,'/',1)
  left join t_sc c on c.code=fx.code and c.key_=left(fx.sg,4) left join t_code d on d.code=fx.code""")
con.execute("create table asg as select t.cc, t.appl_no, coalesce(x.k, t.ksic_txt) ksic from ta t left join fxa x using(cc, appl_no)")
del_ = con.execute("drop table ta; drop table kr; drop table krk; drop table fx")
# mg 정규화(파이썬 정규식) — 고유 ipc_raw 값에 대해
u = con.sql(f"select distinct ipc_raw from read_parquet('{T3}/R3/cache_pub/base_rows.parquet') where ipc_raw is not null").df()
p1 = re.compile(r'^([A-H]\d\d[A-Z])(\d{1,4})/'); p2 = re.compile(r'^([A-H]\d\d[A-Z])(\d{3,6})$')
def mgn(s):
    s = s.replace(' ', '')
    m = p1.match(s)
    if m: return f"{m.group(1)} {int(m.group(2))}"
    m = p2.match(s)
    if m: return f"{m.group(1)} {int(m.group(2)[:-2])}"
    return None
u['mg'] = [mgn(s) for s in u.ipc_raw]; u = u[u.mg.notna()]; con.register('umap', u)
R['n_ipc_raw_distinct'] = len(u)
con.execute(f"""create table mypan as select a.ksic, m.mg, b.cc, b.year, count(distinct b.appl_no) n from read_parquet('{T3}/R3/cache_pub/base_rows.parquet') b
   join asg a on a.cc=b.cc and a.appl_no=b.appl_no join umap m on m.ipc_raw=b.ipc_raw group by all""")
R['mypan_rows'] = con.sql("select count(*), sum(n), min(year), max(year) from mypan").fetchall()
R['panel_cmp'] = con.sql(f"""select count(*) n_cells, sum((p.n is null)::int) only_mine, sum((m.n is null)::int) only_worker, sum((coalesce(p.n,-1)<>coalesce(m.n,-1))::int) n_diff
   from mypan m full join read_parquet('{W}/cache/txt_reg/panel_ccy.parquet') p using(ksic, mg, cc, year)""").fetchall()
R['panel_vs_r8_cells_changed'] = con.sql(f"""select count(*) from mypan m full join read_parquet('{W}/cache/txt_r8/panel_ccy.parquet') p using(ksic, mg, cc, year)
   where coalesce(p.n,-1)<>coalesce(m.n,-1)""").fetchone()[0]
print('panel', R, round(time.time() - t), flush=True)
# ---- CTX 독립 계산 ----
P = con.sql("select ksic, mg, year, sum(n)::double tot, sum(case when cc='KR' then n else 0 end)::double kr, sum(case when cc<>'KR' then n else 0 end)::double fx from mypan where year between 2000 and 2021 group by all").df()
MG = pd.read_parquet(W + '/cache/mg_pub/panel_ccy.parquet', columns=['mg', 'year', 'n'])
mg_first = sorted(MG.mg.unique())[0]   # 작업자 코드: mg 패널에 없는 mg는 인덱스 0(정렬 아님, 등장 순)로 떨어짐 → 아래에서 별도 표시
ctxw = pd.read_parquet(W + '/cache/ctx_txt_reg.parquet')
FE = ['c_ind_kmom', 'c_ind_fmom', 'c_n_ind', 'c_entropy', 'c_cov', 'c_mix_shift', 'c_cell_kmax']
cells = P[['ksic', 'mg']].drop_duplicates()
res = {}; allmine = []
for t0 in range(2005, 2019):
    def win(a, b):
        s = P[(P.year >= a) & (P.year <= b)].groupby(['ksic', 'mg'])[['tot', 'kr', 'fx']].sum()
        return s
    Wd = win(t0 - 2, t0).add_suffix('_W'); Pd = win(t0 - 5, t0 - 3).add_suffix('_P')
    d = cells.set_index(['ksic', 'mg']).join(Wd).join(Pd).fillna(0.0).reset_index()
    ind = d.groupby('ksic')[['tot_W', 'tot_P', 'kr_W', 'kr_P', 'fx_W', 'fx_P']].sum().add_prefix('I_')
    d = d.join(ind, on='ksic')
    d['peer_k'] = np.log((d.I_kr_W - d.kr_W + 1) / (d.I_kr_P - d.kr_P + 1)); d['peer_f'] = np.log((d.I_fx_W - d.fx_W + 1) / (d.I_fx_P - d.fx_P + 1))
    d['cell_k'] = np.log((d.kr_W + 1) / (d.kr_P + 1))
    d = d[(d.tot_W > 0) | (d.tot_P > 0)].copy()
    mt = d.groupby('mg').tot_W.sum(); mp = d.groupby('mg').tot_P.sum()
    d['w'] = d.tot_W / d.mg.map(mt).replace(0, np.nan); d['wp'] = d.tot_P / d.mg.map(mp).replace(0, np.nan)
    o = pd.DataFrame(index=mt.index)
    o['c_ind_kmom'] = (d.w * d.peer_k).groupby(d.mg).sum(min_count=1); o['c_ind_fmom'] = (d.w * d.peer_f).groupby(d.mg).sum(min_count=1)
    o['c_n_ind'] = (d.w >= 0.05).groupby(d.mg).sum().astype(float)
    o['c_entropy'] = d.assign(e=-d.w * np.log(d.w.where(d.w > 0))).groupby('mg').e.sum(min_count=1)
    o['c_mix_shift'] = 0.5 * (d.w.fillna(0) - d.wp.fillna(0)).abs().groupby(d.mg).sum()
    o.loc[(mp == 0) | (mt == 0), 'c_mix_shift'] = np.nan
    o['c_cell_kmax'] = d[d.tot_W >= 10].groupby('mg').cell_k.max()
    o.loc[mt == 0, ['c_ind_kmom', 'c_ind_fmom', 'c_n_ind', 'c_entropy', 'c_cell_kmax']] = np.nan
    mgW = MG[(MG.year >= t0 - 2) & (MG.year <= t0)].groupby('mg').n.sum()
    o['c_cov'] = np.clip(mt / np.maximum(mgW.reindex(mt.index), 1), 0, 1)
    o['in_mgpanel'] = mt.index.isin(MG.mg.unique())
    o['t0'] = t0; allmine.append(o.reset_index())
M = pd.concat(allmine, ignore_index=True)
J = M.merge(ctxw, on=['mg', 't0'], how='outer', suffixes=('_me', '_w'), indicator=True)
R['ctx_rows'] = {str(k): int(v) for k, v in J._merge.value_counts().items()}
R['mg_not_in_mgpanel'] = int((~M.in_mgpanel.astype(bool)).sum())
JJ = J[J._merge == 'both']
for f in FE:
    sub = JJ if f != 'c_cov' else JJ[JJ.in_mgpanel == True]
    a, b = sub[f + '_me'], sub[f + '_w']
    R[f] = dict(nan_mismatch=int((a.isna() != b.isna()).sum()), max_abs=float(np.nanmax(np.abs(a - b))) if len(a) else None, n=int(len(a)))
R['secs'] = round(time.time() - t)
json.dump(R, open(V + '/r2_b.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, default=str))
