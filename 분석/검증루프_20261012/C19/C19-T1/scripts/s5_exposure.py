# C19-T1 s5: (c) T3 최종 시험 셀(t0=2021)별 JP 급증 행 노출 e 와 선택 편향 비교 → exposure_cells.csv
import sys, os; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd, numpy as np
con = con3()
src = open(os.path.join(T3, 'build_panel.py'), encoding='utf-8').read()
macro = src[src.index('con.execute(r"""create macro'):src.index('con.execute(rf"""create view b')]
exec(macro)
BR = ddb(os.path.join(R3, 'cache_pub_final', 'base_rows.parquet'))
NP = ddb(os.path.join(C, 'jp_nopub_rows.parquet'))
con.execute(f"""create table b as select distinct ksic, mgn(ipc_raw) mg, cc, year, appl_no from read_parquet('{BR}') where mgn(ipc_raw) is not null and year between 2019 and 2021""")
con.execute(f"create table npr as select appl_no, py, rgfmt, dup from read_parquet('{NP}')")
cell = con.sql("""select b.ksic||'|'||b.mg cell, count(*) tot, sum((cc='JP')::int) jp, sum((cc='KR')::int) kr,
   sum((cc='JP' and npr.appl_no is not null and b.year=2021)::int) jp_nopub21,
   sum((cc='JP' and npr.appl_no is not null and npr.rgfmt='pubno10' and b.year=2021)::int) jp_pubno10_21,
   sum((cc='JP' and npr.appl_no is not null and npr.dup)::int) jp_dup1921,
   sum((cc='JP' and b.year=2021)::int) jp21, sum((cc='JP' and b.year=2020)::int) jp20
   from b left join npr on b.cc='JP' and npr.appl_no=b.appl_no group by 1""").df()
P = pd.read_parquet(os.path.join(C, 't3res', 'preds_P0_final.parquet'))  # 내 P0 재현 예측(공식 예측과 대조는 s6)
d = P.merge(cell, on='cell', how='left')
assert d.tot.notna().all(), d.tot.isna().sum()
d['e'] = d.jp_nopub21 / d.tot; d['jp_share'] = d.jp / d.tot; d['dup_share'] = d.jp_dup1921 / d.tot
thr = d.e.quantile(0.9); d['hi'] = (d.e >= thr) & (d.e > 0)
g = d.groupby('hi').agg(n=('cell', 'size'), base_rate=('yb', 'mean'), tot_W_med=('tot_W', 'median'), tot_med=('tot', 'median'), jp_share_mean=('jp_share', 'mean'),
                        e_mean=('e', 'mean'), dup_share_mean=('dup_share', 'mean'), kr_W_med=('kr_W', 'median'))
ks = d.groupby(['hi', 'ksic']).size().unstack(0).fillna(0); ks = ks / ks.sum()
tvd = float(0.5 * (ks[True] - ks[False]).abs().sum())
d[['cell', 'ksic', 'mg', 'tot', 'jp', 'kr', 'jp20', 'jp21', 'jp_nopub21', 'jp_pubno10_21', 'jp_dup1921', 'e', 'jp_share', 'dup_share', 'hi', 'yb', 'p', 'p_base', 'tot_W', 'kr_W']].to_csv(
    os.path.join(W, 'exposure_cells.csv'), index=False, encoding='utf-8-sig')
R = dict(n_cells=int(len(d)), e_q90=float(thr), n_hi=int(d.hi.sum()), cells_with_any=int((d.e > 0).sum()), group=g.reset_index().to_dict('records'), ksic_tvd_hi_vs_rest=tvd,
         top_ksic_hi=ks[True].sort_values(ascending=False).head(6).round(3).to_dict(),
         jp21_over_jp20_hi=float(d[d.hi].jp21.sum() / d[d.hi].jp20.sum()), jp21_over_jp20_rest=float(d[~d.hi].jp21.sum() / d[~d.hi].jp20.sum()),
         share_testcells_jp_nopub21_total=int(d.jp_nopub21.sum()))
jdump(R, 's5_exposure.json'); out(R)
