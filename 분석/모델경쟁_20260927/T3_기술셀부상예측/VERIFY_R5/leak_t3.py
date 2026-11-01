# T3 레드팀 — 시점 누수 재탐색(이전 감사가 보지 않은 항목). 읽기 전용(meta parquet, registry sqlite mode=ro, 동결 예측·캐시). 산출 VERIFY_R5/leak_t3.json
import os, sys, json, hashlib, time, re
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
META = os.path.join(T3, '..', '_shared', 'meta').replace(os.sep, '/') + '/*.parquet'
REG = os.path.join(ROOT, '수집파이프라인', 'registry_full.sqlite')
OUT = {}; t0_ = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")

# (a) 제공 데이터 출원인명 = 원출원인인가, 현재 권리자인가 (KR 등록 중 권리 전부이전된 건)
import sqlite3
_rc = sqlite3.connect('file:' + REG.replace(os.sep, '/') + '?mode=ro', uri=True)
_p = pd.read_sql('select rgst_no, applNo appl_no from patent', _rc)
_a = pd.read_sql('select rgst_no, seq, applicantName from applicant', _rc).sort_values(['rgst_no', 'seq']).drop_duplicates('rgst_no')
_o = pd.read_sql('select rgst_no, seq, finalOwnerYn, ownerName, ownerRgstCsDate, ownerRgstCsReason from owner', _rc)
_rc.close()
_f = _o[_o.finalOwnerYn == 'Y'].sort_values(['rgst_no', 'seq']).drop_duplicates('rgst_no', keep='last')[['rgst_no', 'ownerName']]
_t = _o[_o.ownerRgstCsReason.fillna('').str.contains('이전')].groupby('rgst_no').ownerRgstCsDate.min().rename('tdate').reset_index()
TR = _p.merge(_a[['rgst_no', 'applicantName']].rename(columns={'applicantName': 'orig'}), on='rgst_no', how='left')        .merge(_f.rename(columns={'ownerName': 'fin'}), on='rgst_no', how='left').merge(_t, on='rgst_no', how='left')
con.register('tr', TR); del _p, _a, _o, _f, _t
con.execute(f"create temp table mk as select appl_no, trim(split_part(applicant,'|',1)) app1, applicant from read_parquet('{META}') where cc='KR'")
STRIP = r"주식회사|\(주\)|㈜|유한회사|재단법인|사단법인|학교법인|[\s\.\,\-\(\)·&]"
d = con.sql("select t.*, m.app1, m.applicant from tr t join mk m using(appl_no)").df()
nz = lambda s: s.fillna('').astype(str).str.replace(STRIP, '', regex=True).str.lower()
d['o'] = nz(d.orig); d['f'] = nz(d.fin); d['m'] = nz(d.app1); d['mall'] = nz(d.applicant.str.replace('|', ' ', regex=False))
x = d[(d.f != '') & (d.o != '') & (d.o != d.f) & d.tdate.notna()]
OUT['a_applicant_name'] = dict(n_registered_linked=int(len(d)), n_transferred_name_changed=int(len(x)),
                               share_meta_eq_original=float((x.m == x.o).mean()), share_meta_eq_final_owner=float((x.m == x.f).mean()),
                               share_meta_contains_final_owner=float(np.mean([f in m for f, m in zip(x.f, x.mall)])),
                               all_registered_meta_eq_original=float((d.m == d.o).mean()))
print('(a)', OUT['a_applicant_name'], flush=True)
del d, x

# (b) KR 공개 월별 건수(공개연도 레이블 완결성: R3 최종 창 2022–24) 및 (d) 출원연도 피처의 원점 시점 미공개 비중
mon = con.sql(f"""select left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),6) ym, count(distinct appl_no) n
                 from read_parquet('{META}') where cc='KR' group by 1 having ym between '202101' and '202507' order by 1""").df()
yr = mon.assign(y=mon.ym.str[:4]).groupby('y').n.sum()
OUT['b_kr_pub_by_month'] = dict(by_year=yr.to_dict(), last_months=mon.tail(10).set_index('ym').n.to_dict(),
                                dec_over_monthly_mean={y: float(mon[mon.ym == f'{y}12'].n.sum() / mon[mon.ym.str[:4] == y].n.mean()) for y in ('2021', '2022', '2023', '2024')})
print('(b)', OUT['b_kr_pub_by_month'], flush=True)
vis = con.sql(f"""select cc, appl_year::int ay, count(*) n,
     sum((left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)::int > appl_year::int)::int) pub_after_year,
     sum((left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)::int > appl_year::int + 1)::int) pub_after_next
     from read_parquet('{META}') where appl_year between 2010 and 2020 group by 1,2""").df()
def unseen(t0):
    v = vis[(vis.ay >= t0 - 2) & (vis.ay <= t0)].copy()
    # t0년 말 기준 미공개: 출원연도 t0 → 공개 > t0, t0-1 → 공개 > t0, t0-2 → 공개 > t0 (근사: t0 출원은 pub_after_year, t0-1은 pub_after_next)
    a = v[v.ay == t0].pub_after_year.sum() + v[v.ay == t0 - 1].pub_after_next.sum()
    return a / v.n.sum()
OUT['d_appyear_feature_unseen_share'] = {str(t): dict(all_cc=float(unseen(t)), KR=float((lambda v: (v[(v.ay == t)].pub_after_year.sum() + v[v.ay == t - 1].pub_after_next.sum()) / v[(v.ay >= t - 2) & (v.ay <= t)].n.sum())(vis[vis.cc == 'KR']))) for t in (2013, 2016, 2019, 2020)}
print('(d)', OUT['d_appyear_feature_unseen_share'], flush=True)

# (c) R2 출원연도 레이블: 2023 절단의 셀 간 차등(해외 우선권 출원 비중) + 국내 우선권 문서만으로 레이블 재계산 민감도
BR = os.path.join(T3, 'R2', 'cache_2023', 'base_rows.parquet').replace(os.sep, '/')
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
con.execute(f"create temp table kr as select ksic, mgn(ipc_raw) mg, year, appl_no, max(prio_for) pf from read_parquet('{BR}') where cc='KR' and mgn(ipc_raw) is not null group by 1,2,3,4")
py = con.sql("select year, count(*) n, avg(pf) pf_share, sum(pf) n_pf, sum(1-pf) n_dom from kr where year between 2015 and 2023 group by 1 order by 1").df()
OUT['c_kr_app_prio_by_year'] = py.to_dict(orient='list')
print('(c) 연도별', py.to_string(), flush=True)
P2 = pd.read_parquet(os.path.join(T3, 'R2', 'preds', 'FINAL_t0_2019_2020.parquet'))
def hkey(cells): return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in cells], dtype=np.int64)
cres = {}
for t in (2019, 2020):
    c = con.sql(f"""select ksic||'|'||mg cell, sum(case when year between {t-2} and {t} and pf=0 then 1 else 0 end) past, sum(case when year between {t+1} and {t+3} and pf=0 then 1 else 0 end) fut,
                   sum(case when year between {t-2} and {t} then 1 else 0 end) past_all, sum(case when year between {t+1} and {t+3} then pf else 0 end) fut_pf,
                   sum(case when year between {t+1} and {t+3} then 1 else 0 end) fut_all
                   from kr where year between {t-2} and {t+3} group by 1""").df()
    g = P2[P2.t0 == t].merge(c, on='cell', how='left').fillna(0)
    g['y2'] = np.log((g.fut + 1) / (g.past + 1)); k = int(round(0.1 * len(g)))
    o = np.lexsort((hkey(g.cell.values), -(g.fut - g.past).values, -g.y2.values)); g['yb2'] = 0; g.loc[g.index[o[:k]], 'yb2'] = 1
    a, b = roc_auc_score(g.yb2, g.p), roc_auc_score(g.yb2, g.p_base)
    # 부상 셀의 미래창 해외우선권 비중 vs 비부상
    cres[str(t)] = dict(auc_dom_label=a, auc_base_dom_label=b, skill_dom_label=(a - b) / (1 - b), label_overlap=float((g.yb2 & g.yb).sum() / g.yb.sum()),
                        fut_pf_share_breakout=float(g[g.yb == 1].fut_pf.sum() / max(g[g.yb == 1].fut_all.sum(), 1)),
                        fut_pf_share_other=float(g[g.yb == 0].fut_pf.sum() / max(g[g.yb == 0].fut_all.sum(), 1)))
OUT['c_r2_domestic_priority_label'] = cres
print('(c) 국내우선권 레이블', cres, flush=True)

# (e) IPC 신설 코드 셀(재분류 소급) 비중 — R3 t0=2021 피처 창(공개 2019–21) KR·전체 문서 중
NEW_SUB = ('G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H'); NEW_MG = ('H04N 23', 'H04N 25', 'G06F 18')
PB = os.path.join(T3, 'R3', 'cache_pub_final', 'panel_ccy.parquet').replace(os.sep, '/')
e = con.sql(f"""select cc, sum(n) n, sum(case when left(mg,4) in {NEW_SUB} or mg in {NEW_MG} then n else 0 end) n_new
               from read_parquet('{PB}') where year between 2019 and 2021 group by 1""").df()
e['share'] = e.n_new / e.n
P3 = pd.read_parquet(os.path.join(T3, 'R3', 'preds', 'FINAL_R3_t0_2021.parquet'))
nm = P3.mg.str[:4].isin(NEW_SUB) | P3.mg.isin(NEW_MG)
OUT['e_new_ipc_codes'] = dict(doc_share_by_cc=e.set_index('cc').share.to_dict(), cells=int(nm.sum()), breakout_rate_new=float(P3[nm].yb.mean()), breakout_rate_other=float(P3[~nm].yb.mean()),
                              top10pct_share_new=float(nm[P3.p.rank(ascending=False) <= 504].mean()))
print('(e)', OUT['e_new_ipc_codes'], flush=True)
OUT['secs'] = round(time.time() - t0_)
json.dump(OUT, open(os.path.join(HERE, 'leak_t3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('저장 leak_t3.json', OUT['secs'])
