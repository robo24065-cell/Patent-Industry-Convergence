# C19-T1 s2: JP 계열별 연도 건수·공식 대비 포착률(a), 공개일 없는 행 성격(b)·이중 수록, 시점(d), EP 등록 대조
import sys, os, csv; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd, numpy as np
con = con3()
J = f"read_parquet('{ddb(C)}/jp_rows.parquet')"
# 행 분류
con.execute(f"""create table j as select *,
  (pub_date is null) nopub, (rgst_date is not null) hasrg, appl_no like '%U' u,
  case when rgst_no is null then 'none' when regexp_matches(rgst_no,'^[0-9]{{10}}$') then 'pubno10'
       when regexp_matches(rgst_no,'^[0-9]{{7,8}}$') then 'patno' else 'other' end rgfmt,
  case when regexp_matches(appl_no,'^[0-9]{{8}}U?$') then 'a8' when regexp_matches(appl_no,'^[0-9]{{10}}U?$') then 'a10'
       when appl_no like 'PCT/%' then 'pct' else 'aother' end apfmt,
  lower(regexp_replace(coalesce(title,''),'[[:space:]]','','g')) tnorm
  from {J}""")
R = {}
# ---------- (a) 연도별 계열 ----------
q = """select y, sum(n) n from (select {ycol} y, 1 n from j where {cond}) where y between '2010' and '2025' group by y order by y"""
S = {
 'pub_pat': ("left(pub_date,4)", "not nopub and not u"),
 'pub_u': ("left(pub_date,4)", "not nopub and u"),
 'nopub_pat': ("left(rgst_date,4)", "nopub and hasrg and not u"),
 'nopub_pat_pubno10': ("left(rgst_date,4)", "nopub and rgfmt='pubno10' and not u"),
 'nopub_pat_patno': ("left(rgst_date,4)", "nopub and rgfmt='patno' and not u"),
 'nopub_u': ("left(rgst_date,4)", "nopub and hasrg and u"),
 'both_pat': ("left(rgst_date,4)", "not nopub and hasrg and not u"),
 'realreg_pat': ("left(rgst_date,4)", "rgfmt='patno' and not u"),
 'realreg_u': ("left(rgst_date,4)", "rgfmt='patno' and u"),
 'pubevent_pat': ("left(coalesce(pub_date, case when rgfmt='pubno10' then rgst_date end),4)", "not u and (not nopub or rgfmt='pubno10')"),
 't3py_all_a2000': ("py", "appl_year>=2000"),
 't3py_nopub_pat_a2000': ("py", "appl_year>=2000 and nopub and not u"),
}
tab = {}
for k, (yc, cd) in S.items():
    for y, n in con.sql(q.format(ycol=yc, cond=cd)).fetchall():
        tab.setdefault(int(y), {})[k] = int(n)
yrs = sorted(tab)
# 공식 (JPO 특허행정연차보고서 '공보 게재 건수' 표; sources.txt) 2013–2024
OFF = {
 'jpo_tokkyo_koho': {2013: 268800, 2014: 267569, 2015: 190596, 2016: 217778, 2017: 199592, 2018: 191073, 2019: 180399, 2020: 179017, 2021: 181368, 2022: 210998, 2023: 208948, 2024: 200511},
 'jpo_kokai': {2013: 258913, 2014: 241728, 2015: 233414, 2016: 226294, 2017: 229241, 2018: 207786, 2019: 221136, 2020: 205747, 2021: 197906, 2022: 188382, 2023: 184498, 2024: 180270},
 'jpo_kohyo': {2013: 46302, 2014: 35256, 2015: 37504, 2016: 41230, 2017: 39200, 2018: 38787, 2019: 37923, 2020: 37866, 2021: 36725, 2022: 52686, 2023: 54671, 2024: 47211},
 'jpo_saikohyo': {2013: 17932, 2014: 13256, 2015: 18486, 2016: 10496, 2017: 53986, 2018: 23178, 2019: 25079, 2020: 21074, 2021: 31057, 2022: 0, 2023: 0, 2024: 0},
 'jpo_um_koho': {2013: 7379, 2014: 7050, 2015: 6655, 2016: 6482, 2017: 6027, 2018: 5229, 2019: 5052, 2020: 5437, 2021: 5499, 2022: 4841, 2023: 4811, 2024: 4502},
}
rows = []
for y in yrs:
    r = {'year': y, **{k: tab[y].get(k, 0) for k in S}}
    for k, d in OFF.items():
        r[k] = d.get(y)
    if r['jpo_kokai'] is not None:
        pubtot = r['jpo_kokai'] + r['jpo_kohyo'] + r['jpo_saikohyo']
        r['jpo_pub_total'] = pubtot
        r['cap_pub_pat'] = r['pub_pat'] / pubtot
        r['cap_pubevent_pat'] = r['pubevent_pat'] / pubtot
        r['cap_nopub_pat_vs_koho'] = r['nopub_pat'] / r['jpo_tokkyo_koho']
        r['cap_nopub_patno_vs_koho'] = r['nopub_pat_patno'] / r['jpo_tokkyo_koho']
        r['cap_both_vs_koho'] = r['both_pat'] / r['jpo_tokkyo_koho']
        r['cap_realreg_vs_koho'] = r['realreg_pat'] / r['jpo_tokkyo_koho']
        r['cap_realreg_u_vs_um'] = r['realreg_u'] / r['jpo_um_koho']
    rows.append(r)
df = pd.DataFrame(rows)
base = df[(df.year >= 2016) & (df.year <= 2020)]
for c in [c for c in df.columns if c.startswith('cap_')]:
    rb = base[c].mean(); df['rel_' + c[4:]] = df[c] / rb; R['base_' + c] = float(rb)
df.to_csv(os.path.join(W, 'jp_capture.csv'), index=False, encoding='utf-8-sig')
out(df[['year', 'pub_pat', 'nopub_pat', 'nopub_pat_pubno10', 'nopub_pat_patno', 'both_pat', 'realreg_pat', 'pubevent_pat', 'jpo_pub_total', 'jpo_tokkyo_koho']].to_string())
out(df[['year'] + [c for c in df.columns if c.startswith('rel_')]].round(3).to_string())
# 월별 (2019-01~2024-12): 공개일 행 / 공개번호-in-rgst 행 / 진짜 등록(공개일 없음)/ 동시 보유
R['monthly'] = con.sql("""select m, sum(pubd) pub_date_rows, sum(p10) pubno10_rows, sum(nopat) nopub_patno_rows, sum(bothr) both_rows from (
   select left(pub_date,6) m, 1 pubd, 0 p10, 0 nopat, 0 bothr from j where not nopub and not u
   union all select left(rgst_date,6), 0, 1, 0, 0 from j where rgfmt='pubno10' and not u
   union all select left(rgst_date,6), 0, 0, 1, 0 from j where nopub and rgfmt='patno' and not u
   union all select left(rgst_date,6), 0, 0, 0, 1 from j where not nopub and hasrg and not u) where m between '201901' and '202412' group by m order by m""").fetchall()
pd.DataFrame(R['monthly'], columns=['ym', 'pub_date_rows', 'pubno10_rows', 'nopub_patno_rows', 'both_rows']).to_csv(os.path.join(C, 's2_monthly.csv'), index=False)
# ---------- (b) 공개일 없는 행 성격 ----------
con.execute("""create table np as select *, case when left(rgst_date,4) between '2021' and '2022' then 'P2122' when left(rgst_date,4) between '2012' and '2020' then 'P1220' else 'other' end per
  from j where nopub and hasrg""")
R['b_fmt'] = con.sql("select per, u, rgfmt, apfmt, (applicant is null) noappl, (family is null) nofam, (prio_cc is null) noprio, count(*) from np where per<>'other' group by all order by per, 7 desc").fetchall()
R['b_lag'] = con.sql("""select per, rgfmt, count(*), quantile_cont(date_diff('day', strptime(appl_date,'%Y%m%d'), strptime(rgst_date,'%Y%m%d'))/365.25, [0.1,0.25,0.5,0.75,0.9]) from np where per<>'other' and not u group by all order by 1,2""").fetchall()
R['b_applyear'] = con.sql("select per, appl_year, count(*) from np where per<>'other' and not u group by all order by 1,2").fetchall()
ks = con.sql("select per, ksic, count(*) n from np where per<>'other' and not u group by all").df()
pv = ks.pivot(index='ksic', columns='per', values='n').fillna(0); pv = pv / pv.sum()
R['b_ksic_tvd'] = float(0.5 * (pv['P2122'] - pv['P1220']).abs().sum())
R['b_ksic_top'] = pv.assign(d=pv.P2122 - pv.P1220).sort_values('d').iloc[[0, 1, 2, -3, -2, -1]].round(4).reset_index().values.tolist()
# 출원인 유형(규칙 문자열)
R['b_appltype'] = con.sql("""select per, case when applicant is null then '결측' when regexp_matches(applicant,'株式会社|有限会社|合同会社|Corporation|Corp|Inc|Ltd|LTD|GmbH|AG|S\\.A|大学|機構|研究所|法人|Company|CO\\.|LLC|B\\.V|N\\.V|AB') then '법인' else '개인추정' end t, count(*)
  from np where per<>'other' and not u group by all order by 1,2""").fetchall()
# 이중 수록: 같은 정규화 키(jkey) 또는 (정규화 제목, 출원일)로 공개일 가진 다른 행
con.execute("create table pubk as select jkey, ksic, appl_no from j where not nopub")
con.execute("create table pubt as select tnorm, appl_date, ksic, appl_no from j where not nopub and tnorm<>''")
con.execute("""create table npd as select np.appl_no, np.per, np.rgfmt, np.u, np.ksic, np.rgst_date, np.appl_date,
   exists(select 1 from pubk where pubk.jkey=np.jkey and pubk.appl_no<>np.appl_no) k1,
   exists(select 1 from pubk where pubk.jkey=np.jkey and pubk.appl_no<>np.appl_no and pubk.ksic=np.ksic) k1_same_ksic,
   exists(select 1 from pubt where pubt.tnorm=np.tnorm and pubt.appl_date=np.appl_date and pubt.appl_no<>np.appl_no and np.tnorm<>'') k2
   from np""")
R['b_dup'] = con.sql("""select per, rgfmt, u, count(*) n, sum(k1::int) k1, sum(k1_same_ksic::int) k1_same_ksic, sum(k2::int) k2, sum((k1 or k2)::int) k12 from npd where per<>'other' group by all order by 1,2,3""").fetchall()
con.execute(f"copy (select appl_no, per, rgfmt, u, k1, k2 from npd where k1 or k2) to '{ddb(C)}/s2_dup_rows.parquet' (format parquet)")
# 동일 키 충돌 전체(jkey 같은 서로 다른 appl_no) 성격
R['jkey_collisions'] = con.sql("""select a.apfmt, b.apfmt, (a.nopub) a_nopub, (b.nopub) b_nopub, count(*) n, avg((a.tnorm=b.tnorm)::int) title_eq, avg((a.appl_date=b.appl_date)::int) date_eq, avg((a.u=b.u)::int) u_eq
   from j a join j b on a.jkey=b.jkey and a.appl_no<b.appl_no group by all order by n desc limit 12""").fetchall()
# 공개번호-in-rgst(pubno10) 행: 같은 공개번호가 다른 행의 rgst_no/어떤 필드에 있는지 → 다른 행이 같은 출원(제목·출원일)인지
R['pubno10_title_date_match_any'] = con.sql("""select count(*), sum(e::int) from (select exists(select 1 from j b where b.tnorm=a.tnorm and b.appl_date=a.appl_date and b.appl_no<>a.appl_no) e from j a where a.rgfmt='pubno10' and a.tnorm<>'')""").fetchone()
# ---------- (d) 시점: 공개일 없는 행의 출원+18개월 대비 rgst ----------
R['d_shift'] = con.sql("""select per, rgfmt, count(*) n,
   sum((strptime(rgst_date,'%Y%m%d') > strptime(appl_date,'%Y%m%d') + interval 18 month + interval 90 day)::int) late_gt_21m,
   sum((left(rgst_date,4) <> strftime(strptime(appl_date,'%Y%m%d') + interval 18 month,'%Y'))::int) year_differs,
   sum((left(rgst_date,4) > strftime(strptime(appl_date,'%Y%m%d') + interval 18 month,'%Y'))::int) year_later,
   sum((strptime(rgst_date,'%Y%m%d') < strptime(appl_date,'%Y%m%d') + interval 18 month - interval 30 day)::int) early
   from np where per<>'other' and not u and appl_date is not null group by all order by 1,2""").fetchall()
# ---------- EP 등록 대조 ----------
E = f"read_parquet('{ddb(C)}/ep_rows.parquet')"
R['ep_rg_year'] = con.sql(f"select left(rgst_date,4) y, count(*) n, count(distinct appl_no) nu from {E} where rgst_date is not null and left(rgst_date,4) between '2012' and '2025' group by 1 order by 1").fetchall()
R['ep_rg_month_2020_2023'] = con.sql(f"select left(rgst_date,6) m, count(*) from {E} where rgst_date between '20200101' and '20231231' group by 1 order by 1").fetchall()
R['ep_rg_drives_py'] = con.sql(f"""select count(*) n_all, sum((pub_date is null and rgst_date is not null)::int) nopub,
   sum((rgst_date < pub_date)::int) rg_before_pub,
   sum(((pub_date is null or rgst_date < pub_date) and (rgst_date between '20210101' and '20210631' or rgst_date between '20220401' and '20220931'))::int) ve1_rows
   from {E} where appl_year>=2000""").fetchone()
jdump(R, 's2_audit.json')
for k, v in R.items():
    if k == 'monthly':
        continue
    out(k)
    if isinstance(v, list):
        [out('  ', r) for r in v]
    else:
        out('  ', v)
