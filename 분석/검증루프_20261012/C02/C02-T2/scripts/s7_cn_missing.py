# C02-T2 s7: CN main_ipc 결측 행 프로파일(결측 vs 비결측) → missing_ipc_profile.csv, cache/s7_*.csv
# 결측 정의 = T3 build_base_pub 의 ipc_raw(coalesce(main_ipc, ipc_all 첫 값)) 가 비어 있음. 단위 = 제공 행(appl_no×ksic) 고유.
import os, sys, json
import duckdb, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
W = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2"); C = os.path.join(W, 'cache')
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
con.sql(f"""create temp table c as select distinct ksic, appl_no, appl_year::int ay, nullif(pub_date,'') pd, nullif(rgst_date,'') rd,
   coalesce(least(nullif(pub_date,''),nullif(rgst_date,'')),nullif(pub_date,''),nullif(rgst_date,'')) td,
   (coalesce(nullif(main_ipc,''), nullif(split_part(ipc_all,'|',1),'')) is null) miss,
   (nullif(main_cpc,'') is null) cpc_null, (nullif(ipc_all,'') is null) ipcall_null,
   case when length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') then 'inv'
        when length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9') then 'um' else 'other' end dt,
   case when prio_cc is not null and regexp_replace(replace(prio_cc,'CN',''),'\\|','','g')<>'' then 1 else 0 end prio_for,
   (nullif(applicant,'') is null) app_null, (nullif(title,'') is null) title_null, (nullif(family,'') is null) fam_null
  from read_parquet('{META}') where cc='CN'""")
con.sql("alter table c add column ty int; update c set ty = left(td,4)::int")
res = {}
res['rows'] = con.sql("select count(*), count(distinct appl_no), sum(miss::int), sum((miss and ay>=2000)::int) from c").fetchall()
# 연도(T3 공개연도) × 결측 프로파일
prof = con.sql("""select ty, miss, count(*) n, count(distinct appl_no) n_appl,
   avg((dt='inv')::int) sh_inv, avg((dt='um')::int) sh_um, avg((dt='other')::int) sh_other,
   avg((pd is not null)::int) sh_has_pub, avg((rd is not null)::int) sh_has_rgst,
   avg((pd is not null and rd is not null and rd<pd)::int) sh_rgst_before_pub,
   avg(ay) mean_ay, avg((ty-ay=0)::int) sh_lag0, avg((ty-ay=1)::int) sh_lag1, avg((ty-ay>=2)::int) sh_lag2p,
   avg(prio_for) sh_prio_for, avg(app_null::int) sh_app_null, avg(title_null::int) sh_title_null, avg(fam_null::int) sh_fam_null,
   avg(cpc_null::int) sh_cpc_null, avg(ipcall_null::int) sh_ipcall_null
   from c where ty between 2018 and 2025 group by 1,2 order by 1,2""").df()
# 월 분포 (T3 공개일 월)
mon = con.sql("""select ty, substr(td,5,2) m, sum(miss::int) n_miss, count(*) n, avg(miss::int) miss_rate from c where ty between 2020 and 2024 group by 1,2 order by 1,2""").df()
mon.to_csv(os.path.join(C, 's7_cn_month.csv'), index=False)
# 날짜(일) 집중: 결측 상위 공개일
topd = con.sql("""select td, sum(miss::int) n_miss, count(*) n from c where ty in (2021,2022,2024) group by 1 order by 2 desc limit 30""").df()
topd.to_csv(os.path.join(C, 's7_cn_topdates.csv'), index=False)
# doc type × 날짜 출처(pub/rgst 중 어느 것이 T3 날짜인지)
src = con.sql("""select ty, miss, dt, case when pd is not null and (rd is null or pd<=rd) then 'pub' else 'rgst' end src, count(*) n
                 from c where ty between 2019 and 2024 group by all order by all""").df()
src.to_csv(os.path.join(C, 's7_cn_dt_src.csv'), index=False)
# KSIC × 연도 결측률
ks = con.sql("""select ty, ksic, count(*) n, sum(miss::int) n_miss from c where ty between 2015 and 2025 group by 1,2""").df()
ks.to_csv(os.path.join(C, 's7_cn_ksic_year.csv'), index=False)
# 출원연도별 결측(지형도 = 출원 2017–19, 연도 정의 무관 전체 행)
ay = con.sql("""select ay, count(*) n, sum(miss::int) n_miss, sum((miss and ty=2021)::int) n_miss_t2021 from c where ay between 2010 and 2025 group by 1 order by 1""").df()
ay.to_csv(os.path.join(C, 's7_cn_applyear.csv'), index=False)
con.close()

# ---- 판정 지표 (2021, 보조 2022·2024)
rows = []
for y in [2020, 2021, 2022, 2023, 2024]:
    p = prof[prof.ty == y]
    if p.miss.sum() == 0: continue
    a = p[p.miss].iloc[0]; b = p[~p.miss].iloc[0]
    k = ks[ks.ty == y].copy(); k['nm'] = k.n - k.n_miss
    pm = k.n_miss / k.n_miss.sum(); pn = k.nm / k.nm.sum(); tvd = 0.5 * (pm - pn).abs().sum()
    r_all = k.n_miss.sum() / k.n.sum(); k['r'] = k.n_miss / k.n; k['bias'] = (1 - k.r) / (1 - r_all) - 1
    out = dict(year=y, miss_rate=r_all, n_miss=int(k.n_miss.sum()), n=int(k.n.sum()), ksic_tvd=tvd,
               d_sh_inv=a.sh_inv - b.sh_inv, d_sh_um=a.sh_um - b.sh_um, d_sh_prio_for=a.sh_prio_for - b.sh_prio_for,
               n_ksic_absbias_gt10=int((k.bias.abs() > 0.10).sum()), bias_min=k.bias.min(), bias_max=k.bias.max(),
               ksic_r_min=k.r.min(), ksic_r_max=k.r.max(), ksic_r_median=k.r.median())
    out['verdict'] = 'MCAR 근사' if (tvd <= 0.05 and abs(out['d_sh_inv']) <= 0.05 and abs(out['d_sh_prio_for']) <= 0.05) else '비무작위'
    rows.append(out)
J = pd.DataFrame(rows)
# 긴 형식 프로파일 저장
L = prof.melt(id_vars=['ty', 'miss'], var_name='metric', value_name='value')
L['group'] = np.where(L.miss, 'missing', 'non_missing'); L = L.drop(columns='miss').rename(columns={'ty': 'year'})
J2 = J.melt(id_vars=['year'], var_name='metric', value_name='value'); J2['group'] = 'test'
kk = ks[ks.ty == 2021].copy(); r_all = kk.n_miss.sum() / kk.n.sum(); kk['value'] = (1 - kk.n_miss / kk.n) / (1 - r_all) - 1
K2 = pd.DataFrame({'year': 2021, 'metric': 'ksic_count_bias_' + kk.ksic, 'value': kk.value, 'group': 'ksic'})
K3 = pd.DataFrame({'year': 2021, 'metric': 'ksic_miss_rate_' + kk.ksic, 'value': kk.n_miss / kk.n, 'group': 'ksic'})
pd.concat([L, J2, K2, K3], ignore_index=True)[['year', 'group', 'metric', 'value']].to_csv(os.path.join(W, 'missing_ipc_profile.csv'), index=False, encoding='utf-8-sig')
res['tests'] = J.to_dict('records')
json.dump(res, open(os.path.join(C, 's7_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(res['rows']); print(prof.round(3).to_string()); print(J.round(4).to_string())
print(mon[mon.ty.isin([2021, 2022, 2024])].round(3).to_string()); print(topd.head(15).to_string()); print(src.to_string()); print(ay.to_string())
