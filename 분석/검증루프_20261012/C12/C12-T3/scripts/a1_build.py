# -*- coding: utf-8 -*-
"""(A) 1단계: 제공 메타 원천(base_rows)에서 셀 패널·KR 문서표를 자체 코드로 만들고, 메타 rgst_no·등록원부 전체 출원인을 붙인다.
셀 정의(C04-T3 s1과 같은 규칙): mgn(ipc_raw), (ksic, mg, cc, year, appl_no) 고유, year = 공개연도(base_rows.year).
출력: cache/a_panel.parquet, cache/a_krdocs.parquet(셀·연도·출원번호·제공 이름·rgst_no), cache/a_rfapp.parquet(등록원부 출원인, 필요한 rgst_no만)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t = time.time(); log = {}
con = duck()
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
C = CACHE.replace('\\', '/')
con.execute(f"""copy (select ksic, mg, cc, year, count(distinct appl_no) n from (select ksic, mgn(ipc_raw) mg, cc, year, appl_no
   from read_parquet('{BASE_ROWS}') where year between 2014 and 2024) where mg is not null group by all) to '{C}/a_panel.parquet' (format parquet)""")
print('panel', round(time.time() - t), flush=True)
con.execute(f"""copy (with d as (select ksic, mgn(ipc_raw) mg, year, appl_no, app1 from read_parquet('{BASE_ROWS}') where cc='KR' and year between 2016 and 2024),
   d2 as (select ksic, mg, year, appl_no, any_value(app1) app_raw, count(distinct app1) n_app_vals from d where mg is not null group by ksic, mg, year, appl_no),
   m as (select appl_no, any_value(nullif(trim(rgst_no),'')) rgst_no, count(*) n_meta from read_parquet('{META}') where cc='KR' group by appl_no)
   select d2.*, m.rgst_no, m.n_meta from d2 left join m using (appl_no)) to '{C}/a_krdocs.parquet' (format parquet)""")
kd = pd.read_parquet(os.path.join(CACHE, 'a_krdocs.parquet'))
log['krdocs_rows'] = len(kd); log['krdocs_multi_app_vals'] = int((kd.n_app_vals > 1).sum()); log['meta_unmatched'] = int(kd.n_meta.isna().sum())
log['with_rgst_no'] = int(kd.rgst_no.notna().sum()); log['utility_model_rows'] = int(kd.appl_no.str.startswith('20').sum())
print(log, round(time.time() - t), flush=True)
need = set(kd.rgst_no.dropna().unique())
c = ro(RF)
rfp = pd.read_sql("select rgst_no, applNo, rgstDate from patent", c)
rfa = pd.read_sql("select rgst_no, seq, applicantCd, applicantName, applicantNatl from applicant", c)
log['rf_patent_rows'] = len(rfp); log['rf_applicant_rows'] = len(rfa); log['rf_rgstDate_max'] = str(rfp.rgstDate.max()); log['rf_rgstDate_min'] = str(rfp.rgstDate.min())
# 연결 키 점검: 메타 rgst_no ↔ 등록원부 rgst_no, 출원번호 ↔ applNo
log['need_rgst'] = len(need); log['need_in_rf'] = int(len(need & set(rfp.rgst_no)))
am = kd[['appl_no', 'rgst_no']].drop_duplicates().merge(rfp[['rgst_no', 'applNo']], on='rgst_no', how='inner')
log['applno_agree'] = float((am.appl_no == am.applNo).mean()) if len(am) else None
rfa = rfa[rfa.rgst_no.isin(need)].copy()
rfa.to_parquet(os.path.join(CACHE, 'a_rfapp.parquet'), index=False)
rfp[rfp.rgst_no.isin(need)].to_parquet(os.path.join(CACHE, 'a_rfpat.parquet'), index=False)
log['rfapp_kept'] = len(rfa); log['sec'] = round(time.time() - t)
json.dump(log, open(os.path.join(CACHE, 'a1_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(log)
