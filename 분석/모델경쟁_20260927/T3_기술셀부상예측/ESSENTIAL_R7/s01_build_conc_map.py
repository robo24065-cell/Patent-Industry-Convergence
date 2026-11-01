# -*- coding: utf-8 -*-
"""R7 s01: 제공 레코드(5개국, cc+appl_no)마다 공식 연계표 산업코드를 배정하고, 제공 KSIC 라벨과의 일치율을 잰다.
출력(ESSENTIAL_R7/): rec_conc.parquet (cc, appl_no, sg, code, level, p), conc_ksic_map.csv, conc_agreement.json, conc_agreement_by_ksic.csv
제공 데이터는 로컬에서만 읽는다(외부 전송 없음)."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, duckdb
import conc_lib as CL

t = time.time()
cc, bad = CL.load_concordance()
codes = sorted(cc.code.unique())
km = CL.ksic10_to_code(codes)
kmap = pd.DataFrame([dict(ksic=k, **v) for k, v in km.items()])
kmap.to_csv(os.path.join(HERE, 'conc_ksic_map.csv'), index=False, encoding='utf-8-sig')
codes50 = sorted(kmap.code.unique())
print('연계표 코드', len(codes), '/ 50개 KSIC가 걸치는 코드', len(codes50), '| 정규화 실패 비율', round(bad, 5), flush=True)
A1, A2, A3 = CL.argmax_table(cc, 'sg'), CL.argmax_table(cc, 'mg'), CL.argmax_table(cc, 'sc')

con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute(CL.SQL_MACRO)
# 매크로와 파이썬 정규화가 같은지 표본 대조
smp = con.sql(f"""select distinct regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\\s+', ' ', 'g') s
                 from read_parquet('{CL.META}') using sample 20000""").df()
smp['py'] = smp.s.map(CL.norm_sg); smp['sql'] = con.execute("select r7_sg(s) from smp").df().iloc[:, 0].values
mism = int(((smp.py.fillna('~')) != (smp.sql.fillna('~'))).sum())
print('정규화 대조 불일치', mism, '/', len(smp), flush=True)
assert mism == 0
con.register('a1', A1); con.register('a2', A2); con.register('a3', A3); con.register('km', kmap)
OUT = os.path.join(HERE, 'rec_conc.parquet').replace(os.sep, '/')
con.execute(f"""copy (
  with r as (select cc, appl_no, any_value(r7_sg(regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\\s+', ' ', 'g'))) sg
             from read_parquet('{CL.META}') group by 1, 2)
  select r.cc, r.appl_no, r.sg,
     coalesce(a1.code_sg, a2.code_mg, a3.code_sc) code,
     case when a1.code_sg is not null then 1 when a2.code_mg is not null then 2 when a3.code_sc is not null then 3 else 0 end lvl,
     coalesce(a1.p_sg, a2.p_mg, a3.p_sc) p
  from r left join a1 on r.sg = a1.sg left join a2 on split_part(r.sg,'/',1) = a2.mg left join a3 on left(r.sg,4) = a3.sc
) to '{OUT}' (format parquet)""")
print('rec_conc 저장', round(time.time() - t), flush=True)
c50 = "(" + ",".join(f"'{c}'" for c in codes50) + ")"
q = f"""with m as (select m.cc, m.appl_no, m.ksic, m.appl_year, km.code kcode from read_parquet('{CL.META}') m join km using(ksic))
select m.cc, count(*) n_rows,
  avg((r.lvl=1)::int) lv1, avg((r.lvl=2)::int) lv2, avg((r.lvl=3)::int) lv3, avg((r.lvl=0)::int) lv0,
  avg((r.code = m.kcode)::int) agree_code,
  avg((r.code in {c50})::int) in_frame27,
  avg(case when r.code in {c50} then (r.code = m.kcode)::int end) agree_given_inframe
from m join read_parquet('{OUT}') r using(cc, appl_no) where m.appl_year between 2000 and 2021 group by rollup(m.cc) order by m.cc"""
ag = con.sql(q).df()
print(ag.to_string(), flush=True)
byk = con.sql(f"""with m as (select m.cc, m.appl_no, m.ksic, km.code kcode from read_parquet('{CL.META}') m join km using(ksic) where m.appl_year between 2000 and 2021)
  select m.ksic, m.kcode, count(*) n, avg((r.code=m.kcode)::int) agree_all, avg(case when m.cc='KR' then (r.code=m.kcode)::int end) agree_kr,
    (select count(*) from km k2 where k2.code=m.kcode) n_ksic_sharing_code
  from m join read_parquet('{OUT}') r using(cc, appl_no) group by 1, 2 order by 1""").df()
byk.to_csv(os.path.join(HERE, 'conc_agreement_by_ksic.csv'), index=False, encoding='utf-8-sig')
res = dict(created=time.strftime('%Y-%m-%d %H:%M'), n_concordance_codes=len(codes), n_codes_covering_50=len(codes50), codes50=codes50,
           concordance_norm_fail=bad, rule='메인 IPC 서브그룹 최다 → 메인그룹 → 서브클래스 (conc_lib.py)', ksic9_link='국가데이터처 산업·직업분류 연계표(2026-04-30) KSIC 10→9',
           max_ksic_per_code=int(kmap.groupby('code').size().max()), ksic_alone_in_code=int((kmap.groupby('code').size() == 1).sum()),
           by_cc={str(r.cc if r.cc == r.cc else 'ALL'): {k: float(r[k]) for k in ['n_rows', 'lv1', 'lv2', 'lv3', 'lv0', 'agree_code', 'in_frame27', 'agree_given_inframe']} for _, r in ag.iterrows()},
           note='일치율은 연계표 해상도(27개 코드) 기준. 5자리 50개 해상도에서는 연계표가 같은 코드 안 KSIC를 구분할 수 없다(코드당 최대 KSIC 수 참조).')
json.dump(res, open(os.path.join(HERE, 'conc_agreement.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'by_cc'}, ensure_ascii=False), round(time.time() - t))
