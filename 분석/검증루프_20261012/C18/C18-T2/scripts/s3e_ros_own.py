# -*- coding: utf-8 -*-
r"""C18-T2 s3e: T3 RoS 셀 피처(ros_sci·ros_ppp, 동결 스펙 xgbc·nb 성분)의 D2 노출 — US 자국 rgst_no('own') 연결.
 r3ros_01_family.py 규칙 그대로 family 토큰 연결과 own 연결을 따로 만들고, build_ros_pubrule_r3d.py 규칙으로 (KSIC, mg, US특허) 연결표를 재구성.
 (1) P0: 재구성 표 합계가 공식 R3/ros_cell_issueyear_pubrule.parquet 와 같은지(재현) (2) own 연결만 있는(=family 토큰으로는 안 붙는) 연결의 발행연도 분포
 (3) V2r: own-only 연결을 뺀 표 → cache/ros_pubrule_V2r.parquet (4) 공식 build_ext_r3b.ros_feats 를 그 표로 다시 돌려 extB_cell_pub_final 의 ros_* 열만 교체한 사본
     → cache/extB_V2r/extB_cell_pub_final.parquet (P0 재현: 공식 표로 돌린 ros_* 가 공식 파일과 같은지 먼저 확인)
 사용: python s3e_ros_own.py links|feats   (원자료·공식 폴더 읽기만)"""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
X = ddb(MC + r'\_shared\ext')
mode = sys.argv[1]; t = time.time()
if mode == 'links':
    con = con3()
    con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
    con.execute(f"""create table rec as select cc::varchar cc, appl_no, ksic::varchar ksic, any_value(family) famstr, any_value(rgst_no) rgst_no, any_value(rgst_date) rgst_date,
       min(left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4))::int pub_year
      from read_parquet('{META}') group by 1,2,3""")
    con.execute("""create table famlk as select distinct cc, appl_no, ksic, try_cast(substr(m,3) as bigint) us_pn
      from (select cc, appl_no, ksic, unnest(string_split(famstr,'|')) m from rec where famstr is not null) where regexp_full_match(m,'US[0-9]{7,8}')""")
    con.execute("""create table ownlk as select cc, appl_no, ksic, try_cast(rgst_no as bigint) us_pn, rgst_date from rec
      where cc='US' and regexp_full_match(coalesce(rgst_no,''),'[0-9]{7,8}')""")
    R = {}
    R['own_links'] = con.sql("select count(*), count(distinct us_pn) from ownlk").fetchone()
    R['own_also_in_own_family'] = con.sql("select avg((f.us_pn is not null)::int) from ownlk o left join famlk f using(cc, appl_no, ksic, us_pn)").fetchone()[0]
    R['own_by_rgst_ym_pre_post'] = con.sql("select (rgst_date>='20210401') post, count(*) from ownlk group by 1 order by 1").fetchall()
    # build_ros_pubrule_r3d 규칙: (ksic, mg, us_pn) 단위, first_rec_year = min appl_year, first_pub_year = min pub_year
    con.execute(f"create table rr as select * from read_parquet('{X}/r3ros_work/rec.parquet')")
    for nm, src in [('cp_all', "select cc, appl_no, ksic, us_pn from famlk union select cc, appl_no, ksic, us_pn from ownlk"), ('cp_fam', "select cc, appl_no, ksic, us_pn from famlk")]:
        con.execute(f"""create table {nm} as select r.ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year, min(p.pub_year) first_pub_year
           from ({src}) f join rr r using(cc, appl_no, ksic) left join (select cc, appl_no, ksic, pub_year from rec) p using(cc, appl_no, ksic)
           where r.mg is not null and f.us_pn between 3000000 and 13000000 group by 1,2,3""")
    for nm, outp in [('cp_all', 'ros_pubrule_P0re.parquet'), ('cp_fam', 'ros_pubrule_V2r.parquet')]:
        con.execute(f"""copy (select c.ksic, c.mg, s.issue_year, c.first_rec_year, coalesce(c.first_pub_year, 9999) first_pub_year,
           count(*) n_us_pat, sum((s.n_cit>0)::int)::double n_us_pat_sci, sum(s.n_cit)::double n_paper_cit
           from {nm} c join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) where s.issue_year is not null group by all) to '{ddb(C)}/{outp}' (format parquet)""")
    OFF = ddb(R3) + '/ros_cell_issueyear_pubrule.parquet'
    q = lambda f: con.sql(f"select sum(n_us_pat), sum(n_us_pat_sci), sum(n_paper_cit), count(*) from read_parquet('{f}')").fetchone()
    R['sum_official'] = q(OFF); R['sum_P0re'] = q(f'{ddb(C)}/ros_pubrule_P0re.parquet'); R['sum_V2r'] = q(f'{ddb(C)}/ros_pubrule_V2r.parquet')
    R['P0re_equal_official_rows'] = con.sql(f"""select count(*) from (select * from read_parquet('{OFF}') except select * from read_parquet('{ddb(C)}/ros_pubrule_P0re.parquet'))""").fetchone()[0]
    R['P0re_minus_official_rows'] = con.sql(f"""select count(*) from (select * from read_parquet('{ddb(C)}/ros_pubrule_P0re.parquet') except select * from read_parquet('{OFF}'))""").fetchone()[0]
    # own-only 연결(셀×US특허)의 발행연도 분포
    con.execute("create table oo as select a.ksic, a.mg, a.us_pn from cp_all a anti join cp_fam f using(ksic, mg, us_pn)")
    d = con.sql(f"""select s.issue_year, count(*) n_own_only from oo join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) group by 1 order by 1""").df()
    a = con.sql(f"""select s.issue_year, count(*) n_all from cp_all c join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) group by 1 order by 1""").df()
    d = a.merge(d, on='issue_year', how='left').fillna(0); d['own_only_share'] = d.n_own_only / d.n_all
    d.to_csv(os.path.join(C, 's3e_ros_ownonly_by_issue_year.csv'), index=False, encoding='utf-8-sig')
    out(d[(d.issue_year >= 2000) & (d.issue_year <= 2025)].to_string())
    jdump(R, 's3e_links.json'); out(json.dumps(R, ensure_ascii=False, default=str))
else:
    import numpy as np, pandas as pd
    os.environ.update(T3R3_VAR='pub', T3R3_CACHE=os.path.join(C, 't3_P0_fin'), T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1')
    sys.path.insert(0, R3)
    import r3lib as L
    import build_ext_r3b as XB
    P = L.load(); cells, ksic_c, mg_c = P['cells'], P['ksic'], P['mg']
    off = pd.read_parquet(os.path.join(R3, 'extB_cell_pub_final.parquet'))
    t0s = sorted(off.t0.unique()); rc = [c for c in off.columns if c.startswith('ros_')]
    res = {}
    for tag, f in [('P0re', 'ros_pubrule_P0re.parquet'), ('official', None), ('V2r', 'ros_pubrule_V2r.parquet')]:
        XB.ROS_PUBRULE = os.path.join(R3, 'ros_cell_issueyear_pubrule.parquet') if f is None else os.path.join(C, f)
        Rf = XB.ros_feats(t0s, cells, ksic_c, mg_c)
        m = off[['cell', 't0'] + rc].merge(Rf, on=['cell', 't0'], how='left', suffixes=('_off', ''))
        res[tag] = {c: float(np.nanmax(np.abs(m[c] - m[c + '_off']))) for c in rc}
        res[tag]['n_missing'] = int(m[rc[0]].isna().sum())
        if tag == 'V2r':
            new = off.drop(columns=rc).merge(Rf[['cell', 't0'] + rc], on=['cell', 't0'], how='left')[off.columns]
            assert len(new) == len(off)
            os.makedirs(os.path.join(C, 'extB_V2r'), exist_ok=True)
            new.to_parquet(os.path.join(C, 'extB_V2r', 'extB_cell_pub_final.parquet'))
            # t0=2021 vs ≤2018 ros_sci 변화 크기
            res['V2r_change_by_t0'] = {int(k): float(v) for k, v in m.assign(d=np.abs(m.ros_sci - m.ros_sci_off)).groupby('t0').d.mean().items()}
    jdump(res, 's3e_feats.json'); out(json.dumps(res, ensure_ascii=False, default=str))
out('secs', round(time.time() - t))
