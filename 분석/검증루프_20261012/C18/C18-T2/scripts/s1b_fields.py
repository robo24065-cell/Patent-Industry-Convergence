# C18-T2 s1b: 5개국 필드 채움 프로필(등록 연월, 출원연도별) + 번호 형식 이중 표기 + 일반 중복 (읽기 전용) → cache/s1b.json, cache/s1b_*.csv
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
con = con3()
con.execute(f"""create table m as select cc::varchar cc, appl_no, length(appl_no) L, appl_year, rgst_no, rgst_date, pub_date, title, applicant, main_ipc, ipc_all,
   family, prio_cc::varchar prio_cc, appl_date, ksic::varchar ksic from read_parquet('{META}')""")
res = {}
# 등록 연월 프로필
con.execute(f"""copy (select cc, substr(rgst_date,1,6) ym, count(*) n from m where rgst_date is not null group by 1,2 order by 1,2)
   to '{ddb(C)}/s1b_rgst_ym.csv' (header)""")
# 출원연도별 채움률
con.execute(f"""copy (select cc, appl_year, count(*) n, avg((rgst_no is not null)::int) rgst_fill, avg((rgst_date is not null)::int) rgstd_fill,
   avg((pub_date is not null)::int) pub_fill, avg((main_ipc is not null)::int) ipc_fill, avg((applicant is not null)::int) app_fill,
   avg((family is not null)::int) fam_fill, avg((prio_cc is not null)::int) prio_fill, avg((title is not null)::int) title_fill,
   avg((pub_date is null and rgst_date is not null)::int) rgst_only, avg((pub_date is null and rgst_date is null)::int) no_date
   from m where appl_year between 1995 and 2025 group by 1,2 order by 1,2) to '{ddb(C)}/s1b_fill_by_ay.csv' (header)""")
# 공개일 없음 × 등록일 유무 (US 중심, 모든 국가)
res['pub_null_by_rgst'] = con.sql("""select cc, count(*) n, sum((pub_date is null)::int) pub_null, sum((pub_date is null and rgst_date is not null)::int) pub_null_rgst,
   sum((pub_date is not null and rgst_date is not null)::int) both_, sum((rgst_date is not null)::int) rgst_any from m group by 1 order by 1""").fetchall()
# US: 등록일 있는 행의 공개일 유무 × 등록연도
con.execute(f"""copy (select substr(rgst_date,1,4) ry, count(*) n, sum((pub_date is null)::int) pub_null from m where cc='US' and rgst_date is not null group by 1 order by 1)
   to '{ddb(C)}/s1b_us_rgst_year_pub.csv' (header)""")
# US: 공개일 없는 행의 출원연도 분포 × 등록일 유무
con.execute(f"""copy (select appl_year, count(*) n, sum((rgst_date is not null)::int) with_rgst, min(rgst_date) min_rd, max(rgst_date) max_rd
   from m where cc='US' and pub_date is null group by 1 order by 1) to '{ddb(C)}/s1b_us_pubnull_ay.csv' (header)""")
# 번호 길이 분포
res['len_dist'] = con.sql("select cc, L, count(*) from m group by 1,2 order by 1,2").fetchall()
# 정규화 키(숫자만, 앞 0 제거) 충돌: 같은 국가에서 원문이 다른 행 쌍
con.execute(r"""create table nk as select cc, appl_no, ltrim(regexp_replace(appl_no,'[^0-9]','','g'),'0') k from m""")
res['norm_collisions'] = con.sql("""select cc, count(*) n_keys, sum(c) n_rows from (select cc, k, count(distinct appl_no) c from nk group by 1,2 having c>1) group by 1 order by 1""").fetchall()
con.execute(f"""copy (select cc, k, list(distinct appl_no) nos from nk group by 1,2 having count(distinct appl_no)>1 limit 200) to '{ddb(C)}/s1b_norm_collision_examples.csv' (header)""")
# 같은 국가 appl_no 완전 중복(같은 KSIC 안 / 다른 KSIC)
res['exact_dup_appl'] = con.sql("""select cc, count(*) n_appl, sum(c) n_rows, sum((nk>1)::int) multi_ksic from (select cc, appl_no, count(*) c, count(distinct ksic) nk from m group by 1,2 having c>1) group by 1 order by 1""").fetchall()
# 일반 중복: (정규화 제목, 출원일, 메인 IPC) 같고 appl_no 다른 경우 (같은 KSIC 안)
res['gen_dup'] = con.sql(r"""select cc, count(*) n_groups, sum(c) n_rows from (
    select cc, ksic, regexp_replace(lower(title),'\s+',' ','g') t, appl_date, main_ipc, count(distinct appl_no) c from m
    where title is not null and appl_date is not null and main_ipc is not null group by all having c>1) group by 1 order by 1""").fetchall()
con.execute(rf"""copy (select cc, ksic, t, appl_date, main_ipc, nos from (
    select cc, ksic, regexp_replace(lower(title),'\s+',' ','g') t, appl_date, main_ipc, list(distinct appl_no) nos, count(distinct appl_no) c from m
    where title is not null and appl_date is not null and main_ipc is not null group by cc, ksic, t, appl_date, main_ipc having c>1) using sample 150 rows)
    to '{ddb(C)}/s1b_gen_dup_examples.csv' (header)""")
jdump(res, 's1b.json')
out(json.dumps(res, ensure_ascii=False, default=str)[:4000])
