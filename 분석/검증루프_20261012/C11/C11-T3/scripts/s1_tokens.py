# -*- coding: utf-8 -*-
"""s1: 제공 레코드 family 필드의 US 토큰 추출(파일별 청크) -> cache/us_tok.parquet, 레코드 메타 cache/rec.parquet"""
import glob
from common import *
c = con()
files = sorted(glob.glob(METAD + "/*.parquet"))
os.makedirs(W + "/cache/tok", exist_ok=True)
for f in files:
    k = os.path.basename(f)[:-8]
    out = f"{W}/cache/tok/{k}.parquet"
    if os.path.exists(out): continue
    f2 = f.replace(os.sep, '/')
    c.execute(f"""copy (select distinct cc, appl_no, '{k}' ksic, t from (select cc, appl_no, unnest(string_split(family,'|')) t
        from read_parquet('{f2}') where family like '%US%') where t like 'US%') to '{out}' (format parquet)""")
log("tok files done")
c.execute(f"""copy (select cc, appl_no, ksic, any_value(appl_date) appl_date, any_value(appl_year) appl_year, any_value(rgst_no) rgst_no,
    any_value(rgst_date) rgst_date, any_value(pub_date) pub_date, any_value(prio_date) prio_date, any_value(prio_cc) prio_cc,
    any_value(main_ipc) main_ipc, any_value(ipc_all) ipc_all, any_value(applicant) applicant, max((family is not null and family<>'')::int) has_fam
    from read_parquet('{META}') group by 1,2,3) to '{W}/cache/rec.parquet' (format parquet)""")
log("rec", c.sql(f"select count(*), count(distinct cc||appl_no) from '{W}/cache/rec.parquet'").fetchall())
log("tok", c.sql(f"select count(*), count(distinct cc||appl_no) from '{W}/cache/tok/*.parquet'").fetchall())
