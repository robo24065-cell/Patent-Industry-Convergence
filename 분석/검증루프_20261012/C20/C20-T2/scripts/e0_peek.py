# C20-T2 e0: PREREG 전 형식 확인(행 수·형식만, 모형 결과 없음)
import sys, os; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib20 import *
con = con3()
R = {}
spec = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
R['spec_keys'] = {k: v for k, v in spec.items() if k != 'cfg'}
R['vj3_list'] = con.sql(f"select count(*), count(distinct appl_no) from read_parquet('{VJ3_LIST}')").fetchone()
# VJ3 목록 raw appl_no 가 공개일 있는 JP 행과 같은 문자열인 경우(그 행까지 지워지는지)
R['vj3_hits_meta'] = con.sql(f"""select (nullif(pub_date,'') is null) nopub, count(*) from read_parquet('{META}') where cc='JP' and appl_no in (select appl_no from read_parquet('{VJ3_LIST}')) group by 1""").fetchall()
R['cn_applno_fmt'] = con.sql(f"""select regexp_replace(appl_no,'[0-9]','9','g') f, count(*) from read_parquet('{META}') where cc='CN' group by 1 order by 2 desc limit 8""").fetchall()
R['cn_examples'] = con.sql(f"select appl_no, appl_date, pub_date, rgst_date, main_ipc, left(ipc_all,30) from read_parquet('{META}') where cc='CN' limit 5").fetchall()
R['jp_rg10_withpub'] = con.sql(f"""select (nullif(pub_date,'') is null) nopub, count(*) from read_parquet('{META}') where cc='JP' and regexp_matches(coalesce(rgst_no,''),'^[0-9]{{10}}$') group by 1""").fetchall()
R['cols'] = [x[0] for x in con.sql(f"describe select * from read_parquet('{META}')").fetchall()]
jdump(R, 'e0_peek.json'); out(json.dumps(R, ensure_ascii=False, default=str, indent=1))
