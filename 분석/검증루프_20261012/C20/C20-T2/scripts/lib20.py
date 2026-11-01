# C20-T2 공용 경로 (입력은 읽기 전용, 쓰기는 이 폴더만)
import os, sys, json
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
AN = B + r'\분석'
MC = AN + r'\모델경쟁_20260927'
META = (MC + r'\_shared\meta\*.parquet').replace(chr(92), '/')
T3 = MC + r'\T3_기술셀부상예측'
R3 = T3 + r'\R3'
T1 = MC + r'\T1_산업대기실'
MANU = AN + r'\보고서_20261012\manuscript_v3.md'
LOOP = AN + r'\검증루프_20261012'
W = LOOP + r'\C20\C20-T2'
C = W + r'\cache'
C19 = LOOP + r'\C19\C19-T1'
C18 = LOOP + r'\C18\C18-T2'
C18T1 = LOOP + r'\C18\C18-T1'
C17 = LOOP + r'\C17\C17-T2'
META_V1E = (C18 + r'\cache\meta_V1E.parquet').replace(chr(92), '/')
VJ3_LIST = (C19 + r'\cache\lists\VJ3.parquet').replace(chr(92), '/')
# 사전 고정(PREREG §2)
EP_GAP = ['202101', '202102', '202103', '202104', '202105', '202106', '202204', '202205', '202206', '202207', '202208', '202209']
CN_BLOCK_THR = 0.15


def ddb(p):
    return p.replace(chr(92), '/')


def con3(tmp=None):
    import duckdb
    con = duckdb.connect()
    con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    d = tmp or os.path.join(C, 'duck_tmp'); os.makedirs(d, exist_ok=True)
    con.execute(f"set temp_directory='{ddb(d)}'")
    return con


def out(*a):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print(*a, flush=True)


def jdump(o, name):
    json.dump(o, open(os.path.join(C, name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)


# C19-T1 lib19.JPKEY 그대로(E-C19T1-D1 규칙 (2)의 정규화 키)
JPKEY = r"""case
  when regexp_matches(appl_no,'^[0-9]{8}U?$') then cast(1988+cast(substr(appl_no,1,2) as int) as varchar)||substr(appl_no,3,6)||(case when appl_no like '%U' then 'U' else '' end)
  when regexp_matches(appl_no,'^[0-9]{10}U?$') then substr(appl_no,1,10)||(case when appl_no like '%U' then 'U' else '' end)
  else appl_no end"""
PY = "left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)"
PYD = "coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,''))"
IPCRAW = r"regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g')"
