# C19-T1 공용 경로 (입력은 읽기 전용, 쓰기는 이 폴더만)
import os, sys, json
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
AN = B + r'\분석'
MC = AN + r'\모델경쟁_20260927'
META = (MC + r'\_shared\meta\*.parquet').replace(chr(92), '/')
T3 = MC + r'\T3_기술셀부상예측'
R3 = T3 + r'\R3'
MANU = AN + r'\보고서_20261012\manuscript_v3.md'
W = AN + r'\검증루프_20261012\C19\C19-T1'
C = W + r'\cache'
C18 = AN + r'\검증루프_20261012\C18\C18-T2'


def ddb(p):
    return p.replace(chr(92), '/')


def con3():
    import duckdb
    con = duckdb.connect()
    con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    os.makedirs(os.path.join(C, 'duck_tmp'), exist_ok=True)
    con.execute(f"set temp_directory='{ddb(C)}/duck_tmp'")
    return con


def out(*a):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print(*a, flush=True)


def jdump(o, name):
    json.dump(o, open(os.path.join(C, name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)

# JP 출원번호 정규화 키(가설, s1에서 검증): 8자리 = 헤이세이 연호 2자리(+1988) + 일련 6자리, 10자리 = 서기 4자리 + 일련 6자리, 끝 'U' = 실용신안
JPKEY = r"""case
  when regexp_matches(appl_no,'^[0-9]{8}U?$') then cast(1988+cast(substr(appl_no,1,2) as int) as varchar)||substr(appl_no,3,6)||(case when appl_no like '%U' then 'U' else '' end)
  when regexp_matches(appl_no,'^[0-9]{10}U?$') then substr(appl_no,1,10)||(case when appl_no like '%U' then 'U' else '' end)
  else appl_no end"""
