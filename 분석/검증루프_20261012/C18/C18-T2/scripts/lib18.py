# C18-T2 공용 경로 (입력은 읽기 전용, 쓰기는 이 폴더만)
import os, sys, json
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
AN = B + r'\분석'
MC = AN + r'\모델경쟁_20260927'
META = (MC + r'\_shared\meta\*.parquet').replace('\\', '/')
T3 = MC + r'\T3_기술셀부상예측'
R3 = T3 + r'\R3'
N5 = MC + r'\N5_표준특허선언레이더'
MAP5 = MC + r'\_shared_report\fusion_advantage\map5'
LO = MC + r'\_shared_report\label_origin_R10'
MANU = AN + r'\보고서_20261012\manuscript_v3.md'
W = AN + r'\검증루프_20261012\C18\C18-T2'
C = W + r'\cache'


def ddb(p):
    return p.replace('\\', '/')


def con3():
    import duckdb
    con = duckdb.connect()
    con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
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
