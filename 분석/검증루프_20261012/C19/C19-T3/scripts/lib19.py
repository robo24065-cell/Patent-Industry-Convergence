# C19-T3 공용 경로 (입력은 읽기 전용, 쓰기는 이 폴더만)
import os, sys, json, hashlib
BS = chr(92)
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
AN = B + r'\분석'
MC = AN + r'\모델경쟁_20260927'
VL = AN + r'\검증루프_20261012'
META = (MC + r'\_shared\meta\*.parquet').replace(BS, '/')
REP = AN + r'\보고서_20261012'
MANU = REP + r'\manuscript_v3.md'
FACTS = REP + r'\facts'
C18T2 = VL + r'\C18\C18-T2'
C18T3 = VL + r'\C18\C18-T3'
W = VL + r'\C19\C19-T3'
C = W + r'\cache'
MANU_SHA1 = '40406e296a4e0ebce633338176d18706e9fb10e0'
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def ddb(p):
    return p.replace(BS, '/')


def con3():
    import duckdb
    con = duckdb.connect()
    con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    os.makedirs(C + r'\duck_tmp', exist_ok=True)
    con.execute(f"set temp_directory='{ddb(C)}/duck_tmp'")
    return con


def out(*a):
    print(*a, flush=True)


def jdump(o, name):
    json.dump(o, open(os.path.join(C, name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def manu_lines():
    assert sha1(MANU) == MANU_SHA1, 'manuscript changed'
    return open(MANU, encoding='utf-8').read().replace('\r\n', '\n').split('\n')
