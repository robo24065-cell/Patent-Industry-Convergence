# -*- coding: utf-8 -*-
"""C14-T3 민감도 실행기 — 동결 R3E29 spec 그대로, prep 열 일부만 주입값으로 바꿔 동결 train_origin()을 실행.
 python run_inj.py <var> <c>     var: P0, S1(ot_* 7열 0), S1v(같은 7열 NaN), S2, S3_1..S3_5, S4, AN_A, AN_B, AN_TI
 (종료코드 3 = 동결 실행기 단계 종료 → 같은 명령 재실행)
 주입: pandas.read_parquet / pyarrow.parquet.read_table 이 동결 원점 캐시의 prep_tr/prep_te 를 읽을 때, inj/<var>/o<c>_{tr,te}.parquet 의 같은 이름 열로 교체.
 산출: runs/<var>/o<c>.parquet (쓰기 가드: 이 폴더 밖 쓰기 금지)."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'
sys.path.insert(0, str(R3)); os.chdir(str(T1))
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import final_run_r3 as F

var, c = sys.argv[1], int(sys.argv[2])
FCp = R3/'cache_final'/'o2019' if c == 2019 else R3/'cache'/f'o{c}'
OT7 = ['ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr', 'oin50']
TGT = {str((FCp/'prep_tr.parquet').resolve()).lower(): 'tr', str((FCp/'prep_te.parquet').resolve()).lower(): 'te'}
NROW = {w: pq.ParquetFile(FCp/f'prep_{w}.parquet').metadata.num_rows for w in ('tr', 'te')}
SCH = {w: pq.ParquetFile(FCp/f'prep_{w}.parquet').schema_arrow for w in ('tr', 'te')}
if var in ('S1', 'S1v'):
    INJ = {w: None for w in ('tr', 'te')}; ICOLS = set(OT7)
else:
    INJ = {w: HERE/'inj'/var/f'o{c}_{w}.parquet' for w in ('tr', 'te')}
    for w in INJ: assert INJ[w].exists(), INJ[w]
    ICOLS = set(pq.ParquetFile(INJ['tr']).schema.names); assert ICOLS == set(pq.ParquetFile(INJ['te']).schema.names)
_orp, _ort = pd.read_parquet, pq.read_table
LOG = {}
def which(p):
    try: return TGT.get(str(pathlib.Path(p).resolve()).lower())
    except Exception: return None
def colval(w, x):
    if INJ[w] is None:
        dt = SCH[w].field(x).type.to_pandas_dtype()
        return np.zeros(NROW[w], dt) if var == 'S1' else np.full(NROW[w], np.nan, dt)
    v = _orp(INJ[w], columns=[x])[x].values; assert len(v) == NROW[w]; return v
def rp(path, *a, **k):
    df = _orp(path, *a, **k); w = which(path)
    if w:
        assert len(df) == NROW[w]
        for x in [x for x in df.columns if x in ICOLS]:
            v = colval(w, x); assert v.dtype == df[x].dtype, (x, v.dtype, df[x].dtype); df[x] = v
            LOG.setdefault(f'{w}:read_parquet', []).append(x)
    return df
def rt(source, *a, **k):
    tb = _ort(source, *a, **k); w = which(source)
    if w:
        assert tb.num_rows == NROW[w]
        for x in [x for x in tb.column_names if x in ICOLS]:
            v = colval(w, x); tb = tb.set_column(tb.column_names.index(x), x, pa.array(v, type=tb.schema.field(x).type))
            LOG.setdefault(f'{w}:read_table', []).append(x)
    return tb
pd.read_parquet = rp; pq.read_table = rt
RD = HERE/'runs'/var
print(f'[C14-T3] var={var} c={c} FCp={FCp} RD={RD} inj_cols={sorted(ICOLS)}', flush=True)
ec = json.loads(json.dumps(F.EC)); ec['ext'] = list(ec['ext'])
import atexit
def _log():
    if LOG:
        RD.mkdir(parents=True, exist_ok=True)
        with open(RD/f'o{c}_injlog.txt', 'a', encoding='utf-8') as fh: fh.write(json.dumps({k: sorted(set(v)) for k, v in LOG.items()}, ensure_ascii=False) + '\n')
atexit.register(_log)
F.train_origin(c, FCp, 'prep_tr.parquet', RD, ec)
