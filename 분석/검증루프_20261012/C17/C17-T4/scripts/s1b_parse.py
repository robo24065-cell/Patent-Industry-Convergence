# C17-T4 s1b: USPTO MaintFeeEvents 스트리밍 파싱 + 날짜 가드. python -I 로 실행.
# 인자: <zip 경로> <스냅샷 태그 yyyymmdd>
# 출력: cache/ev_dev_<tag>.parquet (제공 US 연결 특허의 개발 구역 이벤트)
#       cache/LOCKED/ev_lock_<tag>.parquet (잠금 구역 이벤트; P0 단계에서 열지 않음)
#       cache/agg_<tag>.json (전체 파일 집계; 개발 구역만)
import sys, os, io, json, zipfile, time, collections
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C17\C17-T4\scripts')
from lib17 import *
import pyarrow as pa, pyarrow.parquet as pq

zp, tag = sys.argv[1], sys.argv[2]
t_start = time.time()
os.makedirs(LOCKDIR, exist_ok=True)
docs = pq.read_table(C + r'\us_docs.parquet', columns=['appl8']).column('appl8').to_pylist()
APPL = set(docs)
del docs
z = zipfile.ZipFile(zp)
name = [n for n in z.namelist() if n.startswith('MaintFeeEvents_') and n.endswith('.txt')][0]
snap = int(name.split('_')[1][:8])
f = io.TextIOWrapper(z.open(name), encoding='ascii', errors='replace', newline='')

schema = pa.schema([('pat', pa.string()), ('appl8', pa.string()), ('ent_now', pa.string()), ('filing', pa.int32()),
                    ('G', pa.int32()), ('e', pa.int32()), ('code', pa.string())])
wd = pq.ParquetWriter(C + rf'\ev_dev_{tag}.parquet', schema)
wl = pq.ParquetWriter(LOCKDIR + rf'\ev_lock_{tag}.parquet', schema)
buf_d = {k: [] for k in schema.names}; buf_l = {k: [] for k in schema.names}


def flush(buf, w):
    if buf['pat']:
        w.write_table(pa.table({k: pa.array(v, type=schema.field(k).type) for k, v in buf.items()}, schema=schema))
        for v in buf.values():
            v.clear()


zcache = {}
seen = set(); seen_link = set()
n = bad = n_lock_all = 0
cnt_gy = collections.Counter(); cnt_gy_link = collections.Counter()
exp_ey = collections.Counter(); pay_ey = collections.Counter(); code_cnt = collections.Counter()
reissue = 0; prev = None; dup_adj = 0
ent_pay = collections.Counter()
for line in f:
    n += 1
    if len(line) < 57:
        bad += 1; continue
    pat = line[0:13]; appl = line[14:22]; ent = line[23]
    try:
        fil = int(line[25:33]); G = int(line[34:42]); e = int(line[43:51])
    except ValueError:
        bad += 1; continue
    code = line[52:57].strip()
    if line == prev:
        dup_adj += 1
    prev = line
    if not pat.isdigit():
        reissue += 1
    key = (G, e >= 20240101)
    if e >= 20240101:
        kz = (G, e)
        lz = zcache.get(kz)
        if lz is None:
            lz = lock_zone(G, e); zcache[kz] = lz
    else:
        lz = False
    linked = appl in APPL
    if lz:
        n_lock_all += 1
        if linked:
            for k, v in zip(schema.names, (pat, appl, ent, fil, G, e, code)):
                buf_l[k].append(v)
    else:
        code_cnt[code] += 1
        if code == 'EXP.':
            exp_ey[(e // 10000, G // 10000)] += 1
        if code in PAY:
            pay_ey[(e // 10000, PAY[code][0])] += 1
            ent_pay[(G // 10000, PAY[code][0], PAY[code][1])] += 1
        if linked:
            for k, v in zip(schema.names, (pat, appl, ent, fil, G, e, code)):
                buf_d[k].append(v)
    if pat not in seen:
        seen.add(pat); cnt_gy[G // 10000] += 1
        if linked and pat not in seen_link:
            seen_link.add(pat); cnt_gy_link[G // 10000] += 1
    if len(buf_d['pat']) >= 500000:
        flush(buf_d, wd)
    if len(buf_l['pat']) >= 200000:
        flush(buf_l, wl)
flush(buf_d, wd); flush(buf_l, wl); wd.close(); wl.close()
agg = {'tag': tag, 'snapshot_file_date': snap, 'file': name, 'n_lines': n, 'bad_lines': bad, 'adjacent_dup_lines': dup_adj,
       'reissue_lines': reissue, 'n_patents_all': len(seen), 'n_patents_linked': len(seen_link), 'n_lock_zone_lines_all': n_lock_all,
       'patents_by_grant_year': {str(k): v for k, v in sorted(cnt_gy.items())},
       'linked_patents_by_grant_year': {str(k): v for k, v in sorted(cnt_gy_link.items())},
       'exp_by_eventyear_grantyear_devzone': {f'{a}|{b}': v for (a, b), v in sorted(exp_ey.items())},
       'pay_by_eventyear_stage_devzone': {f'{a}|{b}': v for (a, b), v in sorted(pay_ey.items())},
       'pay_entity_by_grantyear_stage_devzone': {f'{a}|{b}|{c}': v for (a, b, c), v in sorted(ent_pay.items())},
       'code_counts_devzone': dict(code_cnt.most_common()), 'sec': round(time.time() - t_start, 1)}
json.dump(agg, open(C + rf'\agg_{tag}.json', 'w'), indent=1)
out({k: agg[k] for k in ['tag', 'snapshot_file_date', 'n_lines', 'bad_lines', 'adjacent_dup_lines', 'reissue_lines', 'n_patents_all', 'n_patents_linked', 'n_lock_zone_lines_all', 'sec']})
