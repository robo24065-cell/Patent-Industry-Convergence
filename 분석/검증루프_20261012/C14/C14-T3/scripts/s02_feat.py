# -*- coding: utf-8 -*-
"""s02: 원점 c·패널 t 하나에 대해 범위밖 의존 피처를 변형별로 다시 계산 → inj/parts/<var>_o<c>_c<t>.parquet
 python s02_feat.py <c> <t>      (종료코드 3 = 시간 예산, 같은 명령 재실행 시 남은 변형만)
 P0(무변형) 재계산이 동결 prep 열과 비트 단위로 같지 않으면 다른 변형을 만들지 않고 중단(기록만)."""
import sys, json, time
from c14lib import *
T0 = time.time(); BUD = 500
c, t = int(sys.argv[1]), int(sys.argv[2])
VARS = ['P0', 'S2', 'S3_1', 'S3_2', 'S3_3', 'S3_4', 'S3_5', 'S4', 'AN_A', 'AN_B', 'AN_TI']
if len(sys.argv) > 3: VARS = sys.argv[3].split(',')
PD = HERE/'inj'/'parts'; PD.mkdir(parents=True, exist_ok=True)
def part(v): return PD/f'{v}_o{c}_c{t}.parquet'
todo = [v for v in VARS if not part(v).exists()]
statf = HERE/'cache'/f'feat_stats_o{c}_c{t}.json'
ST = json.load(open(statf, encoding='utf-8')) if statf.exists() else {}
if not todo and ST.get('P0_equal') is not None: print('done already'); sys.exit(0)
cf = cfg(c); S = psrc(c); FCp = fcp(c)
P = np.load(S/'P_base.npy').astype(np.float32); P51 = np.load(S/'P_os51.npy').astype(np.float32); PO = np.load(S/'PO_os51.npy').astype(np.float32)
Pins = np.load(S/'P_ins_cf.npy'); ins_idx = pd.read_parquet(S/'P_ins_cf_index.parquet').rgst_no
cdf = fr.lag_cdf(cf)
D = lib.load()
assert len(D['oos']) == len(P), (len(D['oos']), len(P))
d, ctx = panel(t, c)
print(f'[o{c} c{t}] panel rows {len(d)} firms {len(ctx["firms"])} op {len(ctx["op"])} {time.time()-T0:.0f}s peak {peak_gb():.2f}GB', flush=True)
# 동결 prep 행
prepf = FCp/('prep_te.parquet' if t == c else 'prep_tr.parquet')
import pyarrow.parquet as pq
names = pq.ParquetFile(prepf).schema.names
cols = [x for x in REPL_OP + TI4 if x in names]
pr = pd.read_parquet(prepf, columns=['firm', 'k', 'cut'] + cols)
pr = pr[pr.cut.values == t].reset_index(drop=True)
assert len(pr) == len(d) and (pr.firm.values == d.firm.values).all() and (pr.k.values == d.k.values).all(), '행 순서 불일치'
dt = {x: pr[x].dtype for x in cols}
def cast(df, cc):
    o = pd.DataFrame({x: df[x].values.astype(dt[x]) for x in cc})
    return o
def eq(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype.kind == 'f': return bool(np.array_equal(a, b, equal_nan=True)), float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if len(a) else 0.0
    return bool((a == b).all()), float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if len(a) else 0.0
opc = [x for x in REPL_OP if x in names]
op0 = ctx['op']
fl = pd.read_parquet(HERE/'cache/oos_flags.parquet')
assert (fl.rgst_no.values[op0.row.values] == op0.rgst_no.astype(str).values).all(), 'oos_flags 행 정렬 불일치'
ce = f'{t}1231'
ST.update(c=c, t=t, rows=int(len(d)), firms=int(len(ctx['firms'])), op_rows=int(len(op0)),
          op_rgst_after_cut=int((op0.rgst_date.astype(str) > ce).sum()), op_ay_after_cut=int((op0.ay > t).sum()),
          op_not_in_holder=float((~fl.in_holder.values[op0.row.values]).mean()) if len(op0) else None)
def save(v, df, cc):
    cast(df, cc).to_parquet(part(v), index=False)
def budget(v):
    if time.time() - T0 > BUD:
        json.dump(ST, open(statf, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'[시간 예산] {v} 전에 중단 {time.time()-T0:.0f}s', flush=True); sys.exit(3)
def with_op(op):
    c2 = dict(ctx); c2['op'] = op; return c2
# ---- P0 (무변형) + 비트 일치 점검 ----
if ST.get('P0_equal') is None or 'P0' in todo:
    base = op_feats(d, ctx, P, P51, PO, cdf)
    bad = {}
    for x in opc:
        ok_, mx = eq(base[x].values.astype(dt[x]), pr[x].values)
        if not ok_: bad[x] = mx
    tib = ti_feats(d, ctx, ctx['ip'], Pins, ins_idx)
    for x in [y for y in TI4 if y in names]:
        ok_, mx = eq(tib[x].values.astype(dt[x]), pr[x].values)
        if not ok_: bad[x] = mx
    ST['P0_equal'] = not bad; ST['P0_mismatch'] = bad
    save('P0', base, opc)
    print(f'[o{c} c{t}] P0 equal={not bad} {bad} {time.time()-T0:.0f}s', flush=True)
    del base
if not ST['P0_equal']:
    json.dump(ST, open(statf, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('P0 불일치 → 중단'); sys.exit(2)
keys = (op0.applicant_cd.astype(str) + '|' + op0.rgst_no.astype(str)).values
# 아날로그 행
ipx = ctx['ip']
fiset = set(ctx['fi'].index)
Bt = pd.read_parquet(HERE/'cache/analog_B.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'rgst_date'])
Bt = Bt[Bt.applicant_cd.isin(fiset) & (Bt.ay <= t) & (Bt.ay >= 2000) & (Bt.rgst_date.astype(str) <= ce)]
A10 = ipx[ipx.rgst_no.astype(str).str[:2] == '10'][['applicant_cd', 'rgst_no', 'ay']]
for v in [x for x in todo if x != 'P0']:
    budget(v)
    if v == 'S2':
        keep = fl.in_holder.values[op0.row.values] & (op0.rgst_date.astype(str).values <= ce)
        ST['S2_removed_share'] = float(1 - keep.mean()) if len(keep) else 0.0
        save(v, op_feats(d, with_op(op0[keep]), P, P51, PO, cdf), opc)
    elif v.startswith('S3_'):
        s = int(v.split('_')[1]); dr = cell_rate(op0); keep = u01(keys, f'S3_{s}') >= dr
        ST[f'{v}_removed_share'] = float(1 - keep.mean()) if len(keep) else 0.0
        save(v, op_feats(d, with_op(op0[keep]), P, P51, PO, cdf), opc)
    elif v == 'S4':
        dr = cell_rate(op0); dup = u01(keys, 'S4') < dr / (1 - dr)
        ST['S4_added_share'] = float(dup.mean()) if len(dup) else 0.0
        save(v, op_feats(d, with_op(pd.concat([op0, op0[dup]])), P, P51, PO, cdf), opc)
    elif v == 'AN_A':
        ST['AN_A_rows'] = int(len(A10.drop_duplicates(['applicant_cd', 'rgst_no'])))
        save(v, an_feats(d, ctx, A10, Pins, ins_idx), OT6)
    elif v == 'AN_B':
        ST['AN_B_rows'] = int(len(Bt.drop_duplicates(['applicant_cd', 'rgst_no'])))
        save(v, an_feats(d, ctx, Bt[['applicant_cd', 'rgst_no', 'ay']], Pins, ins_idx), OT6)
    elif v == 'AN_TI':
        um = ipx[ipx.rgst_no.astype(str).str[:2] != '10'][['applicant_cd', 'rgst_no']]
        rows = pd.concat([um, Bt[['applicant_cd', 'rgst_no']]])
        ST['AN_TI_rows'] = int(len(rows.drop_duplicates())); ST['AN_TI_official_rows'] = int(len(ipx.drop_duplicates(['applicant_cd', 'rgst_no'])))
        save(v, ti_feats(d, ctx, rows, Pins, ins_idx), [y for y in TI4 if y in names])
    print(f'[o{c} c{t}] {v} saved {time.time()-T0:.0f}s peak {peak_gb():.2f}GB', flush=True)
json.dump(ST, open(statf, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('all done', round(time.time() - T0))
