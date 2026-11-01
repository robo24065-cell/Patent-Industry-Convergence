# C18-T2 verify_0 r2: 변형 메타를 pyarrow로 독립 생성 (s3a 산출물 미사용)
#  myV1E  = US 7자리(8자리 쌍 있음) 행 + EP 정규화키 쌍의 짧은 쪽 행 제거 (원 발견 규칙과 같음, 독립 구현)
#  myV1Es = 반대 규칙: US 8자리 쌍 행 + EP 긴 쪽 행 제거 (짧은 형식 유지)
import glob, os, sys, json
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc, pandas as pd
META = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta'
D = os.path.dirname(os.path.abspath(__file__))
ue = pd.read_pickle(os.path.join(D, 'us_ep.pkl'))
us = ue[ue.cc == 'US'].appl_no; ep = ue[ue.cc == 'EP'].appl_no
u8 = set(us[us.str.len() == 8]); us7 = set(us[(us.str.len() == 7) & us.str.zfill(8).isin(u8)]); us8 = {s.zfill(8) for s in us7}
e = pd.DataFrame({'a': ep.drop_duplicates()}); e['k'] = e.a.str.replace(r'[^0-9.]', '', regex=True).str.lstrip('0'); e['L'] = e.a.str.len()
g = e[e.duplicated('k', keep=False)].sort_values(['k', 'L'])
eps = set(g.groupby('k').head(1).a); epl = set(g.groupby('k').tail(1).a)
print(len(us7), len(us8), len(eps), len(epl))
cnt = {}
for V, dus, dep in [('myV1E', us7, eps), ('myV1Es', us8, epl)]:
    od = os.path.join(D, 'meta_' + V); os.makedirs(od, exist_ok=True)
    n = {}
    for f in sorted(glob.glob(os.path.join(META, '*.parquet'))):
        t = pq.read_table(f)
        cc = pc.cast(t['cc'], 'string')
        drop = pc.or_(pc.and_(pc.equal(cc, 'US'), pc.is_in(t['appl_no'], pa.array(sorted(dus)))),
                      pc.and_(pc.equal(cc, 'EP'), pc.is_in(t['appl_no'], pa.array(sorted(dep)))))
        drop = pc.fill_null(drop, False)
        t2 = t.filter(pc.invert(drop))
        pq.write_table(t2, os.path.join(od, os.path.basename(f)), compression='zstd')
        n[os.path.basename(f)] = [t.num_rows, t2.num_rows]
    cnt[V] = {'rows_before': sum(a for a, b in n.values()), 'rows_after': sum(b for a, b in n.values())}
    print(V, cnt[V], flush=True)
json.dump(cnt, open(os.path.join(D, 'e_meta.json'), 'w'), indent=1)
