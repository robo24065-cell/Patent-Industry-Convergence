# C14-T4 verify_0 v3: U6×Eb 위험·양성 독립 재집계(duckdb SQL, b1·b2 코드 미사용). 원점 t0=2022·2023(개발)과 2024(잠금 위험만).
# 입력 C13 cache app_{CN,EP}.parquet(공개 < 2025; SQL에서 pub < 20250101 조건). 출력 verify_0/v3_out.json
import os, json, duckdb
HERE = os.path.dirname(os.path.abspath(__file__))
C13C = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T4\cache'
con = duckdb.connect(); con.execute('set threads=2'); con.execute("set memory_limit='3GB'")
res = {}
for cc in ['CN', 'EP']:
    p = (C13C + rf'\app_{cc}.parquet').replace('\\', '/')
    con.execute(f"create or replace temp table a as select appl_no, key, pub, ksic from read_parquet('{p}') where pub < 20250101")
    con.execute("""create or replace temp table e as
        select x.key k1, y.key k2, x.pub from a x join a y on x.appl_no = y.appl_no and x.key < y.key""")
    for t0 in [2022, 2023, 2024]:
        end = t0 * 10000 + 1231
        q = f"""
        with m as (select key, ksic, count(*) n from a where pub > {end - 50000} and pub <= {end} group by key, ksic),
        mo as (select key, ksic from (select key, ksic, row_number() over (partition by key order by n desc, ksic asc) r from m) where r = 1),
        r as (select k1, k2, count(*) n3, sum(case when pub > {end - 10000} then 1 else 0 end) n1 from e where pub > {end - 30000} and pub <= {end} group by k1, k2),
        rk as (select r.* from r join mo f on r.k1 = f.key join mo g on r.k2 = g.key where n3 >= 2 and n1 >= 1 and f.ksic <> g.ksic),
        w as (select k1, k2, count(*) nw from e where pub >= {(t0 + 1) * 10000 + 101} and pub <= {(t0 + 1) * 10000 + 630} group by k1, k2)
        select count(*) risk, sum(case when w.nw is null then 1 else 0 end) stop_ from rk left join w using (k1, k2)"""
        risk, stop = con.execute(q).fetchone()
        res[f'{cc}_{t0}'] = {'risk': int(risk), 'stop_if_window_known': int(stop) if t0 <= 2023 else None}
        print(cc, t0, res[f'{cc}_{t0}'], flush=True)
json.dump(res, open(os.path.join(HERE, 'v3_out.json'), 'w'), indent=1)
