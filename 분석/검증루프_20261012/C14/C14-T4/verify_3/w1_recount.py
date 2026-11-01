# C14-T4 verify_3 w1: U6xEb/Ec/Eg 위험·양성·잠금 예상 독립 재집계(duckdb, 작업 코드 미사용)
# 점검: (appl_no,key) 중복 행, docs appl_no 중복, 중복 제거 전후 수치. 입력은 C13 cache(읽기만), pub < 20250101.
import os, json, duckdb
HERE = os.path.dirname(os.path.abspath(__file__))
C13C = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T4\cache'
con = duckdb.connect(); con.execute('set threads=2'); con.execute("set memory_limit='3GB'")
res = {}
for cc in ['CN', 'EP']:
    p = (C13C + rf'\app_{cc}.parquet').replace('\\', '/')
    d = (C13C + rf'\docs_{cc}.parquet').replace('\\', '/')
    con.execute(f"create or replace temp table raw as select appl_no, key, pub, ksic from read_parquet('{p}') where pub < 20250101")
    chk = {}
    chk['rows'] = con.execute("select count(*) from raw").fetchone()[0]
    chk['max_pub'] = con.execute("select max(pub) from raw").fetchone()[0]
    chk['dup_appl_key_rows'] = con.execute("select count(*) - count(distinct (appl_no, key)) from raw").fetchone()[0]
    chk['appl_multi_pub'] = con.execute("select count(*) from (select appl_no from raw group by appl_no having count(distinct pub) > 1)").fetchone()[0]
    chk['appl_key_multi_ksic'] = con.execute("select count(*) from (select appl_no, key from raw group by appl_no, key having count(distinct ksic) > 1)").fetchone()[0]
    chk['docs_rows'] = con.execute(f"select count(*) from read_parquet('{d}')").fetchone()[0]
    chk['docs_dup_appl'] = con.execute(f"select count(*) - count(distinct appl_no) from read_parquet('{d}')").fetchone()[0]
    res[cc + '_check'] = chk; print(cc, chk, flush=True)
    for variant in ['asis', 'dedup']:
        if variant == 'asis':
            con.execute("create or replace temp table a as select * from raw")
        else:
            con.execute("create or replace temp table a as select appl_no, key, min(pub) pub, min(ksic) ksic from raw group by appl_no, key")
        con.execute("create or replace temp table e as select x.key k1, y.key k2, x.pub from a x join a y on x.appl_no = y.appl_no and x.key < y.key")
        # 안정 패널
        con.execute("""create or replace temp table st as
            select key from a group by key
            having sum(case when pub >= 20130101 and pub < 20160101 then 1 else 0 end) > 0
               and sum(case when pub >= 20220101 and pub < 20250101 then 1 else 0 end) > 0""")
        for t0 in range(2016, 2025):
            end = t0 * 10000 + 1231
            lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 630
            q = f"""
            with m as (select key, ksic, count(*) n from a where pub > {end - 50000} and pub <= {end} group by key, ksic),
            mo as (select key, ksic from (select key, ksic, row_number() over (partition by key order by n desc, ksic asc) r from m) where r = 1),
            r as (select k1, k2, count(*) n3, sum(case when pub > {end - 10000} then 1 else 0 end) n1 from e where pub > {end - 30000} and pub <= {end} group by k1, k2),
            rk as (select r.*, (k1 in (select key from st) and k2 in (select key from st)) stb from r join mo f on r.k1 = f.key join mo g on r.k2 = g.key where n3 >= 2 and n1 >= 1 and f.ksic <> g.ksic),
            w as (select k1, k2, count(*) nw from e where pub >= {lo} and pub <= {hi} group by k1, k2),
            j as (select rk.*, coalesce(w.nw, 0) nw from rk left join w using (k1, k2))
            select count(*), sum(case when stb then 1 else 0 end),
                   sum(case when nw = 0 then 1 else 0 end), sum(case when nw >= 2 and nw > n3 / 6.0 then 1 else 0 end),
                   sum(case when nw = 0 and stb then 1 else 0 end), sum(case when nw > 0 and stb then 1 else 0 end),
                   sum(case when nw >= 2 and nw > n3 / 6.0 and stb then 1 else 0 end)
            from j"""
            r = con.execute(q).fetchone()
            k = f'{cc}_{variant}_{t0}'
            res[k] = dict(zip(['risk', 'risk_stable', 'eb_pos', 'ec_pos', 'eb_pos_st', 'eb_neg_st', 'ec_pos_st'], [int(x or 0) for x in r]))
            print(k, res[k], flush=True)


def lock(variant, pos, den='risk'):
    tot = 0.0
    for cc in ['CN', 'EP']:
        rates = []
        for t0 in [2022, 2023]:
            x = res[f'{cc}_{variant}_{t0}']
            num = x[pos] if pos != 'eb_neg' else x['risk'] - x['eb_pos']
            rates.append(num / x[den])
        tot += res[f'{cc}_{variant}_2024'][den] * sum(rates) / 2
    return tot


for v in ['asis', 'dedup']:
    res[f'lock_{v}'] = {'Eb_pos': lock(v, 'eb_pos'), 'Eb_neg': lock(v, 'eb_neg'), 'Ec_pos': lock(v, 'ec_pos'),
                        'Eb_pos_stable': lock(v, 'eb_pos_st', 'risk_stable'), 'Eb_neg_stable': lock(v, 'eb_neg_st', 'risk_stable'),
                        'Ec_pos_stable': lock(v, 'ec_pos_st', 'risk_stable')}
    dev = lambda c: sum(res[f'{cc}_{v}_{t}'][c] for cc in ['CN', 'EP'] for t in range(2016, 2024))
    res[f'dev_{v}'] = {'eb_pos': dev('eb_pos'), 'eb_neg': dev('risk') - dev('eb_pos'), 'ec_pos': dev('ec_pos')}
    print(v, res[f'lock_{v}'], res[f'dev_{v}'], flush=True)
json.dump(res, open(os.path.join(HERE, 'w1_out.json'), 'w'), indent=1)
