# verify_0 v2: X1' P0(a)를 작업 캐시·lib13 없이 다시 센다. 읽기 = pyarrow.dataset, 이름 정규화 = C12 lib12.norm_key(원본),
# 조직 정규식 = PREREG 변경 1 문구에서 새로 옮김, 집계 = duckdb SQL.
import sys, json, re
import pyarrow.dataset as ds, pyarrow.compute as pc
import pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
sys.path.insert(0, B + '/분석/검증루프_20261012/C12/C12-T4/scripts')
from lib12 import norm_key   # 원본 함수(읽기만)
OUT = B + '/분석/검증루프_20261012/C13/C13-T4/verify_0/v2_out.json'
CN_RE = re.compile('公司|大学|学院|研究|院|所|厂|集团|中心|局|会|株式会社|LTD|INC|CORP')
EP_RE = re.compile(r'\b(LTD|LIMITED|INC|CORP|GMBH|AG|SA|SAS|BV|NV|AB|OY|SPA|PLC|KK|KAISHA|CO|COMPANY|LLC)\b|UNIV|INSTITUT|INSTITUTE|CENTRE|CENTER|FOUND|UNIVERSIT|ECOLE|FRAUNHOFER|CNRS|COMMISSARIAT|S\.A|S\.P\.A')
dset = ds.dataset(B + '/분석/모델경쟁_20260927/_shared/meta', format='parquet')
rows = []
for cc in ['CN', 'EP']:
    t = dset.to_table(columns=['appl_no', 'pub_date', 'ksic', 'applicant'],
                      filter=(pc.field('cc') == cc) & (pc.field('pub_date') >= '20100101') & (pc.field('pub_date') < '20250101')).to_pandas()
    t = t[t.pub_date.str.len() == 8]
    t['pub'] = t.pub_date.astype(int)
    t = t.sort_values('pub').drop_duplicates('appl_no')
    assert t.pub.max() < 20250101
    recs = []
    cache = {}
    rx = CN_RE if cc == 'CN' else EP_RE
    for an, pub, k, ap in zip(t.appl_no.values, t.pub.values, t.ksic.values, t.applicant.values):
        if ap is None:
            continue
        for nm in re.split(r'[|;]', ap):
            nm = nm.strip()
            if not nm or nm.lower() in ('nan', 'none', '-', 'null'):
                continue
            if nm not in cache:
                key = norm_key(nm)
                cache[nm] = key if (rx.search(nm.upper()) and len(key) >= 2) else None
            key = cache[nm]
            if key is not None:
                recs.append((an, int(pub), k, key))
    a = pd.DataFrame(recs, columns=['an', 'pub', 'k', 'key']).drop_duplicates(['an', 'key'])
    del t, recs
    con = duckdb.connect(); con.execute('set threads=2'); con.execute("set memory_limit='3GB'")
    con.register('a', a)
    con.execute("create table mo as select an from a group by an having count(distinct key)>=2")
    con.execute("create table pf as select x.key f, y.key g, min(x.pub) fst from a x join a y on x.an=y.an where x.key<>y.key and x.an in (select an from mo) group by 1,2")
    for t0 in range(2016, 2025):
        end = t0 * 10000 + 1231
        act = f"select key from a where pub>{end - 30000} and pub<={end} group by key having count(distinct an)>=5"
        risk = con.execute(f"select count(*) from ({act})").fetchone()[0]
        r = {'cc': cc, 't0': t0, 'risk': risk}
        if t0 <= 2023:
            md = f"""select key, k from (select key, k, row_number() over (partition by key order by count(*) desc, k asc) rn
                     from a where pub>{end - 50000} and pub<={end} group by key, k) where rn=1"""
            lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 630
            q = f"""select count(distinct pf.f) from pf join ({act}) ac on ac.key=pf.f join ({md}) mf on mf.key=pf.f join ({md}) mg on mg.key=pf.g
                    where pf.fst between {lo} and {hi} and mf.k<>mg.k"""
            r['pos_strict'] = con.execute(q).fetchone()[0]
        rows.append(r); print(r, flush=True)
    con.close()
df = pd.DataFrame(rows)
dev = df[df.t0 <= 2023]
exp = 0.0
for cc in ['CN', 'EP']:
    b = dev[(dev.cc == cc) & dev.t0.isin([2022, 2023])]
    exp += float(df[(df.cc == cc) & (df.t0 == 2024)].risk.iloc[0]) * float((b.pos_strict / b.risk).mean())
res = {'by_origin': rows, 'dev_sum_strict': int(dev.pos_strict.sum()), 'lock_risk': int(df[df.t0 == 2024].risk.sum()), 'lock_expected_strict': exp}
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(res['dev_sum_strict'], res['lock_risk'], round(exp, 2))
