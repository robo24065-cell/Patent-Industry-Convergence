# verify_0 v1: X1 P0(a)를 작업 캐시 없이 sqlite SQL만으로 다시 센다(PREREG §5).
import sqlite3, json, sys
sys.stdout.reconfigure(encoding='utf-8')
REG = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/수집파이프라인/registry_full.sqlite'
OUT = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C13/C13-T4/verify_0/v1_out.json'
con = sqlite3.connect('file:' + REG + '?mode=ro', uri=True)
con.execute("""create temp table o as
  select distinct a.rgst_no, a.applicantCd cd, p.ksic,
         cast(substr(replace(replace(p.rgstDate,'-',''),'.',''),1,8) as integer) rd
  from applicant a join patent p on p.rgst_no=a.rgst_no
  where substr(a.applicantCd,1,1) in ('1','2','3','5')""")
con.execute("delete from o where rd >= 20250101")
con.execute("create temp table mo as select rgst_no from o group by rgst_no having count(distinct cd)>=2")
con.execute("create index io on o(rgst_no)"); con.execute("create index io2 on o(cd)")
con.execute("""create temp table pf as select x.cd f, y.cd g, min(x.rd) first from o x join o y on x.rgst_no=y.rgst_no
               where x.cd<>y.cd and x.rgst_no in (select rgst_no from mo) group by x.cd, y.cd""")
res = []
for t0 in range(2014, 2025):
    end = t0 * 10000 + 1231
    con.execute("drop table if exists el"); con.execute("drop table if exists md")
    con.execute(f"create temp table el as select cd from o where rd<={end} and substr(cd,1,1)='1' group by cd having count(distinct rgst_no)>=3")
    con.execute(f"""create temp table md as select cd, ksic from (
        select cd, ksic, row_number() over (partition by cd order by count(*) desc, ksic asc) rn
        from o where rd<={end} group by cd, ksic) where rn=1""")
    risk = con.execute("select count(*) from el").fetchone()[0]
    r = {'t0': t0, 'risk': risk}
    if t0 <= 2023:
        lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 822
        q = f"""select count(distinct pf.f) from pf join el on el.cd=pf.f
                join md mf on mf.cd=pf.f join md mg on mg.cd=pf.g
                where pf.first between {lo} and {hi} and mf.ksic<>mg.ksic"""
        ql = f"select count(distinct pf.f) from pf join el on el.cd=pf.f where pf.first between {lo} and {hi}"
        r['pos_strict'] = con.execute(q).fetchone()[0]; r['pos_loose'] = con.execute(ql).fetchone()[0]
    res.append(r); print(r, flush=True)
dev = [r for r in res if r['t0'] <= 2023]
b = sum(r['pos_strict'] / r['risk'] for r in dev if r['t0'] in (2022, 2023)) / 2
summ = {'by_origin': res, 'dev_sum_strict': sum(r['pos_strict'] for r in dev), 'lock_risk': res[-1]['risk'],
        'lock_expected_strict': res[-1]['risk'] * b}
json.dump(summ, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(summ['dev_sum_strict'], summ['lock_risk'], round(summ['lock_expected_strict'], 2))
