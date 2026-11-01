# C05-T1 s1: 데이터(동결 최종 캐시 base_rows, 공개연도 ≤2024) 쪽 탐지.
#  (1) 하위클래스×국가×공개연도 고유 출원 수, (2) KR 출현·급감 탐지, (3) 대응표 폐지 기호(구 기호) 잔존률 = 국가별 소급 재분류 정도(D-c),
#  (4) 데이터가 반영한 IPC 버전(2026.01 신설 기호 존재 여부, D-d). 읽기 전용 입력, 쓰기는 C05-T1/cache, C05-T1/detect_*.csv.
import os, sys, json, time, duckdb
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C05\C05-T1"); C = os.path.join(W, 'cache'); os.makedirs(C, exist_ok=True)
BR = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet").replace('\\', '/')
Cs = C.replace('\\', '/')
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
# 기호 정규화: ipc_raw → 14자리 WIPO 기호(서브클래스4 + 메인그룹4 + 서브그룹6, 서브그룹은 왼쪽 정렬 0 채움). 7판 슬래시 없는 표기는 뒤 2자리를 서브그룹으로.
con.execute(r"""create macro sym14(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/[0-9]{1,6}$') then
     substr(replace(s,' ',''),1,4) || lpad(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1),4,'0')
     || rpad(regexp_extract(replace(s,' ',''),'/([0-9]{1,6})$',1),6,'0')
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then
     substr(replace(s,' ',''),1,4) || lpad(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6),4,'0') || rpad(right(replace(s,' ',''),2),6,'0')
  else null end)""")
con.execute(f"""create or replace table d as select distinct cc, year, appl_no, ksic, sym14(ipc_raw) s from read_parquet('{BR}') where year between 2005 and 2024""")
con.execute("create or replace table du as select cc, year, appl_no, any_value(s) s, count(distinct s) ns from d group by all")
info = con.sql("select count(*) n, sum((s is null)::int) n_null, sum((ns>1)::int) multi from du").df().to_dict('records')[0]
print('docs', info)
# (1) 하위클래스×국가×연도
scy = con.sql("select cc, substr(s,1,4) sc, year, count(*) n from du where s is not null group by all").df()
scy.to_parquet(os.path.join(C, 'sc_cc_year.parquet'))
# 메인그룹×국가×연도(대응표 이동 메인그룹 확인용)
mgy = con.sql("select cc, substr(s,1,4)||' '||cast(cast(substr(s,5,4) as int) as varchar) mg, year, count(*) n from du where s is not null group by all").df()
mgy.to_parquet(os.path.join(C, 'mg_cc_year.parquet'))
# 기호×국가×연도(폐지 기호 잔존률)
sy = con.sql("select cc, s, year, count(*) n from du where s is not null group by all").df()
sy.to_parquet(os.path.join(C, 'sym_cc_year.parquet'))
tot = con.sql("select cc, year, count(*) n from du group by all").df()
tot.to_parquet(os.path.join(C, 'tot_cc_year.parquet'))

# (2) KR 출현·급감
kr = scy[scy.cc == 'KR'].pivot_table(index='sc', columns='year', values='n', aggfunc='sum', fill_value=0)
for y in range(2005, 2025):
    if y not in kr: kr[y] = 0
S = lambda a, b: kr[list(range(a, b + 1))].sum(1)
det = pd.DataFrame(dict(s1517=S(2015, 2017), s1921=S(2019, 2021), s2224=S(2022, 2024)))
det['emergent_1517'] = (det.s1517 <= 2) & (det.s2224 >= 10)
det['emergent_final'] = (det.s1921 <= 2) & (det.s2224 >= 10)
krt = tot[tot.cc == 'KR'].set_index('year').n
ratio_all = krt.loc[2022:2024].sum() / krt.loc[2019:2021].sum()
det['ratio'] = det.s2224 / det.s1921.replace(0, np.nan)
det['decline'] = (det.s1921 >= 200) & (det.ratio <= 0.80) & (det.ratio - ratio_all <= -0.15)
det['first_year_ge3'] = [next((y for y in range(2005, 2025) if kr.loc[s, y] >= 3), None) for s in det.index]
det = det.join(kr[list(range(2015, 2025))])
det.reset_index().to_csv(os.path.join(W, 'detect_kr_subclass.csv'), index=False, encoding='utf-8-sig')
print('KR 전체 비율 2022-24/2019-21 =', round(ratio_all, 4))
print('출현(2015-17 ≤2, 2022-24 ≥10):'); print(det[det.emergent_1517][['s1517', 's1921', 's2224', 'first_year_ge3']].to_string())
print('최종창 출현(2019-21 ≤2):'); print(det[det.emergent_final][['s1517', 's1921', 's2224', 'first_year_ge3']].to_string())
print('급감 원천:'); print(det[det.decline][['s1921', 's2224', 'ratio']].sort_values('ratio').to_string())

# (2b) 신설·원천 하위클래스의 국가×연도
NEWS = ['G06V', 'G06K', 'B64U', 'B64C', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H', 'H01L', 'H10P', 'H10W', 'A23B', 'A23L']
x = scy[scy.sc.isin(NEWS) & (scy.year >= 2012)].pivot_table(index=['sc', 'cc'], columns='year', values='n', aggfunc='sum', fill_value=0)
x['first_ge3'] = [next((y for y in x.columns if isinstance(y, (int, np.integer)) and r[y] >= 3), None) for _, r in x.iterrows()]
x.reset_index().to_csv(os.path.join(W, 'detect_newsub_cc_year.csv'), index=False, encoding='utf-8-sig')
print(x.to_string())

# (3) 폐지 기호 잔존률: 버전 v 대응표의 from 기호 중 modification='d'(폐지)인 기호로 아직 분류된 문서 / (그 기호 + 그 기호의 default=Y to 기호 문서)
cc_ = pd.read_csv(os.path.join(W, 'wipo', 'wipo_concordance_all.csv'), dtype=str)
rows = []
for v in ['20220101', '20230101', '20240101', '20250101', '20260101']:
    g = cc_[(cc_.version == v) & (cc_.from_mod == 'd')]
    old = set(g.from_sym); newy = set(g[g.default == 'Y'].to_sym) - old
    for (c, y), h in sy[sy.cc.notna()].groupby(['cc', 'year']):
        n_old = int(h[h.s.isin(old)].n.sum()); n_new = int(h[h.s.isin(newy)].n.sum())
        rows.append(dict(version=v, cc=c, year=int(y), n_old_sym=n_old, n_new_sym=n_new, old_share=(n_old / (n_old + n_new)) if n_old + n_new else np.nan))
ob = pd.DataFrame(rows); ob.to_csv(os.path.join(W, 'detect_obsolete_symbol_share.csv'), index=False, encoding='utf-8-sig')
print(ob[ob.year.isin([2016, 2019, 2021, 2022, 2023, 2024])].pivot_table(index=['version', 'cc'], columns='year', values='old_share').round(3).to_string())
print(ob[ob.year.isin([2016, 2019, 2021, 2022, 2023, 2024])].pivot_table(index=['version', 'cc'], columns='year', values='n_old_sym').to_string())

# (4) 데이터 반영 버전: 2026.01 신설 하위클래스·메인그룹 문서 수
nw = pd.read_csv(os.path.join(W, 'wipo', 'wipo_title_diff.csv'), dtype=str)
for v in ['20220101', '20230101', '20240101', '20250101', '20260101']:
    nmg = set(nw[(nw.version == v) & (nw.kind == 'new_maingroup')].subclass)
    dmg = set(nw[(nw.version == v) & (nw.kind == 'deleted_maingroup')].subclass)
    a = mgy[mgy.mg.isin(nmg)].groupby('cc').n.sum().to_dict(); b = mgy[mgy.mg.isin(dmg)].groupby('cc').n.sum().to_dict()
    print(v, '신설 메인그룹 문서', a, '| 폐지 메인그룹 문서', b)
print('sec', round(time.time() - t, 1))
