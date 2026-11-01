# IPC 2023.01 신설 하위클래스(H10B 메모리, H10K 유기소자, H10N) 재분류 효과 점검: 양성 셀 순증가·삼성·상위1% 몫이 이 셀에 얼마나 실려 있나.
import os, sys, json, re, unicodedata
import pandas as pd, numpy as np, duckdb
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C04\C04-T3")
BR = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet").replace('\\', '/')
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
# KR 연도별 H10B/H10K/H10N 대 H01L 51·H01L 27 대표 IPC 문서 수(대표 ipc_raw 기준, 고유 출원)
q = con.sql(f"""select year, count(distinct case when replace(ipc_raw,' ','') like 'H10K%' then appl_no end) h10k,
  count(distinct case when replace(ipc_raw,' ','') like 'H10B%' then appl_no end) h10b,
  count(distinct case when replace(ipc_raw,' ','') like 'H10N%' then appl_no end) h10n,
  count(distinct case when replace(ipc_raw,' ','') like 'H01L51/%' then appl_no end) h01l51,
  count(distinct case when replace(ipc_raw,' ','') like 'H01L27/%' then appl_no end) h01l27
  from read_parquet('{BR}') where cc='KR' and year>=2015 group by 1 order by 1""").df()
print(q.to_string())
ac = pd.read_csv(os.path.join(W, 'applicant_concentration.csv'), dtype={'ksic': str})
p = ac[ac.yb == 1].copy(); p['bD'] = p.big_fut - p.big_past; p['h10'] = p.mg.str.startswith('H10')
r = dict(kr_by_year=q.to_dict('records'))
r['pos_cells_h10'] = int(p.h10.sum()); r['pos_D_share_h10'] = float(p[p.h10].D.sum() / p.D.sum())
r['big_share_excl_h10'] = float(p[~p.h10].bD.sum() / p[~p.h10].D.sum())
r['h10_cells_all_eligible'] = int(ac.mg.str.startswith('H10').sum()); r['h10_pos_rate'] = float(ac[ac.mg.str.startswith('H10')].yb.mean())
r['h10_pos_cells'] = p[p.h10][['cell', 'kr_W', 'kr_fut', 'D', 'bD', 'top_app_k1', 'p']].to_dict('records')
# 삼성·LG 집단 몫(H10 제외): kr_docs 대신 base_rows 직접
kr = con.sql(f"select ksic, year, appl_no, ipc_raw, app1 from read_parquet('{BR}') where cc='KR' and year between 2019 and 2024").df()
def mgp(s):
    if s is None: return None
    t = s.replace(' ', ''); mm = re.match(r'^([A-H]\d\d[A-Z])(\d{1,4})/', t)
    if mm: return f"{mm.group(1)} {int(mm.group(2))}"
    mm = re.match(r'^([A-H]\d\d[A-Z])(\d{3,6})$', t)
    if mm: return f"{mm.group(1)} {int(t[4:len(t)-2])}"
    return None
u = pd.Series(kr.ipc_raw.unique()); kr['mg'] = kr.ipc_raw.map(dict(zip(u, [mgp(x) for x in u]))); kr = kr[kr.mg.notna()]
kr['cell'] = kr.ksic + '|' + kr.mg; kr = kr.drop_duplicates(['cell', 'year', 'appl_no'])
kr = kr[kr.cell.isin(set(p.cell))]; kr['s'] = np.where(kr.year <= 2021, -1, 1)
def nm(s):
    s = unicodedata.normalize('NFKC', s or '')
    for w in ['주식회사', '(주)', '(株)']: s = s.replace(w, '')
    return re.sub(r'[^0-9a-z가-힣]', '', s.lower())
kr['k'] = kr.app1.fillna('').map(nm); kr['g'] = np.where(kr.k.str.startswith('삼성'), 'SS', np.where(kr.k.str.startswith('엘지') | kr.k.str.startswith('lg'), 'LG', 'other'))
x = kr[~kr.mg.str.startswith('H10')]; tot = x.s.sum()
r['excl_h10_group_share'] = {g: float(x[x.g == g].s.sum() / tot) for g in ['SS', 'LG']}
y = kr; r['all_group_share_check'] = {g: float(y[y.g == g].s.sum() / y.s.sum()) for g in ['SS', 'LG']}
json.dump(r, open(os.path.join(W, 'verify_2', 'v3_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in r.items() if k != 'kr_by_year'}, ensure_ascii=False, indent=1, default=str))
