# C04-T3 verify_2: 'big(상위1%) 몫 69.1%, 삼성 29.8%·LG 12.7%, SAB 대기업 45.9%, SAB 상위10 30.2%' 반박 시도.
# 독립 구현: base_rows.parquet(KR) 직접 → 자체 메인그룹 파싱·자체 이름 정규화. s1/s2/s3 코드·캐시 미사용. 동결 예측 셀 집합·yb만 사용.
import os, sys, re, json, unicodedata
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
OUT = os.path.join(ROOT, r"검증루프_20261012\C04\C04-T3\verify_2")
BR = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet").replace('\\', '/')
P = pd.read_parquet(os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3\preds\FINAL_R3_t0_2021.parquet"))
print(P.columns.tolist(), len(P))
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
kr = con.sql(f"select ksic, year, appl_no, ipc_raw, app1 from read_parquet('{BR}') where cc='KR'").df()
# 자체 메인그룹 파싱: 'A01B 1/00' → 'A01B 1'
def mgp(s):
    if s is None: return None
    t = s.replace(' ', '')
    mm = re.match(r'^([A-H]\d\d[A-Z])(\d{1,4})/', t)
    if mm: return f"{mm.group(1)} {int(mm.group(2))}"
    mm = re.match(r'^([A-H]\d\d[A-Z])(\d{3,6})$', t)
    if mm: return f"{mm.group(1)} {int(t[4:len(t)-2])}"
    return None
u = pd.Series(kr.ipc_raw.unique()); mp = dict(zip(u, [mgp(x) for x in u]))
kr['mg'] = kr.ipc_raw.map(mp); kr = kr[kr.mg.notna()]
kr['cell'] = kr.ksic + '|' + kr.mg
d = kr.drop_duplicates(['cell', 'year', 'appl_no']).copy()
print('KR doc-cell rows', len(d))
# 자체 정규화
def nm(s):
    s = unicodedata.normalize('NFKC', s or '')
    for w in ['주식회사', '(주)', '(株)', '유한회사']: s = s.replace(w, '')
    return re.sub(r'[^0-9a-z가-힣]', '', s.lower())
un = pd.Series(d.app1.fillna('').unique()); m1 = dict(zip(un, [nm(x) for x in un])); d['k'] = d.app1.fillna('').map(m1)

# 셀 레이블 확인
cells = set(P.cell)
e = d[d.cell.isin(cells)]
kw = e[(e.year >= 2019) & (e.year <= 2021)].groupby('cell').size(); kf = e[(e.year >= 2022) & (e.year <= 2024)].groupby('cell').size()
T = P[['cell', 'yb', 'p', 'p_base']].copy() if 'yb' in P else None
T['kr_W'] = T.cell.map(kw).fillna(0); T['kr_fut'] = T.cell.map(kf).fillna(0); T['D'] = T.kr_fut - T.kr_W
T['y2'] = np.log((T.kr_fut + 1) / (T.kr_W + 1))
print('pos', int(T.yb.sum()), 'label threshold y (min over pos)', T.y2[T.yb == 1].min(), 'max over neg', T.y2[T.yb == 0].max(), 'D min pos', T.D[T.yb == 1].min())
pos = set(T.cell[T.yb == 1]); Dpos = T.D[T.yb == 1].sum(); print('sum D pos', Dpos)

res = {}
def big_set(frame, key, by_unique=False):
    f = frame[(frame.year >= 2012) & (frame.year <= 2021) & (frame[key] != '')]
    if by_unique: f = f.drop_duplicates(['appl_no'])
    v = f.groupby(key).size().sort_values(ascending=False); n = int(np.ceil(0.01 * len(v)))
    return set(v.index[:n]), dict(n_app=int(len(v)), n_top=n, share=float(v.iloc[:n].sum() / v.sum()), min_docs=int(v.iloc[n - 1]))
B, info = big_set(d, 'k'); res['big_docrow'] = info
Bu, infou = big_set(d, 'k', True); res['big_unique_appl'] = infou
print(res)
w = e[(e.year >= 2019) & (e.year <= 2024)].copy(); w['s'] = np.where(w.year <= 2021, -1, 1)
def share_inc(cellset, bigset, key='k'):
    x = w[w.cell.isin(cellset)]; tot = x.s.sum(); b = x[x[key].isin(bigset)].s.sum(); return float(b / tot), int(b), int(tot)
def share_docs(cellset, bigset, lo, hi, key='k'):
    x = w[w.cell.isin(cellset) & (w.year >= lo) & (w.year <= hi)]; return float(x[key].isin(bigset).mean())
allc = set(T.cell); neg = allc - pos
res['big_share_pos_increase'] = share_inc(pos, B)
res['big_share_pos_increase_uniqdef'] = share_inc(pos, Bu)
res['big_share_all_increase'] = share_inc(allc, B)
res['big_share_neg_increase'] = share_inc(neg, B)
res['big_share_fut_docs_all'] = share_docs(allc, B, 2022, 2024); res['big_share_fut_docs_pos'] = share_docs(pos, B, 2022, 2024)
res['big_share_past_docs_all'] = share_docs(allc, B, 2019, 2021); res['big_share_past_docs_pos'] = share_docs(pos, B, 2019, 2021)
# 대기업 몫을 양성 셀 '증가 셀만'으로: 셀 단위 big 증가 / 셀 증가
# SAB
g = w.groupby(['cell', 'k']).s.sum().reset_index()
g2 = g.merge(w.groupby(['cell', 'k']).apply(lambda z: int((z.s == 1).sum())).rename('fut').reset_index(), on=['cell', 'k'])
g2 = g2.sort_values(['cell', 's', 'fut', 'k'], ascending=[True, False, False, True])
top = g2.groupby('cell').head(1).set_index('cell')
T['top_k'] = T.cell.map(top.k); T['top_d'] = T.cell.map(top.s).fillna(0)
T['sab'] = (T.yb == 1) & (T.D > 0) & (T.top_d >= 0.5 * T.D)
S = T[T.sab]
res['sab_n'] = int(T.sab.sum()); res['sab_top_is_big'] = float(S.top_k.isin(B).mean())
vc = S.top_k.value_counts(); res['sab_top_applicants'] = vc.head(12).to_dict(); res['sab_top10_share'] = float(vc.head(10).sum() / len(S))
# 집단 키(자체 규칙: 정규화 이름 포함/접두)
GRP = {'삼성': ['삼성'], '엘지': ['엘지', 'lg'], '현대차': ['현대자동차', '기아', '현대모비스', '현대위아', '현대케피코', '현대트랜시스'],
       '에스케이': ['에스케이', 'sk'], }
def grp(k):
    for gname, pres in GRP.items():
        for pr in pres:
            if k.startswith(pr): return 'G:' + gname
    return k
uk = pd.Series(w.k.unique()); gm = dict(zip(uk, [grp(x) for x in uk])); w['g'] = w.k.map(gm)
x = w[w.cell.isin(pos)]; gg = x.groupby('g').s.sum().sort_values(ascending=False)
res['pos_increase_groups'] = {k: float(v / x.s.sum()) for k, v in gg.head(8).items()}
xa = w[w.cell.isin(allc)]; ga = xa.groupby('g').s.sum()
res['all_increase_groups'] = {k: float(ga.get(k, 0) / xa.s.sum()) for k in ['G:삼성', 'G:엘지', 'G:현대차']}
xd = w[w.cell.isin(allc) & (w.year <= 2021)]; res['past_doc_share_groups_allcells'] = {k: float((xd.g == k).mean()) for k in ['G:삼성', 'G:엘지', 'G:현대차']}
xf = w[w.cell.isin(allc) & (w.year >= 2022)]; res['fut_doc_share_groups_allcells'] = {k: float((xf.g == k).mean()) for k in ['G:삼성', 'G:엘지', 'G:현대차']}
xp = w[w.cell.isin(pos) & (w.year >= 2022)]; res['fut_doc_share_groups_pos'] = {k: float((xp.g == k).mean()) for k in ['G:삼성', 'G:엘지', 'G:현대차']}
# '삼성' 접두 이름 목록(그룹 외 회사 섞임 여부)
sam = x[x.g == 'G:삼성'].groupby('k').s.sum().sort_values(ascending=False); res['samsung_names_pos_net'] = sam.head(15).to_dict()
lg = x[x.g == 'G:엘지'].groupby('k').s.sum().sort_values(ascending=False); res['lg_names_pos_net'] = lg.head(10).to_dict()
# 기아/사명 변경 점검: 연도별 이름
for pat in ['기아', '엘지에너지', '엘지화학', '에스케이온', '에스케이이노베이션']:
    q = d[d.k.str.startswith(pat)].groupby(['k', 'year']).size().unstack(fill_value=0)
    q = q.loc[q.sum(1).sort_values(ascending=False).index[:4], [c for c in q.columns if c >= 2016]]
    res[f'name_by_year_{pat}'] = {k: {int(c): int(v) for c, v in row.items()} for k, row in q.iterrows()}
# 셀별 '양성 셀 하나하나'에서 big 몫(중앙값) — 합계가 일부 큰 셀에 좌우되는지
cb = x.assign(b=x.k.isin(B)).groupby('cell').apply(lambda z: pd.Series(dict(D=z.s.sum(), bD=z.s[z.b].sum())))
res['pos_D_top10cells_share_of_sumD'] = float(cb.D.sort_values(ascending=False).head(10).sum() / cb.D.sum())
res['pos_D_top50cells_share_of_sumD'] = float(cb.D.sort_values(ascending=False).head(50).sum() / cb.D.sum())
res['big_share_pos_increase_excl_top10cells'] = float(cb.drop(cb.D.sort_values(ascending=False).head(10).index).bD.sum() / cb.drop(cb.D.sort_values(ascending=False).head(10).index).D.sum())
res['cell_median_big_share'] = float((cb.bD / cb.D).median())
res['frac_pos_cells_big_share_gt50'] = float(((cb.bD / cb.D) > 0.5).mean())
json.dump(res, open(os.path.join(OUT, 'v1_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
