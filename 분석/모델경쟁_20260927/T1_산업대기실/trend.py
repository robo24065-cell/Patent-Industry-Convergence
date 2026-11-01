# 트렌드 산출물 초안(검증 구간): 산업별 '대기실 지수' = 위험집합 법인들의 예측 진입확률 합(기준시점 2005~2009, 동결 v31, 엄격 정보집합)
# 실현 진입(기준시점+1~+5년, 창 끝 ≤2014)과 산업 간 순위상관으로 지수 타당성 점검. 🔒 2015년 이후 진입은 계산하지 않음.
import os; assert os.environ.get('T1_STRICT') == '1'
from score_cut import *
from scipy.stats import spearmanr
K = load()['K']; names = json.load(open(B/'공모전데이터/외부데이터/KSIC해설/ksic50_definitions.json', encoding='utf-8')) if (B/'공모전데이터/외부데이터/KSIC해설/ksic50_definitions.json').exists() else {}
rows = []; t0 = time.time()
for c in (2005, 2006, 2007, 2008, 2009):
    d = load_cut(c); s, px = score(d); d['px'] = px
    g = d.groupby('k').agg(index=('px', 'sum'), index_new=('px', lambda x: x[d.loc[x.index, 'newc']].sum()), realized=('y', 'sum'),
                           realized_new=('y', lambda x: x[d.loc[x.index, 'newc']].sum()), at_risk=('y', 'size'))
    g['ksic'] = [K[i] for i in g.index]; g['cut'] = c
    rho = spearmanr(g['index'], g.realized).correlation; rho_n = spearmanr(g.index_new, g.realized_new).correlation
    base = spearmanr(g.at_risk, g.realized).correlation
    print(c, '산업별 지수-실현진입 Spearman', round(rho, 3), '(신규진입자', round(rho_n, 3), ') | 비교: 위험집합 크기만', round(base, 3), round(time.time() - t0), flush=True)
    g['rho_all'] = rho; g['rho_new'] = rho_n; g['rho_atrisk_only'] = base
    rows.append(g)
    if c == 2009:
        fm = pd.read_parquet(CACHE/'firms.parquet').set_index('applicant_cd').applicant_name
        top = d.assign(s=s).sort_values('s', ascending=False).groupby('k').head(10)
        top = top.assign(ksic=[K[i] for i in top.k], firm_name=top.firm.map(fm))[['ksic', 'firm', 'firm_name', 'newc', 's', 'px', 'y']]
        top.to_csv(T1/'trend_top10_candidates_c2009.csv', index=False, encoding='utf-8-sig')
        print('c=2009 산업별 상위10 후보 적중률', round(top.y.mean(), 3), '(기저율', round(d.y.mean(), 4), ')')
out = pd.concat(rows, ignore_index=True)
out['share_index'] = out.groupby('cut')['index'].transform(lambda x: x / x.sum()); out['share_realized'] = out.groupby('cut').realized.transform(lambda x: x / x.sum())
out[['cut', 'ksic', 'at_risk', 'index', 'index_new', 'share_index', 'realized', 'realized_new', 'share_realized', 'rho_all', 'rho_new', 'rho_atrisk_only']].to_csv(T1/'trend_waiting_room_index_2005_2009.csv', index=False, encoding='utf-8-sig')
print('saved', round(time.time() - t0))
