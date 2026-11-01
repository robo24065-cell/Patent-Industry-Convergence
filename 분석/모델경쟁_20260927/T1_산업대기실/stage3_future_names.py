# 3단계 후처리: future_forecast.csv 의 ksic_name 채우기(final_run의 이름 파서가 dict 형식만 가정해 list 형식 JSON에서 빈 값이 됨). 예측값은 변경하지 않음.
import sys, json, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from common import B, T1
js = json.load(open(B/'공모전데이터/외부데이터/KSIC해설/ksic50_definitions.json', encoding='utf-8'))
nm = {str(d['code']): d.get('codenmKr', '') for d in js}
for f in ['future_forecast.csv', 'future_candidates_top50.csv']:
    g = pd.read_csv(T1/f, encoding='utf-8-sig', dtype={'ksic': str, 'firm': str})
    g['ksic_name'] = g.ksic.map(nm)
    g.to_csv(T1/f, index=False, encoding='utf-8-sig')
g = pd.read_csv(T1/'future_forecast.csv', encoding='utf-8-sig', dtype={'ksic': str})
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(g[['ksic', 'ksic_name', 'at_risk', 'waiting_room_index', 'index_share', 'index_newcomer', 'mean_prob', 'n_high']].round(4).to_string())
print(g[['ksic', 'top5_candidates']].head(5).to_string())
c = pd.read_csv(T1/'future_candidates_top50.csv', encoding='utf-8-sig', dtype={'ksic': str, 'firm': str})
print(c.head(10).to_string()); print('신규진입자 비중(상위50)', round(c.newc.mean(), 4), '| 전체 위험집합 기대 진입 합', round(g.waiting_room_index.sum(), 1))
n = c[c.newc].sort_values('score', ascending=False).head(10); print(n.to_string())
