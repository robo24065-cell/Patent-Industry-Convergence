# C08-T4 s1c: P3(c) 잠금 관측 가능성 — pay 월별 총 행 수만(지연 여부·특허 속성 미조회). 노출로 기록.
import sqlite3, sys, os, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import DB, C
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
m = pd.read_sql("select substr(payDate,1,6) ym, count(*) n from pay where payDate >= '20230101' group by 1 order by 1", con)
m['n'] = m.n.astype(int)
y = m.assign(y=m.ym.str[:4]).groupby('y').n.sum()
mon25 = m[m.ym.str[:4] == '2025'].n.mean()
mon26 = m[(m.ym >= '202601') & (m.ym <= '202607')].n.mean()
out = {'monthly': dict(zip(m.ym, m.n)), 'year_tot': y.to_dict(), 'ratio_2025_2024': float(y['2025'] / y['2024']),
       'mon26_1to7_over_mon25': float(mon26 / mon25), 'max_ym': m.ym.max()}
out['pass'] = bool(out['ratio_2025_2024'] >= 0.85 and out['mon26_1to7_over_mon25'] >= 0.5)
json.dump(out, open(os.path.join(C, 's1c_monthly.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=int)
print(json.dumps(out, ensure_ascii=False, indent=1, default=int))
