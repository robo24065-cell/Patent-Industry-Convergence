# C19-T1 verify_1: 다른 코드 경로(pandas + numpy RandomState(7), 연도별 비복원 표집)로 위약 목록 VRNDJP7 생성
import sys, os, json; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import C, W, out
import pandas as pd, numpy as np
J = pd.read_parquet(os.path.join(C, 'jp_rows.parquet'), columns=['appl_no', 'appl_year', 'pub_date', 'rgst_date', 'py'])
NP = pd.read_parquet(os.path.join(C, 'jp_nopub_rows.parquet'))
NP = NP[NP.py.notna() & NP.py.astype(float).between(2000, 2024)]
quota = NP.groupby('py').size()
# 짝(이중 수록 상대) 제외: s4와 같은 집합을 쓰지 않고 다시 구하지 않는다 — 보수적으로 '공개일 있는 행 중 jp_nopub 쌍의 짝일 수 있는 행'은 제외하지 않고,
# 대신 s4 위약과 결과 차이로 짝 제외 규칙의 영향도 같이 본다(다른 규칙 = 반박 시도).
pool = J[J.pub_date.notna() & (J.appl_year >= 2000) & J.py.notna()]
pool = pool[pool.py.astype(float).between(2000, 2024)]
rs = np.random.RandomState(7); sel = []
for y, n in quota.items():
    a = pool.appl_no[pool.py == y].values
    sel.append(rs.choice(np.sort(a), size=min(n, len(a)), replace=False))
sel = np.concatenate(sel)
os.makedirs(os.path.join(C, 'lists'), exist_ok=True)
pd.DataFrame({'appl_no': sel}).to_parquet(os.path.join(C, 'lists', 'VRNDJP7.parquet'), index=False)
r = dict(n=int(len(sel)), quota_total=int(quota.sum()), n_2021=int(quota.get('2021', 0)))
json.dump(r, open(os.path.join(W, 'verify_1', 'verify_1_lists.json'), 'w'), indent=1); out(r)
