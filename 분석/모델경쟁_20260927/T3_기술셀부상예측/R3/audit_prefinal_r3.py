"""T3 3라운드 최종시험 전 감사: 최종 캐시(cache_pub_final, 공개연도 ≤2024)로 만든 개발 구간(t0 ≤2018) 피처·레이블이
개발 캐시(cache_pub, 공개연도 ≤2021)와 같은지 대조한다. 같으면 → 2022년 이후 공개분이 과거 원점 피처에 섞이지 않는다(원점 이후 공개분 혼입 없음).
🔒 LABEL_MAX=2021 로 고정 → t0 ≥2019 레이블은 계산 자체가 막힌다(r3lib.labels assert). t0=2021 피처·레이블은 만들지 않는다.
출력: audit_prefinal_r3.json
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2021', T3R3_FINAL_UNLOCK='1')
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import r3lib as L
assert L.LABEL_MAX_YEAR == 2021
P = L.load()
out = {}
# 1) 셀 패널 기반 피처(feats) t0 2005..2018 대조
mx = {}
for t0 in range(2005, 2019):
    a = pd.read_parquet(os.path.join(HERE, 'cache_pub', f'feat_{t0}.parquet'))
    b = L.feats(P, t0, 10)
    assert set(a.columns) == set(b.columns), (t0, set(a.columns) ^ set(b.columns))
    m = a.merge(b, on=['cell', 'ksic', 'mg', 't0'], how='outer', suffixes=('_d', '_f'), indicator=True)
    only = m._merge.value_counts().to_dict()
    num = [c for c in a.columns if c not in ('cell', 'ksic', 'mg', 't0')]
    mb = m[m._merge == 'both']
    d = max(float(np.nan_to_num(np.nanmax(np.r_[0.0, np.abs(mb[c + '_d'].astype(float).values - mb[c + '_f'].astype(float).values)]))) for c in num)
    nan_mis = int(sum(((mb[c + '_d'].isna()) != (mb[c + '_f'].isna())).sum() for c in num))
    mx[t0] = dict(n_dev=len(a), n_final=len(b), left_only=int(only.get('left_only', 0)), right_only=int(only.get('right_only', 0)), maxdiff=d, nan_mismatch=nan_mis)
    print(t0, mx[t0], flush=True)
out['feats_t0_le_2018'] = mx
# 2) feat2(출원인·이웃·서브클래스) t0 ≤2018 대조
f2d = pd.read_parquet(os.path.join(HERE, 'cache_pub', 'feat2.parquet')); f2f = pd.read_parquet(os.path.join(HERE, 'cache_pub_final', 'feat2.parquet'))
f2f = f2f[f2f.t0 <= 2018]; f2d = f2d[f2d.t0 <= 2018]
m = f2d.merge(f2f, on=['ksic', 'mg', 't0'], how='outer', suffixes=('_d', '_f'), indicator=True)
num = [c for c in f2d.columns if c not in ('ksic', 'mg', 't0', 'sub')]
mb = m[m._merge == 'both']
out['feat2_t0_le_2018'] = dict(n_dev=len(f2d), n_final=len(f2f), merge=m._merge.value_counts().astype(int).to_dict(),
                               maxdiff=max(float(np.nanmax(np.abs(mb[c + '_d'] - mb[c + '_f']))) for c in num))
print('feat2', out['feat2_t0_le_2018'], flush=True)
# 3) RoS 셀 피처(최종 셀 목록으로 재계산) t0 ≤2018 대조 (개발 산출물 extB_cell_pub.parquet)
import build_ext_r3b as XB
R = XB.ros_feats(list(range(2005, 2019)), P['cells'], P['ksic'], P['mg'])
B0 = pd.read_parquet(os.path.join(HERE, 'extB_cell_pub.parquet'))[['cell', 't0', 'ros_sci', 'ros_ppp']]
mr = B0.merge(R[['cell', 't0', 'ros_sci', 'ros_ppp']], on=['cell', 't0'], how='left', suffixes=('_d', '_f'))
out['ros_t0_le_2018'] = dict(n=len(mr), missing_in_final=int(mr.ros_sci_f.isna().sum()),
                             maxdiff=max(float(np.nanmax(np.abs(mr[c + '_d'] - mr[c + '_f']))) for c in ['ros_sci', 'ros_ppp']))
print('ros', out['ros_t0_le_2018'], flush=True)
# 4) RoS 원천의 발행연도·연결 레코드 규칙 점검 (t0=2021 에 쓰일 행이 발행 ≤2021, 연결 출원 ≤2019 인지)
ci = pd.read_parquet(os.path.join(HERE, '..', '..', '_shared', 'ext', 'r3ros_cell_issueyear.parquet'))
u = ci[(ci.first_rec_year <= 2019) & (ci.issue_year <= 2021) & (ci.issue_year >= 2012)]
out['ros_rule_t0_2021'] = dict(rows=len(u), max_issue_year=int(u.issue_year.max()), max_first_rec_year=int(u.first_rec_year.max()))
ok = all(v['maxdiff'] == 0 and v['left_only'] == 0 and v['right_only'] == 0 and v['nan_mismatch'] == 0 for v in mx.values()) \
     and out['feat2_t0_le_2018']['maxdiff'] < 1e-9 and out['ros_t0_le_2018']['maxdiff'] == 0 and out['ros_t0_le_2018']['missing_in_final'] == 0
out['all_identical'] = bool(ok)
json.dump(out, open(os.path.join(HERE, 'audit_prefinal_r3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('ALL IDENTICAL:', ok)
