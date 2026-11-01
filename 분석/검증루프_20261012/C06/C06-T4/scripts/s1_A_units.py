# s1: C06-A 단위·결과(개발만)·잠금 위험집합 크기, P1·P2(b)(c)·P3 집계.
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from lib6 import *
T0 = time.time()
ca = load_ca(); nm2b6, st = name_map(ca)
docs = load_docs().drop_duplicates('appl_no')
docs = docs[docs.vis.notna() & docs.nname.isin(nm2b6.keys())].copy()
print('linked-name docs', len(docs), 'names', docs.nname.nunique(), round(time.time() - T0), flush=True)
first_vis = docs.groupby('nname').vis.min()
v = pd.read_parquet(os.path.join(C, 'venture_clean.parquet'))
v['s8'] = v.s.dt.strftime('%Y%m%d'); v['e8'] = v.e.dt.strftime('%Y%m%d')
vn = v[v.nname.isin(nm2b6.keys())].copy()
# 같은 정규화 상호의 서로 다른 벤처 행(주소가 다른) → 모호
amb = v.groupby('nname').addr.nunique()
LOG = {'n_ca_unique_corp_names': len(nm2b6), 'n_docnames': int(docs.nname.nunique()),
       'venture_rows_matching_unique_ca_names': int(len(vn)), 'venture_rows_total': int(len(v)),
       'venture_rows_corp_like_match_share': None}
rows = []
for t0, t1 in DEV_A + [LOCK_A]:
    a, b = ymd(t0), ymd(t1)
    is_lock = (t0, t1) == LOCK_A
    names = first_vis[first_vis <= a].index
    valid = set(vn[(vn.s8 <= a) & (vn.e8 >= a)].nname)
    U = pd.DataFrame({'nname': names})
    U['excl_valid'] = U.nname.isin(valid)
    U = U[~U.excl_valid].drop(columns='excl_valid')
    U['t0'] = t0
    if not is_lock:
        assert b <= ymd(LABEL_MAX_A)
        ev = vn[(vn.newre == '신규') & (vn.s8 > a) & (vn.s8 <= b)]
        assert ev.s8.max() <= ymd(LABEL_MAX_A)
        U['y'] = U.nname.isin(set(ev.nname)).astype(int)
        U['amb'] = U.nname.map(amb).fillna(0).gt(1).astype(int)
        # 진단: 원점에 '알 수 없는 유효 확인'(이후 재확인으로 드러남; 시작 ∈ (t0, t0+36개월]) — 진단 전용, 위험집합 구성에는 쓰지 않음
        re_ = vn[(vn.newre == '재확인') & (vn.s8 > a)]
        U['diag_unknown_valid'] = U.nname.isin(set(re_.nname)).astype(int)
    else:
        U['y'] = -1; U['amb'] = -1; U['diag_unknown_valid'] = -1
    rows.append(U)
    print(t0, len(U), int(U.y.clip(lower=0).sum()), round(time.time() - T0), flush=True)
units = pd.concat(rows, ignore_index=True)
units.to_parquet(os.path.join(C, 'A_units.parquet'), index=False)
dev = units[units.y >= 0]
g = dev.groupby('t0').agg(n=('y', 'size'), pos=('y', 'sum'), rate=('y', 'mean'), amb_pos=('amb', lambda s: int((s * dev.loc[s.index, 'y']).sum())),
                          unknown_valid=('diag_unknown_valid', 'mean'))
uniq_pos = int(dev[dev.y == 1].nname.nunique())
lock_n = int((units.t0 == LOCK_A[0]).sum())
last2 = float(g.rate.iloc[-2:].mean())
P1 = dict(dev_by_origin=g.reset_index().to_dict('records'), dev_unique_events=uniq_pos, dev_sum_events=int(g.pos.sum()),
          lock_risk_set=lock_n, last2_rate=last2, lock_expected=round(lock_n * last2, 1),
          pass_a=uniq_pos >= 1000, pass_b=lock_n * last2 >= 1000)
P1['pass'] = bool(P1['pass_a'] and P1['pass_b'])
P1['rate_max_min_ratio'] = float(g.rate.max() / g.rate.min()) if g.rate.min() > 0 else None
LOG['P1'] = P1
# P2(c) 연결 정밀도: 연결된 (단위 상호, 벤처 행)의 시도 일치. 기준: NPS 2024-12 스냅샷(잠금 원점 이전)과 특허 출원인주소.
nps = pd.read_parquet(EXT + r'\firm_nps_wkpl_202412.parquet', columns=['name', 'biz6', 'stat', 'ldong'])
nps = nps[nps.stat.astype(str) == '1'].copy(); nps['nname'] = nps.name.map(norm)
nps['key'] = nps.biz6.astype(str) + '|' + nps.nname
nps['sido'] = nps.ldong.astype(str).str[:2].map(SIDO)
ks = nps.groupby('key').sido.agg(lambda s: set(s.dropna()))
pairs = vn[vn.nname.isin(set(units.nname))][['nname', 'region', 'addr']].copy()
pairs['key'] = pairs.nname.map(nm2b6).astype(str) + '|' + pairs.nname
pairs['nps_sido'] = pairs.key.map(ks)
known = pairs[pairs.nps_sido.map(lambda x: isinstance(x, set) and len(x) > 0)]
agree_nps = known.apply(lambda r: r.region in r.nps_sido, axis=1)
# 특허 출원인주소 시도(최근 문서의 주소 첫 토큰)
def addr_sido(a):
    a = str(a or '').strip()
    for k2, nm in [('서울', '서울'), ('부산', '부산'), ('대구', '대구'), ('인천', '인천'), ('광주', '광주'), ('대전', '대전'), ('울산', '울산'), ('세종', '세종'),
                   ('경기', '경기'), ('강원', '강원'), ('충청북', '충북'), ('충북', '충북'), ('충청남', '충남'), ('충남', '충남'), ('전라북', '전북'), ('전북', '전북'),
                   ('전라남', '전남'), ('전남', '전남'), ('경상북', '경북'), ('경북', '경북'), ('경상남', '경남'), ('경남', '경남'), ('제주', '제주')]:
        if a.startswith(k2):
            return nm
    return None
lastaddr = docs.sort_values('vis').groupby('nname').addr.last().map(addr_sido)
pairs['pat_sido'] = pairs.nname.map(lastaddr)
kp = pairs[pairs.pat_sido.notna()]
LOG['P2c'] = dict(pairs=int(len(pairs)), nps_known=int(len(known)), agree_nps=round(float(agree_nps.mean()), 4) if len(known) else None,
                  pat_addr_known=int(len(kp)), agree_pat_addr=round(float((kp.region == kp.pat_sido).mean()), 4) if len(kp) else None,
                  amb_names_share_pairs=round(float(pairs.nname.map(amb).gt(1).mean()), 4))
LOG['P2c']['pass'] = bool(LOG['P2c']['agree_nps'] is not None and LOG['P2c']['agree_nps'] >= 0.80)
LOG['P2b'] = dict(rate_max_min_ratio=P1['rate_max_min_ratio'], fail=bool(P1['rate_max_min_ratio'] and P1['rate_max_min_ratio'] > 2.0),
                  warn=bool(P1['rate_max_min_ratio'] and 1.5 <= P1['rate_max_min_ratio'] <= 2.0))
LOG['P3'] = dict(n_dev_origins=len(g), min_pos=int(g.pos.min()), pass_=bool(len(g) >= 4 and g.pos.min() >= 100), warn_lt200=int((g.pos < 200).sum()))
json.dump(LOG, open(os.path.join(C, 's1_A_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(LOG, ensure_ascii=False, indent=1, default=str))
print('done', round(time.time() - T0))
