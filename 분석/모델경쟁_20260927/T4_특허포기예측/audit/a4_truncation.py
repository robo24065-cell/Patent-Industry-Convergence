# -*- coding: utf-8 -*-
"""감사 4: 절단 불변성 검사 — 기준일 L 이후 사건(납부·소멸·권리사건·권리자이전·실시권)을 모두 지운 자료로
피처를 다시 계산해도 캐시된 피처와 같아야 한다(같지 않으면 L 이후 정보가 새고 있다는 뜻).
대상: t4lib.features, soc.soc_features, soc.add_ak, feat2.agent_feats, feat2.owner_feats, feat2.fam_dated(연도 절단), feat2.macro_at.
정적 서지(청구항 수·최종처분·IPC·만료일·대리인 수)는 스냅샷이라 이 검사로 잡을 수 없다 → 별도 기록."""
import sys, os, shutil
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import t4lib as T, soc as SO, feat2 as F2

L = sys.argv[1] if len(sys.argv) > 1 else '2009-07-01'
Ld = np.datetime64(L)
ref_f = T.features(L); ref_s = SO.soc_features(L); ref_a = F2.agent_feats(L); ref_o = F2.owner_feats(L)
ref_ak = SO.add_ak(ref_f.assign(acd=T.load()['pt'].acd.values[ref_f.pid.values]), 'next_annl')[['ak_rate', 'ak_n']]

D = T.load()
TR = dict(D)
TR['pay'] = D['pay'][D['pay'].pd <= Ld]
TR['rt'] = D['rt'][D['rt'].cd <= Ld]
TR['lap'] = D['lap'][D['lap'].cd <= Ld]
TR['ev'] = {k: v[v.cd <= Ld] for k, v in D['ev'].items()}
TR['ow'] = D['ow'][D['ow'].od <= Ld]
TR['lic'] = D['lic'][D['lic'].sd <= Ld]
T._D.clear(); T._D.update(TR)
SO._AK.clear(); SO._INV.clear(); F2._C.clear()
# 소유자 시기표: owner.parquet 에서 L 이후 행 제거한 사본을 감사용 캐시로
AUD = os.path.join(HERE, 'audit', 'cache_trunc'); os.makedirs(AUD, exist_ok=True)
for fn in ['inventor.parquet', 'agent.parquet', 's2_fam_pid.parquet']:
    shutil.copy(os.path.join(T.CACHE, fn), os.path.join(AUD, fn))
o = pd.read_parquet(os.path.join(T.CACHE, 'owner.parquet'))
o = o[pd.to_datetime(o.ownerRgstCsDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce') <= pd.Timestamp(L)]
o.to_parquet(os.path.join(AUD, 'owner.parquet'), index=False)
T.CACHE = AUD; F2.CACHE = AUD
new_f = T.features(L, use_cache=False)
new_s = SO.soc_features(L); new_a = F2.agent_feats(L); new_o = F2.owner_feats(L)
new_ak = SO.add_ak(new_f.assign(acd=T.load()['pt'].acd.values[new_f.pid.values]), 'next_annl')[['ak_rate', 'ak_n']]


def cmp(nm, a, b):
    assert len(a) == len(b), (nm, len(a), len(b))
    bad = []
    for c in a.columns:
        if c not in b: continue
        x, y = a[c].values, b[c].values
        if x.dtype.kind in 'fiub':
            ne = ~np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-6)
        else:
            ne = x.astype(str) != y.astype(str)
        if ne.any(): bad.append((c, int(ne.sum())))
    print(f'{nm}: 행 {len(a):,} 불일치 열 {bad if bad else "없음"}')


cmp('features', ref_f, new_f); cmp('soc', ref_s, new_s); cmp('ak', ref_ak, new_ak); cmp('agent', ref_a, new_a); cmp('owner', ref_o, new_o)
# 패밀리: 추정 공개연도 <= L연도-1 만 쓰므로 연도 절단 토큰표와 동일해야
fa = F2.fam_dated(L, ref_f.pid.values)
print('fam_dated: 사용 토큰 최대 연도 <=', int(L[:4]) - 1, '(정의상)')
# 경기지표: L-2개월 이후 행을 지워도 같아야
m1 = F2.macro_at(L)
mac = F2._macro(); last = pd.Period(L[:7], freq='M') - 2
F2._C['mac'] = {k: v[v.index <= last] for k, v in mac.items()}
m2 = F2.macro_at(L)
print('macro: 불일치', int((~np.isclose(m1.values, m2.values, equal_nan=True)).sum()))
shutil.rmtree(AUD)
