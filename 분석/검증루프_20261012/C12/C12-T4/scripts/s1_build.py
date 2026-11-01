# 개발 원점 단위 표. 사용: python -I s1_build.py CC Y [Y ...] → cache/u_{CC}_{Y}.parquet, cache/info_{CC}_{Y}.json
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feat12 import *

cc = sys.argv[1]; YS = [int(x) for x in sys.argv[2:]]
t = time.time()
Cn = Country(cc)
out(cc, 'loaded', Cn.nF, round(time.time() - t, 1))
for Y in YS:
    assert Y < 2025
    for info, df in Cn.gen_origin(Y, 'dev'):
        # 변형 키 거짓 사건(P5c): 양성 행에서 같은 공격적 키의 다른 키 변형이 10년 이력에 s 보유
        t0 = ymd(Y)
        wl = Cn.mwin(addy(t0, -10), t0) & (Cn.m_sc >= 0)
        ah = pd.DataFrame({'ak': Cn.akey_of[Cn.m_aid[wl]], 's': Cn.m_sc[wl]}).drop_duplicates()
        ahs = set(zip(ah.ak.values, ah.s.values))
        pos = df.y.values == 1
        akp = Cn.akey_of[df.aid.values[pos]]
        vk = np.zeros(len(df), np.int8)
        vk[np.where(pos)[0]] = [1 if (a, s) in ahs else 0 for a, s in zip(akp, df.s.values[pos])]
        df['vk'] = vk
        info['vk_rate'] = float(vk[pos].mean()) if pos.any() else None
        info['cover'] = Cn._fc['cover']
        info['secs'] = round(time.time() - t, 1)
        df.to_parquet(C + rf'\u_{cc}_{Y}.parquet', index=False)
        json.dump(info, open(C + rf'\info_{cc}_{Y}.json', 'w'), indent=1)
        out(json.dumps(info))
    Cn._fc = None
out('done', round(time.time() - t, 1))
