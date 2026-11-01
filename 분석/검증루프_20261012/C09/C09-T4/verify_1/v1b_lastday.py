# C09-T4 verify_1b(사후 민감도, 판정 밖): 마지막 합법일 납부끼리 비교 — B0 평일 기한 당일 납부, W1 주말·공휴일 기한의 보정 기한(R2) 당일 납부, D1 평일 기한의 다음 영업일 납부(1영업일 지연), W2 보정 기한 다음 영업일 이후 1–30일.
# 원 주석: C09-T4 verify_1(반박 시도, PREREG §6): '주말 기한 뒤 다음 영업일 납부(보정으로 기한 내로 바뀌는 납부)가 실제로는 진짜 지연과 같은 위험을 나른다
#  → 보정이 신호를 지운다'를 검정한다. 반대 방향('주말 기한 납부자는 원래 성실하다')도 같은 비교로 드러난다.
#  개발 구간: 갱신 납부일 2008-01-01 ~ 2017-12-31, 결과 = 그 특허가 납부 뒤 3년 안(실효일 E ∈ (납부일, 납부일+3년])에 등록료 불납 소멸.
#  소멸 사건은 트랙 적재 캐시(loaded_r2dev.pkl, E < 2022-01-01 만 적재)에서 읽는다(읽기 전용) — 결과 창 끝 ≤ 2020-12-31 이라 절단 없음.
#  집단: A 평일 기한·4일 이상 일찍, B 평일 기한·당일~3일 전, W 주말·공휴일 기한·기한 후지만 R2 보정 기한 내, D 평일 기한·1–30일 지연, (참고) W0 주말·공휴일 기한·당일~3일 전.
#  표준화: 출원인 유형 × 연차(4–6, 7–9, 10–12, 13+) 층, 네 집단 합친 분포로 직접 표준화. 출원인 군집 부트스트랩 300회(시드 0).
import os, sys, json, pickle, gc
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
hol = load_hol()
D = pickle.load(open(os.path.join(T4D, 'R2', 'cache', 'loaded_r2dev.pkl'), 'rb'))
lap = D['lap'][['pid', 'E']].copy(); del D; gc.collect()
p = pd.read_parquet(os.path.join(C, 't4_pay.parquet'))
pt = pd.read_parquet(os.path.join(C, 't4_pt.parquet'))
p = p[(p.seq > 0) & (p.pd >= '2008-01-01') & (p.pd <= '2017-12-31')].copy()
due = p.due.values.astype('datetime64[D]')
p['off'] = (dow(due) >= 5) | np.isin(due, hol)
e, e2 = p.early.values, p.earlyR2.values
grp = np.full(len(p), '', dtype=object)
nb = np.busday_offset(due, 1, roll='forward', weekmask='1111100', holidays=hol)   # 기한 다음 영업일(기한이 영업일일 때)
pdd0 = p.pd.values.astype('datetime64[D]')
dR2 = p.dueR2.values.astype('datetime64[D]') if 'dueR2' in p else None
grp[(~p.off.values) & (e >= 4)] = 'A'
grp[(~p.off.values) & (e == 0)] = 'B0'
grp[(~p.off.values) & (pdd0 == nb)] = 'D1'
grp[(p.off.values) & (e < 0) & (e2 == 0)] = 'W1'
grp[(p.off.values) & (e2 <= -1) & (e2 >= -30)] = 'W2'
grp[(~p.off.values) & (e <= -2) & (e >= -30) & (pdd0 != nb)] = 'D2'
p['g'] = grp
p = p[p.g != ''].copy()
p = p.reset_index(drop=True); p['rid'] = np.arange(len(p))
pdd = p.pd.values.astype('datetime64[D]')
p['end'] = (pd.DatetimeIndex(pdd) + pd.DateOffset(years=3)).values.astype('datetime64[D]')
mm = p[['rid', 'pid', 'pd', 'end']].merge(lap, on='pid')
hit = mm[(mm.E.values.astype('datetime64[D]') > mm.pd.values.astype('datetime64[D]')) & (mm.E.values.astype('datetime64[D]') <= mm.end.values.astype('datetime64[D]'))].rid.unique()
p['y'] = np.isin(p.rid.values, hit).astype(float)
p['atype'] = p.pid.map(pt.set_index('pid').atype).fillna('other')
p['acd'] = p.pid.map(pt.set_index('pid').acd).fillna('NA')
p['sb'] = pd.cut(p.s, [0, 6, 9, 12, 99], labels=['4-6', '7-9', '10-12', '13+']).astype(str)
p['st'] = p.atype + '|' + p.sb
main = p[p.g.isin(['A', 'B0', 'W1', 'D1', 'W2', 'D2'])]
wst = main.st.value_counts(normalize=True)


def std_rates(df):
    r = df.groupby(['g', 'st']).y.mean().unstack()
    r = r.reindex(columns=wst.index)
    # 층에 값이 없으면 그 집단 전체 평균으로 채움
    r = r.apply(lambda row: row.fillna(df[df.g == row.name].y.mean()), axis=1)
    return (r * wst.values).sum(axis=1)


raw = p.groupby('g').y.agg(['size', 'mean'])
sr = std_rates(p)
# 군집 부트스트랩
rng = np.random.default_rng(0)
cl = np.where(p.acd.values == 'NA', 'P' + p.pid.astype(str).values, p.acd.values)
codes, uniq = pd.factorize(cl); K = len(uniq)
idx_by = pd.Series(np.arange(len(p))).groupby(codes).apply(lambda s: s.values).values
bs = []
for b in range(200):
    pick = rng.integers(0, K, K)
    ii = np.concatenate([idx_by[j] for j in pick])
    s_ = std_rates(p.iloc[ii])
    bs.append(dict(W1B0=s_['W1'] - s_['B0'], W1D1=s_['W1'] - s_['D1'], D1B0=s_['D1'] - s_['B0']))
bs = pd.DataFrame(bs)
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = dict(window='납부 2008–2017, 결과 3년(사후 민감도)', n_by_group=raw['size'].astype(int).to_dict(), raw_rate=raw['mean'].round(5).to_dict(),
           std_rate=sr.round(5).to_dict(), W1_minus_B0=float(sr['W1'] - sr['B0']), W1_minus_B0_ci=q(bs.W1B0), W1_minus_D1=float(sr['W1'] - sr['D1']), W1_minus_D1_ci=q(bs.W1D1),
           D1_minus_B0=float(sr['D1'] - sr['B0']), D1_minus_B0_ci=q(bs.D1B0))
# 보조: 대형 권리자 상위 20 제외판
big = p[p.acd != 'NA'].acd.value_counts().head(20).index
pp = p[~p.acd.isin(big)]
out['std_rate_excl_top20'] = std_rates(pp).round(5).to_dict()
json.dump(out, open(os.path.join(W, 'verify_1', 'v1b_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
