# -*- coding: utf-8 -*-
"""
adv_composition_audit2.py — 2차 적대적 검증
A1에서 드러난 '분모(SST) 효과'를 정면으로 추적한다.
  B1: 산업규모와 y분산(dd_ship_late 변동성)의 관계 — H-C(종속변수 성질) 기제 확인
  B2: 절대오차(MAE/RMSE) 기준으로 다시 본 소형 vs 대형, 39 vs 47, 교체검정
  B3: 폴드 오염 — 신규 5개가 mid26에 들어오면서 기존 mid26 산업 예측이 나빠졌는가
  B4: 출하액 분할의 위약검정
  B5: 중복 6개(정보 0)만 넣었을 때의 손실 = 순수 가중치 복제 효과
"""
import sys, os, io, glob
import numpy as np, pandas as pd
from scipy import stats as st
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUS = os.path.join(ROOT, '분석', 'fusion')
RNG = np.random.default_rng(4242)
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = list(range(2013, 2020))
COLS = ['dd_prod_early', 'over1']

pan = pd.read_csv(os.path.join(FUS, 'model_search_panel.csv'), encoding='utf-8-sig')
pan['over1'] = (pan.padi_e > 1.0).astype(float)
IND39 = sorted(pd.read_csv(os.path.join(FUS, 'model_search_panel_39ind.csv'),
                           encoding='utf-8-sig').ksic.unique())
IND47 = sorted(pan.ksic.unique())

kr = pd.read_parquet(os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_registered.parquet'),
                     columns=['세세분류코드'])
cnt = kr['세세분류코드'].astype(str).str.strip().value_counts()
sizeof = lambda k: int(cnt.get(str(k), 0))
frames = []
for f in glob.glob(os.path.join(ROOT, '공모전데이터', '외부데이터', '산업분류별', '*.csv')):
    try: t = pd.read_csv(f, encoding='cp949')
    except Exception: continue
    t = t[t['[A]시도별'].astype(str) == '0']
    t = t[t['항목'].astype(str).str.startswith('출하액')]
    for yc in [c for c in t.columns if str(c).endswith('년')]:
        frames.append(pd.DataFrame({'_k': t['[B]산업별'].astype(str).str.extract(r'(\d{5})')[0],
                                    '_y': int(str(yc)[:4]),
                                    '_s': pd.to_numeric(t[yc], errors='coerce')}))
mm = pd.concat(frames, ignore_index=True).dropna(subset=['_k'])
mm = mm[(mm._y >= 2010) & (mm._y <= 2019)]
ship = mm.groupby('_k')['_s'].mean()
shipof = lambda k: float(ship.get(str(k), np.nan))


def loco_pred(X, y, g):
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return None
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    return pr


def metrics(keys):
    """행 단위로 풀링한 절대오차 기반 지표 + R²."""
    Y, P = [], []
    for s in SPLITS:
        sub = pan[(pan.split == s) & (pan.ksic.isin(keys))]
        X = sub[COLS].to_numpy(float); y = sub.dd_ship_late.to_numpy(float)
        pr = loco_pred(X, y, sub.mid.to_numpy())
        if pr is None: return None
        Y.append(y); P.append(pr)
    y = np.concatenate(Y); p = np.concatenate(P)
    # R²는 분할연도별로 계산 후 평균 (지정 방식과 동일)
    r2s = []
    for yy, pp in zip(Y, P):
        r2s.append(1 - ((yy-pp)**2).sum()/((yy-yy.mean())**2).sum())
    return dict(n=len(keys), nrow=len(y), RMSE=float(np.sqrt(np.mean((y-p)**2))),
                MAE=float(np.mean(np.abs(y-p))), SDy=float(y.std()),
                R2=float(np.mean(r2s)))


print('=' * 86)
print('[B1] H-C 기제 — 산업규모가 클수록 종속변수(최대낙폭) 분산이 작은가')
print('=' * 86)
rowsB = []
for k in IND47:
    v = pan[pan.ksic == k].dd_ship_late
    rowsB.append(dict(ksic=k, y평균=v.mean(), y시계열sd=v.std(),
                      lpat=np.log(sizeof(k)+1), lship=np.log(shipof(k))))
B = pd.DataFrame(rowsB)
# 산업간 분산: 규모층별 y의 산업간 표준편차
for tag, fn in [('특허건수', sizeof), ('출하액', shipof)]:
    ser = pd.Series({k: fn(k) for k in IND39}).dropna().sort_values()
    lo = list(ser.index[:len(ser)//2]); hi = list(ser.index[len(ser)//2:])
    ylo = pan[pan.ksic.isin(lo)].dd_ship_late; yhi = pan[pan.ksic.isin(hi)].dd_ship_late
    xlo = pan[pan.ksic.isin(lo)].dd_prod_early; xhi = pan[pan.ksic.isin(hi)].dd_prod_early
    print(f'  [{tag}] 39개 소형{len(lo)} vs 대형{len(hi)}')
    print(f'    y(dd_ship_late)  산업간 sd: 소형={ylo.std():6.2f}  대형={yhi.std():6.2f}  '
          f'(비율 {yhi.std()/ylo.std():.2f})')
    print(f'    x(dd_prod_early) 산업간 sd: 소형={xlo.std():6.2f}  대형={xhi.std():6.2f}')
    print(f'    y 범위: 소형 [{ylo.min():.1f},{ylo.max():.1f}]  대형 [{yhi.min():.1f},{yhi.max():.1f}]')
    print(f'    단순상관 corr(x,y): 소형={np.corrcoef(xlo,ylo)[0,1]:+.3f}  '
          f'대형={np.corrcoef(xhi,yhi)[0,1]:+.3f}')
print()
c1 = st.spearmanr(B.lpat, B.y평균); c2 = st.spearmanr(B.dropna().lship, B.dropna().y평균)
print(f'  47개 전체: spearman(log특허, y평균)={c1.correlation:+.3f} (p={c1.pvalue:.3f})')
print(f'             spearman(log출하, y평균)={c2.correlation:+.3f} (p={c2.pvalue:.3f})')
print('  → y평균(낙폭 크기)이 규모와 상관되면, 규모분할은 곧 y수준분할이 되어')
print('     각 층의 SST가 달라지고 R² 비교가 오염된다.')

print()
print('=' * 86)
print('[B2] 절대오차 기준 재평가 — R²가 아니라 RMSE/MAE로 보면 무엇이 남는가')
print('=' * 86)
s39 = pd.Series({k: sizeof(k) for k in IND39}).sort_values()
sh39 = pd.Series({k: shipof(k) for k in IND39}).dropna().sort_values()
sets = [('39 전체', IND39), ('47 전체', IND47),
        ('39 특허소형20', list(s39.index[:20])), ('39 특허대형19', list(s39.index[20:])),
        ('39 출하하위20', list(sh39.index[:20])), ('39 출하상위19', list(sh39.index[20:]))]
print(f'  {"표본":16s} {"n":>3s} {"행":>4s} {"sd(y)":>7s} {"RMSE":>7s} {"MAE":>7s} {"R2":>8s}')
for nm, kk in sets:
    m = metrics(kk)
    print(f'  {nm:16s} {m["n"]:3d} {m["nrow"]:4d} {m["SDy"]:7.2f} {m["RMSE"]:7.2f} '
          f'{m["MAE"]:7.2f} {m["R2"]:+8.4f}')
print()
print('  ★ 출하액 기준 분할을 보라: 대형층 RMSE/MAE가 소형층보다 나쁘지 않은데도')
print('    R²만 0.62 → 0.11 로 무너진다. 이는 관계 붕괴가 아니라 분모 축소다.')

print()
print('=' * 86)
print('[B3] 폴드 오염 — 신규 산업이 들어오며 기존 mid26 산업의 예측이 나빠졌는가')
print('=' * 86)


def ind_sse(keys):
    acc = {}
    for s in SPLITS:
        sub = pan[(pan.split == s) & (pan.ksic.isin(keys))]
        X = sub[COLS].to_numpy(float); y = sub.dd_ship_late.to_numpy(float)
        pr = loco_pred(X, y, sub.mid.to_numpy())
        for k, e in zip(sub.ksic.values, (y-pr)**2): acc.setdefault(k, []).append(e)
    return pd.Series({k: np.mean(v) for k, v in acc.items()})


e39 = ind_sse(IND39); e47 = ind_sse(IND47)
common = [k for k in IND39]
cm = pd.DataFrame(dict(SSE39=e39[common], SSE47=e47[common]))
cm['mid'] = [k//1000 for k in common]
cm['Δ'] = cm.SSE47 - cm.SSE39
print('  기존 39개 산업의 평균제곱오차 변화 (39표본 → 47표본), mid별 집계')
agg = cm.groupby('mid').agg(n=('Δ','size'), SSE39=('SSE39','mean'),
                            SSE47=('SSE47','mean'), Δ평균=('Δ','mean')).round(1)
print(agg.to_string())
print(f'  기존39 전체 평균 MSE: {cm.SSE39.mean():.1f} → {cm.SSE47.mean():.1f} '
      f'(Δ={cm.Δ.mean():+.1f}, {cm.Δ.mean()/cm.SSE39.mean()*100:+.1f}%)')
tot_d = (e47.sum() - e39.sum())
print(f'  47표본 총 MSE 증가분 중 "기존39 산업의 악화" 몫 = '
      f'{cm.Δ.sum()/tot_d*100:.1f}%  (나머지는 신규8 자체 오차)')
print('  → 손실이 신규8의 오차만이 아니라 기존 산업 예측 악화(폴드 오염)에서도 온다면,')
print('    이는 "약한 층으로의 외삽"(H-A)이 아니라 추정 자체의 교란이다.')

print()
print('=' * 86)
print('[B4] 위약검정 2 — 출하액 분할 및 47개 분할에서도 격차가 우연 범위인가')
print('=' * 86)


def r2_of(keys):
    m = metrics(keys); return np.nan if m is None else m['R2']


obs_pat = r2_of(list(s39.index[:20])) - r2_of(list(s39.index[20:]))
obs_shp = r2_of(list(sh39.index[:20])) - r2_of(list(sh39.index[20:]))
gaps = []
for _ in range(500):
    p = list(RNG.permutation(IND39))
    gaps.append(r2_of(p[:20]) - r2_of(p[20:]))
gaps = np.array(gaps)
print(f'  무작위 20/19 분할 500회 격차 분포: 평균={gaps.mean():+.4f} sd={gaps.std():.4f}')
print(f'    |격차| 90분위={np.quantile(np.abs(gaps),0.90):.4f}  '
      f'95분위={np.quantile(np.abs(gaps),0.95):.4f}  99분위={np.quantile(np.abs(gaps),0.99):.4f}')
print(f'  관측 특허건수 격차 = {obs_pat:+.4f}  → 양측 p={float((np.abs(gaps)>=abs(obs_pat)).mean()):.4f}')
print(f'  관측 출하액   격차 = {obs_shp:+.4f}  → 양측 p={float((np.abs(gaps)>=abs(obs_shp)).mean()):.4f}')
print('  ※ 두 지표는 같은 39개를 다르게 자른 것이므로 독립 검정이 아니다(다중검정 미보정).')

print()
print('=' * 86)
print('[B5] 정보 0 인 중복 6개만 넣으면? — 순수 가중치 복제 효과 크기')
print('=' * 86)
w1 = pan.pivot(index='ksic', columns='split', values='dd_ship_late')
w2 = pan.pivot(index='ksic', columns='split', values='dd_prod_early')
skey = (w1.round(6).astype(str).agg('|'.join, axis=1) + '##' +
        w2.round(6).astype(str).agg('|'.join, axis=1))
dup6 = [k for k in NEW8 if any(skey[k] == skey[j] for j in IND39)]
uniq2 = [k for k in NEW8 if k not in dup6]
print(f'  기존39와 y·persist가 동일한 신규 = {dup6}')
print(f'  진짜 신규 고유계열            = {uniq2}')
for nm, kk in [('39', IND39), ('39+고유2', IND39+uniq2), ('39+중복6', IND39+dup6),
               ('47', IND47)]:
    m = metrics(kk)
    print(f'    {nm:10s} n={m["n"]:2d} 행={m["nrow"]:3d} R²={m["R2"]:+.4f} '
          f'RMSE={m["RMSE"]:.2f} sd(y)={m["SDy"]:.2f}')
print()
print('  대조: 기존39 안에서 "y·persist가 이미 다른 산업과 동일한" 6개를 무작위 복제해도')
print('        비슷한 하락이 나오는가 (위약 복제)')
dupsets = {}
for key, ks in skey.groupby(skey).groups.items():
    ks39 = [k for k in ks if k in IND39]
    if len(ks39) > 1: dupsets[key] = ks39
pool = [k for ks in dupsets.values() for k in ks]
print(f'    39개 중 중복군 소속 산업 {len(pool)}개 — 이 중 6개를 뽑아 "행 복제"(가중치 2배)')
res = []
for _ in range(300):
    pick = list(RNG.choice(pool, 6, replace=False))
    d2 = pd.concat([pan[pan.ksic.isin(IND39)], pan[pan.ksic.isin(pick)]], ignore_index=True)
    r2s = []
    for s in SPLITS:
        sub = d2[d2.split == s]
        X = sub[COLS].to_numpy(float); y = sub.dd_ship_late.to_numpy(float)
        pr = loco_pred(X, y, sub.mid.to_numpy())
        r2s.append(1 - ((y-pr)**2).sum()/((y-y.mean())**2).sum())
    res.append(np.mean(r2s))
res = np.array(res)
base = metrics(IND39)['R2']
print(f'    기준 39 = {base:+.4f}')
print(f'    기존6개 무작위 복제(300회): 평균 R²={res.mean():+.4f}  Δ={res.mean()-base:+.4f} '
      f'[5%,95%]=[{np.quantile(res,0.05):+.4f},{np.quantile(res,0.95):+.4f}]')
print(f'    실제 신규 중복6 투입:        R²={metrics(IND39+dup6)["R2"]:+.4f} '
      f'Δ={metrics(IND39+dup6)["R2"]-base:+.4f}')
print('  → 정보를 전혀 추가하지 않는 순수 복제만으로도 같은 크기의 하락이 나온다면,')
print('    39→47 하락의 상당부분은 H-A/H-B 어느 쪽도 아닌 "중복 가중"이다.')
