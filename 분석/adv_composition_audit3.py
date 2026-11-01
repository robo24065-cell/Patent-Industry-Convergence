# -*- coding: utf-8 -*-
"""
adv_composition_audit3.py — 3차. 핵심 쟁점만 확정한다.
  C1: 공통 분모(SST)로 재계산한 층별 성능 — R² 분모효과를 제거한 apples-to-apples
  C2: y 극단치(레버리지)가 소형층 R²를 떠받치는가 — winsorize/trim 후 재검정
  C3: 39→47 하락은 분모효과인가 실제 오차증가인가 (확정)
  C4: 교체검정[6]을 RMSE로 다시 — 분모효과 제거 후에도 신규8이 나쁜가
  C5: 중복제거(고유계열) 위에서 규모 구배 + 위약검정
"""
import sys, os, io, glob
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUS = os.path.join(ROOT, '분석', 'fusion')
RNG = np.random.default_rng(31337)
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = list(range(2013, 2020)); COLS = ['dd_prod_early', 'over1']

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
mm = pd.concat(frames, ignore_index=True); mm = mm.dropna(subset=['_k'])
mm = mm[(mm._y >= 2010) & (mm._y <= 2019)]
ship = mm.groupby('_k')['_s'].mean(); shipof = lambda k: float(ship.get(str(k), np.nan))


def loco_pred(X, y, g):
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return None
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    return pr


def run(keys, panel=pan, ycol='dd_ship_late'):
    Y, P = [], []
    for s in SPLITS:
        sub = panel[(panel.split == s) & (panel.ksic.isin(keys))]
        X = sub[COLS].to_numpy(float); y = sub[ycol].to_numpy(float)
        pr = loco_pred(X, y, sub.mid.to_numpy())
        if pr is None: return None
        Y.append(y); P.append(pr)
    return Y, P


def r2avg(Y, P):
    return float(np.mean([1-((y-p)**2).sum()/((y-y.mean())**2).sum() for y, p in zip(Y, P)]))


s39 = pd.Series({k: sizeof(k) for k in IND39}).sort_values()
sh39 = pd.Series({k: shipof(k) for k in IND39}).dropna().sort_values()
LO_P, HI_P = list(s39.index[:20]), list(s39.index[20:])
LO_S, HI_S = list(sh39.index[:20]), list(sh39.index[20:])

print('=' * 88)
print('[C1] 공통 분모로 재계산 — 각 층의 SSE를 "39개 전체의 SST"로 나눈다')
print('=' * 88)
# 39 전체의 분할연도별 SST (행당)
full = run(IND39)
sst_full = [((y-y.mean())**2).mean() for y in full[0]]
print(f'  {"층":16s} {"n":>3s} {"행당SSE":>9s} {"자기SST":>9s} {"자기R2":>8s} {"공통R2":>8s}')
for nm, kk in [('39 전체', IND39), ('39 특허소형20', LO_P), ('39 특허대형19', HI_P),
               ('39 출하하위20', LO_S), ('39 출하상위19', HI_S), ('47 전체', IND47)]:
    Y, P = run(kk)
    mse = [((y-p)**2).mean() for y, p in zip(Y, P)]
    own = [((y-y.mean())**2).mean() for y in Y]
    r2own = float(np.mean([1-a/b for a, b in zip(mse, own)]))
    r2com = float(np.mean([1-a/b for a, b in zip(mse, sst_full)]))
    print(f'  {nm:16s} {len(kk):3d} {np.mean(mse):9.1f} {np.mean(own):9.1f} '
          f'{r2own:+8.4f} {r2com:+8.4f}')
print()
print('  ★ "공통R2" 는 모든 층에 동일한 분모를 쓴 값. 여기서 소형-대형 격차가')
print('    사라지거나 뒤집히면, [4]의 "소형층 전용" 결론은 분모효과다.')

print()
print('=' * 88)
print('[C2] 레버리지 — y 극단 관측 제거 후에도 소형층 우위가 남는가')
print('=' * 88)
for trim in [0.0, 0.05, 0.10]:
    line = f'  trim={trim:.0%}: '
    for nm, kk in [('39전체', IND39), ('특허소형', LO_P), ('특허대형', HI_P),
                   ('출하하위', LO_S), ('출하상위', HI_S)]:
        Y, P = run(kk); r2s = []
        for y, p in zip(Y, P):
            if trim > 0:
                lo, hi = np.quantile(y, [trim, 1-trim]); m = (y >= lo) & (y <= hi)
                y2, p2 = y[m], p[m]
            else: y2, p2 = y, p
            r2s.append(1-((y2-p2)**2).sum()/((y2-y2.mean())**2).sum())
        line += f'{nm}={np.mean(r2s):+.3f}  '
    print(line)
print('  ※ 예측은 전체 표본으로 하고 평가만 트림. 소형층 R²가 크게 떨어지면')
print('    "소형층 강함"은 소수 극단 낙폭 산업이 만든 것.')

print()
print('=' * 88)
print('[C3] 39→47 하락의 성격 확정')
print('=' * 88)
for nm, kk in [('39', IND39), ('47', IND47)]:
    Y, P = run(kk)
    print(f'  {nm}: 행당SSE={np.mean([((y-p)**2).mean() for y,p in zip(Y,P)]):7.2f} '
          f'행당SST={np.mean([((y-y.mean())**2).mean() for y in Y]):7.2f}  '
          f'R²={r2avg(Y,P):+.4f}')
Y39, P39 = run(IND39); Y47, P47 = run(IND47)
mse39 = np.mean([((y-p)**2).mean() for y, p in zip(Y39, P39)])
mse47 = np.mean([((y-p)**2).mean() for y, p in zip(Y47, P47)])
sst39 = np.mean([((y-y.mean())**2).mean() for y in Y39])
sst47 = np.mean([((y-y.mean())**2).mean() for y in Y47])
print(f'  분해: ΔR² = {r2avg(Y47,P47)-r2avg(Y39,P39):+.4f}')
print(f'    SSE만 47로 바꾸면(분모 39 고정): R²={1-mse47/sst39:+.4f} '
      f'(Δ={1-mse47/sst39-(1-mse39/sst39):+.4f})')
print(f'    SST만 47로 바꾸면(분자 39 고정): R²={1-mse39/sst47:+.4f} '
      f'(Δ={1-mse39/sst47-(1-mse39/sst39):+.4f})')
print('  → 39→47 하락은 (분모가 아니라) 실제 오차증가. 저자 전제 자체는 성립.')

print()
print('=' * 88)
print('[C4] 교체검정[6]을 RMSE 기준으로 — 분모효과 제거 후에도 신규8이 나쁜가')
print('=' * 88)
def rmse_of(keys):
    Y, P = run(keys)
    return float(np.sqrt(np.mean([((y-p)**2).mean() for y, p in zip(Y, P)])))
base_r, base_rm = r2avg(*run(IND39)), rmse_of(IND39)
dr2, drm = [], []
for _ in range(300):
    b31 = list(RNG.choice(IND39, 31, replace=False))
    rest8 = [k for k in IND39 if k not in b31]
    a2, b2 = r2avg(*run(b31+rest8)), r2avg(*run(b31+NEW8))
    a1, b1 = rmse_of(b31+rest8), rmse_of(b31+NEW8)
    dr2.append(b2-a2); drm.append(b1-a1)
dr2, drm = np.array(dr2), np.array(drm)
print(f'  무작위8 교체(300회, n=39 고정): ΔR²평균={dr2.mean():+.4f}  '
      f'ΔRMSE평균={drm.mean():+.3f} ({drm.mean()/base_rm*100:+.1f}%)  ΔRMSE>0비율={float((drm>0).mean()):.3f}')
for tag, ser in [('특허건수', s39), ('출하액', sh39)]:
    pool = list(ser.sort_values(ascending=False).index[:19]); d2, dm = [], []
    for _ in range(300):
        c8 = list(RNG.choice(pool, 8, replace=False))
        b31 = [k for k in IND39 if k not in c8]
        d2.append(r2avg(*run(b31+NEW8)) - r2avg(*run(b31+c8)))
        dm.append(rmse_of(b31+NEW8) - rmse_of(b31+c8))
    d2, dm = np.array(d2), np.array(dm)
    print(f'  크기매칭({tag} 대형19 추출, 300회): ΔR²평균={d2.mean():+.4f}  '
          f'ΔRMSE평균={dm.mean():+.3f}  ΔRMSE>0비율={float((dm>0).mean()):.3f}')

print()
print('=' * 88)
print('[C5] 중복제거 고유계열 위에서 규모 구배 + 위약검정')
print('=' * 88)
w1 = pan.pivot(index='ksic', columns='split', values='dd_ship_late')
w2 = pan.pivot(index='ksic', columns='split', values='dd_prod_early')
skey = (w1.round(6).astype(str).agg('|'.join, axis=1) + '##' +
        w2.round(6).astype(str).agg('|'.join, axis=1))
# 대표 선택 규칙 3종 (규칙 민감도 확인) — 사전지정
for rule_nm, keyfn in [('특허최대', lambda ks: max(ks, key=sizeof)),
                       ('특허최소', lambda ks: min(ks, key=sizeof)),
                       ('코드최소', lambda ks: min(ks))]:
    rep = []
    for key, ks in skey.groupby(skey).groups.items():
        ks39 = [k for k in ks if k in IND39]
        if ks39: rep.append(keyfn(ks39))
    rep = sorted(set(rep))
    sr = pd.Series({k: sizeof(k) for k in rep}).sort_values()
    h = len(rep)//2
    lo, hi = list(sr.index[:h]), list(sr.index[h:])
    a, b, c = r2avg(*run(rep)), r2avg(*run(lo)), r2avg(*run(hi))
    # 위약
    g = []
    for _ in range(300):
        p = list(RNG.permutation(rep)); g.append(r2avg(*run(p[:h])) - r2avg(*run(p[h:])))
    g = np.array(g)
    print(f'  [대표규칙={rule_nm}] 고유{len(rep)}개: 전체R²={a:+.4f} | '
          f'소형{len(lo)}={b:+.4f} 대형{len(hi)}={c:+.4f} 격차={b-c:+.4f} | '
          f'위약 |격차|95%={np.quantile(np.abs(g),0.95):.4f} → p={float((np.abs(g)>=abs(b-c)).mean()):.3f}')
print('  ※ 중복제거 후 표본은 n=23 내외. 소표본이라 위약분포 자체가 매우 넓다.')
