# -*- coding: utf-8 -*-
"""
diag_composition.py
표본 구성 효과 분해 — H-A(표본편향) vs H-B(신규 8개 이질성) 판별.

기존 스크립트는 일절 수정하지 않음. 결과는 분석/fusion/diag_comp_*.csv 로 저장.
"""
import sys, os, io, json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUS = os.path.join(ROOT, '분석', 'fusion')
OUT = FUS

RNG = np.random.default_rng(20260813)

NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]


# ---------------------------------------------------------------- 평가 함수 (지정된 것 그대로)
def loco(X, y, g):
    """leave-one-cluster-out 예측 R². g는 클러스터 라벨(보통 mid)."""
    import numpy as np
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try: b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception: return np.nan
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pr)
    sse = ((y[ok]-pr[ok])**2).sum(); sst = ((y[ok]-y[ok].mean())**2).sum()
    return float(1 - sse/sst) if sst > 0 else np.nan


def prep(df):
    d = df.copy()
    d['over1'] = (d['padi_e'] > 1.0).astype(float)
    return d


def design(d, cols=('dd_prod_early', 'over1')):
    return d[list(cols)].to_numpy(float), d['dd_ship_late'].to_numpy(float), d['mid'].to_numpy()


def ols(X, y):
    Xt = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xt, y, rcond=None)
    r = y - Xt @ b
    sse = (r ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return b, r2, sse


def loco_bysplit(d, cols=('dd_prod_early', 'over1')):
    """분할연도별 LOCO 후 평균/최솟값."""
    vals = []
    for s in SPLITS:
        sub = d[d.split == s]
        X, y, g = design(sub, cols)
        vals.append(loco(X, y, g))
    v = np.array(vals, float)
    return np.nanmean(v), np.nanmin(v) if np.isfinite(v).any() else np.nan, v


def fit_bysplit(d, cols=('dd_prod_early', 'over1')):
    """분할연도별 OLS 계수/R² 평균."""
    B, R = [], []
    for s in SPLITS:
        sub = d[d.split == s]
        X, y, _ = design(sub, cols)
        if len(y) < X.shape[1] + 2:
            continue
        b, r2, _ = ols(X, y)
        B.append(b); R.append(r2)
    B = np.array(B); R = np.array(R)
    return B.mean(axis=0), B.std(axis=0), R.mean(), len(R)


# ================================================================ 데이터 로드
pan47 = prep(pd.read_csv(os.path.join(FUS, 'model_search_panel.csv'), encoding='utf-8-sig'))
pan39f = prep(pd.read_csv(os.path.join(FUS, 'model_search_panel_39ind.csv'), encoding='utf-8-sig'))

IND47 = sorted(pan47.ksic.unique())
IND39 = sorted(pan39f.ksic.unique())
assert set(IND47) - set(IND39) == set(NEW8)

pan39 = pan47[pan47.ksic.isin(IND39)].copy()   # ★ 동일 데이터원(47패널)에서 잘라낸 39
pan8 = pan47[pan47.ksic.isin(NEW8)].copy()

print('=' * 78)
print('[0] 기준선 재현 — dd_ship_late ~ dd_prod_early + over1, LOCO(mid), 분할 7개')
print('=' * 78)
rows0 = []
for nm, d in [('39개(39ind파일 원본)', pan39f), ('39개(47패널에서 부분추출)', pan39),
              ('47개(전체)', pan47)]:
    m, mn, v = loco_bysplit(d)
    rows0.append(dict(표본=nm, n산업=d.ksic.nunique(), n행=len(d),
                      LOCO평균=round(m, 4), LOCO최소=round(mn, 4),
                      분할별=' '.join(f'{x:+.3f}' for x in v)))
    print(f'  {nm:26s} n산업={d.ksic.nunique():2d} n행={len(d):3d} '
          f'평균={m:+.4f} 최소={mn:+.4f}')
pd.DataFrame(rows0).to_csv(os.path.join(OUT, 'diag_comp_baseline.csv'),
                           index=False, encoding='utf-8-sig')

# 참고: 지속성만 사양
print('  [참고] 지속성만(dd_prod_early 단독):')
for nm, d in [('39개(47패널부분추출)', pan39), ('47개', pan47)]:
    m, mn, _ = loco_bysplit(d, cols=('dd_prod_early',))
    print(f'    {nm:22s} 평균={m:+.4f} 최소={mn:+.4f}  n산업={d.ksic.nunique()}')


# ================================================================ [1] 계수 비교
print()
print('=' * 78)
print('[1] 같은 사양을 세 표본에서 — β_persist, β_over1, R², n')
print('=' * 78)
rows1 = []
for nm, d in [('(a) 기존 39', pan39), ('(b) 전체 47', pan47), ('(c) 신규 8', pan8)]:
    bm, bs, r2, k = fit_bysplit(d)
    if d.ksic.nunique() >= 20:
        lm, lmn, _ = loco_bysplit(d)
    else:
        lm, lmn = np.nan, np.nan   # n=8 은 LOCO 불가(클러스터 3개, 학습표본 부족)
    rows1.append(dict(표본=nm, n산업=d.ksic.nunique(), n행=len(d),
                      절편=round(bm[0], 3),
                      beta_persist=round(bm[1], 4), sd_persist=round(bs[1], 4),
                      beta_over1=round(bm[2], 3), sd_over1=round(bs[2], 3),
                      R2_insample=round(r2, 4),
                      LOCO평균=None if np.isnan(lm) else round(lm, 4),
                      LOCO최소=None if np.isnan(lmn) else round(lmn, 4)))
    print(f'  {nm:12s} n산업={d.ksic.nunique():2d} n행={len(d):3d} | '
          f'β_persist={bm[1]:+.4f}(sd{bs[1]:.3f})  β_over1={bm[2]:+7.3f}(sd{bs[2]:5.2f})  '
          f'R²={r2:.4f}  LOCO={"n/a" if np.isnan(lm) else f"{lm:+.4f}"}')
print('  ※ (c)는 산업 n=8 (연도별 관측 8개). 유의성 주장 없음 — 방향/크기만.')
print('  ※ 계수는 분할연도 7개 각각의 OLS 계수 평균, sd는 분할 간 표준편차(표준오차 아님).')
pd.DataFrame(rows1).to_csv(os.path.join(OUT, 'diag_comp_coef3.csv'),
                           index=False, encoding='utf-8-sig')

# 원자료 수준 기술통계
print()
print('  [기술통계] dd_ship_late / dd_prod_early / padi_e (분할연도 풀링)')
for nm, d in [('39', pan39), ('8', pan8)]:
    print(f'    {nm:>3s}: dd_ship_late 평균={d.dd_ship_late.mean():7.2f} sd={d.dd_ship_late.std():6.2f} | '
          f'dd_prod_early 평균={d.dd_prod_early.mean():7.2f} sd={d.dd_prod_early.std():6.2f} | '
          f'over1비율={d.over1.mean():.2f}')


# ================================================================ [2] Chow 검정
print()
print('=' * 78)
print('[2] Chow 검정 — 기존39 vs 신규8 계수 동일성')
print('=' * 78)


def chow_F(d, newset, cols=('dd_prod_early', 'over1')):
    """분할연도별 Chow F. 반환: (평균F, 분할별 F리스트, df1, df2)"""
    Fs = []
    for s in SPLITS:
        sub = d[d.split == s]
        X, y, _ = design(sub, cols)
        D = sub.ksic.isin(newset).to_numpy(float)
        _, _, sse_r = ols(X, y)
        Xf = np.column_stack([X, D, X * D[:, None]])
        _, _, sse_f = ols(Xf, y)
        k_r = X.shape[1] + 1
        k_f = Xf.shape[1] + 1
        q = k_f - k_r
        df2 = len(y) - k_f
        F = ((sse_r - sse_f) / q) / (sse_f / df2)
        Fs.append(F)
    return float(np.mean(Fs)), Fs, q, df2


F_obs, Fs, q, df2 = chow_F(pan47, set(NEW8))
from scipy import stats
p_par = [1 - stats.f.cdf(f, q, df2) for f in Fs]
print(f'  자유도: q={q}, df2={df2} (분할연도별 n=47)')
for s, f, p in zip(SPLITS, Fs, p_par):
    print(f'    split={s}  F={f:6.3f}  p={p:.4f}')
print(f'  평균 F = {F_obs:.4f} | 분할별 p 중앙값 = {np.median(p_par):.4f}')

# 순열검정 — 산업 단위로 8개를 무작위 재배치, 통계량 = 평균F
NPERM = 2000
inds = np.array(IND47)
perm_F = np.empty(NPERM)
for i in range(NPERM):
    fake = set(RNG.choice(inds, size=8, replace=False).tolist())
    perm_F[i], _, _, _ = chow_F(pan47, fake)
p_perm = float((perm_F >= F_obs).mean())
print(f'  순열검정({NPERM}회, 산업 8개 무작위 재배치): '
      f'평균F 관측={F_obs:.3f}, 귀무분포 중앙값={np.median(perm_F):.3f}, '
      f'95분위={np.quantile(perm_F,0.95):.3f}  → p={p_perm:.4f}')

# 계수별 상호작용 방향
print('  [그룹별 계수 — 상호작용 모형에서]')
rows2i = []
for s in SPLITS:
    sub = pan47[pan47.split == s]
    X, y, _ = design(sub)
    D = sub.ksic.isin(NEW8).to_numpy(float)
    Xf = np.column_stack([X, D, X * D[:, None]])
    b, _, _ = ols(Xf, y)
    rows2i.append(b)
rows2i = np.array(rows2i)
lab = ['절편', 'persist', 'over1', 'D', 'D×persist', 'D×over1']
for j, l in enumerate(lab):
    print(f'    {l:10s} 평균={rows2i[:,j].mean():+9.3f}  (분할간 sd={rows2i[:,j].std():.3f})')
pd.DataFrame(dict(zip(lab, rows2i.T)), index=SPLITS).to_csv(
    os.path.join(OUT, 'diag_comp_chow_coef.csv'), encoding='utf-8-sig')
pd.DataFrame(dict(split=SPLITS, F=Fs, p_param=p_par)).assign(
    F_mean=F_obs, p_perm=p_perm, nperm=NPERM, n_ind=47).to_csv(
    os.path.join(OUT, 'diag_comp_chow.csv'), index=False, encoding='utf-8-sig')


# ================================================================ [3] 하나씩 추가 궤적
print()
print('=' * 78)
print('[3] 신규 8개를 하나씩 추가하며 LOCO 궤적 (추가 순서 무작위 200회)')
print('=' * 78)

NORD = 200
curve = np.full((NORD, 9), np.nan)          # 0..8개 추가
marg = {k: [] for k in NEW8}                # 산업별 한계기여

base_sets = pan47.set_index('ksic')
for r in range(NORD):
    order = list(RNG.permutation(NEW8))
    cur = list(IND39)
    d = pan47[pan47.ksic.isin(cur)]
    prev, _, _ = loco_bysplit(d)
    curve[r, 0] = prev
    for step, k in enumerate(order, start=1):
        cur.append(k)
        d = pan47[pan47.ksic.isin(cur)]
        v, _, _ = loco_bysplit(d)
        curve[r, step] = v
        marg[k].append(v - prev)
        prev = v

cm = np.nanmean(curve, axis=0)
cs = np.nanstd(curve, axis=0)
c05 = np.nanquantile(curve, 0.05, axis=0)
c95 = np.nanquantile(curve, 0.95, axis=0)
print('  추가개수  LOCO평균   sd     5%      95%   (직전대비 Δ)')
for i in range(9):
    dlt = '' if i == 0 else f'  {cm[i]-cm[i-1]:+.4f}'
    print(f'    +{i}     {cm[i]:+.4f}  {cs[i]:.4f}  {c05[i]:+.4f} {c95[i]:+.4f}{dlt}')
pd.DataFrame(dict(n_added=range(9), n_ind=[39+i for i in range(9)],
                  loco_mean=cm, loco_sd=cs, q05=c05, q95=c95)).to_csv(
    os.path.join(OUT, 'diag_comp_curve.csv'), index=False, encoding='utf-8-sig')

print()
print('  [산업별 한계기여] — 그 산업을 넣는 순간의 LOCO 변화 평균 (모든 위치·순서 평균)')
padi_map = {26111: 1.356, 26211: 1.310, 26519: 1.159, 27199: 1.086,
            28111: 1.063, 27301: 1.040, 26121: 0.798, 26212: 0.484}
rows3 = []
for k in sorted(NEW8, key=lambda x: np.mean(marg[x])):
    a = np.array(marg[k])
    rows3.append(dict(ksic=k, padi_e=padi_map[k], over1=int(padi_map[k] > 1),
                      한계기여평균=round(a.mean(), 4), sd=round(a.std(), 4),
                      최악=round(a.min(), 4), 최선=round(a.max(), 4), n관측=len(a)))
    print(f'    {k}  PADI={padi_map[k]:.3f}  Δ평균={a.mean():+.4f} (sd={a.std():.4f}, '
          f'범위 {a.min():+.4f}~{a.max():+.4f})')
pd.DataFrame(rows3).to_csv(os.path.join(OUT, 'diag_comp_marginal.csv'),
                           index=False, encoding='utf-8-sig')

# 단일 제외(leave-one-out from 47) 확인 — 한계기여의 다른 각도
print()
print('  [교차확인] 47개에서 신규 1개만 빼면? (기준 47개 LOCO)')
base47, _, _ = loco_bysplit(pan47)
for k in NEW8:
    d = pan47[pan47.ksic != k]
    v, _, _ = loco_bysplit(d)
    print(f'    -{k} 제외 → LOCO={v:+.4f}  (기준 {base47:+.4f}, Δ={v-base47:+.4f})')


# ================================================================ [4] 39개 내부 크기 분할
print()
print('=' * 78)
print('[4] 역방향 확인 — 기존 39개를 크기로 잘라본다')
print('=' * 78)

kr = pd.read_parquet(os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_registered.parquet'),
                     columns=['세세분류코드'])
cnt = kr['세세분류코드'].astype(str).str.strip().value_counts()
cnt.index = cnt.index.astype(str)


def sizeof(k):
    return int(cnt.get(str(k), 0))


size39 = pd.Series({k: sizeof(k) for k in IND39}).sort_values()
size47 = pd.Series({k: sizeof(k) for k in IND47}).sort_values()
print('  등록특허 건수 기준 (kr_registered 세세분류코드 카운트)')
print(f'  39개 범위: {size39.min()} ~ {size39.max()}, 중앙값 {size39.median():.0f}')
print(f'  신규 8개: ' + ', '.join(f'{k}={sizeof(k)}' for k in NEW8))
print(f'  신규 8개의 39개 분포 내 백분위: ' +
      ', '.join(f'{k}={(size39 < sizeof(k)).mean()*100:.0f}%' for k in NEW8))


def report_sub(nm, keys, panel):
    d = panel[panel.ksic.isin(keys)]
    bm, bs, r2, _ = fit_bysplit(d)
    m, mn, v = loco_bysplit(d)
    nna = int(np.isnan(v).sum())
    print(f'  {nm:24s} n산업={len(keys):2d} n행={len(d):3d} | '
          f'β_persist={bm[1]:+.4f} β_over1={bm[2]:+7.3f} R²={r2:.4f} | '
          f'LOCO평균={m:+.4f} 최소={mn:+.4f} (분할중 nan {nna}개)')
    return dict(구분=nm, n산업=len(keys), n행=len(d),
                beta_persist=round(bm[1], 4), beta_over1=round(bm[2], 3),
                R2=round(r2, 4), LOCO평균=round(m, 4), LOCO최소=round(mn, 4), nan분할=nna)


rows4 = []
med = size39.median()
lo39 = [k for k in IND39 if sizeof(k) <= med]
hi39 = [k for k in IND39 if sizeof(k) > med]
print()
print('  A. 39개 → 등록특허 건수 중위 분할')
rows4.append(report_sub('39 전체', IND39, pan47))
rows4.append(report_sub('39 하위(소형)', lo39, pan47))
rows4.append(report_sub('39 상위(대형)', hi39, pan47))

# 3분위
print()
print('  B. 39개 → 3분위')
q3 = np.quantile(size39.values, [1/3, 2/3])
t1 = [k for k in IND39 if sizeof(k) <= q3[0]]
t2 = [k for k in IND39 if q3[0] < sizeof(k) <= q3[1]]
t3 = [k for k in IND39 if sizeof(k) > q3[1]]
for nm, kk in [('39 T1 최소', t1), ('39 T2 중간', t2), ('39 T3 최대', t3)]:
    rows4.append(report_sub(nm, kk, pan47))

# 47개 전체에서 크기 분할 (신규8 포함)
print()
print('  C. 47개 전체 → 중위 분할 (신규 8개 포함)')
med47 = size47.median()
lo47 = [k for k in IND47 if sizeof(k) <= med47]
hi47 = [k for k in IND47 if sizeof(k) > med47]
rows4.append(report_sub('47 하위(소형)', lo47, pan47))
rows4.append(report_sub('47 상위(대형)', hi47, pan47))
print(f'    ※ 47 상위 {len(hi47)}개 중 신규8은 {sum(1 for k in hi47 if k in NEW8)}개')

# D. 39개 대형과 신규8을 합친 '대형군' vs 39개 소형
print()
print('  D. 대형군 통합 검증')
rows4.append(report_sub('39대형 + 신규8', hi39 + NEW8, pan47))

# E. 출하액 기준(광업제조업조사) 대안 크기지표 — 특허건수와 독립인 규모 측정
import glob
fs = glob.glob(os.path.join(ROOT, '공모전데이터', '외부데이터', '산업분류별', '*.csv'))
frames = []
for f in fs:
    try:
        t = pd.read_csv(f, encoding='cp949')
    except Exception:
        continue
    t = t[t['[A]시도별'].astype(str) == '0']          # 전국만
    t = t[t['항목'].astype(str).str.startswith('출하액')]
    ycols = [c for c in t.columns if str(c).endswith('년')]
    for yc in ycols:
        frames.append(pd.DataFrame({
            '_k': t['[B]산업별'].astype(str).str.extract(r'(\d{5})')[0],
            '_y': int(str(yc)[:4]),
            '_s': pd.to_numeric(t[yc], errors='coerce')}))
mm = pd.concat(frames, ignore_index=True).dropna(subset=['_k'])
mm = mm[(mm._y >= 2010) & (mm._y <= 2019)]
ship = mm.groupby('_k')['_s'].mean()
sh39 = pd.Series({k: ship.get(str(k), np.nan) for k in IND39}).dropna()
print(f'  E. 39개 → 출하액(2010-19 평균, 백만원) 중위 분할 — 매칭 {len(sh39)}/39개')
print(f'     신규8 출하액: ' + ', '.join(
    f'{k}={ship.get(str(k), float("nan"))/1e6:.1f}조' for k in NEW8))
print(f'     39개 출하액 범위: {sh39.min()/1e6:.2f}조 ~ {sh39.max()/1e6:.1f}조, '
      f'중앙값 {sh39.median()/1e6:.2f}조')
print(f'     신규8의 39개 분포 내 백분위: ' +
      ', '.join(f'{k}={(sh39 < ship.get(str(k), np.nan)).mean()*100:.0f}%' for k in NEW8))
if len(sh39) >= 30:
    m2 = sh39.median()
    lo_s = [k for k in sh39.index if sh39[k] <= m2]
    hi_s = [k for k in sh39.index if sh39[k] > m2]
    rows4.append(report_sub('39 출하액 하위', lo_s, pan47))
    rows4.append(report_sub('39 출하액 상위', hi_s, pan47))

pd.DataFrame(rows4).to_csv(os.path.join(OUT, 'diag_comp_sizesplit.csv'),
                           index=False, encoding='utf-8-sig')

# 크기와 잔차의 관계 (39개 내부)
print()
print('  F. 39개 내부: 산업규모(log 등록건수)와 모형 절대예측오차 상관')
res_rows = []
for s in SPLITS:
    sub = pan47[(pan47.split == s) & (pan47.ksic.isin(IND39))]
    X, y, g = design(sub)
    # LOCO 예측
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    for kk, e in zip(sub.ksic.values, np.abs(y - pr)):
        res_rows.append(dict(split=s, ksic=kk, abserr=e, lsize=np.log(sizeof(kk) + 1)))
rr = pd.DataFrame(res_rows)
c = rr[['abserr', 'lsize']].corr().iloc[0, 1]
print(f'    corr(|LOCO오차|, log규모) = {c:+.3f}  (n={len(rr)} 산업×분할, 39개 산업)')
rr.groupby('ksic').agg(abserr=('abserr', 'mean'), lsize=('lsize', 'first')).to_csv(
    os.path.join(OUT, 'diag_comp_residsize.csv'), encoding='utf-8-sig')


# ================================================================ [5] 교란 분리: 규모 vs 업종
print()
print('=' * 78)
print('[5] 교란 분리 — "대형이라 약한 것"인가 "전자·기계라 약한 것"인가')
print('=' * 78)
ELEC = {26, 27, 28}          # 신규8이 속한 업종군 (전자·의료광학·전기장비)
rows5 = []
e39 = [k for k in IND39 if (k // 1000) in ELEC]
n39 = [k for k in IND39 if (k // 1000) not in ELEC]
print(f'  39개 중 전자·기계(mid 26/27/28) {len(e39)}개, 그 외 {len(n39)}개')
rows5.append(report_sub('39 전자기계(26/27/28)', e39, pan47))
rows5.append(report_sub('39 비전자기계', n39, pan47))

# 업종 고정: 비전자기계 안에서만 크기 분할 → 규모 효과만 남음
sub_n = pd.Series({k: sizeof(k) for k in n39})
mn_ = sub_n.median()
print(f'  [업종 고정] 비전자기계 {len(n39)}개 안에서만 크기 중위 분할')
rows5.append(report_sub('  비전자·소형', [k for k in n39 if sub_n[k] <= mn_], pan47))
rows5.append(report_sub('  비전자·대형', [k for k in n39 if sub_n[k] > mn_], pan47))

# 규모 고정: 대형군 안에서만 업종 분할
hi39s = set(hi39)
print(f'  [규모 고정] 39개 대형 {len(hi39)}개 안에서만 업종 분할')
rows5.append(report_sub('  대형·전자기계', [k for k in hi39 if (k // 1000) in ELEC], pan47))
rows5.append(report_sub('  대형·비전자기계', [k for k in hi39 if (k // 1000) not in ELEC], pan47))

print(f'  [규모 고정] 39개 소형 {len(lo39)}개 안에서만 업종 분할')
rows5.append(report_sub('  소형·전자기계', [k for k in lo39 if (k // 1000) in ELEC], pan47))
rows5.append(report_sub('  소형·비전자기계', [k for k in lo39 if (k // 1000) not in ELEC], pan47))
pd.DataFrame(rows5).to_csv(os.path.join(OUT, 'diag_comp_confound.csv'),
                           index=False, encoding='utf-8-sig')

# 규모를 조절변수로 넣으면 47개 LOCO가 회복되나? (H-A 처방의 직접 검증)
print()
print('  [처방 검증] 47개에 규모 조절변수를 추가하면 LOCO가 회복되는가')
p47 = pan47.copy()
p47['lsize'] = p47.ksic.map(lambda k: np.log(sizeof(k) + 1))
p47['lsize'] = (p47['lsize'] - p47['lsize'].mean()) / p47['lsize'].std()
p47['big'] = (p47.ksic.map(sizeof) > size47.median()).astype(float)
p47['px_size'] = p47.dd_prod_early * p47.lsize
p47['px_big'] = p47.dd_prod_early * p47.big
p47['elec'] = p47.ksic.map(lambda k: float((k // 1000) in ELEC))
p47['px_elec'] = p47.dd_prod_early * p47.elec
specs = [
    ('기준: persist+over1', ('dd_prod_early', 'over1')),
    ('+ log규모', ('dd_prod_early', 'over1', 'lsize')),
    ('+ log규모 + persist×log규모', ('dd_prod_early', 'over1', 'lsize', 'px_size')),
    ('+ 대형더미 + persist×대형', ('dd_prod_early', 'over1', 'big', 'px_big')),
    ('+ 전자더미 + persist×전자', ('dd_prod_early', 'over1', 'elec', 'px_elec')),
]
rows6 = []
for nm, cols in specs:
    m, mn, _ = loco_bysplit(p47, cols)
    m39, mn39, _ = loco_bysplit(p47[p47.ksic.isin(IND39)], cols)
    rows6.append(dict(사양=nm, n변수=len(cols), n산업_47=47, LOCO47평균=round(m, 4),
                      LOCO47최소=round(mn, 4), n산업_39=39, LOCO39평균=round(m39, 4)))
    print(f'    {nm:30s} [n=47] 평균={m:+.4f} 최소={mn:+.4f} | [n=39] 평균={m39:+.4f}')
print('    ※ 탐색 규모: 위 5개 사양만 시도(사전 지정). 그 외 사양 탐색 없음.')
pd.DataFrame(rows6).to_csv(os.path.join(OUT, 'diag_comp_rx.csv'),
                           index=False, encoding='utf-8-sig')


# ================================================================ [6] 등가표본 교체검정 (핵심)
print()
print('=' * 78)
print('[6] 등가표본 교체검정 — n을 39로 고정하고 "신규8" vs "기존8"을 맞바꾼다')
print('=' * 78)
print('  논리: 31개 공통기반 + 8개. 8개를 (기존39에서 뽑은 8) vs (신규8) 로 교체.')
print('        n=39 로 고정되므로 표본크기 효과가 제거된다.')
print('        차이 없음 → 신규8은 평범 (H-A: 크기층 확장 탓)')
print('        신규8이 유의하게 나쁨 → 신규8 자체가 이질 (H-B)')

NDRAW = 300
size_ser = pd.Series({k: sizeof(k) for k in IND39})
ship_ser = pd.Series({k: float(ship.get(str(k), np.nan)) for k in IND39})

res6 = {'무작위8': [], '특허건수상위8': [], '출하액상위8': []}
for i in range(NDRAW):
    base31 = list(RNG.choice(IND39, size=31, replace=False))
    rest8 = [k for k in IND39 if k not in base31]
    a, _, _ = loco_bysplit(pan47[pan47.ksic.isin(base31 + rest8)])
    b, _, _ = loco_bysplit(pan47[pan47.ksic.isin(base31 + NEW8)])
    res6['무작위8'].append((a, b))
for tag, ser in [('특허건수상위8', size_ser), ('출하액상위8', ship_ser)]:
    top8 = list(ser.sort_values(ascending=False).index[:8])
    base31 = [k for k in IND39 if k not in top8]
    a, _, _ = loco_bysplit(pan47[pan47.ksic.isin(base31 + top8)])
    b, _, _ = loco_bysplit(pan47[pan47.ksic.isin(base31 + NEW8)])
    res6[tag].append((a, b))

arr = np.array(res6['무작위8'])
d = arr[:, 1] - arr[:, 0]
print()
print(f'  A. 대조군 = 기존39에서 무작위 8개 ({NDRAW}회, 각각 n=39 고정)')
print(f'     기존8 유지: LOCO 평균={arr[:,0].mean():+.4f} (sd {arr[:,0].std():.4f})')
print(f'     신규8 교체: LOCO 평균={arr[:,1].mean():+.4f} (sd {arr[:,1].std():.4f})')
print(f'     차이 Δ 평균={d.mean():+.4f}  Δ>0 비율={float((d>0).mean()):.3f}  '
      f'[5%,95%]=[{np.quantile(d,0.05):+.4f}, {np.quantile(d,0.95):+.4f}]')
for tag in ['특허건수상위8', '출하액상위8']:
    a, b = res6[tag][0]
    top8 = list((size_ser if tag.startswith('특허') else ship_ser)
                .sort_values(ascending=False).index[:8])
    print(f'  B. 대조군 = 기존39의 {tag} {sorted(top8)} (n=39 고정)')
    print(f'     기존8 유지: LOCO={a:+.4f} | 신규8 교체: LOCO={b:+.4f} | Δ={b-a:+.4f}')
pd.DataFrame(dict(loco_keep=arr[:, 0], loco_swap=arr[:, 1], delta=d)).to_csv(
    os.path.join(OUT, 'diag_comp_swap.csv'), index=False, encoding='utf-8-sig')

# 산업별 예측오차 — 신규8 vs 크기매칭 기존군
print()
print('  C. 47개 모형에서 산업별 LOCO 예측오차 (RMSE) — 규모 층별')
err = {}
for s in SPLITS:
    sub = pan47[pan47.split == s]
    X, y, g = design(sub)
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        bb = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ bb
    for kk, e in zip(sub.ksic.values, (y - pr) ** 2):
        err.setdefault(kk, []).append(e)
rmse = pd.Series({k: np.sqrt(np.mean(v)) for k, v in err.items()})
tab = pd.DataFrame(dict(ksic=rmse.index, rmse=rmse.values))
tab['신규'] = tab.ksic.isin(NEW8)
tab['특허건수'] = tab.ksic.map(sizeof)
tab['출하액조'] = tab.ksic.map(lambda k: float(ship.get(str(k), np.nan)) / 1e6)
tab = tab.sort_values('rmse', ascending=False)
print('     RMSE 상위 12개 산업:')
for _, r in tab.head(12).iterrows():
    print(f'       {int(r.ksic)} RMSE={r.rmse:6.2f}  특허={int(r.특허건수):6d}  '
          f'출하={r.출하액조:6.1f}조  {"[신규]" if r.신규 else ""}')
g39top = [k for k in IND39 if sizeof(k) > size39.median()]
print(f'     그룹 평균 RMSE: 신규8={tab[tab.신규].rmse.mean():.2f} | '
      f'기존39대형({len(g39top)})={tab[tab.ksic.isin(g39top)].rmse.mean():.2f} | '
      f'기존39소형({39-len(g39top)})={tab[tab.ksic.isin([k for k in IND39 if k not in g39top])].rmse.mean():.2f}')
tab.to_csv(os.path.join(OUT, 'diag_comp_rmse_by_ind.csv'), index=False, encoding='utf-8-sig')

# 신규8 내부: 한계기여 vs 규모 방향 (n=8, 유의성 주장 없음)
mg = pd.Series({k: np.mean(marg[k]) for k in NEW8})
print()
print('  D. 신규8 내부: 한계기여와 규모의 방향 (n=8 — 유의성 주장 없음, 부호만)')
print(f'     corr(한계기여, log 특허건수) = {np.corrcoef(mg.values, np.log([sizeof(k) for k in NEW8]))[0,1]:+.3f}')
sv = np.array([float(ship.get(str(k), np.nan)) for k in NEW8])
print(f'     corr(한계기여, log 출하액)   = {np.corrcoef(mg.values, np.log(sv))[0,1]:+.3f}')
print('     → H-A 예측은 "클수록 한계기여가 나쁘다"(음의 상관). 실제 부호를 확인.')


# ---- [6B'] 크기매칭 교체검정 분포 (핵심 판정 근거)
print()
print("  B'. 크기매칭 교체검정 — 대조 8개를 '기존39의 대형 절반'에서만 추출")
for tag, ser in [('특허건수', size_ser), ('출하액', ship_ser)]:
    pool = list(ser.sort_values(ascending=False).index[:19])   # 기존39의 대형 19개
    ds = []
    for i in range(NDRAW):
        c8 = list(RNG.choice(pool, size=8, replace=False))
        base31 = [k for k in IND39 if k not in c8]
        a, _, _ = loco_bysplit(pan47[pan47.ksic.isin(base31 + c8)])
        b, _, _ = loco_bysplit(pan47[pan47.ksic.isin(base31 + NEW8)])
        ds.append(b - a)
    ds = np.array(ds)
    print(f'     [{tag} 기준 대형19에서 추출, {NDRAW}회, n=39 고정] '
          f'Δ(신규8-대조8) 평균={ds.mean():+.4f}  Δ>0비율={float((ds>0).mean()):.3f}  '
          f'[5%,95%]=[{np.quantile(ds,0.05):+.4f}, {np.quantile(ds,0.95):+.4f}]')
print('     ※ Δ≈0 이면 신규8은 "그냥 대형 산업"과 구별 불가 → H-A')
print('     ※ Δ<<0 이면 대형이라는 것만으로 설명 안 됨 → H-B 잔여분 존재')

# ---- [7] 데이터 품질 점검: 26519 / 26529 이상 유사성
print()
print('=' * 78)
print('[7] 데이터 점검 — 최악 예측오차 산업 26519(신규) / 26529(기존)')
print('=' * 78)
chk = pan47[pan47.ksic.isin([26519, 26529])][
    ['ksic', 'split', 'padi_e', 'dd_prod_early', 'dd_ship_late', 'O', 'R']]
print(chk.to_string(index=False))
a = pan47[pan47.ksic == 26519].sort_values('split')
b = pan47[pan47.ksic == 26529].sort_values('split')
for c in ['dd_prod_early', 'dd_ship_late', 'padi_e']:
    print(f'  {c}: 두 산업 값이 전 분할에서 동일한가 = '
          f'{np.allclose(a[c].values, b[c].values)}')
print('  ※ 두 산업이 동일 시계열이면 KSIC 매핑 중복 가능성 — 별도 확인 필요.')


# ================================================================ [8] 종속변수 중복 진단
print()
print('=' * 78)
print('[8] 결정적 데이터 진단 — 종속변수(출하/생산지수)가 5자리 수준에서 중복')
print('=' * 78)
w = pan47.pivot(index='ksic', columns='split', values='dd_ship_late')
w2 = pan47.pivot(index='ksic', columns='split', values='dd_prod_early')
skey = (w.round(6).astype(str).agg('|'.join, axis=1) + '##' +
        w2.round(6).astype(str).agg('|'.join, axis=1))
pan47['skey'] = pan47.ksic.map(skey)
grp = {k: sorted(v) for k, v in skey.groupby(skey).groups.items()}
dups = [g for g in grp.values() if len(g) > 1]
print(f'  (y, persist) 쌍이 완전히 동일한 산업 그룹: {len(dups)}개, '
      f'관련 산업 {sum(len(g) for g in dups)}/47개')
u39 = pan47[pan47.ksic.isin(IND39)].skey.nunique()
u47 = pan47.skey.nunique()
print(f'  고유 시계열 수: 39개 표본 → {u39}개 | 47개 표본 → {u47}개')
print(f'  ★ 신규 8개가 실제로 추가한 고유 시계열은 {u47-u39}개뿐 '
      f'(나머지 {8-(u47-u39)}개는 기존 산업과 동일한 y·persist)')
rows8 = []
for k in NEW8:
    g = grp[skey[k]]
    old = [x for x in g if x in IND39]
    rows8.append(dict(ksic=k, 그룹크기=len(g), 그룹=';'.join(map(str, g)),
                      기존39내_동일계열=';'.join(map(str, old)) if old else '없음',
                      신규고유=len(old) == 0,
                      한계기여=round(float(np.mean(marg[k])), 4)))
    print(f'    {k}: 기존39에 동일계열 {old if old else "없음(고유)"} '
          f'| 한계기여={np.mean(marg[k]):+.4f}')
pd.DataFrame(rows8).to_csv(os.path.join(OUT, 'diag_comp_dupdv.csv'),
                           index=False, encoding='utf-8-sig')
mixed = 0
for s in SPLITS:
    sub = pan47[pan47.split == s]
    gg = sub.groupby('skey').over1.nunique()
    mixed += int((gg > 1).sum())
print(f'  같은 y·persist 그룹 안에서 over1 깃발이 엇갈리는 사례: '
      f'분할 7개 합계 {mixed}건 (그룹 {u47}개 기준)')
print('  → 39→47 확장의 상당 부분은 정보 추가가 아니라 기존 관측의 가중치 복제.')

print()
print('=' * 78)
print('저장: diag_comp_baseline / coef3 / chow / chow_coef / curve / marginal / '
      'sizesplit / residsize / confound / rx / swap / rmse_by_ind / dupdv .csv')
print('=' * 78)
