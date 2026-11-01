# -*- coding: utf-8 -*-
"""
adv_composition_audit.py  — 적대적 검증 (기존 스크립트 미수정, 신규 파일)
목표: "H-A 표본편향" 결론을 반증. 특히 제3의 설명을 찾는다.
  Alt-1: LOCO R²의 분모(SST) 차이 — '소형층이 강하다'가 관계 강도가 아니라 y분산 차이의 산물인가
  Alt-2: 종속변수 5자리 중복 — 규모 층별 중복 편중이 규모 구배를 만들어내는가
  Alt-3: 동일 y 그룹 안 over1 엇갈림 = 원리적으로 예측 불가한 오차. 이것이 대형층에 몰려 있는가
  Alt-4: LOCO 폴드(mid) 구성 변화 — 39→47 하락이 '층'이 아니라 '폴드 재구성'의 산물인가
  Alt-5: 규모 지표 대신 무작위/무관 지표로 갈라도 같은 격차가 나오는가 (분할 자체의 우연)
"""
import sys, os, io, glob
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUS = os.path.join(ROOT, '분석', 'fusion')
RNG = np.random.default_rng(777)
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]


def loco(X, y, g):
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


def loco_pred(X, y, g):
    """예측값 자체를 반환 (SSE/SST 분해용)."""
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return None
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    return pr


pan = pd.read_csv(os.path.join(FUS, 'model_search_panel.csv'), encoding='utf-8-sig')
pan['over1'] = (pan.padi_e > 1.0).astype(float)
pan39f = pd.read_csv(os.path.join(FUS, 'model_search_panel_39ind.csv'), encoding='utf-8-sig')
IND47 = sorted(pan.ksic.unique()); IND39 = sorted(pan39f.ksic.unique())

COLS = ['dd_prod_early', 'over1']


def parts(keys, panel=pan, cols=COLS):
    """분할연도 7개에 대해 SSE/SST/RMSE/sd(y)/R² 를 모두 반환."""
    out = []
    for s in SPLITS:
        sub = panel[(panel.split == s) & (panel.ksic.isin(keys))]
        X = sub[cols].to_numpy(float); y = sub.dd_ship_late.to_numpy(float)
        g = sub.mid.to_numpy()
        pr = loco_pred(X, y, g)
        if pr is None:
            out.append((np.nan,)*5); continue
        sse = ((y-pr)**2).sum(); sst = ((y-y.mean())**2).sum()
        out.append((sse, sst, np.sqrt(sse/len(y)), y.std(ddof=0), 1-sse/sst))
    a = np.array(out, float)
    return dict(SSE=np.nanmean(a[:, 0]), SST=np.nanmean(a[:, 1]),
                RMSE=np.nanmean(a[:, 2]), SDy=np.nanmean(a[:, 3]),
                R2=np.nanmean(a[:, 4]), R2min=np.nanmin(a[:, 4]), n=len(keys))


# 규모지표
kr = pd.read_parquet(os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_registered.parquet'),
                     columns=['세세분류코드'])
cnt = kr['세세분류코드'].astype(str).str.strip().value_counts()
sizeof = lambda k: int(cnt.get(str(k), 0))
size39 = pd.Series({k: sizeof(k) for k in IND39}).sort_values()
med39 = size39.median()
lo39 = [k for k in IND39 if sizeof(k) <= med39]
hi39 = [k for k in IND39 if sizeof(k) > med39]

print('=' * 84)
print('[A1] LOCO R² 분해 — 소형층 우위는 "오차가 작아서"인가 "설명할 분산이 커서"인가')
print('=' * 84)
print(f'{"층":22s} {"n산업":>5s} {"sd(y)":>8s} {"LOCO RMSE":>10s} {"RMSE/sd":>9s} {"LOCO R2":>9s}')
rows = []
for nm, kk in [('39 전체', IND39), ('39 소형20', lo39), ('39 대형19', hi39),
               ('47 전체', IND47), ('신규8', NEW8)]:
    p = parts(kk)
    rows.append(dict(층=nm, **p))
    print(f'{nm:22s} {p["n"]:5d} {p["SDy"]:8.2f} {p["RMSE"]:10.2f} '
          f'{p["RMSE"]/p["SDy"]:9.3f} {p["R2"]:9.4f}')
print()
print('  판독: R² = 1-(RMSE/sd)². 소형층 sd(y)와 대형층 sd(y)를 비교하라.')
print('  만약 RMSE가 비슷한데 sd(y)만 다르면 "관계 강도" 차이가 아니라 척도 효과다.')

# 출하액 기준도
fs = glob.glob(os.path.join(ROOT, '공모전데이터', '외부데이터', '산업분류별', '*.csv'))
frames = []
for f in fs:
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
sh39 = pd.Series({k: float(ship.get(str(k), np.nan)) for k in IND39}).dropna()
lo_s = list(sh39[sh39 <= sh39.median()].index); hi_s = list(sh39[sh39 > sh39.median()].index)
for nm, kk in [('39 출하액하위20', lo_s), ('39 출하액상위19', hi_s)]:
    p = parts(kk)
    print(f'{nm:22s} {p["n"]:5d} {p["SDy"]:8.2f} {p["RMSE"]:10.2f} '
          f'{p["RMSE"]/p["SDy"]:9.3f} {p["R2"]:9.4f}')

# ---------------------------------------------------------------- A2 중복 진단
print()
print('=' * 84)
print('[A2] 종속변수 중복이 규모 층에 편중돼 있는가 (H-A의 규모 구배 = 중복 산물?)')
print('=' * 84)
w1 = pan.pivot(index='ksic', columns='split', values='dd_ship_late')
w2 = pan.pivot(index='ksic', columns='split', values='dd_prod_early')
skey = (w1.round(6).astype(str).agg('|'.join, axis=1) + '##' +
        w2.round(6).astype(str).agg('|'.join, axis=1))
pan['skey'] = pan.ksic.map(skey)
grpsize = skey.map(skey.value_counts())

print('  47개 전체 중복 그룹 구조:')
vc = skey.value_counts().value_counts().sort_index()
for sz, n in vc.items():
    print(f'    그룹크기 {sz}: {n}개 그룹 ({sz*n}개 산업)')
print()
print('  39개 내부: 규모 층별 "중복 산업" 비율 및 평균 그룹크기')
for nm, kk in [('소형20', lo39), ('대형19', hi39)]:
    gs = grpsize.loc[kk]
    print(f'    {nm:8s} 중복산업(그룹크기>1) {int((gs>1).sum())}/{len(kk)}개, 평균그룹크기={gs.mean():.2f}')
print()
print('  ★ 39개 내부에서 고유 시계열(그룹당 1개 대표)만 남기고 재검정')
# 대표 선택: 그룹 안에서 특허건수 최대 산업 1개 (규칙 사전지정)
rep = {}
for key, ks in skey.groupby(skey).groups.items():
    ks39 = [k for k in ks if k in IND39]
    if ks39: rep[key] = max(ks39, key=sizeof)
uniq39 = sorted(set(rep.values()))
print(f'    39개 → 고유대표 {len(uniq39)}개')
su = pd.Series({k: sizeof(k) for k in uniq39}).sort_values()
mu = su.median()
lu = [k for k in uniq39 if sizeof(k) <= mu]; hu = [k for k in uniq39 if sizeof(k) > mu]
for nm, kk in [('고유39 전체', uniq39), ('고유39 소형', lu), ('고유39 대형', hu)]:
    p = parts(kk)
    print(f'    {nm:16s} n={p["n"]:2d} sd(y)={p["SDy"]:6.2f} RMSE={p["RMSE"]:6.2f} '
          f'LOCO R²={p["R2"]:+.4f} (최소 {p["R2min"]:+.4f})')
print('    ※ 중복 제거 후에도 소형>대형 격차가 남는가? 사라지면 H-A는 중복의 산물.')

# ---------------------------------------------------------------- A3 불가피오차
print()
print('=' * 84)
print('[A3] 동일 y 그룹 내 over1 엇갈림 = 원리적 예측불가 오차. 규모층별 분포')
print('=' * 84)
# 47개에서 산업별 LOCO 제곱오차
sq = {}
for s in SPLITS:
    sub = pan[pan.split == s]
    X = sub[COLS].to_numpy(float); y = sub.dd_ship_late.to_numpy(float)
    pr = loco_pred(X, y, sub.mid.to_numpy())
    for k, e in zip(sub.ksic.values, (y-pr)**2):
        sq.setdefault(k, []).append(e)
sse_ind = pd.Series({k: np.mean(v) for k, v in sq.items()})
mixed_ind = set()
for s in SPLITS:
    sub = pan[pan.split == s]
    for key, gg in sub.groupby('skey'):
        if gg.over1.nunique() > 1:
            mixed_ind |= set(gg.ksic.tolist())
print(f'  같은 y·persist인데 over1이 엇갈리는 산업: {len(mixed_ind)}/47개')
print(f'    → {sorted(mixed_ind)}')
tot = sse_ind.sum()
print(f'  47개 총 평균SSE 중 이 산업들이 차지하는 비중 = '
      f'{sse_ind.loc[sorted(mixed_ind)].sum()/tot*100:.1f}%')
for nm, kk in [('39소형20', lo39), ('39대형19', hi39), ('신규8', NEW8)]:
    ov = [k for k in kk if k in mixed_ind]
    print(f'    {nm:8s}: 엇갈림 산업 {len(ov)}/{len(kk)}개 '
          f'({len(ov)/len(kk)*100:.0f}%), 평균SSE={sse_ind.loc[kk].mean():.1f}')

# ---------------------------------------------------------------- A4 폴드 구성
print()
print('=' * 84)
print('[A4] LOCO 폴드(mid) 구성 — 39→47 하락이 특정 폴드에서만 오는가')
print('=' * 84)


def fold_sse(keys):
    acc = {}
    for s in SPLITS:
        sub = pan[(pan.split == s) & (pan.ksic.isin(keys))]
        X = sub[COLS].to_numpy(float); y = sub.dd_ship_late.to_numpy(float)
        g = sub.mid.to_numpy(); pr = loco_pred(X, y, g)
        for m, e in zip(g, (y-pr)**2):
            acc.setdefault(m, []).append(e)
    return pd.Series({m: np.sum(v)/7 for m, v in acc.items()})


f39 = fold_sse(IND39); f47 = fold_sse(IND47)
cmp = pd.DataFrame(dict(SSE39=f39, SSE47=f47)).fillna(0)
cmp['Δ'] = cmp.SSE47 - cmp.SSE39
cmp['n39'] = [sum(1 for k in IND39 if k//1000 == m) for m in cmp.index]
cmp['n47'] = [sum(1 for k in IND47 if k//1000 == m) for m in cmp.index]
cmp = cmp.sort_values('Δ', ascending=False)
print('  mid 폴드별 평균SSE (분할 7개 평균 합계)')
print(cmp.head(8).round(1).to_string())
print(f'  전체 SSE: 39={f39.sum():.0f} → 47={f47.sum():.0f}  (SST: '
      f'{parts(IND39)["SST"]:.0f} → {parts(IND47)["SST"]:.0f})')
print('  ※ SST(분모)도 같이 변한다. R² 하락이 SSE증가인지 SST감소인지 확인:')
p39, p47 = parts(IND39), parts(IND47)
print(f'    39: SSE={p39["SSE"]:.0f} SST={p39["SST"]:.0f} R²={p39["R2"]:+.4f}')
print(f'    47: SSE={p47["SSE"]:.0f} SST={p47["SST"]:.0f} R²={p47["R2"]:+.4f}')
print(f'    가상: 47의 SSE를 39의 표본크기로 정규화(행당 SSE) '
      f'{p39["SSE"]/273:.2f} → {p47["SSE"]/329:.2f}  '
      f'| 행당 SST {p39["SST"]/273:.2f} → {p47["SST"]/329:.2f}')

# ---------------------------------------------------------------- A5 위약 분할
print()
print('=' * 84)
print('[A5] 위약검정 — 규모가 아닌 무작위 20/19 분할에서도 격차가 나오는가')
print('=' * 84)
gaps = []
for i in range(500):
    perm = list(RNG.permutation(IND39))
    a = loco_bys = None
    A = parts(perm[:20])['R2']; B = parts(perm[20:])['R2']
    gaps.append(A - B)
gaps = np.array(gaps)
obs = parts(lo39)['R2'] - parts(hi39)['R2']
print(f'  관측 격차(소형-대형) = {obs:+.4f}')
print(f'  무작위 분할 500회: |격차| 평균={np.abs(gaps).mean():.4f}, '
      f'격차 sd={gaps.std():.4f}, 95분위(절대값)={np.quantile(np.abs(gaps),0.95):.4f}')
print(f'  양측 p(무작위 |격차| >= 관측 |격차|) = {float((np.abs(gaps)>=abs(obs)).mean()):.4f}')
print(f'  참고: 무작위 분할 R² 범위 5~95% = '
      f'[{np.quantile([parts(list(RNG.permutation(IND39))[:20])["R2"] for _ in range(60)],0.05):+.3f}, ...]')

# ---------------------------------------------------------------- A6 대안 지표
print()
print('=' * 84)
print('[A6] 척도 무관 지표로 재평가 — Spearman(예측,실제) 및 MAE/MAD')
print('=' * 84)
from scipy import stats as st
def alt_metrics(keys):
    sp, mae_r = [], []
    for s in SPLITS:
        sub = pan[(pan.split == s) & (pan.ksic.isin(keys))]
        X = sub[COLS].to_numpy(float); y = sub.dd_ship_late.to_numpy(float)
        pr = loco_pred(X, y, sub.mid.to_numpy())
        if pr is None: continue
        sp.append(st.spearmanr(pr, y).correlation)
        mad = np.mean(np.abs(y - np.median(y)))
        mae_r.append(np.mean(np.abs(y-pr))/mad)
    return np.nanmean(sp), np.nanmean(mae_r)
for nm, kk in [('39 소형20', lo39), ('39 대형19', hi39), ('39 전체', IND39),
               ('47 전체', IND47), ('39 출하하위20', lo_s), ('39 출하상위19', hi_s)]:
    a, b = alt_metrics(kk)
    print(f'  {nm:16s} Spearman(pred,y)={a:+.3f}   MAE/MAD={b:.3f}')
print('  ※ Spearman은 척도무관. 소형/대형 순위상관 격차가 R² 격차만큼 큰가?')

# ---------------------------------------------------------------- A7 신규8 재정의
print()
print('=' * 84)
print('[A7] "신규8은 전부 대형" 주장의 성격 — 수집순서 tautology 확인')
print('=' * 84)
print(f'  39개 특허건수 최대 = {size39.max()} ({size39.idxmax()})')
print(f'  신규8 특허건수 최소 = {min(sizeof(k) for k in NEW8)}')
print('  수집 우선순위가 "작은 것부터"였다면 신규군이 100백분위인 것은 설계상 필연.')
print('  → [4]의 "100백분위 밖 외삽" 은 경험적 발견이 아니라 수집설계의 항진명제.')
print('  검증: 47개를 특허건수 순으로 정렬해 실제로 39개가 하위 39개인가?')
s47 = pd.Series({k: sizeof(k) for k in IND47}).sort_values()
bot39 = set(s47.index[:39])
print(f'    하위39 == 기존39 ?  {bot39 == set(IND39)}  '
      f'(불일치 {len(bot39 ^ set(IND39))//2}개)')

# ---------------------------------------------------------------- A8 잔여 검정
print()
print('=' * 84)
print('[A8] H-B 잔여: 중복 제거 후 신규8의 순수 기여')
print('=' * 84)
# 47 고유대표만 (그룹당 1개) → 25개
rep47 = {}
for key, ks in skey.groupby(skey).groups.items():
    rep47[key] = max(list(ks), key=sizeof)
u47 = sorted(set(rep47.values()))
u39only = sorted(set(u47) & set(IND39))
newuniq = [k for k in u47 if k in NEW8]
print(f'  47 고유대표 {len(u47)}개 중 신규8 출신 대표 = {newuniq}')
print('  ※ 대표선택 규칙(그룹내 특허건수 최대)에 따라 신규가 기존을 밀어낼 수 있음 — 규칙은 사전지정.')
for nm, kk in [('고유47 전체', u47), ('고유47 - 신규대표', [k for k in u47 if k not in NEW8])]:
    p = parts(kk)
    print(f'    {nm:22s} n={p["n"]:2d} LOCO R²={p["R2"]:+.4f} (최소 {p["R2min"]:+.4f}) '
          f'sd(y)={p["SDy"]:.2f} RMSE={p["RMSE"]:.2f}')
# 고유 시계열 기준: 27199, 27301만 진짜 신규
print()
print('  진짜 신규 고유계열은 27199, 27301 뿐. 39 + 이 2개 만 넣으면?')
p = parts(IND39 + [27199, 27301]); print(f'    39+2 (n=41): LOCO R²={p["R2"]:+.4f}')
p = parts(IND39 + [k for k in NEW8 if k not in (27199, 27301)])
print(f'    39+중복6 (n=45): LOCO R²={p["R2"]:+.4f}')
print(f'    39 (n=39): LOCO R²={parts(IND39)["R2"]:+.4f} | 47: {parts(IND47)["R2"]:+.4f}')
