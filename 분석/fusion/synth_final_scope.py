# -*- coding: utf-8 -*-
"""
4개 진단 각도 통합 - 최종 사양(적용범위) 확정용 재계산.
기존 스크립트는 수정하지 않고 새 파일로만 작성.

핵심 원칙(반증 검토에서 공통 지적된 사항 반영):
  - LOCO R2 는 층마다 분모(SST)가 달라 층 간 비교 불가.
    따라서 동일 폴드/동일 행에서 '절편만 모형(훈련폴드 평균)' 을 함께 돌려
    skill = 1 - MSE_model / MSE_naive 를 병기한다.
  - 모든 표에 n산업 / n행 / 고유결과셀 을 명기한다.
"""
import numpy as np, pandas as pd, itertools, json

PAN = 'model_search_panel.csv'
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = list(range(2013, 2020))


def loco(X, y, g):
    """leave-one-cluster-out 예측 R². g는 클러스터 라벨(보통 mid)."""
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
    """지정 함수와 동일한 폴드 구조로 예측값과 '절편만' 예측값을 반환."""
    uq = np.unique(g); pr = np.full(len(y), np.nan); pn = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return None, None
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try: b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception: return None, None
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
        pn[te] = y[tr].mean()
    return pr, pn


def prep(df):
    d = df.copy()
    d['over1'] = (d['padi_e'] > 1.0).astype(float)
    return d


def evaluate(d, cols=('dd_prod_early', 'over1'), tag=''):
    """분할 7개 각각 LOCO(mid). R2 / RMSE / naive RMSE / skill 을 평균."""
    r2s, rmses, nrmses, skills, sdys = [], [], [], [], []
    for s in SPLITS:
        sub = d[d['split'] == s].dropna(subset=list(cols) + ['dd_ship_late'])
        if len(sub) < 5: continue
        X = sub[list(cols)].values.astype(float)
        y = sub['dd_ship_late'].values.astype(float)
        g = sub['mid'].values
        r2 = loco(X, y, g)
        pr, pn = loco_pred(X, y, g)
        if pr is None or np.isnan(r2):
            r2s.append(np.nan); continue
        sse = ((y-pr)**2).sum(); ssn = ((y-pn)**2).sum()
        r2s.append(r2)
        rmses.append(np.sqrt(sse/len(y)))
        nrmses.append(np.sqrt(ssn/len(y)))
        skills.append(1 - sse/ssn if ssn > 0 else np.nan)
        sdys.append(y.std(ddof=1))
    f = lambda a: (float(np.nanmean(a)) if len(a) else np.nan)
    return dict(tag=tag,
                n_ind=d['ksic'].nunique(), n_row=len(d),
                n_cell=d[['dd_prod_early', 'dd_ship_late']].drop_duplicates().shape[0]
                if len(d) else 0,
                loco_mean=f(r2s), loco_min=(float(np.nanmin(r2s)) if len(r2s) else np.nan),
                rmse=f(rmses), rmse_naive=f(nrmses), skill=f(skills), sd_y=f(sdys),
                n_G=int(d['mid'].nunique()))


def cellcount(d):
    """분할연도별 (dd_prod_early, dd_ship_late) 고유 조합 수의 평균(=고유 결과셀)."""
    v = []
    for s in SPLITS:
        sub = d[d['split'] == s]
        v.append(sub[['dd_prod_early', 'dd_ship_late']].drop_duplicates().shape[0])
    return float(np.mean(v))


pan = prep(pd.read_csv(PAN))
pan39 = pan[~pan['ksic'].isin(NEW8)].copy()
pan_orig39 = prep(pd.read_csv('model_search_panel_39ind.csv'))

rows = []
# ---- 0. 기준선 --------------------------------------------------------
rows.append(evaluate(pan, tag='(가) 47산업 전체'))
rows.append(evaluate(pan39, tag='[참조] 47패널에서 신규8 제외(39)'))
rows.append(evaluate(pan_orig39, tag='[참조] 39산업 원본 파일'))

# ---- 1. 적용범위 후보 -------------------------------------------------
elec = pan['mid'].isin([26, 27, 28])
rows.append(evaluate(pan[~elec], tag='(다-1) 전자·기계(mid 26/27/28) 전면 제외'))
rows.append(evaluate(pan[pan['mid'] != 26], tag='(다-2) mid26 만 제외'))
rows.append(evaluate(pan39[~pan39['mid'].isin([26, 27, 28])],
                     tag='[참조] 39산업 기준 전자·기계 제외'))
# 중복 쌍둥이(26519/26529) 제외
rows.append(evaluate(pan[~pan['ksic'].isin([26519, 26529])],
                     tag='(다-3) 중복쌍 26519+26529 제외'))

# ---- 2. 규모 층화(등록특허 건수 대용: 출하액이 없으므로 패널 내 정보만 사용) --
# 규모 지표: 광업제조업조사 출하액을 쓰지 않고, 여기서는 층화 자체가 아니라
# '전자·기계 vs 그 외' 라는 관측 가능한 절단면만 사용한다(반증 검토 반영).

# ---- 3. 조절변수 확장 (동일 47산업 고정) ------------------------------
p = pan.copy()
p['elec'] = p['mid'].isin([26, 27, 28]).astype(float)
p['ov_elec'] = p['over1'] * p['elec']
p['new8'] = p['ksic'].isin(NEW8).astype(float)
p['ov_new8'] = p['over1'] * p['new8']
rows.append(evaluate(p, cols=('dd_prod_early', 'over1', 'elec', 'ov_elec'),
                     tag='(나-1) 47 + 전자더미 + over1×전자'))
rows.append(evaluate(p, cols=('dd_prod_early', 'over1', 'new8', 'ov_new8'),
                     tag='(나-2) 47 + 신규8더미 + over1×신규8 (오라클)'))
rows.append(evaluate(p, cols=('dd_prod_early',), tag='[참조] 47 지속성만(over1 없음)'))
rows.append(evaluate(pan39, cols=('dd_prod_early',), tag='[참조] 39 지속성만'))
rows.append(evaluate(p[~elec.values], cols=('dd_prod_early',),
                     tag='[참조] 비전자27 지속성만'))

res = pd.DataFrame(rows)
res['n_cell_split'] = [cellcount(pan), cellcount(pan39), cellcount(pan_orig39),
                       cellcount(pan[~elec]), cellcount(pan[pan['mid'] != 26]),
                       cellcount(pan39[~pan39['mid'].isin([26, 27, 28])]),
                       cellcount(pan[~pan['ksic'].isin([26519, 26529])]),
                       cellcount(p), cellcount(p), cellcount(p), cellcount(pan39),
                       cellcount(p[~elec.values])]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(res[['tag', 'n_ind', 'n_row', 'n_cell_split', 'n_G', 'loco_mean', 'loco_min',
           'rmse', 'rmse_naive', 'skill', 'sd_y']].round(4).to_string(index=False))
res.to_csv('synth_final_scope.csv', index=False, encoding='utf-8-sig')

# ---- 4. 블록별 오차 분해 (동일 LOCO 실행, 행만 분할) -------------------
print('\n[블록별 오차 분해] 같은 47산업 LOCO 실행에서 행만 나눔')
blocks = {}
for s in SPLITS:
    sub = pan[pan['split'] == s].dropna(subset=['dd_prod_early', 'over1', 'dd_ship_late'])
    X = sub[['dd_prod_early', 'over1']].values.astype(float)
    y = sub['dd_ship_late'].values.astype(float); g = sub['mid'].values
    pr, pn = loco_pred(X, y, g)
    sub = sub.assign(pred=pr, naive=pn)
    for name, m in [('신규8', sub['ksic'].isin(NEW8)),
                    ('기존39·전자기계', (~sub['ksic'].isin(NEW8)) & sub['mid'].isin([26, 27, 28])),
                    ('기존39·비전자', (~sub['ksic'].isin(NEW8)) & ~sub['mid'].isin([26, 27, 28]))]:
        q = sub[m]
        blocks.setdefault(name, []).append(
            (((q['dd_ship_late']-q['pred'])**2).sum(),
             ((q['dd_ship_late']-q['naive'])**2).sum(), len(q),
             q['dd_ship_late'].std(ddof=1)))
for name, v in blocks.items():
    sse = sum(a for a, b, c, d in v); ssn = sum(b for a, b, c, d in v)
    n = sum(c for a, b, c, d in v)
    sd = np.mean([d for a, b, c, d in v])
    sub0 = pan[pan['ksic'].isin(NEW8)] if name == '신규8' else None
    print(f'  {name:16s} n행={n:4d} RMSE={np.sqrt(sse/n):6.2f} '
          f'절편만RMSE={np.sqrt(ssn/n):6.2f} skill={1-sse/ssn:+.3f} sd_y={sd:5.2f}')

# ---- 5. 중복 결과셀 실태 ----------------------------------------------
print('\n[중복 결과셀]')
for nm, d in [('47산업', pan), ('39산업', pan39), ('비전자27', pan[~elec])]:
    print(f'  {nm}: n산업={d["ksic"].nunique()}, 분할당 고유 결과셀={cellcount(d):.1f}')
s = pan[pan['split'] == 2016]
dup = s.groupby(['dd_prod_early', 'dd_ship_late'])['ksic'].apply(list)
dup = dup[dup.apply(len) > 1]
print('  중복 그룹(split=2016):')
for k, v in dup.items():
    flag = s[s['ksic'].isin(v)].set_index('ksic')['over1'].to_dict()
    print(f'    {v}  over1={flag}')
