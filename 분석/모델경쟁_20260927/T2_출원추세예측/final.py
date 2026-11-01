# T2 최종 후보(v29): 3-모형 동일가중 앙상블 = 전역 Ridge+우선권선행(v22) + 우선권 흐름·실용신안 분리(v27) + 계층 하향식(v15)
# 1) 검증(원점 2012–2015, h=1..3) 재현·진단  2) 예측구간 보정  3) 트렌드 산출물 초안
# 🔒 원점 2018 예측(2019–2021)은 "동결 예측"으로 저장만 하며 실제값은 계산·조회하지 않는다(패널 자체가 ≤2018).
import os, warnings
import numpy as np, pandas as pd
from common import *
from ridge2 import r2_fn
from flow import flow_fn, FY, SRC, DOM, CCI, KSI, ROW
from hier import topdown_fn

warnings.filterwarnings('ignore')
OUT = os.path.join(T, 'outputs'); os.makedirs(OUT, exist_ok=True)
MEMBERS = {
    'ridge_lead': (r2_fn, {'fs': ('lead',), 'alpha': 10.0, 'tmin': 2005}),
    'flow_um': (flow_fn, {'k': 8, 'phi': 0.8, 'lag': 1, 'rk': 1, 'um_split': True}),
    'topdown': (topdown_fn, {'kt': 8, 'pt': 1.0, 'ks': 8, 'ps': 0.8}),
}


def final_fn(hist, o, return_members=False):
    Fs = {m: np.asarray(fn(hist.copy(), o, **kw), float) for m, (fn, kw) in MEMBERS.items()}
    F = np.mean(list(Fs.values()), 0)
    return (F, Fs) if return_members else F


def boot_ci(r, nb, B=2000, seed=0):
    """KSIC 군집 부트스트랩(50개 산업 재표집) — 같은 산업의 5개국·원점·h는 함께 움직임"""
    rng = np.random.default_rng(seed)
    ks = r.ksic.unique()
    a = r.groupby('ksic').ase.agg(['sum', 'count']); b = nb.groupby('ksic').ase.agg(['sum', 'count'])
    out = []
    for _ in range(B):
        s = rng.choice(ks, len(ks), replace=True)
        out.append(1 - (a.loc[s, 'sum'].sum() / a.loc[s, 'count'].sum()) / (b.loc[s, 'sum'].sum() / b.loc[s, 'count'].sum()))
    return np.percentile(out, [2.5, 97.5])


if __name__ == '__main__':
    Yw = load_panel()
    r = evaluate(final_fn)
    s, bsum, skill = report(r, 'v29_final', seg=True)
    save_pred(r, 'v29_final')
    nb = naive_ref()
    lo, hi = boot_ci(r, nb)
    print(f'skill 95% CI (KSIC 군집 부트스트랩 2000회): [{lo:.3f}, {hi:.3f}]')
    ex = r.ksic != '31312'
    print('31312(드론) 제외 skill:', round(1 - r[ex].ase.mean() / nb[ex].ase.mean(), 4),
          '| 31312 오차 점유율:', round(r[~ex].ase.sum() / r.ase.sum(), 3))
    ser = r.groupby(['ksic', 'cc']).ase.mean() < nb.groupby(['ksic', 'cc']).ase.mean()
    print('naive보다 나은 계열 비율:', round(ser.mean(), 3))
    for h in range(1, H + 1):
        print(f'h{h} skill', round(1 - r[r.h == h].ase.mean() / nb[nb.h == h].ase.mean(), 4))
    # 규모 3분위
    size = pd.qcut(r.groupby(['ksic', 'cc']).ylast.transform('mean'), 3, labels=['소', '중', '대'])
    print('규모별 MASE(모형/naive):', {str(k): (round(r[size == k].ase.mean(), 3), round(nb[size == k].ase.mean(), 3)) for k in ['소', '중', '대']})

    # 예측구간: 검증 로그비 오차(log(y+1)-log(f+1))의 h별·국가별 분위수 (검증 내 보정 → 낙관적 가능)
    r['lerr'] = np.log1p(r.y) - np.log1p(r.f)
    q = r.groupby(['cc', 'h']).lerr.quantile([0.1, 0.9]).unstack()
    q.columns = ['q10', 'q90']
    q.to_csv(os.path.join(OUT, 'interval_logerr_quantiles.csv'), encoding='utf-8-sig')
    rr = r.join(q, on=['cc', 'h'])
    cov = ((rr.lerr >= rr.q10) & (rr.lerr <= rr.q90)).mean()
    print('80% 구간 검증 내 적중률(정의상 ~0.8):', round(cov, 3))

    # 트렌드 산출물 초안 1: 검증 그림용 — 원점 2015 → 2016–2018 예측 vs 실제
    v = r[r.origin == 2015].copy()
    v = v.join(q, on=['cc', 'h'])
    v['lo80'] = np.expm1(np.log1p(v.f) + v.q10); v['hi80'] = np.expm1(np.log1p(v.f) + v.q90)
    v[['ksic', 'cc', 'target', 'ylast', 'y', 'f', 'lo80', 'hi80']].to_csv(os.path.join(OUT, 'val_origin2015_forecast_vs_actual.csv'), index=False, encoding='utf-8-sig')

    # 트렌드 산출물 초안 2: 원점 2018 동결 예측(2019–2021) — 실제값 미사용, 홀드아웃 평가자가 대조
    hist = Yw.values.astype(float)
    assert hist.shape[1] == 2018 - 2000 + 1
    F, Fs = final_fn(hist, 2018, return_members=True)
    rows = []
    for i, (k, c) in enumerate(Yw.index):
        for h in range(1, H + 1):
            ql, qh = q.loc[(c, h)]
            rows.append(dict(ksic=k, cc=c, origin=2018, target=2018 + h, y2018=hist[i, -1], f=F[i, h - 1],
                             lo80=np.expm1(np.log1p(F[i, h - 1]) + ql), hi80=np.expm1(np.log1p(F[i, h - 1]) + qh),
                             **{f'f_{m}': Fs[m][i, h - 1] for m in Fs}))
    fz = pd.DataFrame(rows)
    fz.to_csv(os.path.join(OUT, 'frozen_forecast_origin2018.csv'), index=False, encoding='utf-8-sig')

    # 트렌드 산출물 초안 3: 가속/감속 순위 — 예측 CAGR(2018→2021) − 과거 CAGR(2015→2018)
    past = (np.log(hist[:, -1] + 1) - np.log(hist[:, -4] + 1)) / 3
    fut = (np.log(F[:, 2] + 1) - np.log(hist[:, -1] + 1)) / 3
    acc = pd.DataFrame({'ksic': Yw.index.get_level_values(0), 'cc': Yw.index.get_level_values(1),
                        'n2018': hist[:, -1], 'cagr_2015_18': np.expm1(past), 'cagr_2018_21_pred': np.expm1(fut),
                        'accel_pp': (np.expm1(fut) - np.expm1(past)) * 100})
    import json
    kd = {d['code']: d['codenmKr'] for d in json.load(open(os.path.join(T, '..', '..', '..', '공모전데이터', '외부데이터', 'KSIC해설', 'ksic50_definitions.json'), encoding='utf-8'))}
    acc['ksic_name'] = acc.ksic.map(kd)
    acc.sort_values('accel_pp', ascending=False).to_csv(os.path.join(OUT, 'accel_ranking_origin2018.csv'), index=False, encoding='utf-8-sig')
    print('\n[가속 상위 10 (ksic×cc, 2018년 ≥100건)]')
    a2 = acc[acc.n2018 >= 100].sort_values('accel_pp', ascending=False)
    print(a2.head(10)[['ksic', 'ksic_name', 'cc', 'n2018', 'cagr_2015_18', 'cagr_2018_21_pred', 'accel_pp']].round(3).to_string(index=False))
    print('[감속 상위 10]')
    print(a2.tail(10)[['ksic', 'ksic_name', 'cc', 'n2018', 'cagr_2015_18', 'cagr_2018_21_pred', 'accel_pp']].round(3).to_string(index=False))

    # 트렌드 산출물 초안 4: 국가 간 선행 관계(≤2018) — (a) 우선권 흐름 비중 (b) 패널 회귀 g_X(t+1) ~ g_X(t) + g_Y(t)
    L = np.log1p(hist)
    g = np.diff(L, axis=1)  # g[:, j] = 연도 2001+j 성장
    lead_rows = []
    for X in CCS:
        for Y in CCS:
            if X == Y:
                continue
            ix = np.where(CCI == CCS.index(X))[0]
            iy = [ROW[(KSI[i], CCS.index(Y))] for i in ix]
            yy, x1, x2 = [], [], []
            for j in range(5, g.shape[1] - 1):  # 성장 연도 2006..2017 → 목표 2007..2018
                a1, a2, a3 = g[ix, j + 1], g[ix, j], g[iy, j]
                # 연도 고정효과(연도별 평균 제거): 전 세계 공통 경기 충격을 제거하고 산업 간 차이로만 선행 추정
                yy.append(a1 - a1.mean()); x1.append(a2 - a2.mean()); x2.append(a3 - a3.mean())
            yy = np.concatenate(yy); x1 = np.concatenate(x1); x2 = np.concatenate(x2)
            Xm = np.column_stack([np.ones_like(x1), x1, x2])
            beta, *_ = np.linalg.lstsq(Xm, yy, rcond=None)
            res = yy - Xm @ beta; s2 = res @ res / (len(yy) - 3)
            se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
            share = FY[Y][ix][:, -3:].sum() / hist[ix][:, -3:].sum()
            lead_rows.append(dict(target_cc=X, source_cc=Y, prio_share_2016_18=share, beta_lead=beta[2], t_lead=beta[2] / se[2], n=len(yy)))
    ld_ = pd.DataFrame(lead_rows).sort_values('t_lead', ascending=False)
    ld_.to_csv(os.path.join(OUT, 'cross_country_lead.csv'), index=False, encoding='utf-8-sig')
    print('\n[국가 간 선행: 목적국 성장(t+1) ~ 자기성장(t) + 원천국 같은KSIC 성장(t), 2007–2018, 연도FE, OLS(군집보정 없음)]')
    print(ld_.round(3).to_string(index=False))
    # 국가별 예측 합계
    tot = fz.groupby(['cc', 'target']).f.sum().unstack().round(0)
    print('\n[원점2018 동결 예측 국가합(50 KSIC)]'); print(tot)
