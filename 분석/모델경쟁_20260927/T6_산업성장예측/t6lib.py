# T6 공용 함수: 패널 로드, 롤링 1년 앞 검증(목표연도 2012-2019), 지표, 리더보드 기록.
# 🔒 홀드아웃(2021-2024) 결과변수는 이 파일에서 다루지 않는다(패널 자체가 <=2019).
import pandas as pd, numpy as np, os, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
VAL_YEARS = list(range(2012, 2020))


def load_panel(with_patent=True):
    F = pd.read_parquet(os.path.join(HERE, 'cache/panel_base.parquet'))
    assert F.year.max() <= 2019
    if with_patent:
        P = pd.read_parquet(os.path.join(HERE, 'cache/patent_feats.parquet'))
        F = F.merge(P, on=['ksic', 'year'], how='left')
    # 파생 피처(모두 T-1 이전 값의 조합)
    F['dmargin1'] = F.gva1 - F.g1          # 부가가치율 변화
    F['inv_rel1'] = F.gasset1 - F.g1       # 출하 대비 유형자산 증가(설비투자 선행)
    F['prod_lab1'] = F.g1 - F.gemp1        # 노동생산성 변화
    F['div_sector1'] = F.g1 - F.is_g3      # 5자리 출하성장 - 3자리 출하지수 성장(괴리 → 평균회귀?)
    F['div_peer1'] = F.g1 - F.peer3_1      # 5자리 - 3자리 동료(자기 제외) 괴리
    F['g12'] = F.g1 + F.g2                 # 2년 누적 성장
    F['estab_size1'] = F.gestab1 - F.gemp1 # 사업체 수 대비 종사자 변화(진입·소형화)
    return F


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    sp, sy = np.sign(p), np.sign(y)
    dir_acc = np.mean(np.where(sp == 0, 0.5, (sp == sy).astype(float)))
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))), DIR=float(dir_acc),
                MedAE=float(np.median(np.abs(e))), n=int(len(y)))


def rolling_eval(F, fit_predict, train_filter=None, years=VAL_YEARS, min_train_year=2009):
    """fit_predict(train_df, test_df) -> np.array 예측. train: year<T & y notna (train_filter 적용), test: in50 & year==T & y notna."""
    out = []
    for T in years:
        tr = F[(F.year < T) & (F.year >= min_train_year) & F.y.notna()]
        if train_filter is not None:
            tr = train_filter(tr)
        te = F[(F.year == T) & (F.in50 == 1) & F.y.notna()]
        p = fit_predict(tr, te)
        o = te[['ksic', 'year', 'y']].copy()
        o['pred'] = np.asarray(p, float)
        out.append(o)
    return pd.concat(out, ignore_index=True)


BASE_MAE = None


def baseline_preds(F):
    te = F[(F.year.isin(VAL_YEARS)) & (F.in50 == 1) & F.y.notna()]
    return te


def boot_ci(pred_df, base_df, B=2000, seed=0):
    """산업 군집 부트스트랩: skill = 1 - MAE_m/MAE_b 의 95% CI"""
    rng = np.random.default_rng(seed)
    m = pred_df.merge(base_df[['ksic', 'year', 'pred']].rename(columns={'pred': 'pb'}), on=['ksic', 'year'])
    m['am'] = (m.y - m.pred).abs()
    m['ab'] = (m.y - m.pb).abs()
    g = m.groupby('ksic')[['am', 'ab']].sum()
    k = len(g)
    vals = []
    A, Bb = g.am.values, g.ab.values
    for _ in range(B):
        idx = rng.integers(0, k, k)
        vals.append(1 - A[idx].sum() / Bb[idx].sum())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def log_version(version, change, hypothesis, pred_df, baseline_mae, kept, note='', metric='MAE'):
    os.makedirs(os.path.join(HERE, 'preds'), exist_ok=True)
    pred_df.to_parquet(os.path.join(HERE, f'preds/{version}.parquet'), index=False)
    mt = metrics(pred_df.y, pred_df.pred)
    skill = 1 - mt['MAE'] / baseline_mae if baseline_mae else 0.0
    note2 = f"RMSE={mt['RMSE']:.4f}; DIR={mt['DIR']:.3f}; MedAE={mt['MedAE']:.4f}; n={mt['n']}" + (('; ' + note) if note else '')
    row = dict(version=version, date=datetime.date.today().isoformat(), change=change, hypothesis=hypothesis,
               metric=metric, value=round(mt['MAE'], 5), baseline_value=round(baseline_mae, 5),
               skill=round(skill, 4), kept=kept, note=note2)
    path = os.path.join(HERE, 'leaderboard.csv')
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=row.keys())
    df = df[df.version != version]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"[{version}] MAE={mt['MAE']:.4f} RMSE={mt['RMSE']:.4f} DIR={mt['DIR']:.3f} skill={skill:.4f}")
    return mt, skill


def by_year(pred_df):
    return pred_df.assign(ae=(pred_df.y - pred_df.pred).abs()).groupby('year').ae.mean()
