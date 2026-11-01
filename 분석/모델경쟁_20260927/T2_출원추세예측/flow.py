# 우선권 흐름 분해 모형: n_X = 자국기원 D_X + Σ_Y 유입 F_{Y→X} + 기타국 유입 O_X
# F_{Y→X}(t) ≈ rate_{YX} × D_Y(t-lag)  (파리조약 12개월 → 원천국 선행 신호)
import sys, json, warnings
import numpy as np, pandas as pd
from common import *

warnings.filterwarnings('ignore')
MAIN = load_panel()
IDX = MAIN.index
KS = sorted(set(IDX.get_level_values('ksic')))
CCI = np.array([CCS.index(c) for c in IDX.get_level_values('cc')])
KSI = np.array([KS.index(k) for k in IDX.get_level_values('ksic')])
p = pd.read_parquet(os.path.join(CACHE, 'panel_prio.parquet'))
assert p.year.max() <= 2018


def arr(df):
    w = df.pivot_table(index=['ksic', 'cc'], columns='year', values='n', aggfunc='sum')
    return w.reindex(IDX).reindex(columns=YEARS).fillna(0.0).values


FY = {}
for Y in CCS:
    FY[Y] = arr(p[(p.p1 == Y) & (p.cc != Y)])
OTHER = arr(p[(~p.p1.isin(CCS))])
DOM = MAIN.values - sum(FY.values()) - OTHER  # 자국우선권 + 우선권 없음
UMA = load_all()['n_um'].values
assert DOM.min() >= -1e-9
# 원천국 Y 의 같은 KSIC 자국기원 출원 D_Y 를 목적국 X 행에 정렬: SRC[Y][i] = DOM[(ksic_i, Y)]
ROW = {(KSI[i], CCI[i]): i for i in range(len(IDX))}
SRC = {Y: DOM[[ROW[(KSI[i], CCS.index(Y))] for i in range(len(IDX))]] for Y in CCS}


def ld(M, k, phi, h=H):
    L = np.log1p(M)
    g = (L[:, -1] - L[:, -1 - k]) / k
    steps = np.cumsum(phi ** np.arange(1, h + 1))
    return np.expm1(L[:, -1:] + g[:, None] * steps[None, :])


def flow_fn(hist, o, k=8, phi=0.8, lag=1, rk=3, comp_only=False, rate_trend=False, um_split=False):
    T_ = hist.shape[1]
    D = DOM[:, :T_]; O = OTHER[:, :T_]
    FD = ld(D, k, phi)            # 자국기원 예측 (250 x H)
    if um_split:  # 자국기원을 실용신안/발명으로 분리 외삽 (CN 실용신안 급증 분리)
        U = np.minimum(UMA[:, :T_], D)
        has = U[:, -k - 1:].min(1) >= 20
        FDs = ld(D - U, k, phi) + ld(U, k, phi)
        FD[has] = FDs[has]
    FO = ld(O, k, phi)
    out = FD + FO
    for Y in CCS:
        Fy = FY[Y][:, :T_]
        if comp_only:
            out += ld(Fy, k, phi)
            continue
        S = SRC[Y][:, :T_]
        # 원천 D_Y 예측을 원천국 행에서 가져옴 (자국기원 예측 FD 재사용)
        srcF = FD[[ROW[(KSI[i], CCS.index(Y))] for i in range(len(IDX))]]
        Sx = np.concatenate([S, srcF], axis=1)  # 관측 + 예측
        num = Fy[:, -rk:].sum(1)
        den = Sx[:, T_ - rk - lag:T_ - lag].sum(1)
        rate = np.where(den > 0, num / np.maximum(den, 1), 0)
        for h in range(1, H + 1):
            srcv = Sx[:, T_ - 1 + h - lag]
            out[:, h - 1] += rate * srcv
    return np.maximum(out, 0)


if __name__ == '__main__':
    tot = MAIN.values
    print('자국기원 비중(2012-15):', {c: round(DOM[CCI == j][:, 12:16].sum() / tot[CCI == j][:, 12:16].sum(), 3) for j, c in enumerate(CCS)})
    runs = [('flow_componly', {'comp_only': True}), ('flow_lag1', {}), ('flow_lag0', {'lag': 0}),
            ('flow_lag1_rk5', {'rk': 5}), ('flow_lag1_rk1', {'rk': 1})]
    if len(sys.argv) > 1:
        runs = [(a, json.loads(b)) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]
    for nm, kw in runs:
        r = evaluate(flow_fn, **kw)
        report(r, nm)
        save_pred(r, nm)
