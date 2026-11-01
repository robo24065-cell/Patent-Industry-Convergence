# -*- coding: utf-8 -*-
"""N2 패널(산업×연도) 기초: 결과변수 + 거시 피처. 🔒 개발 모드는 로드 단계에서 연도 ≤ 2023 필터(duckdb WHERE)로 2024 조사값을 메모리에 올리지 않는다.
결과 g(k,t) = log(출하액_t/출하액_{t−1}) (T04), 10차(t≤2020)/11차(t≥2021) 같은 차수 안에서만.
피처(모두 t+1년 4월 이전 공표):
  g_idx     광공업생산지수 출하지수 3자리(없으면 2자리) 연간합 성장 (t+1년 1월 말 공표)
  g_idx_h2  하반기 합 성장, g_idx_q4 4분기 합 성장, g_pidx 생산지수 성장, g_inv 재고지수 12월 전년동월비
  g_idx2    2자리 출하지수 성장, g_mfg 제조업(C) 출하지수 성장
  g_dom/g_exp 내수·수출 출하지수(2자리, C26은 3자리) 성장
  g_lag     t−1년 조사 성장(t년 12월 공표 → 이용 가능)
  g_x_hs    한국 HS6 수출(UN Comtrade, KSIC-HS6 연계) t년 성장 — 수집 범위 ≤2021(+N2 추가 2022·2023 있으면 사용)
출력: cache/panel_base.parquet"""
import os, sys, glob
import numpy as np, pandas as pd, duckdb
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
MAXY = 2023   # 개발 모드 고정. 최종시험은 final_run_r2.py 에서만 2024 로드.


def build(maxy=MAXY, feature_years=(), idx_maxy=None):
    """maxy: 조사(결과) 로드 상한. feature_years: 결과 없이 피처만 만들 연도(미래 나우캐스트). idx_maxy: 지수·무역 로드 상한."""
    idx_maxy = idx_maxy or maxy
    f = os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv").replace("\\", "/")
    k = duckdb.sql(f"select 기준, cast(세세분류코드 as varchar) ksic, 항목코드, 연도, 값 from read_csv_auto('{f}') "
                   f"where 연도 <= {maxy} and 항목코드 in ('T04','T06','T02')").df()
    assert k.연도.max() <= maxy
    out = {}
    for it, nm in [("T04", "g"), ("T06", "g_va"), ("T02", "g_emp")]:
        rows = []
        for (ks, bs), g in k[k.항목코드 == it].groupby(["ksic", "기준"]):
            s = g.set_index("연도").값.sort_index()
            for t in s.index:
                if t - 1 in s.index and s[t - 1] > 0 and s[t] > 0 and ((bs == "10차" and t <= 2020) or (bs == "11차" and t >= 2021)):
                    rows.append((ks, t, np.log(s[t] / s[t - 1]), s[t - 1]))
        out[nm] = pd.DataFrame(rows, columns=["ksic", "t", nm, f"{nm}_lvlprev"])
    Y = out["g"].rename(columns={"g_lvlprev": "ship_prev"})
    for nm in ["g_va", "g_emp"]:
        Y = Y.merge(out[nm][["ksic", "t", nm]], on=["ksic", "t"], how="left")
    if feature_years:
        sk = pd.DataFrame([(kk, t) for kk in Y.ksic.unique() for t in feature_years], columns=["ksic", "t"])
        Y = pd.concat([Y, sk], ignore_index=True)
    Y["g_lag"] = Y.set_index(["ksic", "t"]).g.reindex(pd.MultiIndex.from_arrays([Y.ksic, Y.t - 1])).values
    # 생산지수(월별, 3자리)
    p = os.path.join(BASE, r"공모전데이터\외부데이터\경기지표\kosis_광공업생산지수_월별.csv").replace("\\", "/")
    pi = duckdb.sql(f"select 항목명, 산업코드, 연월, try_cast(값 as double) v from read_csv_auto('{p}', all_varchar=true) "
                    f"where cast(연월 as integer) <= {idx_maxy}12").df()
    pi["연월"] = pi.연월.astype(int); pi["y"] = pi.연월 // 100; pi["m"] = pi.연월 % 100

    def annual(item, months=range(1, 13), agg="sum"):
        d = pi[(pi.항목명 == item) & pi.m.isin(list(months))]
        a = d.groupby(["산업코드", "y"]).v.agg([agg, "size"]).reset_index()
        a = a[a["size"] == len(list(months))]
        return a.set_index(["산업코드", "y"])[agg]
    A = {"g_idx": annual("출하지수(원지수)"), "g_idx_h2": annual("출하지수(원지수)", range(7, 13)),
         "g_idx_q4": annual("출하지수(원지수)", range(10, 13)), "g_pidx": annual("생산지수(원지수)"), "g_pidx_h2": annual("생산지수(원지수)", range(7, 13)),
         "g_inv": annual("재고지수(원지수)", [12])}

    def idx_g(S, code, t, levels=(3, 2)):
        for L in levels:
            c = "C" + code[:L]
            if (c, t) in S.index and (c, t - 1) in S.index and S[(c, t - 1)] > 0:
                return np.log(S[(c, t)] / S[(c, t - 1)])
        return np.nan
    for nm, S in A.items():
        Y[nm] = [idx_g(S, a, b) for a, b in zip(Y.ksic, Y.t)]
    Y["g_idx2"] = [idx_g(A["g_idx"], a, b, (2,)) for a, b in zip(Y.ksic, Y.t)]
    Y["g_pidx2"] = [idx_g(A["g_pidx"], a, b, (2,)) for a, b in zip(Y.ksic, Y.t)]
    Sm = A["g_idx"]
    Y["g_mfg"] = [np.log(Sm[("C", b)] / Sm[("C", b - 1)]) if ("C", b) in Sm.index else np.nan for b in Y.t]
    # 내수·수출 출하지수
    h = os.path.join(BASE, r"공모전데이터\외부데이터\경기지표\kosis_내수수출출하지수.csv").replace("\\", "/")
    dx = duckdb.sql(f"select ITM_ID, C1, PRD_DE, try_cast(DT as double) v from read_csv_auto('{h}', all_varchar=true) "
                    f"where cast(PRD_DE as integer) <= {idx_maxy}12").df()
    dx["y"] = dx.PRD_DE.astype(int) // 100
    for itm, nm in [("T10", "g_dom"), ("T11", "g_exp")]:
        a = dx[dx.ITM_ID == itm].groupby(["C1", "y"]).v.agg(["sum", "size"]).reset_index()
        a = a[a["size"] == 12].set_index(["C1", "y"])["sum"]
        Y[nm] = [idx_g(a, kk, b, (3, 2)) for kk, b in zip(Y.ksic, Y.t)]
    # 무역: 한국 HS6 수출 → KSIC 5자리
    m = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str); m["hs6"] = m.hs6.str.zfill(6)
    xs = [pd.read_csv(os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\R2global_kr_export_hs6.csv"), dtype={"hs6": str})]
    for fn in glob.glob(os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\R2N2_kr_export_hs6_*.csv")):
        xs.append(pd.read_csv(fn, dtype={"hs6": str}))
    x = pd.concat(xs); x["hs6"] = x.hs6.str.zfill(6); x = x[x.year <= idx_maxy].drop_duplicates(["year", "hs6"])
    xk = x.merge(m[["k5", "hs6"]].drop_duplicates(), on="hs6")
    # 두 해 모두 보고된 HS6 만으로 성장(품목 구성 변화 제거)
    rows = []
    for (k5, t), g in xk.groupby(["k5", "year"]):
        rows.append((k5, t, dict(zip(g.hs6, g.value))))
    D = {(a, b): c for a, b, c in rows}
    gx = []
    for kk, t in zip(Y.ksic, Y.t):
        a, b = D.get((kk, t)), D.get((kk, t - 1))
        if a and b:
            com = set(a) & set(b); sa = sum(a[h] for h in com); sb = sum(b[h] for h in com)
            gx.append(np.log(sa / sb) if sa > 0 and sb > 0 else np.nan)
        else:
            gx.append(np.nan)
    Y["g_x_hs"] = gx
    Y["k3"] = Y.ksic.str[:3]
    return Y


if __name__ == "__main__":
    Y = build()
    assert Y.t.max() <= 2023
    Y.to_parquet(os.path.join(C, "panel_base.parquet"), index=False)
    d = Y[Y.t.between(2016, 2023)]
    print("rows", len(Y), "dev rows", len(d), "ksic", Y.ksic.nunique(), "3digit groups", Y.k3.nunique())
    print(d[["g", "g_idx", "g_idx_h2", "g_pidx", "g_inv", "g_idx2", "g_mfg", "g_dom", "g_exp", "g_x_hs", "g_lag"]].describe().round(3).to_string())
    print("corr with g (dev)", d[["g", "g_idx", "g_idx_h2", "g_idx_q4", "g_pidx", "g_inv", "g_idx2", "g_mfg", "g_dom", "g_exp", "g_x_hs", "g_lag"]].corr().g.round(3).to_dict())
    print("g_x_hs available by year", d.groupby("t").g_x_hs.apply(lambda s: s.notna().mean()).round(2).to_dict())
