# -*- coding: utf-8 -*-
"""N3 기술-수출 괴리 조기경보 — 공용 로더·피처 빌더.
파일럿(_ideas_R2/global/common.py)을 복사·확장. 파일럿 폴더는 수정하지 않는다.
🔒 최종시험 t0=2021 → 결과 2022–24 (F=2022). 무역 로더는 TRADE_MAX_YEAR=2021로 로드 직후 필터.
   KOSIS는 ≤2021만 로드. 특허는 최초공표연도 dy ≤ t0만.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
EXT = os.path.join(BASE, "분석", "모델경쟁_20260927", "_shared", "ext")
IO_DIR = os.path.join(BASE, "공모전데이터", "외부데이터", "산업연관표")
KOSIS_F = os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_2008_2024.csv")
CACHE = os.path.join(HERE, "cache")
R2 = os.path.join(HERE, "R2")
os.makedirs(CACHE, exist_ok=True); os.makedirs(R2, exist_ok=True)

TRADE_MAX_YEAR = 2021     # 개발·학습 레이블 ≤2021, 최종시험 피처 ≤2021
KOSIS_MAX_YEAR = 2021
COUNTRIES = {410: "KR", 156: "CN", 392: "JP", 842: "US", 276: "DE"}


# ---------------------------------------------------------------- 무역
def load_map():
    m = pd.read_csv(os.path.join(FUS, "ksic_hs6_map.csv"), encoding="utf-8-sig", dtype=str)
    m["hs6"] = m.hs6.str.zfill(6)
    return m[["io", "k5", "hs6"]].drop_duplicates(["k5", "hs6"])


def load_imports(max_year=TRADE_MAX_YEAR):
    d = pd.read_csv(os.path.join(FUS, "hs6_import_yearly.csv"), encoding="utf-8-sig", dtype={"hs6": str})
    d = d[d.year <= max_year]
    d["hs6"] = d.hs6.str.zfill(6)
    return d


KR_EXPORT_FINAL = os.path.join(EXT, "R2N3_kr_export_hs6_2022_2024.csv")   # 🔒 최종시험 때 1회 수집(final_run_r2.py)


def load_exports(max_year=TRADE_MAX_YEAR):
    d = pd.read_csv(os.path.join(EXT, "R2global_kr_export_hs6.csv"), encoding="utf-8-sig", dtype={"hs6": str})
    d = d[d.year <= TRADE_MAX_YEAR]
    if max_year > TRADE_MAX_YEAR:   # final/future 모드에서만
        e = pd.read_csv(KR_EXPORT_FINAL, dtype={"hs6": str})
        d = pd.concat([d, e[e.year > TRADE_MAX_YEAR][["year", "flow", "hs6", "value"]]], ignore_index=True)
    d = d[d.year <= max_year]
    d["hs6"] = d.hs6.str.zfill(6)
    return d


def load_5c_exports(max_year=TRADE_MAX_YEAR):
    """5개국(KR·CN·JP·US·DE) HS6 대세계 수출. R2ind 파일 + N3 보충 수집(R2N3_). 연도별로 완결된 연도만 반환."""
    a = pd.read_csv(os.path.join(EXT, "R2ind_comtrade_exports_5c_hs6_2004_2021.csv"), dtype={"hs6": str})
    a = a[["year", "reporter", "hs6", "export_usd"]]
    f = os.path.join(EXT, "R2N3_comtrade_exports_5c_hs6.csv")
    if os.path.exists(f):
        b = pd.read_csv(f, dtype={"hs6": str})[["year", "reporter", "hs6", "export_usd"]]
        a = pd.concat([a, b], ignore_index=True)
    a["hs6"] = a.hs6.str.zfill(6)
    a = a[a.year <= max_year].drop_duplicates(["year", "reporter", "hs6"])
    a["cc"] = a.reporter.map(COUNTRIES)
    # 완결 연도만: 청크 11개 모두 수집된 연도(행 수 기준 ≥ 4,000 = 5개국×~800)
    cnt = a.groupby("year").size()
    good = cnt[cnt >= 4000].index
    return a[a.year.isin(good)]


def trade_panel(d, partner=None):
    if partner is not None:
        d = d[d.partner == partner]
    return d.pivot_table(index="hs6", columns="year", values="value", aggfunc="sum")


# ---------------------------------------------------------------- 특허
def load_pat_panel():
    return pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))


def patent_feats(t0, pan):
    """KSIC별 특허 피처 — 최초공표연도 dy ≤ t0 (원점 시점 정보). 창 A = t0-2..t0, 창 B = t0-5..t0-3. (파일럿과 동일)"""
    p = pan[(pan.dy <= t0) & (pan.dy >= t0 - 5)].copy()
    p["w"] = np.where(p.dy >= t0 - 2, "A", "B")
    out = {}

    def agg(mask, name):
        s = p[mask].groupby(["ksic", "w"]).n.sum().unstack(fill_value=0).reindex(columns=["A", "B"], fill_value=0)
        out[name + "_A"] = s["A"]; out[name + "_B"] = s["B"]
    for c in ["KR", "US", "JP", "EP", "CN"]:
        agg(p.cc == c, c)
    agg((p.cc == "KR") & (p.p1 == "KR"), "KRdom")
    agg((p.cc == "KR") & (p.p1 != "KR"), "KRin")
    agg((p.cc != "KR") & (p.p1 == "KR"), "KRout")
    agg((p.cc != "KR") & (p.p1 == "CN"), "CNout")
    agg((p.cc != "KR") & (p.p1 != "KR"), "FOR")
    agg((p.cc != "KR") & (p.p1 == "JP"), "JPout")
    agg((p.cc != "KR") & (p.p1 == "US"), "USout")
    agg((p.cc != "KR") & (p.p1 == "EU"), "EUout")
    f = pd.DataFrame(out).fillna(0)
    r = pd.DataFrame(index=f.index)
    for k in ["KR", "US", "JP", "EP", "CN", "KRdom", "KRin", "KRout", "CNout", "FOR"]:
        r["pm_" + k] = np.log((f[k + "_A"] + 1) / (f[k + "_B"] + 1))
        r["pl_" + k] = np.log1p(f[k + "_A"])
    r["p_gap_for_kr"] = r.pm_FOR - r.pm_KR
    r["p_domshare"] = f.KRdom_A / (f.KR_A + 1)
    r["p_domshare_chg"] = r.p_domshare - f.KRdom_B / (f.KR_B + 1)
    r["p_outshare"] = f.KRout_A / (f.FOR_A + f.KRout_A + 1)
    r["p_outshare_chg"] = r.p_outshare - f.KRout_B / (f.FOR_B + f.KRout_B + 1)
    r["p_krshare5"] = f.KR_A / (f[["KR_A", "US_A", "JP_A", "EP_A"]].sum(axis=1) + 1)
    # 기원국별 해외출원(비한국 4개국 관청) 점유 — 수출 점유와 짝지을 '기술 점유'
    tot = f.FOR_A + f.KRout_A + 1
    for k in ["KRout", "CNout", "JPout", "USout", "EUout"]:
        r["q_" + k] = f[k + "_A"] / tot
        r["q_" + k + "_B"] = f[k + "_B"] / (f.FOR_B + f.KRout_B + 1)
    r["n_tot_A"] = f[[c + "_A" for c in ["KR", "US", "JP", "EP", "CN"]]].sum(axis=1)
    r.index.name = "k5"
    return r.reset_index()


# ---------------------------------------------------------------- KOSIS 광업제조업조사 (KSIC 5자리)
def load_kosis(max_year=KOSIS_MAX_YEAR):
    """10차(≤2020) 기준 연속 시계열, 2021+는 11차 값 × (10차2020/11차2020) 연결. 2010은 경제총조사로 결측. 기본 ≤2021만."""
    d = pd.read_csv(KOSIS_F, dtype={"세세분류코드": str})
    d = d[d.연도 <= max_year]
    items = {"T01": "est", "T02": "emp", "T04": "ship", "T06": "va", "T07": "cap"}
    d = d[d.항목코드.isin(items)]
    d["it"] = d.항목코드.map(items)
    a = d[d.기준 == "10차"].pivot_table(index=["세세분류코드", "it"], columns="연도", values="값")
    b = d[d.기준 == "11차"].pivot_table(index=["세세분류코드", "it"], columns="연도", values="값")
    if 2020 in b.columns:
        link = (a[2020] / b[2020]).replace([np.inf, -np.inf], np.nan)
        for y in [y for y in b.columns if y > 2020]:
            a[y] = (b[y] * link.reindex(b.index)).reindex(a.index)
    a = a.where(a > 0)
    return a  # index (k5, it), columns year


def kosis_feats(t0, K):
    """원점 t0 시점: 최신 관측연도 y1 ≤ t0, 3년 전 y0 ≤ y1-3(관측 있는 가장 가까운 해)."""
    yrs = [y for y in K.columns if y <= t0]
    if not yrs:
        return None
    y1 = max(yrs)
    y0s = [y for y in K.columns if y <= y1 - 3]
    out = pd.DataFrame(index=K.index.get_level_values(0).unique())
    for it in ["ship", "emp", "va", "est", "cap"]:
        s1 = K.xs(it, level=1)[y1]
        out["ks_l_" + it] = np.log(s1)
        if y0s:
            y0 = max(y0s)
            s0 = K.xs(it, level=1)[y0]
            out["ks_m_" + it] = np.log(s1 / s0) * 3.0 / (y1 - y0)
    out["ks_va_ratio"] = out.ks_l_va - out.ks_l_ship
    out["ks_ship_per_emp"] = out.ks_l_ship - out.ks_l_emp
    out["ks_cap_int"] = out.ks_l_cap - out.ks_l_ship
    out.index.name = "k5"
    return out.reset_index()


# ---------------------------------------------------------------- 산업연관표 (2010 접속표, 통합소분류 165)
def load_io():
    """반환: Z(165×165 중간거래, 행=공급 상품, 열=수요 산업), 산출 export 열, KSIC5→소분류 대응(가중)."""
    x = os.path.join(IO_DIR, "2010-2015-2020_생산자가격(경상)_통합소분류_거래표 및 각종 계수표.xlsx")
    t = pd.read_excel(x, "10-A표_총거래표(생산자)", header=None)
    codes = t.iloc[6:171, 0].astype(str).str.strip().tolist()
    Z = t.iloc[6:171, 2:167].astype(float).values
    Z = pd.DataFrame(Z, index=codes, columns=codes)
    hdr = t.iloc[4].astype(str).str.strip().tolist()
    tot_out = t.iloc[6:171, [i for i, h in enumerate(hdr) if h in ("9140",)][0]].astype(float).values
    exp_col = pd.Series(tot_out, index=codes, name="io_export")
    # 기본부문 → 소분류
    c = pd.read_excel(os.path.join(IO_DIR, "2010-2015-2020_부문분류표.xlsx"), header=None, skiprows=3)
    c = c[[0, 2]].copy(); c.columns = ["basic", "sub"]
    c["sub"] = c["sub"].ffill()
    c = c.dropna(subset=["basic"])
    c["basic"] = c.basic.astype(str).str.strip().str.zfill(4)
    c["sub"] = c["sub"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(3)
    # 기본부문 → KSIC
    k = pd.read_excel(os.path.join(IO_DIR, "IO코드와 KSIC 및 HS코드 대조표.xlsx"), "IO코드와 KSIC 비교표", header=None, skiprows=2)
    k = k[[0, 2]].copy(); k.columns = ["basic", "k5"]
    k["basic"] = k.basic.ffill().astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(4)
    k = k.dropna(subset=["k5"])
    k["k5"] = k.k5.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(5).str[:5]
    ks = k.merge(c, on="basic", how="left")
    ks = ks.dropna(subset=["sub"]).drop_duplicates(["k5", "sub"])
    ks["w"] = 1.0 / ks.groupby("k5").sub.transform("count")
    return Z, exp_col, ks[["k5", "sub", "w"]]


# ---------------------------------------------------------------- 대중 수입점유 (KSIC5)
def load_cn_ksic():
    d = pd.read_csv(os.path.join(FUS, "cn_import_yearly_ksic5.csv"), dtype={"ksic": str})
    d = d[(d.year <= TRADE_MAX_YEAR) & (d.ksic_level == 5)]
    d["k5"] = d.ksic.str.zfill(5)
    return d.pivot_table(index="k5", columns="year", values="cn_share_imp") / 100.0


def spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).correlation
