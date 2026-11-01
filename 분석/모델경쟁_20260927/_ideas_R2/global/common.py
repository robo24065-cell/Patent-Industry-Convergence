# -*- coding: utf-8 -*-
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
EXT = os.path.join(BASE, "분석", "모델경쟁_20260927", "_shared", "ext")
CACHE = os.path.join(HERE, "cache")

# 🔒 G1/G2 최종시험: t0=2021 → 2022–24 (F=2022). 개발 레이블은 ≤2021 → 무역 데이터 2022+ 로드 금지
TRADE_MAX_YEAR = 2021


def load_map():
    m = pd.read_csv(os.path.join(FUS, "ksic_hs6_map.csv"), encoding="utf-8-sig", dtype=str)
    m["hs6"] = m.hs6.str.zfill(6)
    return m[["k5", "hs6"]].drop_duplicates()


def load_imports():
    d = pd.read_csv(os.path.join(FUS, "hs6_import_yearly.csv"), encoding="utf-8-sig", dtype={"hs6": str})
    d = d[d.year <= TRADE_MAX_YEAR]
    d["hs6"] = d.hs6.str.zfill(6)
    return d


def load_exports():
    d = pd.read_csv(os.path.join(EXT, "R2global_kr_export_hs6.csv"), encoding="utf-8-sig", dtype={"hs6": str})
    d = d[d.year <= TRADE_MAX_YEAR]
    d["hs6"] = d.hs6.str.zfill(6)
    return d


def patent_feats(t0, pan=None):
    """KSIC별 특허 피처 — 최초공표연도 dy ≤ t0 인 건만 (원점 시점에 알 수 있던 정보).
    창 A = 공표 t0-2..t0, 창 B = t0-5..t0-3."""
    if pan is None:
        pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
    p = pan[(pan.dy <= t0) & (pan.dy >= t0 - 5)].copy()
    p["w"] = np.where(p.dy >= t0 - 2, "A", "B")
    out = {}
    def agg(mask, name):
        s = p[mask].groupby(["ksic", "w"]).n.sum().unstack(fill_value=0).reindex(columns=["A", "B"], fill_value=0)
        out[name + "_A"] = s["A"]; out[name + "_B"] = s["B"]
    for c in ["KR", "US", "JP", "EP", "CN"]:
        agg(p.cc == c, c)
    agg((p.cc == "KR") & (p.p1 == "KR"), "KRdom")            # 한국 출원 중 국내 우선권(국산 기술)
    agg((p.cc == "KR") & (p.p1 != "KR"), "KRin")             # 해외→한국 유입 출원
    agg((p.cc != "KR") & (p.p1 == "KR"), "KRout")            # 한국 기원 해외 출원(수출 의도)
    agg((p.cc != "KR") & (p.p1 == "CN"), "CNout")            # 중국 기원 해외 출원
    agg((p.cc != "KR") & (p.p1 != "KR"), "FOR")               # 해외 4개국 비한국 출원
    f = pd.DataFrame(out).fillna(0)
    r = pd.DataFrame(index=f.index)
    for k in ["KR", "US", "JP", "EP", "CN", "KRdom", "KRin", "KRout", "CNout", "FOR"]:
        r["pm_" + k] = np.log((f[k + "_A"] + 1) / (f[k + "_B"] + 1))
        r["pl_" + k] = np.log1p(f[k + "_A"])
    r["p_gap_for_kr"] = r.pm_FOR - r.pm_KR
    r["p_domshare"] = f.KRdom_A / (f.KR_A + 1)
    r["p_domshare_chg"] = r.p_domshare - f.KRdom_B / (f.KR_B + 1)
    r["p_outshare"] = f.KRout_A / (f.FOR_A + f.KRout_A + 1)     # 해외 4개국 출원 중 한국 기원 점유
    r["p_outshare_chg"] = r.p_outshare - f.KRout_B / (f.FOR_B + f.KRout_B + 1)
    r["p_krshare5"] = f.KR_A / (f[["KR_A", "US_A", "JP_A", "EP_A"]].sum(axis=1) + 1)
    r.index.name = "k5"
    return r.reset_index()


def trade_panel(d, partner=None, flow_col=None):
    """HS6 × 연도 wide (값, 결측 NaN)."""
    if partner is not None:
        d = d[d.partner == partner]
    return d.pivot_table(index="hs6", columns="year", values="value", aggfunc="sum")


def spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).correlation
