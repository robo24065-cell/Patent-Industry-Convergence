# -*- coding: utf-8 -*-
"""C01-T1 공용: 방식 A·B 문서 가중치 (s3_rta_sensitivity.py 와 동일 규칙의 사본, PREREG.txt)."""
import os
import numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(OUT, "cache")


def build_tables():
    m = pd.read_parquet(os.path.join(C, "cov_r.parquet"))
    tot = m.groupby(["cc", "year"])[["prov", "wipo"]].sum(); tot["r_tot"] = tot.prov / tot.wipo; tot = tot.reset_index()
    gt = tot[tot.year >= 2010].groupby("year").r_tot.apply(lambda s: np.exp(np.log(s).mean())).rename("g_tot")
    tot = tot.merge(gt, on="year", how="left"); tot["q_tot"] = tot.r_tot / tot.g_tot
    m = m.merge(tot[["cc", "year", "r_tot", "q_tot"]], on=["cc", "year"])
    m["rA"] = np.where(m.small | (m.r <= 0), m.r_tot, m.r)
    g = m.groupby(["year", "f"]).rA.apply(lambda s: np.exp(np.log(s.clip(lower=1e-6)).mean())).rename("g")
    m = m.merge(g, on=["year", "f"]); m["qB"] = np.where(m.small | (m.r <= 0), m.q_tot, m.r / m.g)
    m["wA"] = 1 / m.rA.clip(0.02, 1.5); m["wB"] = 1 / m.qB.clip(0.25, 4)
    W = m[["cc", "year", "f", "wA", "wB"]]
    T = tot[["cc", "year", "r_tot", "q_tot"]].copy(); T["wA_tot"] = 1 / T.r_tot.clip(0.02, 1.5); T["wB_tot"] = 1 / T.q_tot.clip(0.25, 4)
    return W, T


def weights(df, W, T):
    d = df.copy()
    d["wy"] = d.pub_year.clip(upper=2022)
    miss = d.pub_year.isna() | (d.pub_year < 2003)
    d.loc[miss, "wy"] = (d.loc[miss, "appl_year"] + 1).clip(2003, 2022)
    d["wy"] = d.wy.astype(int)
    d = d.merge(W.rename(columns={"year": "wy", "f": "wipo_field"}), on=["cc", "wy", "wipo_field"], how="left")
    d = d.merge(T[["cc", "year", "wA_tot", "wB_tot"]].rename(columns={"year": "wy"}), on=["cc", "wy"], how="left")
    miss = d.pub_year.isna() | (d.pub_year < 2003)
    use_tot = miss | (d.wipo_field == 0) | d.wA.isna()
    d.loc[use_tot, "wA"] = d.loc[use_tot, "wA_tot"]; d.loc[use_tot, "wB"] = d.loc[use_tot, "wB_tot"]
    d["nA"] = d.n * d.wA; d["nB"] = d.n * d.wB
    return d
