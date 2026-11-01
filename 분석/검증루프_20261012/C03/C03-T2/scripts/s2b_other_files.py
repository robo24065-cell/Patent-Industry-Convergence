# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
mp = load_map(); codes = set(mp.hs6)
for f in ["trade_kr_export_hs6_uv_2000_2021.csv", "R2global_kr_export_hs6.csv", "R2N2_kr_export_hs6_2022_2023.csv", "R2N2_kr_export_hs6_2024.csv",
          "R2N3_kr_export_hs6_2022_2024.csv", "N1final_kr_export_hs6_2019_2024.csv"]:
    d = pd.read_csv(os.path.join(EXT, f), dtype={"hs6": str})
    d["hs6"] = d.hs6.str.zfill(6)
    yc = "year" if "year" in d.columns else d.columns[0]
    print("==", f, d.shape, d.columns.tolist())
    print("  years", sorted(d[yc].unique())[:30])
    print("  codes", d.hs6.nunique(), "in map", d.hs6.isin(codes).mean().round(4), "map codes covered", len(set(d.hs6) & codes))
    print(d.head(2).to_string())
mp2 = mp.copy()
print("map rows", len(mp2), "k5", mp2.k5.nunique(), "hs6", mp2.hs6.nunique(), "io", mp2.io.nunique(), "dup k5-hs6", mp2.duplicated(["k5", "hs6"]).sum())
nk = mp2.drop_duplicates(["k5", "hs6"]).groupby("hs6").k5.nunique()
print("hs6 by n ksic", nk.value_counts().sort_index().to_dict())
print(mp2.groupby("io").k5.nunique().value_counts().to_dict())
prof = pd.read_csv(os.path.join(MAP5, "m2_ksic_profile.csv"), dtype={"ksic": str})
print("profile ksic", len(prof), "not in map", sorted(set(prof.ksic) - set(mp2.k5)))
print("map k5 not in profile", sorted(set(mp2.k5) - set(prof.ksic)))
