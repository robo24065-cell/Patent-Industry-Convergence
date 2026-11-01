# -*- coding: utf-8 -*-
"""I6 진단: 양성이 몇 개 거래(날짜×양수인)에 몰리는가 — 사건 < 2023만."""
import re, pandas as pd, numpy as np
from r4_common import *
D = pd.read_parquet(f"{CACHE}/I6_panel.parquet")
ow = pd.read_parquet(f"{CACHE}/reg_owner_lt2023.parquet"); ow["d"] = pd.to_numeric(ow.d, errors="coerce")
for t0 in [2014, 2017, 2020]:
    end = t0 * 10000 + 1231; end2 = end + 20000
    pos = D[(D.t0 == t0) & (D.y_tr == 1)]
    o = ow[ow.rgst_no.isin(set(pos.rgst_no)) & (ow.d > end) & (ow.d <= end2)]
    deals = o.groupby(["d", "ownerName"]).rgst_no.nunique().sort_values(ascending=False)
    print(t0, "pos_tr", len(pos), "deals", len(deals), "top10 share", round(deals.head(10).sum() / len(pos), 3))
    print(deals.head(12).to_string())
    by_inst = pos.okey.value_counts(); print("inst with pos", len(by_inst), "top5 share", round(by_inst.head(5).sum() / len(pos), 3)); print(by_inst.head(6).to_string())
