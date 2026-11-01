# -*- coding: utf-8 -*-
"""NPS 연말 스냅샷(2015-12~2023-12만) → 기업키(biz6+정규화상호) 단위 집계. 2024-12 이후는 읽지 않는다(봉인)."""
import os, time
import pandas as pd, numpy as np
from i00_common import *

t = time.time()
C = ["name", "biz6", "stat", "ldong", "indcd", "indnm", "adpt_dt", "n_emp", "amt", "n_new", "n_loss"]
for s in SNAPS_OK:
    fp = os.path.join(EXT, f"firm_nps_wkpl_{s.replace('-', '')}.parquet")
    d = pd.read_parquet(fp, columns=C)
    d["nname"] = d.name.map(norm)
    d["key"] = d.biz6.fillna("") + "|" + d.nname
    d["act"] = d.stat == "1"
    d["mfg"] = d.indnm.fillna("").str.contains("제조")
    d = d.sort_values("n_emp", ascending=False)
    big = d.drop_duplicates("key")[["key", "biz6", "nname", "ldong", "indcd", "mfg"]]
    g = d.groupby("key").agg(n_emp=("n_emp", "sum"), n_wkpl=("name", "size"), amt=("amt", "sum"), n_new=("n_new", "sum"),
                             n_loss=("n_loss", "sum"), active=("act", "any"), adpt=("adpt_dt", "min")).reset_index()
    g = g.merge(big, on="key")
    g["sgg"] = g.ldong.fillna("").str[:5]
    g["snap"] = s
    g.to_parquet(os.path.join(CACHE, f"nps_firm_{s}.parquet"), index=False)
    print(s, len(g), "active", int(g.active.sum()), "mfg active", int((g.active & g.mfg).sum()), round(time.time() - t), flush=True)
