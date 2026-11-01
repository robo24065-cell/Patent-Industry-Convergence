# -*- coding: utf-8 -*-
"""K1·K2 패널에서 wo_vis=1 단위만 뽑아 K1P·K2P 패널을 만든다(피처·레이블은 그대로, 🔒 봉인 재확인)."""
import os, pyarrow.parquet as pq
from r6_common import CACHE
for a, b, oy in (("K1", "K1P", "fy"), ("K2", "K2P", "ky")):
    T = pq.read_table(os.path.join(CACHE, f"{a}_panel.parquet"), filters=[("wo_vis", "=", 1)]).to_pandas()
    assert T.t0.max() <= 2020 and T.loc[T.y == 1, oy].max() <= 2022
    T.to_parquet(os.path.join(CACHE, f"{b}_panel.parquet"), index=False)
    print(b, len(T), T.groupby("t0").y.agg(["size", "sum", "mean"]).round(3).T.to_string())
