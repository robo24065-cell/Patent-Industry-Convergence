# -*- coding: utf-8 -*-
"""dev 패널 빌드: 원점 t0=2012..2020 (결과 반영 ≤2022 < F=2023). 인자: KSIC(기본 26429). 26422는 표본 확대 실험용(학습에만 추가)."""
import sys, time, os
import pandas as pd
from n5lib import *

t = time.time()
ks5 = sys.argv[1] if len(sys.argv) > 1 else KS_MAIN
k = load_kr(ks5); tlog(t, "KR", ks5, len(k))
ex = load_family(k, ks5); tlog(t, "family pairs", len(ex), "해외 선언 매칭", int(ex.frefl.notna().sum()))
E = load_ext(ks5); tlog(t, "ext", {kk: len(v) for kk, v in E.items()})
rows = []
for t0 in range(2012, 2021):
    U = build_panel(t0, ks5, "dev", k, ex, E)
    rows.append(U)
    tlog(t, t0, len(U), int(U.y.sum()), round(U.y.mean(), 4), "ros_sci", round(U.ros_sci.mean(), 3), "ntis", round(U.ntis_gov.mean(), 3),
         "oa_cov", round(U.oa_lv3.notna().mean(), 3), "etsi_map", round((U.etsi_co_tot > 0).mean(), 3))
D = pd.concat(rows, ignore_index=True)
D.to_parquet(os.path.join(CACHE, f"panel_dev_{ks5}.parquet"), index=False)
tlog(t, "saved", D.shape)
