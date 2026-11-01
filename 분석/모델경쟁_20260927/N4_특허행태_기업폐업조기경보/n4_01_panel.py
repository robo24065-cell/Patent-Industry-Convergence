# -*- coding: utf-8 -*-
"""dev 기본 패널(M0+M1+레이블) t0=2015..2021 → cache/base_dev.parquet. 파일럿 패널과 행·레이블 일치 확인."""
import os, sys
import pandas as pd, numpy as np
from n4lib import *

MODE = "dev"
AF = load_app_firm()
AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
log("AF", len(AF))
R = load_registry(); log("registry")
ns = load_nts(); NPS = load_nps(MODE); A2K = a2key(); log("nts/nps")
rows = []
for t0 in range(TRAIN_MIN_T0, 2022):
    assert label_allowed(t0, MODE)
    f = build_base(t0, MODE, AF, R, ns, NPS, A2K)
    rows.append(f)
    log(t0, len(f), int(f.y.sum()), round(f.y.mean(), 4), "NPS", round(f.nps_present.mean(), 3))
D = pd.concat(rows, ignore_index=True)
D.to_parquet(os.path.join(CACHE, "base_dev.parquet"), index=False)
# 파일럿 대조
pp = os.path.join(MC, r"_ideas_R3\cache\I2b_panel.parquet")
if os.path.exists(pp):
    P = pd.read_parquet(pp, columns=["applicantCd", "t0", "y", "log_emp", "pay_1y"])
    m = D.merge(P, on=["applicantCd", "t0"], how="outer", suffixes=("", "_p"), indicator=True)
    print(m._merge.value_counts())
    b = m[m._merge == "both"]
    print("레이블 일치", (b.y == b.y_p).mean(), "log_emp 일치", np.allclose(b.log_emp.fillna(-9), b.log_emp_p.fillna(-9)),
          "pay_1y 일치", np.allclose(b.pay_1y, b.pay_1y_p))
