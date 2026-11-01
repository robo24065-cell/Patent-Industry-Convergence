# -*- coding: utf-8 -*-
# 탐색: WIPO 35분야별 특허 심판청구(전 종류) 연도 비교(2016–2025 자료연도). 출력 cache/wipo_field_trials.csv
import sys, glob, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = {"f_267_2": 2017, "f_283_2": 2018, "f_312_2": 2019, "f_328_2": 2020, "f_360_1": 2021, "f_380_2": 2022, "f_400_2": 2023, "f_416_2": 2024, "f_437_1": 2025}
rows = []
for tag, y in PUB.items():
    for f in glob.glob(os.path.join(W, "download", "moip_unzip", tag, "*")):
        if "심판" not in os.path.basename(f): continue
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            if "WIPO" not in sh: continue
            df = xl.parse(sh, header=None)
            for i in range(len(df)):
                lab = str(df.iat[i, 1]).split("\n")[0].strip() if pd.notna(df.iat[i, 1]) else ""
                v = pd.to_numeric(df.iloc[i, 2:8], errors="coerce").values
                if lab and not np.isnan(v[2]) and "소계" not in lab:
                    rows.append(dict(year=y, field=lab, pat_total=v[2]))
D = pd.DataFrame(rows).drop_duplicates(["year", "field"])
P = D.pivot_table(index="field", columns="year", values="pat_total")
P.to_csv(os.path.join(W, "cache", "wipo_field_trials.csv"), encoding="utf-8-sig")
P["d23"] = P[2023] - (P[2022] + P[2024]) / 2
print(P.sort_values("d23", ascending=False).head(8).to_string())
print(P.sum().to_string())
