# -*- coding: utf-8 -*-
"""탐색(봉인 캐시 < 2023-01-01만): 사건명 분포, 분쟁 사건 사유, 청구항 사후값 누수 점검 재료."""
import pandas as pd, numpy as np
R4C = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_ideas_R4\cache"
rt = pd.read_parquet(f"{R4C}/reg_right_lt2023.parquet"); assert rt.d.max() < "20230101"
rt["y"] = rt.d.str[:4]
ev = rt[rt.cs.str.contains("예고|심판|정정|무효|권리범위", na=False)]
print(ev.cs.value_counts().head(40))
print(rt.cs.value_counts().head(40))
for nm in ["무효(심판)예고등록", "권리범위(심판)예고등록", "정정(심판)예고등록"]:
    s = rt[rt.cs == nm]
    print(nm, len(s)); print(s.y.value_counts().sort_index().tail(20).to_dict())
    print(s.rs.value_counts().head(10))
