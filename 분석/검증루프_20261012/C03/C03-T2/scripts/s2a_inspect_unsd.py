# -*- coding: utf-8 -*-
import os, sys
import pandas as pd
D = sys.argv[1]
for f in sorted(os.listdir(D)):
    if not f.endswith(".xlsx"): continue
    x = pd.ExcelFile(os.path.join(D, f))
    print("=====", f, x.sheet_names)
    for s in x.sheet_names:
        df = pd.read_excel(x, s, header=None, dtype=str, nrows=8)
        print("--", s, pd.read_excel(x, s, header=None, dtype=str).shape)
        print(df.to_string()[:1500])
