# -*- coding: utf-8 -*-
"""R5 레드팀(N4) 2단계: final_run_r4.py와 같은 순서로 final 모드 패널(t0 2015–2021 + 2023)을 새 캐시로 만든다."""
import sys, os, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import n4lib
n4lib.CACHE = os.path.join(HERE, "cache")
from n4lib import load_app_firm, load_registry, load_nts, load_nps, a2key, build_base, TRAIN_MIN_T0, FINAL_T0, log
from n4feat import load_ctx, build_all
import pandas as pd
MODE = "final"
T0S = list(range(TRAIN_MIN_T0, 2022)) + [FINAL_T0]
AF = load_app_firm(); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
R = load_registry(); ns = load_nts(); NPS = load_nps(MODE); A2K = a2key()
rows = []
for t0 in T0S:
    B = build_base(t0, MODE, AF, R, ns, NPS, A2K); rows.append(B); log("base", t0, len(B), B.y.sum())
D = pd.concat(rows, ignore_index=True)
ctx = load_ctx(MODE, set(D.applicantCd))
X = pd.concat([build_all(B, t0, ctx, AF, ns) for t0, B in D.groupby("t0")]).reindex(D.index)
D = pd.concat([D, X], axis=1)
D.to_parquet(os.path.join(HERE, "cache", "final_panel.parquet"), index=False)
log("panel", D.shape)
