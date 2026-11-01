# -*- coding: utf-8 -*-
"""C04-T1 s4: P1 보조 — N4 모집단 2년 폐업률(원점별) 대 공식 법인 2년 폐업률(폐업_{t+1}+폐업_{t+2})/가동_t."""
import os, pandas as pd
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T1"
N4 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N4_특허행태_기업폐업조기경보"
off = pd.read_csv(os.path.join(H, "cache", "official_bizstat_2825.csv"), index_col=0)
lr = pd.read_csv(os.path.join(N4, "link_rates_dev.csv"), encoding="utf-8-sig")[["t0", "n", "pos", "rate"]]
lr = pd.concat([lr, pd.DataFrame([dict(t0=2023, n=24840, pos=1109, rate=1109 / 24840)])], ignore_index=True)
lr["off_2y_rate"] = [(off.corp_close[t + 1] + off.corp_close[t + 2]) / off.corp_stock[t] for t in lr.t0]
lr["ratio_n4_over_off"] = lr.rate / lr.off_2y_rate
lr.to_csv(os.path.join(H, "cache", "s4_hazard.csv"), index=False, encoding="utf-8-sig")
print(lr.round(4).to_string())
