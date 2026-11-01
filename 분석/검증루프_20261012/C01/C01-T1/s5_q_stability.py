# -*- coding: utf-8 -*-
"""C01-T1 s5(사후 보조, PREREG 밖): 국가 고유 분야 커버리지 q(c,y,f)가 해마다 같은 모양인지(구조적) 확인.
q가 해마다 안정적이면 '수집 누락'보다 '분야 안 구성 차이(50 KSIC에 드는 하위기술 비중)'에 가깝다. 단, 이 검정만으로 둘을 완전히 구분하지는 못한다.
산출: q_stability.csv"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = os.path.dirname(os.path.abspath(__file__))
m = pd.read_parquet(os.path.join(OUT, "cache", "cov_r.parquet")); m = m[~m.small & (m.year >= 2015)]
k = m.groupby(["year", "f"]).r.transform("size"); m = m[k == 5].copy()
m["q"] = m.r / m.groupby(["year", "f"]).r.transform(lambda s: np.exp(np.log(s).mean()))
rows = []
for c in ["KR", "US", "JP", "CN", "EP"]:
    Q = m[m.cc == c].pivot(index="f", columns="year", values="q").dropna()
    rows.append(dict(cc=c, n_fields=len(Q), rho_q_2015_vs_2022=spearmanr(Q[2015], Q[2022]).statistic,
                     rho_q_2017_vs_2021=spearmanr(Q[2017], Q[2021]).statistic,
                     sd_logq_2017_19_mean=np.log(Q[[2017, 2018, 2019]]).std().mean(),
                     sd_logq_2017_19_excl_f16=np.log(Q.drop(16, errors="ignore")[[2017, 2018, 2019]]).std().mean()))
t = pd.DataFrame(rows).round(3); t.to_csv(os.path.join(OUT, "q_stability.csv"), index=False, encoding="utf-8-sig"); print(t.to_string())
