# -*- coding: utf-8 -*-
"""아이디어 B: 특허 증분 검증 — 무역만(cnsh,dcnsh) 대비 특허 피처 추가 효과(원점별 Δρ, 승률, 산업 부트스트랩 CI)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
exec(open("p25_B_final_pilot.py", encoding="utf-8").read().split("# 기준선")[0])
T = ["cnsh", "dcnsh"]; oT = preds(T); sT = score(oT)
for lab, F in [("+특허 최종4(L2 모멘텀·점유변화·규모·해외)", T+["mom_KR_L2", "dsh_KR_L2", "lvl_KR", "frgn_mom"]),
               ("+mom_KR_L2", T+["mom_KR_L2"]), ("+dsh_KR_L2", T+["dsh_KR_L2"]), ("+mom_KR_L2,dsh_KR_L2", T+["mom_KR_L2", "dsh_KR_L2"]),
               ("+frgn_mom", T+["frgn_mom"]), ("+lvl_KR", T+["lvl_KR"])]:
    o = preds(F); s = score(o); d = s - sT
    rng = np.random.default_rng(1); ks = P.ksic.unique(); bd = []
    for b in range(500):
        samp = rng.choice(ks, len(ks), replace=True); a, c = [], []
        for v in VS:
            x = o[v].set_index("ksic"); z = oT[v].set_index("ksic")
            kk = [k for k in samp if k in x.index]
            a.append(spearmanr(x.loc[kk].y, x.loc[kk].p)[0]); c.append(spearmanr(z.loc[kk].y, z.loc[kk].p)[0])
        bd.append(np.nanmean(a)-np.nanmean(c))
    print(f"{lab:38s} ρ {s.mean():.3f} (무역만 {sT.mean():.3f}) Δ {d.mean():+.3f} 승률 {(d>0).mean():.2f} ΔCI {np.percentile(bd,[2.5,97.5]).round(3)}")
print("무역만 원점별", sT.round(3).to_dict())
