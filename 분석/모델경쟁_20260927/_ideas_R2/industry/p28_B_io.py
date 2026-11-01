# -*- coding: utf-8 -*-
"""아이디어 B + 산업연관표(외부데이터 3종째): 전후방 산업의 과거 성장·특허 모멘텀 피처 추가 실험(챔피언 = p25 최종 6피처)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
exec(open("p25_B_final_pilot.py", encoding="utf-8").read().split("# 기준선")[0])
Ka = load_kosis_all("T04"); IOF = []
for t0 in sorted(P.t0.unique()):
    f = io_feats(t0, Ka, pv); f["t0"] = t0; IOF.append(f.reset_index())
IOF = pd.concat(IOF); IOF.to_parquet(f"{CACHE}/io_feats.parquet")
print("하류 커버리지(출하 관측 가중 비중) 평균", round(IOF.io_dn_cov.mean(), 3))
P = P.merge(IOF, on=["ksic", "t0"], how="left")
CH = ["mom_KR_L2", "dsh_KR_L2", "cnsh", "dcnsh", "lvl_KR", "frgn_mom"]
sC = score(preds(CH, data=P))
print(f"챔피언 ρ {sC.mean():.3f}")
for lab, add in [("+io_dn_g3(하류 성장)", ["io_dn_g3"]), ("+io_up_g3(상류 성장)", ["io_up_g3"]), ("+io_dn_pat(하류 특허)", ["io_dn_pat"]),
                 ("+io_up_pat(상류 특허)", ["io_up_pat"]), ("+하류 성장·특허", ["io_dn_g3", "io_dn_pat"])]:
    s = score(preds(CH+add, data=P)); d = s - sC
    print(f"{lab:22s} ρ {s.mean():.3f} Δ {d.mean():+.3f} 승률 {(d>0).mean():.2f} → {'ACCEPT' if d.mean()>=0.003 and (d>0).mean()>=0.6 else 'ROLLBACK'}")
