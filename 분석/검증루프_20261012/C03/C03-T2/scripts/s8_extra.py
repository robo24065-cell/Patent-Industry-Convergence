# -*- coding: utf-8 -*-
"""s8 (보고만): 3.6.4 우선권 KR 해외출원 RTA ↔ 수출 RCA ρ(0.545)의 대안별 값, 사분면 경계 취약성, 경계값 부트스트랩 안정도."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s5_sensitivity as S5
from lib import *
prof = S5.prof
ab = prof.kr_origin_abroad_rta_P3
out = {}
for lab in ["B_original", "A1_hs_align_equal", "P_A1c_continuity(사후)", "A2a_weight_KRpatent", "A3_alltrade_denominator", "P_A1c+A3(사후)"]:
    out[lab] = sp_ci(ab, S5.P2[lab])
# KSIC 부트스트랩: 원본 '기술 강·수출 약' 수의 분포는 정의상 고정이지만, RTA·RCA 경계 근접도
rta = S5.rta["P2"]; rca = S5.P2["B_original"]
d = pd.DataFrame({"rta": rta, "rca": rca}).dropna()
d["dist"] = np.minimum((np.log(d.rta)).abs(), (np.log(d.rca)).abs())
sw = d[(d.rta >= 1) & (d.rca < 1)]
out["SW_margin_log"] = sw.dist.round(3).sort_values().to_dict()
out["SW_within_10pct_of_boundary"] = sw.index[sw.dist < np.log(1.1)].tolist()
# 대안 전체에서 '기술 강·수출 약'에 늘 남는 KSIC
Q = S5.Q
always = [k for k in d.index if all(Q[v].get(k) == "SW" for v in Q)]
ever = [k for k in d.index if any(Q[v].get(k) == "SW" for v in Q)]
out["SW_in_all_variants"] = always; out["SW_in_any_variant"] = ever
json.dump(out, open(os.path.join(CACHE, "s8_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
