# -*- coding: utf-8 -*-
"""N2 대조·진단(보고서 핵심 표). 챔피언 고정, 한 요소씩만 바꿔 채점(decision=CONTROL).
+ 원점별 표, KSIC 군집 부트스트랩 CI, 실시간성(원본 보고서 연도) 분석, 연계 커버리지(특허배분 vs 업종코드)."""
import json, os
import numpy as np, pandas as pd
import n_lib as L
A = L.Arena()
f = A.champ["feats"]
K = {}
K["K1_nopatent"] = A.control("K1", "특허 브리지 제거(지수+수출만) — 특허배분 기여 확인", feats=[x for x in f if x != "gd_f"])
K["K2_induty"] = A.control("K2", "[필수 대조 E11] 특허배분 → DART 업종코드(스냅샷) 배분 단독", induty=True, feats=[("gd_induty_f" if x == "gd_f" else x) for x in f])
K["K3_snap"] = A.control("K3", "시점판 → 스냅샷판 가중(w_snap: right_list 현재 권리자 분모)", w="snap", conf_c=2)
K["K4_pitfull"] = A.control("K4", "최근 8년 구성 → 생존 전체 포트폴리오(w_pit)", w="pit")
K["K5_first"] = A.control("K5", "실시간 재무 → 파일럿 재무(fin_first, 이후 보고서 비교수치 포함=누수)", fin="first")
K["K6_ols"] = A.control("K6", "QR → OLS", model="ols")
K["K7_noconf"] = A.control("K7", "라벨 신뢰도 수축 제거", conf_c=None)
K["K8_noexport"] = A.control("K8", "HS6 수출 제거", feats=[x for x in f if x != "g_x_hs"])
K["K9_idx"] = A.control("K9", "g_pidx → g_idx(고정 기준선과 같은 지수)", feats=[("g_idx" if x == "g_pidx" else x) for x in f])
base = A.base
ci, pp = L.boot_ci(A.cP, base)
out = {"champion": A.champ_id, "spec": A.champ, "mean_skill": A.cm["mean_skill"], "mean_mae": A.cm["mean_mae"],
       "boot_ci_skill": ci.round(4).tolist(), "P_skill_gt0": pp,
       "per_origin": {t: {"mae_model": round(A.cm["mae"][t], 4), "mae_M1": round(L.mae(base)[t], 4), "skill": round(A.cm["skill"][t], 4)} for t in L.DEV},
       "rho": A.cm["rho"], "dir": A.cm["dir"], "medae": A.cm["medae"]}
for k, (m, P) in K.items():
    c2, p2 = L.boot_ci(P, base)
    out[k] = {"mean_skill": round(m["mean_skill"], 4), "ci": c2.round(4).tolist()}
# 특허배분 vs 업종코드 차이의 CI (챔피언 vs K2)
_, P2 = K["K2_induty"]
cd, pd_ = L.boot_ci(A.cP, P2)
out["champ_vs_induty_skill_ci"] = cd.round(4).tolist(); out["champ_vs_induty_P_gt0"] = pd_
cn, pn = L.boot_ci(A.cP, K["K1_nopatent"][1])
out["champ_vs_nopatent_skill_ci"] = cn.round(4).tolist(); out["champ_vs_nopatent_P_gt0"] = pn
# 실시간성: 파일럿 수집 보고서 연도(2018·2021·2023)와 N2 추가 수집 연도
rep = [2018, 2021, 2023]
out["skill_pilot_report_years"] = round(float(np.mean([A.cm["skill"][t] for t in rep])), 4)
out["skill_other_years"] = round(float(np.mean([A.cm["skill"][t] for t in L.DEV if t not in rep])), 4)
# 연계 커버리지
Y = L.build(A.champ); Yi = L.build({**A.champ, "induty": True})
d = Y[Y.t.isin(L.DEV)]; di = Yi[Yi.t.isin(L.DEV)]
out["coverage_patent"] = round(float(d.gd.notna().mean()), 3); out["nf_patent_median"] = float(d.nf_gd.median())
out["coverage_induty"] = round(float(di.gd_induty.notna().mean()), 3); out["nf_induty_median"] = float(di.nf_induty.fillna(0).median())
out["corr_g_gd"] = round(float(d[["g", "gd"]].corr().iloc[0, 1]), 3)
out["corr_g_gdinduty"] = round(float(di[["g", "gd_induty"]].corr().iloc[0, 1]), 3)
out["corr_g_gidx"] = round(float(d[["g", "g_idx"]].corr().iloc[0, 1]), 3)
json.dump(out, open(os.path.join(L.R2, "dev_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
# 세그먼트 진단: 산업별 평균 오차 개선
e = pd.concat([A.cP[t].assign(t=t) for t in L.DEV]); b = pd.concat([base[t].assign(t=t) for t in L.DEV])
e["err"] = (e.y - e.p).abs(); e["err_M1"] = (b.y - b.p).abs().values
seg = e.groupby("ksic")[["err", "err_M1"]].mean(); seg["gain"] = seg.err_M1 - seg.err
seg = seg.join(d.groupby("ksic").nf_gd.median().rename("nf_median")).sort_values("gain")
seg.round(4).to_csv(os.path.join(L.R2, "segment_by_ksic.csv"), encoding="utf-8-sig")
print(json.dumps({k: v for k, v in out.items() if k != "spec"}, ensure_ascii=False, indent=1, default=str))
print("KSIC with gain>0:", int((seg.gain > 0).sum()), "/", len(seg), "; corr(gain, log nf)", round(np.corrcoef(seg.gain, np.log1p(seg.nf_median))[0, 1], 3))
