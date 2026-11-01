# -*- coding: utf-8 -*-
"""1단계: 고정 기준선·참고 기준선·출발 챔피언(파일럿 M7) 재채점 + M14 재도전 + 특허 증분."""
from engine import *
o0 = b0(); log_exp("B0", "-", "고정 기준선: 과거3년 종사자성장(t0-3→t0) 원점 내 순위(적합 없음)", o0, note="파일럿 ρ 0.151 AUC 0.578")
log_exp("B1", "-", "참고: 무역만(cnsh, dcnsh) 순위 Ridge", preds(["cnsh", "dcnsh"]), note="파일럿 ρ 0.320")
log_exp("M0", "-", "참고: 특허만(KR 모멘텀+KR 점유변화, 시차1)", preds(["mom_KR", "dsh_KR"]), note="파일럿 ρ 0.186")
log_exp("SELF", "-", "참고: 자기이력 Ridge(g3·g1·lv 종사자, 노동생산성)", preds(["g3_emp", "g1_emp", "lv_emp", "prod"]), note="파일럿 ρ -0.098")
M7 = ["mom_KR_L2", "dsh_KR_L2", "cnsh", "dcnsh", "lvl_KR", "frgn_mom"]
oC = preds(M7); _, sC = log_exp("C0", "-", "출발 챔피언 = 파일럿 M7 재채점(6피처, 순위 Ridge α=5)", oC, note="파일럿 ρ 0.346")
save_champ({"id": "C0", "F": M7, "alpha": 5.0, "train_tg": None, "rho": sC.round(5).to_dict()})
acc, s14 = log_exp("E01", "C0", "+kab_mom(한국발 해외출원 모멘텀, 시차3) — 파일럿 M14 재도전", preds(M7+["kab_mom"]), champ=sC, note="파일럿 경계선 채택 Δ+0.004 승률 62%")
if acc: save_champ({"id": "E01", "F": M7+["kab_mom"], "alpha": 5.0, "train_tg": None, "rho": s14.round(5).to_dict()})
# 특허 증분(무역만 대비)
st = champ_state(); F = st["F"]; oF = preds(F); oT = preds([c for c in F if not is_patent(c)])
r = boot(oF, oT); rb = boot(oF, o0)
d = score(oF) - score(oT)
print("챔피언", st["id"], F, "ρ", round(score(oF).mean(), 4), "CI", r["ci"], "| vs B0 ΔCI", rb["dci"])
print("특허 증분(동일 모형 − 특허 제거):", round(d.mean(), 4), "승률", round((d > 0).mean(), 2), "ΔCI", r["dci"], "P(Δ≤0)", round(r["p_le0"], 3))
json.dump({"champ": st["id"], "F": F, "rho": round(score(oF).mean(), 4), "ci": r["ci"], "vsB0_dci": rb["dci"], "pat_inc": round(d.mean(), 4),
           "pat_inc_win": round((d > 0).mean(), 3), "pat_inc_ci": r["dci"], "p_le0": r["p_le0"]}, open(f"{R2}/increment_E01_from_s01.json", "w"), ensure_ascii=False)
