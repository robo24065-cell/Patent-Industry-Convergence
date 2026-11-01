# -*- coding: utf-8 -*-
"""모듈 D(보조 타깃): KSIC 48개 한국 수출 3년 합 전후 로그비 순위 예측. 원점 t0=2005..2018, 평가 v=2008..2018, 학습 t0 ≤ v−3.
yD = log(Σ수출[t0+1..t0+3] / Σ수출[t0−2..t0]) − 원점 중앙값 (수출 ≤2021만 로드). 기준선 = 과거 3년 수출성장 순위(적합 없음).
실험 기록: R2/experiments_D.csv, 챔피언 R2/champion_D.json"""
import sys
from engine import *
X = kr_export_ksic()   # ≤2021
rows = []
for t0 in range(2005, 2019):
    fut = X[[t0+1, t0+2, t0+3]].sum(1); now = X[[t0-2, t0-1, t0]].sum(1); past = X[[t0-5, t0-4, t0-3]].sum(1)
    d = pd.DataFrame({"ksic": X.index, "t0": t0, "yD": np.log(fut/now).values, "momX": np.log(now/past).values, "xlvD": np.log(now).values})
    rows.append(d)
D = pd.concat(rows); D["yD"] = D.yD - D.groupby("t0").yD.transform("median")
PD = D.merge(P_ALL.drop(columns=[c for c in P_ALL.columns if c.startswith("y_")]), on=["ksic", "t0"], how="left")
PD = PD[PD.ksic.isin(cn_share().ksic.unique())]   # 48개(수출·수입 연결 산업)
VD = list(range(2008, 2019)); EXPD = f"{R2}/experiments_D.csv"; CHD = f"{R2}/champion_D.json"
def b0D():
    return {v: pd.DataFrame({"ksic": g.ksic.values, "p": g.momX.rank(pct=True).fillna(.5).values, "y": g.yD.values}) for v, g in PD[PD.t0.isin(VD)].groupby("t0")}
oB = b0D(); BASE_D = (score(oB), auc(oB), oB)
def pD(F, alpha=5.0): return preds(F, alpha=alpha, tg="yD", data=PD, vs=VD)
def chal(eid, change, F2, alpha=None, note=""):
    st = champ_state(CHD); sC = pd.Series({int(k): v for k, v in st["rho"].items()}); a = alpha or st["alpha"]
    acc, s = log_exp(eid, st["id"], change, pD(F2, a), champ=sC, note=note, file=EXPD, base=BASE_D)
    if acc: save_champ({"id": eid, "F": F2, "alpha": a, "rho": s.round(5).to_dict()}, CHD)
if __name__ == "__main__":
    step = sys.argv[1]
    if step == "base":
        print("행", len(PD), "원점당", PD.groupby("t0").yD.count().to_dict())
        log_exp("DB0", "-", "고정 기준선: 과거 3년 수출성장 순위", oB, note="파일럿 ρ 0.311", file=EXPD, base=BASE_D)
        log_exp("DD0", "-", "참고: 무역만(cnsh, dcnsh, 과거 수출성장)", pD(["cnsh", "dcnsh", "momX"]), note="파일럿 ρ 0.377", file=EXPD, base=BASE_D)
        log_exp("DM0", "-", "참고: 특허만(KR L2 모멘텀·점유변화, 규모, 해외 모멘텀)", pD(["mom_KR_L2", "dsh_KR_L2", "lvl_KR", "frgn_mom"]), file=EXPD, base=BASE_D)
        F = ["cnsh", "dcnsh", "momX", "frgn_mom", "xlvD"]
        _, s = log_exp("DC0", "-", "출발 챔피언 = 파일럿 D6 재채점(무역3 + frgn_mom + 수출규모)", pD(F), note="파일럿 ρ 0.390", file=EXPD, base=BASE_D)
        save_champ({"id": "DC0", "F": F, "alpha": 5.0, "rho": s.round(5).to_dict()}, CHD)
    elif step == "ch1":
        st = champ_state(CHD); F = st["F"]
        chal("D14", "+dffam_pit_Q2(원점 시점 해외패밀리 비중 변화)", F + ["dffam_pit_Q2"], note="주 타깃 채택 피처 이식")
        F = champ_state(CHD)["F"]; chal("D15", "+pi_g3(생산지수 3자리 3년 성장)", F + ["pi_g3"], note="주 타깃 채택 피처 이식")
        F = champ_state(CHD)["F"]; chal("D16", "+dsh_KR_L2(KR 특허 점유변화, 시차2)", F + ["dsh_KR_L2"], note="파일럿 D1(시차1) +0.004 롤백 → 시차2 재시도")
        F = champ_state(CHD)["F"]; chal("D17", "Ridge α 5→2", F, alpha=2.0, note="주 타깃에서 채택된 변경")
        F = champ_state(CHD)["F"]; chal("D18", "+kab_mom(한국발 해외출원 모멘텀, 시차3)", F + ["kab_mom"], note="파일럿 D10 롤백, 새 챔피언에서 재도전")
        F = champ_state(CHD)["F"]; chal("D19", "+io_exr(산업연관표 수출/총산출)", F + ["io_exr"], note="외부: 산업연관표")
        F = champ_state(CHD)["F"]; chal("D20", "+ap_ui(대학·연구기관 출원 비중)", F + ["ap_ui"], note="특허 출원인 구성")
        F = champ_state(CHD)["F"]; chal("D21", "+cninv_mom_L2(CN 발명특허 모멘텀)", F + ["cninv_mom_L2"], note="중국 기술 추격")
        F = champ_state(CHD)["F"]; chal("D22", "+g3_shp(KOSIS 출하 3년 성장)", F + ["g3_shp"], note="외부: 광업제조업조사")
        print("D 챔피언", champ_state(CHD))
    elif step == "ch2":
        F = champ_state(CHD)["F"]; chal("D23", "+mom_KR_L2(KR 특허 모멘텀, 시차2)", F + ["mom_KR_L2"], note="특허 추가")
        F = champ_state(CHD)["F"]; chal("D24", "+tsi(무역특화지수 수준)", F + ["tsi"], note="외부: 무역(수출·수입)")
        F = champ_state(CHD)["F"]; chal("D25", "+dkrcn_lr_L2(KR/CN 특허 상대규모 이동)", F + ["dkrcn_lr_L2"], note="경쟁국 격차")
        F = champ_state(CHD)["F"]; chal("D26", "+pi_g1(생산지수 1년 성장)", F + ["pi_g1"], note="외부: 생산지수 최근 모멘텀")
        print("D 챔피언", champ_state(CHD))
    elif step == "inc":
        st = champ_state(CHD); F = st["F"]; oF = pD(F, st["alpha"]); oT = pD([c for c in F if not is_patent(c)], st["alpha"])
        r = boot(oF, oT); rb = boot(oF, oB); d = score(oF) - score(oT)
        out = {"champ": st["id"], "F": F, "rho": round(score(oF).mean(), 4), "ci": r["ci"], "rho_noPatent": round(score(oT).mean(), 4), "pat_inc": round(d.mean(), 4),
               "pat_inc_win": round((d > 0).mean(), 3), "pat_inc_ci": r["dci"], "p_le0": r["p_le0"], "skill_vs_B0": round(score(oF).mean() - score(oB).mean(), 4), "skill_ci": rb["dci"],
               "auc": round(auc(oF).mean(), 4), "auc_B0": round(auc(oB).mean(), 4), "per_origin": score(oF).round(3).to_dict(), "per_origin_B0": score(oB).round(3).to_dict()}
        print(out); json.dump(out, open(f"{R2}/increment_D.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
