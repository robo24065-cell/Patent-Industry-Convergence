# -*- coding: utf-8 -*-
"""C12-T1 s7: P0 재현(a 레이블·b 점수·c 재적합), 보조 S1a·S2a(동결 예측), 시나리오 판정 → sens_results.csv, cache/s7_out.json"""
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import n5lib
from n5lib import auc, skill, cluster_boot_ci
TR = os.path.join(n5lib.MC, "N5_표준특허선언레이더")
O = {}
OFF = json.load(open(os.path.join(TR, "final_result_r4.json"), encoding="utf-8"))
FP = pd.read_parquet(os.path.join(TR, "final_preds_r4.parquet"))
OFF_SK, OFF_CI = OFF["skill_champ_vs_M0"], OFF["ci95_cluster_app"]
OFF_ST, OFF_STCI = 0.00019, [-0.0721, 0.0701]
# ---------------- P0-a 레이블: 이 작업 독립 파서(엄격 키) 최초 반영일로 시험 단위·레이블
k = pd.read_parquet(os.path.join(W, "cache", "n5", "kr_26429.parquet"))
M = pd.read_parquet(os.path.join(W, "cache", "link_matches.parquet"), columns=["appl_no", "refl", "strict", "lvl"])
rs = M[M.strict == 1].groupby("appl_no").refl.min(); rf = M[M.lvl <= 2].groupby("appl_no").refl.min()
k["refl_c12"] = k.appl_no.map(rs).astype(float)
O["P0a_refl_26429_trade_vs_c12strict"] = dict(n_declared_trade=int(k.refl.notna().sum()), n_declared_c12=int(k.refl_c12.notna().sum()),
                                              diff=int((k.refl.fillna(0) != k.refl_c12.fillna(0)).sum()))
end = 20221231


def units(refl):
    U = k[(k.known <= end) & k.ay.between(2017, 2022) & ~(refl <= end)].copy()
    U["yy"] = ((refl.loc[U.index] > end) & (refl.loc[U.index] <= 20241231)).astype(int)
    return U


U = units(k.refl_c12)
mg = FP.merge(U[["appl_no", "yy"]], on="appl_no", how="outer", indicator=True)
O["P0a"] = dict(n=int(len(U)), pos=int(U.yy.sum()), only_frozen=int((mg._merge == "left_only").sum()), only_c12=int((mg._merge == "right_only").sum()),
                label_mismatch=int((mg[mg._merge == "both"].y != mg[mg._merge == "both"].yy).sum()))
O["P0a"]["pass"] = O["P0a"]["n"] == 10051 and O["P0a"]["pos"] == 442 and O["P0a"]["only_frozen"] == 0 and O["P0a"]["only_c12"] == 0 and O["P0a"]["label_mismatch"] == 0
# ---------------- P0-b 점수(동결 예측)
a1, a0 = auc(FP.y, FP.p_champ), auc(FP.y, FP.p_M0)
(lo, hi), _ = cluster_boot_ci(FP.y.values, FP.p_champ.values, FP.p_M0.values, FP.app.values, n=1000, seed=2022)
O["P0b"] = dict(auc_champ=a1, auc_M0=a0, skill=skill(a1, a0), d_skill=skill(a1, a0) - OFF_SK, ci=[lo, hi], ci_official=OFF_CI,
                pass_=bool(abs(skill(a1, a0) - OFF_SK) < 1e-9 and round(lo, 4) == round(OFF_CI[0], 4) and round(hi, 4) == round(OFF_CI[1], 4)))
# ---------------- P0-c 재적합
PO = pd.read_parquet(os.path.join(W, "cache", "preds_off.parquet"))
j = FP.merge(PO, on="appl_no", suffixes=("_f", "_r"))
O["P0c"] = dict(n_join=int(len(j)), max_diff_champ=float((j.p_champ_f - j.p_champ_r).abs().max()), max_diff_M0=float((j.p_M0_f - j.p_M0_r).abs().max()),
                y_mismatch=int((j.y_f != j.y_r).sum()))
sp = os.path.join(TR, "VERIFY_R5", "cache", "sens_preds.parquet")
if os.path.exists(sp):
    S = pd.read_parquet(sp, columns=["appl_no", "p_M0plus"]).merge(PO[["appl_no", "p_M0plus"]], on="appl_no", suffixes=("_v5", "_c12"))
    O["P0c"]["max_diff_M0plus_vs_VERIFY_R5"] = float((S.p_M0plus_v5 - S.p_M0plus_c12).abs().max())
SO = json.load(open(os.path.join(W, "cache", "score_off.json"), encoding="utf-8"))["main"]
O["P0c"]["skill_refit"] = SO["skill_M0"]; O["P0c"]["d_skill"] = SO["skill_M0"] - OFF_SK
O["P0c"]["strong_refit"] = SO["skill_strong"]; O["P0c"]["ci_strong_refit"] = SO["ci_strong"]
O["P0c"]["pass"] = bool(O["P0c"]["max_diff_champ"] < 1e-6 and O["P0c"]["max_diff_M0"] < 1e-6 and abs(O["P0c"]["d_skill"]) < 1e-6)
C04 = json.load(open(os.path.join(n5lib.BASE, r"분석\검증루프_20261012\C04\C04-T2\n5_rescore.json"), encoding="utf-8"))
O["C04_compare"] = dict(c04_skill=C04["scores"]["skill"], c04_ext_n=C04["labels"]["ext_rule"]["n_indep"], c04_ext_pos=C04["labels"]["ext_rule"]["pos_indep"],
                        c04_kr26429_ext=C04["labels"]["kr26429_matched_ext"])
# ---------------- S1a·S2a(동결 예측 + 레이블/모집단 교체; 강한 기준선은 C12 재적합 p_M0plus — P0c에서 VERIFY_R5와 대조)
D = FP.merge(PO[["appl_no", "p_M0plus"]], on="appl_no")


def score(d, label, kind):
    y = d.y.values
    A = {m: float(auc(y, d[f"p_{m}"].values)) for m in ["champ", "M0", "M0plus"]}
    (l1, h1), _ = cluster_boot_ci(y, d.p_champ.values, d.p_M0.values, d.app.values, n=1000, seed=2022)
    (l2, h2), _ = cluster_boot_ci(y, d.p_champ.values, d.p_M0plus.values, d.app.values, n=1000, seed=55)
    return dict(scen=label, kind=kind, n=int(len(d)), pos=int(y.sum()), auc_champ=A["champ"], auc_M0=A["M0"], auc_strong=A["M0plus"],
                skill_M0=skill(A["champ"], A["M0"]), ci_M0_lo=l1, ci_M0_hi=h1, skill_strong=skill(A["champ"], A["M0plus"]), ci_st_lo=l2, ci_st_hi=h2)


rows = []
d1 = D.copy(); rfm = d1.appl_no.map(rf)
d1 = d1[~(rfm <= end)].copy(); d1["y"] = ((d1.appl_no.map(rf) > end) & (d1.appl_no.map(rf) <= 20241231)).astype(int)
rows.append(score(d1, "S1a 완전 키 레이블(동결 예측, 재적합 없음)", "보조"))
TOP5 = ["삼성전자", "광동오포모바일텔레커뮤니케이션즈코포레이션리미티드", "퀄컴인코포레이티드", "엘지전자", "비보모바일커뮤니케이션컴퍼니리미티드"]
rows.append(score(D[~D.app.isin(TOP5)].copy(), "S2a 상위 5사 출원인 제외(동결 예측)", "보조"))
lab = {"off": ("P0 공식 재적합", "재현"), "s1": ("S1 키 파싱 완전판(재적합)", "주"), "s2": ("S2 상위 5사 제외(재적합)", "주"), "s4": ("S4 2011 이관분만 선언 출원 제외(재적합)", "주"),
       "s4b": ("S4b 이관 기록 삭제 후 반영일 재계산(재적합)", "정보"), "s3": ("S3 모집단 확장 7개 KSIC×IPC 핵심(재적합)", "정보")}
for sc, (lb, kd) in lab.items():
    p = os.path.join(W, "cache", f"score_{sc}.json")
    if not os.path.exists(p):
        continue
    J = json.load(open(p, encoding="utf-8"))
    for part in ["main", "sub26429"]:
        if part not in J:
            continue
        m = J[part]
        rows.append(dict(scen=lb + ("" if part == "main" else " — 26429 부분집합"), kind=kd, n=m["n"], pos=m["pos"], auc_champ=m["auc"]["champ"], auc_M0=m["auc"]["M0"],
                         auc_strong=m["auc"]["M0plus"], skill_M0=m["skill_M0"], ci_M0_lo=m["ci_M0"][0], ci_M0_hi=m["ci_M0"][1], skill_strong=m["skill_strong"],
                         ci_st_lo=m["ci_strong"][0], ci_st_hi=m["ci_strong"][1], n_train=m["n_train"], pos_train=m["pos_train"]))
X = pd.DataFrame(rows)
TH = 0.25 * (OFF_CI[1] - OFF_CI[0])


def cat(lo, hi):
    return "이득" if lo > 0 else ("손해" if hi < 0 else "이득 없음(0 포함)")


def judge(r):
    ds = r.skill_M0 - OFF_SK
    sign_off, sign_r = np.sign(round(OFF_ST, 3)), np.sign(round(r.skill_strong, 3))
    cat_off, cat_r = cat(*OFF_STCI), cat(r.ci_st_lo, r.ci_st_hi)
    cond = (r.ci_st_lo <= 0 <= r.ci_st_hi) or r.skill_strong <= 0.005
    flags = []
    if abs(ds) >= TH:
        flags.append("|Δskill|≥0.088")
    if cat_r != cat_off:
        flags.append("강한 기준선 CI 범주 변화")
    if not cond:
        flags.append("'조건부' 해제")
    wording_change = (r.ci_M0_lo <= 0) or (r.ci_st_lo > 0)
    if r.kind == "주" and r.scen.startswith("S1") and wording_change:
        v = "오류"
    elif wording_change:
        v = "단서 필요(강)"
    elif flags:
        v = "단서 필요"
    else:
        v = "영향 없음"
    return pd.Series(dict(d_skill=ds, strong_round_sign_change=bool(sign_r != sign_off), strong_ci_cat=cat_r, conditional_kept=bool(cond),
                          flags=";".join(flags), verdict=v if r.kind in ("주", "보조") else v + "(정보)"))


X = pd.concat([X, X.apply(judge, axis=1)], axis=1)
X.loc[X.kind == "재현", "verdict"] = np.where(X.loc[X.kind == "재현", "d_skill"].abs() < 1e-6, "재현(차 0)", "재현 실패")
X.to_csv(os.path.join(W, "sens_results.csv"), index=False, encoding="utf-8-sig")
order = {"오류": 3, "단서 필요(강)": 2, "단서 필요": 1, "영향 없음": 0}
main = X[X.kind == "주"]
O["overall"] = main.verdict.map(order).max()
O["overall_label"] = [k_ for k_, v in order.items() if v == O["overall"]][0] if len(main) else None
O["threshold"] = TH
json.dump(O, open(os.path.join(W, "cache", "s7_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(json.dumps(O, ensure_ascii=False, indent=1, default=float))
print(X[["scen", "kind", "n", "pos", "auc_champ", "auc_M0", "auc_strong", "skill_M0", "ci_M0_lo", "ci_M0_hi", "skill_strong", "ci_st_lo", "ci_st_hi", "d_skill", "verdict", "flags"]].round(4).to_string())
