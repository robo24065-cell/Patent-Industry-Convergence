# -*- coding: utf-8 -*-
r"""N8 R7 필수성 검증: 동결 챔피언(E01, XGB, 음성 5% 표본 seed0, 가중 1/0.05) 설정 그대로 피처만 바꿔 개발 원점 2012–2020 재학습.
python e7_n8.py <VARIANT ...>  (CH R_KSIC R_FAM R_PROV S_KSIC S_M0)
읽기: ..\cache\panel_dev.parquet, registry_full.sqlite(mode=ro, 사건 < 2023-01-01만), 공용 연계표 배정. 쓰기: 이 폴더만."""
import os, sys, json, time, sqlite3
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); MC = os.path.dirname(TR)
sys.path.insert(0, TR); sys.path.insert(0, os.path.join(MC, "T4_특허포기예측", "ESSENTIAL_R7", "common"))
import numpy as np, pandas as pd
import n8lib as L
from e_run import score, M0, M1
from xwalk import load_app_xw

SPEC = json.load(open(os.path.join(TR, "final_spec_r6.json"), encoding="utf-8"))
FE = SPEC["champion"]["feats"]; CFG = {"neg_frac": SPEC["champion"]["neg_frac"]}
KS = ["ksic_i", "ksic_disp5"]; FAM = ["fam_offices", "fam_us"]


def ksic_x():
    fx = os.path.join(HERE, "ksic_x_dev.parquet")
    if os.path.exists(fx):
        return pd.read_parquet(fx)
    c = sqlite3.connect(f"file:{L.REG}?mode=ro", uri=True)
    pt = pd.read_sql("select rgst_no, ksic, applNo, rgstDate from patent", c)
    rt = pd.read_sql(f"select rgst_no, rgstCsDate d, rgstCsName cs from right where rgstCsDate < '{L.SEAL}'", c)
    rt["d"] = pd.to_numeric(rt.d, errors="coerce")
    pt["rg"] = pd.to_numeric(pt.rgstDate, errors="coerce")
    pt["ext_d"] = pt.rgst_no.map(rt[rt.cs.isin(L.EXTINCT)].groupby("rgst_no").d.min())
    xw = load_app_xw().set_index("appl_no").xw
    pt["xw"] = pt.applNo.astype(str).map(xw).fillna("XNA")
    disp = rt[rt.cs.isin(L.DISP)][["rgst_no", "d"]]
    rows = []
    for t0 in range(L.TRAIN_MIN, 2021):
        E = t0 * 10000 + 1231
        d5 = set(disp[(disp.d > E - 50000) & (disp.d <= E)].rgst_no)
        alive5 = pt[(pt.rg <= E) & ~(pt.ext_d <= E - 50000)]
        dis5 = alive5.rgst_no.isin(d5)
        o = {}
        for key in ("ksic", "xw"):
            g = pd.DataFrame({"k": alive5[key].values, "d": dis5.values}).groupby("k").d.agg(["sum", "size"])
            o[key] = np.log((g["sum"] + 0.5) / (g["size"] + 500))
        P = pt[["rgst_no", "ksic", "xw"]].copy(); P["t0"] = t0
        P["ksic_disp5_chk"] = P.ksic.map(o["ksic"]).astype(float); P["ksic_disp5_x"] = P.xw.map(o["xw"]).astype(float)
        rows.append(P[["rgst_no", "t0", "xw", "ksic_disp5_chk", "ksic_disp5_x"]])
    X = pd.concat(rows, ignore_index=True)
    X["ksic_i_x"] = pd.to_numeric(X.xw.str[1:], errors="coerce")
    X.to_parquet(fx, index=False)
    return X


cols = list(dict.fromkeys(["rgst_no", "t0", "y", "ksic"] + FE + M0))
D = pd.read_parquet(os.path.join(TR, "cache", "panel_dev.parquet"), columns=cols)
X = ksic_x()
D = D.merge(X, on=["rgst_no", "t0"], how="left", validate="one_to_one")
chk = float(np.mean(np.isclose(D.ksic_disp5.astype(float), D.ksic_disp5_chk.astype(float), atol=1e-5)))
SX = {"ksic_i": "ksic_i_x", "ksic_disp5": "ksic_disp5_x"}
VAR = {
    "CH": (FE, "동결 챔피언 재현"),
    "R_KSIC": ([f for f in FE if f not in KS], "제거: 제공 KSIC 라벨 피처(ksic_i 범주 코드, ksic_disp5 KSIC 최근 5년 분쟁률)"),
    "R_FAM": ([f for f in FE if f not in FAM], "제거(5개국): 제공 5개국 패밀리 피처(fam_offices, fam_us) → KR 레코드만"),
    "R_PROV": ([f for f in FE if f not in KS + FAM], "제거: KSIC + 5개국 패밀리 모두"),
    "S_KSIC": ([SX.get(f, f) for f in FE], "대체: ksic_i·ksic_disp5를 공식 IPC-KSIC 연계표 배정 코드로 다시 계산"),
    "S_M0": ([SX.get(f, f) for f in M0], "참고: 기준선 M0의 ksic_i도 연계표 코드로 바꾼 M0(공식 skill 분모는 바꾸지 않음)"),
    "M0": (M0, "참고: 공식 기준선 M0 재현"),
}
ref = dict(zip(SPEC["dev_reference"]["origins"], SPEC["dev_reference"]["auc_M0"]))
refc = dict(zip(SPEC["dev_reference"]["origins"], SPEC["dev_reference"]["auc_champ"]))
fb = os.path.join(HERE, "base_M0_preds.parquet")
if os.path.exists(fb):
    bp = pd.read_parquet(fb)
else:
    p0 = score(D, M0, CFG, "xgb")
    bp = pd.concat([pd.DataFrame({"t0": v, "p": p0[v]}) for v in L.DEV], ignore_index=True)
    bp.to_parquet(fb, index=False)
BP = {v: bp[bp.t0 == v].p.values for v in L.DEV}
for name in sys.argv[1:]:
    t = time.time(); feats, desc = VAR[name]
    P = score(D, feats, CFG, "xgb")
    res = {}
    for v in L.DEV:
        y = D.y.values[D.t0.values == v]
        a = float(L.auc(y, P[v])); a0 = float(L.auc(y, BP[v]))
        prec, hit, k = L.topk(y, P[v], frac=0.01)
        res[str(v)] = dict(auc=a, auc_M0=a0, skill=float(L.skill(a, a0)), n=int(len(y)), pos=int(y.sum()), top1pct_capture=hit / max(1, int(y.sum())))
    o = dict(variant=name, desc=desc, feats=feats, by_origin=res, mean_auc=float(np.mean([r["auc"] for r in res.values()])),
             mean_skill=float(np.mean([r["skill"] for r in res.values()])), ksic_disp5_recompute_match=chk,
             frozen_auc_champ=refc, frozen_auc_M0=ref, sec=round(time.time() - t))
    json.dump(o, open(os.path.join(HERE, f"res_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(name, round(o["mean_auc"], 4), round(o["mean_skill"], 4), {k: round(r["auc"], 4) for k, r in res.items()}, o["sec"], "s", flush=True)
print("ksic_disp5 recompute match", chk, "xw coverage", float((D.xw != "XNA").mean()))
