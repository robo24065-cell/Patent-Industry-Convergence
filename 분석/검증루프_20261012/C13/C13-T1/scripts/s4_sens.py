# -*- coding: utf-8 -*-
"""C13-T1 s4: N8 판정 민감도(공식 동결 예측만, 재학습 없음). 레이블·가중만 바꾼다.
사용: python -I s4_sens.py point            → 점추정 전 시나리오 + S4/S5 반복(200회)
      python -I s4_sens.py boot <part> <B>  → 권리자 군집 부트스트랩(S0,S1,S1b,S2) 일부
      python -I s4_sens.py merge <nparts>   → CI 합치기, sens_results.csv 작성
입력(읽기만): registry_full.sqlite, final_preds_r6.parquet, C01-T3 _clusters_internal.parquet(g_owner), capture_rate.csv"""
import os, sys, json, time, sqlite3
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(BASE, r"분석\모델경쟁_20260927\N8_특허분쟁표적레이더")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
CLF = os.path.join(BASE, r"분석\검증루프_20261012\C01\C01-T3\_clusters_internal.parquet")
t_start = time.time()
mode = sys.argv[1]
INV, SCO = "무효(심판)예고등록", "권리범위(심판)예고등록"
P = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no", "y", "p_champ", "p_M0", "p_SB"])
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
ev = pd.read_sql(f"select rgst_no, rgstCsDate, rgstCsName from right where rgstCsName in ('{INV}','{SCO}') and rgstCsDate between '20230101' and '20241231'", c)
ev["d"] = ev.rgstCsDate.astype(int)
ev = ev[ev.rgst_no.isin(set(P.rgst_no))]
first = ev.sort_values(["d", "rgstCsName"], ascending=[True, False]).groupby("rgst_no").first()   # 같은 날 둘 다면 '무효' 우선(내림차순: 무효 > 권리범위)
pos_inv = set(ev[ev.rgstCsName == INV].rgst_no); pos_sco = set(ev[ev.rgstCsName == SCO].rgst_no)
y_re = P.rgst_no.isin(set(ev.rgst_no)).values
assert (y_re == (P.y.values == 1)).all(), "레이블 재현 실패"
n = len(P); posall = y_re
fy = P.rgst_no.map(first.d // 10000).values
fk = P.rgst_no.map(first.rgstCsName.map({INV: "무효", SCO: "권리범위"})).values
CR = pd.read_csv(os.path.join(W, "capture_rate.csv"))
main = CR[(CR.kind == "무효+권리범위") & (CR.right == "특허+실용")].iloc[0]
rk = {k: CR[(CR.kind == k) & (CR.right == "특허+실용")].iloc[0] for k in ["무효", "권리범위"]}
r_y = {2023: main.r_2023, 2024: main.r_2024}
r_yk = {(y, k): rk[k][f"r_{y}"] for y in (2023, 2024) for k in ("무효", "권리범위")}
inv = P.rgst_no.isin(pos_inv).values; sco = P.rgst_no.isin(pos_sco).values
low = "권리범위" if rk["권리범위"].r_all_2008_2024 < rk["무효"].r_all_2008_2024 else "무효"

def norm(w, pos):
    w = w.copy(); w[pos] = w[pos] / w[pos].mean(); w[~pos] = 1.0; return w
w_S1 = np.ones(n); w_S1[posall] = [1.0 / r_y[int(v)] for v in fy[posall]]; w_S1 = norm(w_S1, posall)
w_S1b = np.ones(n); w_S1b[posall] = [1.0 / r_yk[(int(a), b)] for a, b in zip(fy[posall], fk[posall])]; w_S1b = norm(w_S1b, posall)
SC = {  # 이름: (y, eval_mask, base_weight)
    "S0_official": (posall, np.ones(n, bool), np.ones(n)),
    "S1_inv_capture_year": (posall, np.ones(n, bool), w_S1),
    "S1b_inv_capture_year_type": (posall, np.ones(n, bool), w_S1b),
    "S2_invalid_only": (inv, ~(posall & ~inv), np.ones(n)),
}
if low == "권리범위":
    SC["S3_drop_low_capture_type(=S2)"] = SC["S2_invalid_only"]
    SC["S3x_scope_only_as_negative(참고)"] = (inv, np.ones(n, bool), np.ones(n))
else:
    SC["S3_drop_low_capture_type"] = (sco, ~(posall & ~sco), np.ones(n))
MODELS = ["p_champ", "p_M0", "p_SB"]
S = {k: P[k].values.astype(float) for k in MODELS}
IV = {}; ORD = {}
for k in MODELS:
    u, iv = np.unique(S[k], return_inverse=True); IV[k] = (iv, len(u)); ORD[k] = np.argsort(-S[k], kind="stable")

def wauc(y, k, w):
    iv, G = IV[k]
    pos = np.bincount(iv, weights=w * y, minlength=G); neg = np.bincount(iv, weights=w * (1 - y), minlength=G)
    cn = np.cumsum(neg) - neg
    return float((pos * (cn + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))

def wcap(y, k, w, frac=0.01):
    o = ORD[k]; wo = w[o]; cw = np.cumsum(wo); top = (cw - wo) < frac * cw[-1]
    return float((wo * y[o] * top).sum() / (w * y).sum())

def metrics(y, w):
    y = y.astype(float)
    a = {k: wauc(y, k, w) for k in MODELS}; cp = {k: wcap(y, k, w) for k in MODELS}
    return dict(auc_champ=a["p_champ"], auc_M0=a["p_M0"], auc_SB=a["p_SB"],
                skill_M0=(a["p_champ"] - a["p_M0"]) / (1 - a["p_M0"]), skill_SB=(a["p_champ"] - a["p_SB"]) / (1 - a["p_SB"]),
                cap_champ=cp["p_champ"], cap_M0=cp["p_M0"], cap_SB=cp["p_SB"])

OFF_SKILL = 0.408925728460638; OFF_CIW = 0.540097309715566 - 0.24297136524642474
if mode == "point":
    res = {}
    for nm, (y, msk, w0) in SC.items():
        res[nm] = metrics(y, w0 * msk); res[nm]["pos_eval"] = int((y & msk).sum()); res[nm]["n_eval"] = int(msk.sum())
    # S4 시점 균등화, S5 완전 포착 최악: 무작위 가짜 양성 추가(점수와 무관), 200회
    rng = np.random.default_rng(20261017)
    rT, rF = main.r_T, main.r_F
    negidx = np.where(~posall)[0]
    sims = {}
    specs = {"S4_time_equalize": 108 * (rT / rF - 1) if rF < rT else -108 * (1 - rT / rF),
             "S5_complete_capture_worst": min(108 * (1 / rF - 1), 3 * 108)}
    adj = json.load(open(os.path.join(W, "cache", "s3_results.json"), encoding="utf-8"))["adjusted_surges"]["무효+권리범위"]
    specs["S4adj_time_equalize_surge_adjusted(참고)"] = 108 * (adj["r_T"] / adj["r_F"] - 1) if adj["r_F"] < adj["r_T"] else -108 * (1 - adj["r_T"] / adj["r_F"])
    for nm, m_add in specs.items():
        k = int(round(abs(m_add))); out = []
        for b in range(200):
            y = posall.copy()
            if m_add > 0:
                y[rng.choice(negidx, k, replace=False)] = True
            elif k > 0:
                y[rng.choice(np.where(posall)[0], k, replace=False)] = False
            out.append(metrics(y, np.ones(n)))
        D = pd.DataFrame(out)
        sims[nm] = dict(n_change=int(np.sign(m_add) * k), **{f"{c}_med": float(D[c].median()) for c in D.columns},
                        **{f"{c}_p025": float(D[c].quantile(0.025)) for c in ["skill_M0", "cap_champ"]},
                        **{f"{c}_p975": float(D[c].quantile(0.975)) for c in ["skill_M0", "cap_champ"]})
    res["_sims"] = sims
    res["_meta"] = dict(r_y=r_y, r_yk={f"{a}|{b}": v for (a, b), v in r_yk.items()}, low_capture_type=low,
                        weights_S1={str(k): float(1 / v) for k, v in r_y.items()},
                        first_year_counts=pd.Series(fy[posall]).value_counts().to_dict(),
                        first_type_counts=pd.Series(fk[posall]).value_counts().to_dict(),
                        w_S1_pos_range=[float(w_S1[posall].min()), float(w_S1[posall].max())],
                        w_S1b_pos_range=[float(w_S1b[posall].min()), float(w_S1b[posall].max())])
    json.dump(res, open(os.path.join(W, "cache", "s4_point.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float)); sys.exit()

if mode == "boot":
    part, B = int(sys.argv[2]), int(sys.argv[3])
    CL = pd.read_parquet(CLF)[["rgst_no", "g_owner"]]
    g = P[["rgst_no"]].merge(CL, on="rgst_no", how="left").g_owner
    assert g.notna().all()
    codes, uniq = pd.factorize(g.values); G = len(uniq)
    rng = np.random.default_rng(20261017 + 1000 * part)
    draws = []
    names = [k for k in SC if not k.startswith("S3x")]
    for b in range(B):
        cw = rng.multinomial(G, np.full(G, 1.0 / G))[codes].astype(float)
        draws.append({nm: metrics(SC[nm][0], cw * SC[nm][1] * SC[nm][2]) for nm in names})
        if time.time() - t_start > 480:
            print("time stop", b + 1); break
    pd.to_pickle(draws, os.path.join(W, "cache", f"_boot_p{part}.pkl"))
    print("part", part, len(draws), round(time.time() - t_start)); sys.exit()

if mode == "merge":
    k = int(sys.argv[2]); draws = []
    for p_ in range(k):
        fp = os.path.join(W, "cache", f"_boot_p{p_}.pkl")
        if os.path.exists(fp): draws += pd.read_pickle(fp)
    pt = json.load(open(os.path.join(W, "cache", "s4_point.json"), encoding="utf-8"))
    rows = []
    for nm, v in pt.items():
        if nm.startswith("_"): continue
        row = dict(scenario=nm, **v)
        vals = [d[nm] for d in draws if nm in d]
        if vals:
            for mt in ["skill_M0", "skill_SB", "cap_champ", "auc_champ"]:
                row[f"{mt}_lo"] = float(np.percentile([x[mt] for x in vals], 2.5)); row[f"{mt}_hi"] = float(np.percentile([x[mt] for x in vals], 97.5))
            row["B"] = len(vals)
        row["d_skill_M0"] = v["skill_M0"] - OFF_SKILL
        row["impact_threshold"] = 0.25 * OFF_CIW
        row["verdict"] = ("영향 없음" if abs(row["d_skill_M0"]) < 0.25 * OFF_CIW else "영향 있음") if nm.startswith(("S1", "S3_")) else ("재현" if nm.startswith(("S0", "S2")) else "참고")
        rows.append(row)
    for nm, v in pt["_sims"].items():
        rows.append(dict(scenario=nm, n_change=v["n_change"], auc_champ=v["auc_champ_med"], auc_M0=v["auc_M0_med"], auc_SB=v["auc_SB_med"],
                         skill_M0=v["skill_M0_med"], skill_SB=v["skill_SB_med"], cap_champ=v["cap_champ_med"], cap_M0=v["cap_M0_med"], cap_SB=v["cap_SB_med"],
                         skill_M0_lo=v["skill_M0_p025"], skill_M0_hi=v["skill_M0_p975"], cap_champ_lo=v["cap_champ_p025"], cap_champ_hi=v["cap_champ_p975"],
                         d_skill_M0=v["skill_M0_med"] - OFF_SKILL, impact_threshold=0.25 * OFF_CIW, verdict="보고만(판정 밖, 200회 중앙값·2.5–97.5%)"))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(W, "sens_results.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(R[["scenario", "pos_eval", "auc_champ", "auc_M0", "auc_SB", "skill_M0", "skill_M0_lo", "skill_M0_hi", "skill_SB", "cap_champ", "cap_M0", "cap_SB", "d_skill_M0", "verdict"]].round(4).to_string())
