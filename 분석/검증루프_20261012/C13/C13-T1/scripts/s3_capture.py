# -*- coding: utf-8 -*-
"""C13-T1 s3: 등록원부 예고등록 포착률 r = N_reg / (N_off × s) 계산과 PREREG J1–J3 판정, 내부 포착 r_int, 데이터 점검 보조값.
입력(읽기만): registry_full.sqlite(mode=ro), cache/official_trials.csv, cache/inforce_17_by_edition.csv, cache/wipo_field_trials.csv
출력: capture_rate.csv, cache/s3_results.json, cache/reg_alive_by_year.csv, cache/s3_internal.csv"""
import os, sys, json, sqlite3
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from scipy import stats
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
INV, SCO = "무효(심판)예고등록", "권리범위(심판)예고등록"
OUT = {}
pt = pd.read_sql("select rgst_no, rgstDate, cndrtExptnDate from patent", c)
rt = pd.read_sql("select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason from right", c)
rt["d"] = pd.to_numeric(rt.rgstCsDate, errors="coerce")
rt["y"] = (rt.d // 10000)
pt["j"] = np.where(pt.rgst_no.str[:2] == "10", "특허", "실용")
pt["rg"] = pd.to_numeric(pt.rgstDate, errors="coerce"); pt["ex"] = pd.to_numeric(pt.cndrtExptnDate, errors="coerce")
OUT["patent_rows"] = int(len(pt)); OUT["rgstDate_missing"] = int(pt.rg.isna().sum()); OUT["exp_missing"] = int(pt.ex.isna().sum())
ext = rt[rt.rgstCsName.isin(["소멸등록", "권리의 말소등록", "말소등록"])].groupby("rgst_no").d.min()
pt["extd"] = pt.rgst_no.map(ext)
J = {r: g for r, g in pt.groupby("j")}
# ---------------------------------------------------------------- 1) 제공 범위 연말 존속(N8 모집단 규칙)
alive = []
for y in range(2005, 2026):
    E = y * 10000 + 1231
    for j, g in J.items():
        n = int(((g.rg <= E) & (g.ex > E) & ~(g.extd <= E)).sum())
        alive.append(dict(year=y, right=j, alive_prov=n))
A = pd.DataFrame(alive)
INF = pd.read_csv(os.path.join(W, "cache", "inforce_17_by_edition.csv"))
INF = INF[INF.edition == INF.edition.max()][["year", "right", "value"]].rename(columns={"value": "inforce_off"})
A = A.merge(INF, on=["year", "right"], how="left")
A["s"] = A.alive_prov / A.inforce_off
A.to_csv(os.path.join(W, "cache", "reg_alive_by_year.csv"), index=False, encoding="utf-8-sig")
# 재현 점검: t0=2022 모집단 317,840
OUT["alive_2022_total"] = int(A[A.year == 2022].alive_prov.sum())
print(A[A.year >= 2006].to_string())

# ---------------------------------------------------------------- 2) 예고등록 건수
ev = rt[rt.rgstCsName.isin([INV, SCO, "심판예고등록", "정정(심판)예고등록"])].copy()
ev["j"] = np.where(ev.rgst_no.str[:2] == "10", "특허", "실용")
ev["k"] = ev.rgstCsName.map({INV: "무효", SCO: "권리범위", "심판예고등록": "미상", "정정(심판)예고등록": "정정"})
ev["dup"] = ev.duplicated(["rgst_no", "d", "k"])
NR = ev.groupby(["y", "k", "j"]).agg(n_reg=("rgst_no", "size"), n_reg_dedup=("dup", lambda s: int((~s).sum()))).reset_index().rename(columns={"y": "year"})
NR["year"] = NR.year.astype(int)
O = pd.read_csv(os.path.join(W, "cache", "official_trials.csv")).rename(columns={"trial": "k", "right": "j"})
O = O[O.k.isin(["무효", "권리범위", "정정"])][["year", "k", "j", "request", "basis"]]
G = pd.MultiIndex.from_product([range(2006, 2026), ["무효", "권리범위", "정정"], ["특허", "실용"]], names=["year", "k", "j"]).to_frame(index=False)
G = G.merge(NR, on=["year", "k", "j"], how="left").merge(O, on=["year", "k", "j"], how="left")
G[["n_reg", "n_reg_dedup"]] = G[["n_reg", "n_reg_dedup"]].fillna(0)
G = G.merge(A.rename(columns={"right": "j"})[["year", "j", "alive_prov", "inforce_off", "s"]], on=["year", "j"], how="left")
G["expected"] = G.request * G.s
G["r"] = G.n_reg / G.expected
# 미상(심판예고등록) 건수 따로
unk = NR[NR.k == "미상"].groupby("year").n_reg.sum()
OUT["generic_trial_notice_by_year"] = {int(k): int(v) for k, v in unk.items()}

def agg(df, years, ks, js=("특허", "실용"), col="n_reg", off_shift=0):
    d = df[df.year.isin(years) & df.k.isin(ks) & df.j.isin(js)]
    if off_shift:
        d2 = df.copy(); d2["year"] = d2.year - off_shift   # 공식 연도를 옮긴 기대값
        e = d2[d2.year.isin(years) & d2.k.isin(ks) & d2.j.isin(js)].expected.sum()
    else:
        e = d.expected.sum()
    return float(d[col].sum()), float(e)

T = range(2008, 2023); F = range(2023, 2025)
def ratio_ci(xF, eF, xT, eT):
    n = xF + xT
    lo = stats.beta.ppf(0.025, xF, xT + 1) if xF > 0 else 0.0
    hi = stats.beta.ppf(0.975, xF + 1, xT) if xT > 0 else np.inf
    f = lambda p: (p / (1 - p)) * (eT / eF)
    return f(lo), f(hi)

rows = []
for lab, ks in [("무효+권리범위", ["무효", "권리범위"]), ("무효", ["무효"]), ("권리범위", ["권리범위"]), ("정정(참고)", ["정정"])]:
    for jl, js in [("특허+실용", ("특허", "실용")), ("특허", ("특허",)), ("실용", ("실용",))]:
        xT, eT = agg(G, T, ks, js); xF, eF = agg(G, F, ks, js)
        xA, eA = agg(G, range(2008, 2025), ks, js)
        rT, rF = xT / eT, xF / eF
        lo, hi = ratio_ci(xF, eF, xT, eT)
        yr = {}
        for y in range(2006, 2026):
            x, e = agg(G, [y], ks, js)
            yr[y] = x / e if e > 0 else np.nan
        rows.append(dict(kind=lab, right=jl, n_reg_T=xT, exp_T=eT, r_T=rT, r_T_yearmean=float(np.nanmean([yr[y] for y in T])),
                         n_reg_F=xF, exp_F=eF, r_F=rF, ratio_F_T=rF / rT, ratio_ci_lo=lo, ratio_ci_hi=hi, r_all_2008_2024=xA / eA,
                         **{f"r_{y}": yr[y] for y in range(2006, 2026)}))
CR = pd.DataFrame(rows)
# 연도별 상세(긴 형식)도 붙인다
G.to_csv(os.path.join(W, "cache", "capture_detail_long.csv"), index=False, encoding="utf-8-sig")
CR.to_csv(os.path.join(W, "capture_rate.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(CR[["kind", "right", "n_reg_T", "exp_T", "r_T", "r_T_yearmean", "n_reg_F", "exp_F", "r_F", "ratio_F_T", "ratio_ci_lo", "ratio_ci_hi", "r_all_2008_2024"]].round(3).to_string())
print(CR[CR.right == "특허+실용"][["kind"] + [f"r_{y}" for y in range(2006, 2026)]].round(2).T.to_string())

# ---------------------------------------------------------------- 3) 판정 J1–J3
m = CR[(CR.kind == "무효+권리범위") & (CR.right == "특허+실용")].iloc[0]
J1 = abs(m.ratio_F_T - 1) > 0.25
J2 = (m.r_T < 0.5) or (m.r_F < 0.5)
ri = CR[(CR.kind == "무효") & (CR.right == "특허+실용")].iloc[0].r_all_2008_2024
rs = CR[(CR.kind == "권리범위") & (CR.right == "특허+실용")].iloc[0].r_all_2008_2024
J3 = (ri / rs > 2) or (ri / rs < 0.5)
OUT["J"] = dict(r_T=m.r_T, r_F=m.r_F, ratio=m.ratio_F_T, ratio_ci=[m.ratio_ci_lo, m.ratio_ci_hi], J1_shift_warning=bool(J1),
                J2_incomplete_warning=bool(J2), r_inv_all=ri, r_scope_all=rs, inv_over_scope=ri / rs, J3_type_bias=bool(J3))
for kind in ["무효", "권리범위"]:
    mm = CR[(CR.kind == kind) & (CR.right == "특허+실용")].iloc[0]
    OUT["J"][f"{kind}_r_T"] = mm.r_T; OUT["J"][f"{kind}_r_F"] = mm.r_F; OUT["J"][f"{kind}_ratio"] = mm.ratio_F_T
    OUT["J"][f"{kind}_ratio_ci"] = [mm.ratio_ci_lo, mm.ratio_ci_hi]
print(json.dumps(OUT["J"], ensure_ascii=False, indent=1))

# ---------------------------------------------------------------- 4) 보고용 변형(판정 밖): 대표성 급증 해 보정
# 2015 무효 급증(의약 허가-특허 연계 시행), 2023 권리범위 급증(WIPO '유기정밀화학' 542건, 전후 해 평균 90.5건)
WF = pd.read_csv(os.path.join(W, "cache", "wipo_field_trials.csv"), index_col=0)
WF.columns = [int(float(x)) for x in WF.columns]
oc = WF.loc["유기정밀화학"]; ph = WF.loc["의약"]
OUT["wipo_organic_fine_chem"] = {int(k): float(v) for k, v in oc.items()}
OUT["wipo_pharma"] = {int(k): float(v) for k, v in ph.items()}
excess23 = float(oc[2023] - (oc[2022] + oc[2024]) / 2)
OUT["excess_2023_organic"] = excess23
G2 = G.copy()
sel = (G2.year == 2023) & (G2.k == "권리범위") & (G2.j == "특허")
G2.loc[sel, "expected"] = (G2.loc[sel, "request"] - excess23) * G2.loc[sel, "s"]
# 2015 무효: 직전·직후 해 평균을 넘는 부분을 범위 밖 급증으로 간주
inv = G[(G.k == "무효") & (G.j == "특허")].set_index("year").request
ex15 = float(inv[2015] - (inv[2014] + inv[2016]) / 2)
OUT["excess_2015_invalid"] = ex15
sel = (G2.year == 2015) & (G2.k == "무효") & (G2.j == "특허")
G2.loc[sel, "expected"] = (G2.loc[sel, "request"] - ex15) * G2.loc[sel, "s"]
adj = {}
for lab, ks in [("무효+권리범위", ["무효", "권리범위"]), ("무효", ["무효"]), ("권리범위", ["권리범위"])]:
    xT, eT = agg(G2, T, ks); xF, eF = agg(G2, F, ks)
    lo, hi = ratio_ci(xF, eF, xT, eT)
    adj[lab] = dict(r_T=xT / eT, r_F=xF / eF, ratio=(xF / eF) / (xT / eT), ci=[lo, hi])
OUT["adjusted_surges"] = adj
# 창에서 2015·2023을 빼기
exy = {}
for lab, ks in [("무효+권리범위", ["무효", "권리범위"]), ("권리범위", ["권리범위"]), ("무효", ["무효"])]:
    xT, eT = agg(G, [y for y in T if y != 2015], ks); xF, eF = agg(G, [2024], ks)
    exy[lab] = dict(r_T_wo2015=xT / eT, r_2024=xF / eF, ratio=(xF / eF) / (xT / eT))
OUT["window_without_surge_years"] = exy
print(json.dumps({k: OUT[k] for k in ["adjusted_surges", "window_without_surge_years", "excess_2023_organic", "excess_2015_invalid"]}, ensure_ascii=False, indent=1))

# ---------------------------------------------------------------- 5) 내부 포착 r_int: 확정 사유가 무효/권리범위인 심결 확정 → 같은 특허에 그 전 같은 종류 예고등록?
D = rt[rt.rgstCsName.str.contains("확정", na=False) & ~rt.rgstCsName.str.contains("기술평가|이의신청", na=False)].copy()
D["rs"] = D.rgstCsReason.fillna("").str.replace(" ", "")
D["kt"] = np.where(D.rs.str.contains("속함|무속"), "권리범위", np.where(D.rs.str.contains("무효") & ~D.rs.str.contains("정정"), "무효", None))
D = D[D.kt.notna()]
first_notice = {}
for k, nm in [("무효", INV), ("권리범위", SCO)]:
    first_notice[k] = rt[rt.rgstCsName == nm].groupby("rgst_no").d.apply(list).to_dict()
generic = rt[rt.rgstCsName == "심판예고등록"].groupby("rgst_no").d.apply(list).to_dict()
res = []
for r, d, k, nm in zip(D.rgst_no, D.d, D.kt, D.rgstCsName):
    prior = [x for x in first_notice[k].get(r, []) if x <= d]
    anyprior = prior or [x for x in generic.get(r, []) if x <= d] or [x for kk in first_notice for x in first_notice[kk].get(r, []) if x <= d]
    res.append(dict(rgst_no=r, d=d, y=int(d // 10000) if d == d else None, kt=k, stage=nm, prior_same=int(bool(prior)), prior_any=int(bool(anyprior))))
RI = pd.DataFrame(res)
RI.drop(columns=["rgst_no"]).to_csv(os.path.join(W, "cache", "s3_internal.csv"), index=False, encoding="utf-8-sig")
ri_tab = RI.groupby(["kt"]).agg(n=("prior_same", "size"), same=("prior_same", "mean"), anyp=("prior_any", "mean"))
ri_y = RI.assign(per=pd.cut(RI.y, [0, 2007, 2015, 2022, 2030], labels=["≤2007", "2008–15", "2016–22", "2023–"])).groupby(["kt", "per"], observed=True).agg(n=("prior_same", "size"), same=("prior_same", "mean"))
print(ri_tab.to_string()); print(ri_y.to_string())
OUT["r_int"] = {k: dict(n=int(v.n), prior_same=float(v.same), prior_any=float(v.anyp)) for k, v in ri_tab.iterrows()}
OUT["r_int_by_period"] = {f"{a}|{b}": dict(n=int(v.n), prior_same=float(v.same)) for (a, b), v in ri_y.iterrows()}
# 확정 사유 미상(기각·각하 등) 심결확정 수
OUT["decisions_scope_or_invalid_reason"] = int(len(RI))
json.dump(OUT, open(os.path.join(W, "cache", "s3_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("done")
