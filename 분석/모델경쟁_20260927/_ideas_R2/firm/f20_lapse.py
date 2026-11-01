# -*- coding: utf-8 -*-
"""아이디어 3' 파일럿: 기업 재무 스트레스(DART) × 산업 경기(KOSIS·무역, KSIC 연결) → 보유 특허 포기(등록료 불납 소멸) 예측.
단위 = (DART 재무 보유 법인의 생존 특허, 기준일 L=1월 1일). 결과 = [L, L+2년) 안 소멸등록(존속만료 임박 특허 제외: 만료연도 > L+2).
정보집합: 특허·기업 사건 < L, 재무 = 회계연도 L−2 (L−1 사업보고서는 3월 공시라 1월 1일엔 미공개), KOSIS = L−2 조사, 무역 = L−1.
🔒 최종시험 L=2023(결과 2023–24, F=2023). 개발 L=2017..2021(결과 ≤2022), 학습 L' ≤ v−2 (L' ≥ 2015).
기준선(고정) = 로지스틱[특허 연령, 연령², log 청구항, log 기업 생존특허]."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalkit import walk, skill
import sqlite3
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
LS = list(range(2015, 2022))
cache = os.path.join(C, "lapse_panel.parquet")
if not os.path.exists(cache):
    pf = pd.read_parquet(os.path.join(C, "pat_firm.parquet"))
    pf = pf[pf.corp_code.notna()].drop_duplicates("rgst_no")          # 공동출원은 첫 연결 법인
    F = pd.read_parquet(os.path.join(C, "fin.parquet"))
    pf = pf[pf.corp_code.isin(F.corp_code)]
    con = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
    con.execute("create temp table ids(r text primary key)")
    con.executemany("insert into ids values(?)", [(r,) for r in pf.rgst_no])
    right = pd.read_sql("select rgst_no, rgstCsDate d, rgstCsName n from right where rgst_no in (select r from ids)", con)
    pat = pd.read_sql("select rgst_no, cndrtExptnDate ex, claimCount cc from patent where rgst_no in (select r from ids)", con)
    pay = pd.read_sql("select rgst_no, payDate d from pay where rgst_no in (select r from ids)", con)
    own = pd.read_sql("select rgst_no, ownerRgstCsDate d, ownerRgstCsName n from owner where rgst_no in (select r from ids) and ownerRgstCsName is not null", con)
    con.close()
    yr = lambda s: pd.to_numeric(s.astype(str).str[:4], errors="coerce")
    pf = pf.merge(pat, on="rgst_no")
    pf["ry"] = yr(pf.rgstDate); pf["exy"] = yr(pf.ex); pf["cc"] = pd.to_numeric(pf.cc, errors="coerce")
    lap = right[right.n.fillna("").str.contains("소멸")].copy(); lap["y"] = yr(lap.d)
    pf["lapy"] = pf.rgst_no.map(lap.groupby("rgst_no").y.min())
    pf["try"] = pf.rgst_no.map(own.assign(y=yr(own.d)).groupby("rgst_no").y.min())   # 최초 권리 변동(이전 후엔 원 기업 재무와 무관)
    pay["y"] = yr(pay.d)
    rows = []
    for L in LS + [2022]:   # 2022 는 피처만(결과 미계산) — 아래에서 y=NaN
        d = pf[(pf.ry < L) & (pf.lapy.isna() | (pf.lapy >= L)) & (pf.exy > L + 2) & (pf["try"].isna() | (pf["try"] >= L))].copy()
        d["L"] = L
        if L + 1 <= 2022:
            d["y"] = d.lapy.between(L, L + 1).astype(float)
        else:
            d["y"] = np.nan
        d["age"] = L - d.ry
        pc = pay[pay.y < L].groupby("rgst_no").size()
        d["n_pay"] = d.rgst_no.map(pc).fillna(0)
        rows.append(d[["rgst_no", "corp_code", "ksic", "L", "y", "age", "cc", "n_pay"]])
    P = pd.concat(rows, ignore_index=True)
    # 기업 특허행동(< L): f03 patfeat 의 t0 = L−1
    PF = pd.read_parquet(os.path.join(C, "patfeat.parquet"))
    PF = PF[["corp_code", "t0", "n_alive", "lap_rate", "reg_mom", "n_reg3", "tr_rate", "hhi"]].assign(L=lambda x: x.t0 + 1).drop(columns="t0")
    P = P.merge(PF, on=["corp_code", "L"], how="left")
    # 재무 FY L−2
    Pn = pd.read_parquet(os.path.join(C, "panel12.parquet"))[["corp_code", "t0", "log_rev", "log_assets", "g_rev1", "g_rev2", "opm", "roa", "lev", "eq_cap", "op_neg", "op_neg_m1", "re_assets"]]
    P = P.merge(Pn.assign(L=Pn.t0 + 2).drop(columns="t0"), on=["corp_code", "L"], how="left")
    # 산업 신호: ksic_ext 의 t0 = L−1 (KOSIS t0−1 = L−2, 무역 t0 = L−1)
    X = pd.read_parquet(os.path.join(C, "ksic_ext.parquet"))[["ksic", "t0", "g_ship", "g_emp", "g_imp", "pmom_KR", "pmom_foreign", "kr_gap"]]
    P = P.merge(X.assign(L=X.t0 + 1).drop(columns="t0"), on=["ksic", "L"], how="left")
    P.to_parquet(cache, index=False)
P = pd.read_parquet(cache)
P = P.rename(columns={"L": "t0"})
P["age2"] = P.age ** 2; P["log_cc"] = np.log1p(P.cc); P["log_alive"] = np.log1p(P.n_alive)
P["pay_gap"] = P.age - P.n_pay
print(P[P.y.notna()].groupby("t0").agg(n=("y", "size"), rate=("y", "mean"), firms=("corp_code", "nunique")).round(4).to_string())
DEV = list(range(2017, 2022)); H = 2
base_f = ["age", "age2", "log_cc", "log_alive"]
cands = [("firm_pat", ["lap_rate", "reg_mom", "tr_rate"]), ("pay", ["n_pay", "pay_gap"]),
         ("fin_stress", ["op_neg", "op_neg_m1", "opm", "roa"]), ("fin_struct", ["lev", "eq_cap", "re_assets"]),
         ("fin_growth", ["g_rev1", "g_rev2", "log_rev"]), ("ind_kosis", ["g_ship", "g_emp"]), ("ind_trade", ["g_imp"]),
         ("ind_patent", ["pmom_KR", "pmom_foreign", "kr_gap"])]
out = []
ab, _ = walk(P, "y", base_f, DEV, H, "logit"); champ = list(base_f); ac = dict(ab)
out.append({"exp": "B0", "change": "baseline", "mean_auc": round(np.mean(list(ab.values())), 4), "decision": "BASE", **{f"auc_{v}": round(ab[v], 4) for v in DEV}})
for nm, f in cands:
    am, _ = walk(P, "y", champ + f, DEV, H, "logit")
    s_new = np.mean(list(skill(am, ab).values())); s_ch = np.mean(list(skill(ac, ab).values()))
    win = np.mean([am[v] > ac[v] for v in DEV])
    dec = "ACCEPT" if (s_new - s_ch >= 0.003 and win >= 0.6) else "ROLLBACK"
    if dec == "ACCEPT": champ += f; ac = am
    out.append({"exp": nm, "change": "+" + ",".join(f), "mean_auc": round(np.mean(list(am.values())), 4), "skill_vs_base": round(s_new, 4),
                "d_skill": round(s_new - s_ch, 4), "win": round(win, 2), "decision": dec, **{f"auc_{v}": round(am[v], 4) for v in DEV}})
    print(out[-1], flush=True)
am, _ = walk(P, "y", champ, DEV, H, "xgb", n_estimators=300, max_depth=5)
out.append({"exp": "xgb_champ", "change": "XGB on champion", "mean_auc": round(np.mean(list(am.values())), 4),
            "skill_vs_base": round(np.mean(list(skill(am, ab).values())), 4), "win": round(np.mean([am[v] > ac[v] for v in DEV]), 2),
            "decision": "INFO", **{f"auc_{v}": round(am[v], 4) for v in DEV}})
# 재무 없는 챔피언 대비 재무 기여(절제)
nofin = [c for c in champ if c not in sum([f for n, f in cands if n.startswith("fin")], [])]
an, _ = walk(P, "y", nofin, DEV, H, "logit")
out.append({"exp": "ablate_fin", "change": "champion − 재무", "mean_auc": round(np.mean(list(an.values())), 4),
            "skill_vs_base": round(np.mean(list(skill(an, ab).values())), 4), "decision": "INFO"})
L = pd.DataFrame(out); L.to_csv(os.path.join(HERE, "iter_lapse.csv"), index=False)
pd.set_option("display.width", 250)
print(L.to_string()); print("champion", champ)
