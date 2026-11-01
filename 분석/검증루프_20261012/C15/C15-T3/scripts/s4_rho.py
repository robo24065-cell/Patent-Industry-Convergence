# -*- coding: utf-8 -*-
r"""C15-T3 s4: c5_R10 c2_rta.py 정의 그대로 PKR(진단) ρ(해외 출원 RTA↔수출 RCA)·보조창·지형도 pkr_H·한국 특허 RTA↔RCA pkr_H를 팔별로 재계산.
팔(pkr 열만 교체): stored(alt_assign k_pkr) / r8 / reg(대안 A) / pure(대안 B). cweq·txt·free는 저장 alt_assign 값 그대로.
팔마다 rng를 seed 20261005로 새로 만들고 c2 호출 순서(cweq→txt→free→pkr, 이어서 보조창 같은 순서, 지형도, 특허 RTA)를 지킨다.
부트스트랩 지수는 n(=48)에만 의존 → pkr 결과는 txt·free 팔 선택과 무관(n=48 assert).
읽기만: c5_R10/alt_assign, C15-T3/cache/pkr_alt_*, map5 m4_exports_ksic_5rep.csv, conc_ksic_map.csv, c2_results.json. 쓰기: res/s4_rho.json, res/s4_rho_table.csv"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C15/C15-T3"
C5 = MC + "/_shared_report/fusion_advantage/c5_R10"
MAP5 = MC + "/_shared_report/fusion_advantage/map5"
CCS = ["KR", "US", "JP", "CN", "EP"]
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
km = pd.read_csv(MC + "/T3_기술셀부상예측/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]
km["w"] = 1 / km.groupby("code").ksic.transform("size"); con.register("km", km); K50 = sorted(km.ksic)
A0 = C5 + "/alt_assign/*.parquet"; AP = W + "/cache/pkr_alt_*.parquet"
base0 = f"(select * from read_parquet('{A0}') where is_inv and appl_year between 2015 and 2019)"
baseP = f"(select * from read_parquet('{AP}') where is_inv and appl_year between 2015 and 2019)"
q = lambda s: con.sql(f"select k, cc, y, o, sum(w) n from ({s}) group by all").df()
FIX = {"prov": q(f"select ksic_prov k, cc, appl_year y, kr_origin o, 1.0 w from {base0}"),
       "txt": q(f"select k_txt k, cc, appl_year y, kr_origin o, 1.0 w from {base0} where k_txt is not null"),
       "free": q(f"select k_free k, cc, appl_year y, kr_origin o, 1.0 w from {base0} where k_free is not null"),
       "cweq": q(f"select km.ksic k, b.cc, b.appl_year y, b.kr_origin o, km.w w from {base0} b join km on km.code=b.code")}
ARMS = ["stored", "r8", "reg", "pure"]
VAR = {"stored": q(f"select k_pkr k, cc, appl_year y, kr_origin o, 1.0 w from {base0} where k_pkr is not null")}
for a in ("r8", "reg", "pure"):
    # c2의 pkr 표는 KR 문서(k_pkr = 제공)도 포함한다 → KR 행은 저장값 그대로 붙인다
    kr = VAR["stored"][VAR["stored"].cc == "KR"]
    VAR[a] = pd.concat([kr, q(f"select k_{a} k, cc, appl_year y, kr_origin o, 1.0 w from {baseP} where k_{a} is not null")], ignore_index=True)
chk = {}
s0 = VAR["stored"].set_index(["k", "cc", "y", "o"]).n.sort_index(); s8 = VAR["r8"].set_index(["k", "cc", "y", "o"]).n.sort_index()
chk["r8_vs_stored_maxabs"] = float((s0 - s8.reindex(s0.index).fillna(0)).abs().max()); chk["r8_extra_keys"] = int(len(set(s8.index) - set(s0.index)))
log("R8 집계 == 저장 집계", chk); assert chk["r8_vs_stored_maxabs"] == 0 and chk["r8_extra_keys"] == 0
ek = pd.read_csv(MAP5 + "/m4_exports_ksic_5rep.csv", dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); s_ = v / v.sum()
RCA = s_.div(s_.mean(1), axis=0).KR.reindex(K50)
sp = lambda x, y: spearmanr(x, y).statistic
B = 2000


def run(arm):
    C = dict(FIX); C["pkr"] = VAR[arm]
    rng = np.random.default_rng(20261005)

    def counts(lk, lf, y0, y1):
        out = {}
        for cc in CCS:
            d = C[lk if cc == "KR" else lf]; out[cc] = d[(d.cc == cc) & d.y.between(y0, y1)].groupby("k").n.sum()
        return pd.DataFrame(out).reindex(K50).fillna(0)

    def rta_ew(n):
        sh = n / n.sum(); return sh.div(sh.mean(1), axis=0)

    def abroad(lab, y0, y1):
        d = C[lab]; d = d[(d.cc != "KR") & d.y.between(y0, y1)]
        k = d[d.o].groupby("k").n.sum().reindex(K50).fillna(0); a = d.groupby("k").n.sum().reindex(K50)
        return (k / a) / (k.sum() / a.sum())

    def paired(xp, xa, y):
        d = pd.concat([xp, xa, y], axis=1).dropna(); a, b, c = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
        rp, ra = sp(a, c), sp(b, c); n = len(d); assert n == 48, n
        I = rng.integers(0, n, (B, n))
        bp = np.array([sp(a[i], c[i]) for i in I]); ba = np.array([sp(b[i], c[i]) for i in I]); bd = bp - ba
        ci = lambda z: [round(float(np.nanpercentile(z, 2.5)), 3), round(float(np.nanpercentile(z, 97.5)), 3)]
        dd = rp - ra; lo, hi = ci(bd)
        verdict = "제공 우위" if (dd > 0 and lo > 0) else ("대체 우위" if (dd < 0 and hi < 0) else "같다")
        return dict(n=n, rho_prov=round(rp, 3), rho_alt=round(ra, 3), rho_alt_raw=float(ra), ci_alt=ci(ba), d=round(dd, 3), d_raw=float(dd),
                    ci_d=[lo, hi], verdict=verdict, alt_keeps_alignment=bool(ci(ba)[0] > 0), rho_prov_vs_alt=round(sp(a, b), 3))
    R = {}
    ab3 = {lab: abroad(lab, 2017, 2019) for lab in C}; ab2 = {lab: abroad(lab, 2015, 2017) for lab in C}
    R["main"] = {lab: paired(ab3["prov"], ab3[lab], RCA) for lab in ["cweq", "txt", "free", "pkr"]}
    R["sub_P2"] = {lab: paired(ab2["prov"], ab2[lab], RCA) for lab in ["cweq", "txt", "free", "pkr"]}
    R["abroad_kr_docs_P3_pkr"] = round(float(C["pkr"][(C["pkr"].cc != "KR") & C["pkr"].o & C["pkr"].y.between(2017, 2019)].n.sum()))
    rt_prov = rta_ew(counts("prov", "prov", 2017, 2019)); lp = rt_prov.idxmax(1); krp = set(lp.index[lp == "KR"])
    rt = rta_ew(counts("prov", "pkr", 2017, 2019)); ld = rt.idxmax(1); krl = set(ld.index[ld == "KR"])
    R["land_pkr_H"] = dict(kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=int(len(krl)), kept=int(len(krl & krp)),
                           leaders={c: int((ld == c).sum()) for c in CCS}, rho_vs_prov={c: round(sp(rt[c], rt_prov[c]), 3) for c in CCS},
                           rho_vs_prov_raw_KR=float(sp(rt.KR, rt_prov.KR)), kr_rta_gt1_agree=float(((rt.KR > 1) == (rt_prov.KR > 1)).mean()),
                           big_change=bool(len(krl & krp) < 4 or min(sp(rt[c], rt_prov[c]) for c in CCS) < 0.7), kr_leader_list=sorted(krl))
    # c2 순서: 지형도 다음 특허 RTA(cweq_F, cweq_H, txt_F, txt_H, free_F, free_H, pkr_H) — rng 소비 순서를 맞춘다
    rtp2 = rta_ew(counts("prov", "prov", 2015, 2017)).KR; sub = {}
    KRLAB = {"cweq": "cweq", "txt": "txt", "free": "free", "pkr": "prov"}
    for lab in ["cweq", "txt", "free", "pkr"]:
        for scen, lk in (("F", KRLAB[lab]), ("H", "prov")):
            if lab == "pkr" and scen == "F": continue
            sub[f"{lab}_{scen}"] = paired(rtp2, rta_ew(counts(lk, lab, 2015, 2017)).KR, RCA)
    R["patRTA_P2_pkr_H"] = sub["pkr_H"]
    return R


OUT = {a: run(a) for a in ARMS}
c2 = json.load(open(C5 + "/c2_results.json", encoding="utf-8"))
rep = {}
for arm in ("stored", "r8"):
    o, s = c2["main_abroadRTA_P3_vs_RCA"]["pkr"], OUT[arm]["main"]["pkr"]
    rep[f"{arm}_main"] = dict(orig=(o["rho_alt"], o["ci_alt"], o["d"], o["ci_d"]), mine=(s["rho_alt"], s["ci_alt"], s["d"], s["ci_d"]))
    o, s = c2["sub_abroadRTA_P2_vs_RCA"]["pkr"], OUT[arm]["sub_P2"]["pkr"]
    rep[f"{arm}_sub"] = dict(orig=(o["rho_alt"], o["d"], o["ci_d"]), mine=(s["rho_alt"], s["d"], s["ci_d"]))
    o, s = c2["landscape_P3"]["pkr_H"], OUT[arm]["land_pkr_H"]
    rep[f"{arm}_land"] = dict(orig=(o["kr_rta_gt1"], o["kr_leader"], o["kr_leader_kept_of_prov7"], o["rho_vs_prov"]), mine=(s["kr_rta_gt1"], s["kr_leader"], s["kept"], s["rho_vs_prov"]))
    o, s = c2["sub_patRTA_P2_vs_RCA"]["pkr_H"], OUT[arm]["patRTA_P2_pkr_H"]
    rep[f"{arm}_pat"] = dict(orig=(o["rho_alt"], o["d"], o["ci_d"]), mine=(s["rho_alt"], s["d"], s["ci_d"]))
for k in rep: rep[k]["same"] = json.dumps(rep[k]["orig"]) == json.dumps(rep[k]["mine"])
# 비-pkr 결과도 stored 팔에서 c2와 같아야 rng 순서가 맞은 것
for lab in ("cweq", "txt", "free"):
    o, s = c2["main_abroadRTA_P3_vs_RCA"][lab], OUT["stored"]["main"][lab]
    rep[f"stored_main_{lab}"] = dict(orig=(o["rho_alt"], o["ci_d"]), mine=(s["rho_alt"], s["ci_d"]), same=(o["rho_alt"], o["ci_d"]) == (s["rho_alt"], s["ci_d"]))
OUT["repro_vs_c2"] = rep; OUT["repro_all_same"] = all(v["same"] for v in rep.values())
log("재현", OUT["repro_all_same"], {k: v for k, v in rep.items() if not v["same"]})
rows = []
for a in ARMS:
    m, s2, L, pt = OUT[a]["main"]["pkr"], OUT[a]["sub_P2"]["pkr"], OUT[a]["land_pkr_H"], OUT[a]["patRTA_P2_pkr_H"]
    rows.append(dict(arm=a, rho_pkr=m["rho_alt"], rho_pkr_raw=round(m["rho_alt_raw"], 5), ci_pkr=m["ci_alt"], d_prov_minus_pkr=m["d"], d_raw=round(m["d_raw"], 5), ci_d=m["ci_d"],
                     verdict=m["verdict"], keeps=m["alt_keeps_alignment"], rho_prov_vs_pkr=m["rho_prov_vs_alt"], sub_rho=s2["rho_alt"], sub_d=s2["d"], sub_ci=s2["ci_d"], sub_verdict=s2["verdict"],
                     abroad_kr_docs_P3=OUT[a]["abroad_kr_docs_P3_pkr"],
                     land_H=f"{L['kr_rta_gt1']}/{L['kr_leader']}/{L['kept']}/{L['rho_vs_prov']['KR']}", land_rho_KR_raw=round(L["rho_vs_prov_raw_KR"], 5),
                     land_leaders=json.dumps(L["leaders"]), land_min_rho=min(L["rho_vs_prov"].values()), big_change=L["big_change"],
                     patH_rho=pt["rho_alt"], patH_d=pt["d"], patH_ci=pt["ci_d"]))
T = pd.DataFrame(rows); T.to_csv(W + "/res/s4_rho_table.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 300); print(T.to_string())
json.dump(OUT, open(W + "/res/s4_rho.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done")
