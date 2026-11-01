# -*- coding: utf-8 -*-
r"""C10-T3 s3: c5_R10 c2_rta.py 정의 그대로 ρ(해외 출원 RTA↔수출 RCA)·보조창·지형도·한국 특허 RTA↔RCA를 팔별로 재계산.
팔 세트마다 rng를 seed 20261005로 새로 만들고 c2와 같은 호출 순서(cweq→txt→free→pkr …)를 지켜 같은 재표본을 쓴다.
'stored' 세트는 c5_R10 c2_results.json과 같아야 한다(재현 점검).
읽기만: c5_R10/alt_assign(prov·cweq·pkr), C10-T3/cache/alt_fix_*(txt·free 팔), map5 m4_exports_ksic_5rep.csv, conc_ksic_map.csv, c2_results.json
쓰기: res/s3_rho.json, res/s3_rho_table.csv"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C10/C10-T3"
C5 = MC + "/_shared_report/fusion_advantage/c5_R10"
MAP5 = MC + "/_shared_report/fusion_advantage/map5"
CCS = ["KR", "US", "JP", "CN", "EP"]
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
km = pd.read_csv(MC + "/T3_기술셀부상예측/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]
km["w"] = 1 / km.groupby("code").ksic.transform("size"); con.register("km", km); K50 = sorted(km.ksic)
A0 = C5 + "/alt_assign/*.parquet"; AF = W + "/cache/alt_fix_*.parquet"
base0 = f"(select * from read_parquet('{A0}') where is_inv and appl_year between 2015 and 2019)"
baseF = f"(select * from read_parquet('{AF}') where is_inv and appl_year between 2015 and 2019)"
q = lambda s: con.sql(f"select k, cc, y, o, sum(w) n from ({s}) group by all").df()
FIX = {"prov": q(f"select ksic_prov k, cc, appl_year y, kr_origin o, 1.0 w from {base0}"),
       "pkr": q(f"select k_pkr k, cc, appl_year y, kr_origin o, 1.0 w from {base0} where k_pkr is not null"),
       "cweq": q(f"select km.ksic k, b.cc, b.appl_year y, b.kr_origin o, km.w w from {base0} b join km on km.code=b.code")}
ARMS = ["stored", "pure", "reg", "pub", "uni"]
VAR = {}
for a in ARMS:
    VAR[("txt", a)] = q(f"select k_txt_{a} k, cc, appl_year y, kr_origin o, 1.0 w from {baseF} where k_txt_{a} is not null")
    VAR[("free", a)] = q(f"select k_free_{a} k, cc, appl_year y, kr_origin o, 1.0 w from {baseF} where k_free_{a} is not null")
# 저장 alt_assign txt/free와 alt_fix stored 동일성
chk = {}
for lab, col in (("txt", "k_txt"), ("free", "k_free")):
    o = q(f"select {col} k, cc, appl_year y, kr_origin o, 1.0 w from {base0} where {col} is not null").set_index(["k", "cc", "y", "o"]).n
    s = VAR[(lab, "stored")].set_index(["k", "cc", "y", "o"]).n
    chk[lab] = float((o - s.reindex(o.index)).abs().max())
log("stored == alt_assign 집계 최대차", chk); assert max(chk.values()) == 0
ek = pd.read_csv(MAP5 + "/m4_exports_ksic_5rep.csv", dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); s_ = v / v.sum()
RCA = s_.div(s_.mean(1), axis=0).KR.reindex(K50)
sp = lambda x, y: spearmanr(x, y).statistic
B = 2000


def run(arm):
    C = dict(FIX); C["txt"] = VAR[("txt", arm)]; C["free"] = VAR[("free", arm)]
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
        rp, ra = sp(a, c), sp(b, c); n = len(d); I = rng.integers(0, n, (B, n))
        bp = np.array([sp(a[i], c[i]) for i in I]); ba = np.array([sp(b[i], c[i]) for i in I]); bd = bp - ba
        ci = lambda z: [round(float(np.nanpercentile(z, 2.5)), 3), round(float(np.nanpercentile(z, 97.5)), 3)]
        dd = rp - ra; lo, hi = ci(bd)
        verdict = "제공 우위" if (dd > 0 and lo > 0) else ("대체 우위" if (dd < 0 and hi < 0) else "같다")
        return dict(n=n, rho_prov=round(rp, 3), ci_prov=ci(bp), rho_alt=round(ra, 3), rho_alt_raw=float(ra), ci_alt=ci(ba), d=round(dd, 3), d_raw=float(dd),
                    ci_d=[lo, hi], verdict=verdict, alt_keeps_alignment=bool(ci(ba)[0] > 0))
    R = {}
    ab3 = {lab: abroad(lab, 2017, 2019) for lab in C}; ab2 = {lab: abroad(lab, 2015, 2017) for lab in C}
    R["rho_prov_P3"] = round(sp(*pd.concat([ab3["prov"], RCA], axis=1).dropna().values.T), 4)
    R["main"] = {lab: paired(ab3["prov"], ab3[lab], RCA) for lab in ["cweq", "txt", "free", "pkr"]}
    R["sub_P2"] = {lab: paired(ab2["prov"], ab2[lab], RCA) for lab in ["cweq", "txt", "free", "pkr"]}
    R["abroad_kr_docs_P3"] = {lab: round(float(C[lab][(C[lab].cc != "KR") & C[lab].o & C[lab].y.between(2017, 2019)].n.sum())) for lab in C}
    KRLAB = {"cweq": "cweq", "txt": "txt", "free": "free", "pkr": "prov"}
    rt_prov = rta_ew(counts("prov", "prov", 2017, 2019)); lp = rt_prov.idxmax(1); krp = set(lp.index[lp == "KR"])
    land = {}
    for lab in ["cweq", "txt", "free", "pkr"]:
        for scen, lk in (("F", KRLAB[lab]), ("H", "prov")):
            if lab == "pkr" and scen == "F": continue
            rt = rta_ew(counts(lk, lab, 2017, 2019)); ld = rt.idxmax(1); krl = set(ld.index[ld == "KR"])
            land[f"{lab}_{scen}"] = dict(kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=int(len(krl)), kept=int(len(krl & krp)),
                                        rho_vs_prov={c: round(sp(rt[c], rt_prov[c]), 3) for c in CCS},
                                        big_change=bool(len(krl & krp) < 4 or min(sp(rt[c], rt_prov[c]) for c in CCS) < 0.7))
    R["landscape"] = land
    rtp2 = rta_ew(counts("prov", "prov", 2015, 2017)).KR; sub = {}
    for lab in ["cweq", "txt", "free", "pkr"]:
        for scen, lk in (("F", KRLAB[lab]), ("H", "prov")):
            if lab == "pkr" and scen == "F": continue
            sub[f"{lab}_{scen}"] = paired(rtp2, rta_ew(counts(lk, lab, 2015, 2017)).KR, RCA)
    R["patRTA_P2"] = sub
    R["_abroad_P3"] = {lab: ab3[lab].round(5).to_dict() for lab in ("prov", "txt", "free")}
    return R


OUT = {a: run(a) for a in ARMS}
# 재현 점검: stored == c2_results.json
c2 = json.load(open(C5 + "/c2_results.json", encoding="utf-8"))
rep = {}
for lab in ["cweq", "txt", "free", "pkr"]:
    o, s = c2["main_abroadRTA_P3_vs_RCA"][lab], OUT["stored"]["main"][lab]
    rep[f"main_{lab}"] = dict(rho=(o["rho_alt"], s["rho_alt"]), d=(o["d"], s["d"]), ci_d=(o["ci_d"], s["ci_d"]), ci_alt=(o["ci_alt"], s["ci_alt"]),
                              same=o["rho_alt"] == s["rho_alt"] and o["ci_d"] == s["ci_d"] and o["ci_alt"] == s["ci_alt"])
    o2, s2 = c2["sub_abroadRTA_P2_vs_RCA"][lab], OUT["stored"]["sub_P2"][lab]
    rep[f"sub_{lab}"] = dict(d=(o2["d"], s2["d"]), ci=(o2["ci_d"], s2["ci_d"]), same=o2["d"] == s2["d"] and o2["ci_d"] == s2["ci_d"])
for k, v in c2["landscape_P3"].items():
    if k == "prov": continue
    s = OUT["stored"]["landscape"][k]
    rep[f"land_{k}"] = dict(orig=(v["kr_rta_gt1"], v["kr_leader"], v["kr_leader_kept_of_prov7"], v["rho_vs_prov"]["KR"]), mine=(s["kr_rta_gt1"], s["kr_leader"], s["kept"], s["rho_vs_prov"]["KR"]))
    rep[f"land_{k}"]["same"] = rep[f"land_{k}"]["orig"] == rep[f"land_{k}"]["mine"]
for k, v in c2["sub_patRTA_P2_vs_RCA"].items():
    s = OUT["stored"]["patRTA_P2"][k]; rep[f"pat_{k}"] = dict(orig=(v["rho_alt"], v["d"], v["ci_d"]), mine=(s["rho_alt"], s["d"], s["ci_d"]))
    rep[f"pat_{k}"]["same"] = rep[f"pat_{k}"]["orig"] == rep[f"pat_{k}"]["mine"]
OUT["repro_vs_c2"] = rep; OUT["repro_all_same"] = all(v["same"] for v in rep.values())
log("재현", OUT["repro_all_same"], {k: v for k, v in rep.items() if not v["same"]})
rows = []
for a in ARMS:
    for lab in ("txt", "free"):
        m = OUT[a]["main"][lab]; s2 = OUT[a]["sub_P2"][lab]
        rows.append(dict(arm=a, lab=lab, rho_alt=m["rho_alt"], rho_alt_raw=round(m["rho_alt_raw"], 5), ci_alt=m["ci_alt"], d=m["d"], ci_d=m["ci_d"], verdict=m["verdict"],
                         keeps=m["alt_keeps_alignment"], sub_d=s2["d"], sub_ci=s2["ci_d"], sub_verdict=s2["verdict"],
                         abroad_kr_docs=OUT[a]["abroad_kr_docs_P3"][lab],
                         F=(lambda L: f"{L['kr_rta_gt1']}/{L['kr_leader']}/{L['kept']}/{L['rho_vs_prov']['KR']}")(OUT[a]["landscape"][f"{lab}_F"]),
                         H=(lambda L: f"{L['kr_rta_gt1']}/{L['kr_leader']}/{L['kept']}/{L['rho_vs_prov']['KR']}")(OUT[a]["landscape"][f"{lab}_H"]),
                         patF=OUT[a]["patRTA_P2"][f"{lab}_F"]["rho_alt"], patH=OUT[a]["patRTA_P2"][f"{lab}_H"]["rho_alt"],
                         patH_d=OUT[a]["patRTA_P2"][f"{lab}_H"]["d"], patH_ci=OUT[a]["patRTA_P2"][f"{lab}_H"]["ci_d"]))
T = pd.DataFrame(rows); T.to_csv(W + "/res/s3_rho_table.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 300); print(T.to_string())
json.dump(OUT, open(W + "/res/s3_rho.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done")
