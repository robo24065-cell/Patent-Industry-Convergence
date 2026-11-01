# -*- coding: utf-8 -*-
r"""c5_R10 c2: map5 지표(한국 기원 해외 출원 RTA ↔ 수출 RCA, 5개국 RTA 지형도)를 해외 라벨 대체안으로 다시 계산.
입력(읽기만): alt_assign/*.parquet(c1), map5/m4_exports_ksic_5rep.csv, map5/m2_ksic_profile.csv(재현 대조), T3 conc_ksic_map.csv
정의·판정은 PREREG.txt (2)."""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr, rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(HERE, r"..\..\.."))
MAP5 = os.path.join(MC, r"_shared_report\fusion_advantage\map5")
CCS = ["KR", "US", "JP", "CN", "EP"]; FOR = ["US", "JP", "CN", "EP"]
t = time.time(); log = lambda *a: print(*a, f"[{time.time() - t:.0f}s]", flush=True)
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
A = os.path.join(HERE, "alt_assign", "*.parquet").replace("\\", "/")
km = pd.read_csv(os.path.join(MC, r"T3_기술셀부상예측\ESSENTIAL_R7\conc_ksic_map.csv"), dtype=str)[["ksic", "code"]]
km["w"] = 1 / km.groupby("code").ksic.transform("size"); con.register("km", km)
K50 = sorted(km.ksic)
# 문서 단위 → (라벨링, ksic, cc, year, kr_origin) 가중 건수. 발명특허, 출원 2015–2019만(지표 창).
base = f"(select * from read_parquet('{A}') where is_inv and appl_year between 2015 and 2019)"
parts = {
    "prov": f"select ksic_prov k, cc, appl_year y, kr_origin o, 1.0 w from {base}",
    "txt": f"select k_txt k, cc, appl_year y, kr_origin o, 1.0 w from {base} where k_txt is not null",
    "free": f"select k_free k, cc, appl_year y, kr_origin o, 1.0 w from {base} where k_free is not null",
    "pkr": f"select k_pkr k, cc, appl_year y, kr_origin o, 1.0 w from {base} where k_pkr is not null",
    "cweq": f"select km.ksic k, b.cc, b.appl_year y, b.kr_origin o, km.w w from {base} b join km on km.code=b.code",
}
C = {}
for lab, q in parts.items():
    C[lab] = con.sql(f"select k, cc, y, o, sum(w) n from ({q}) group by all").df()
    log(lab, round(C[lab].n.sum()), "건(가중)")
ek = pd.read_csv(os.path.join(MAP5, "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); s = v / v.sum()
RCA = s.div(s.mean(1), axis=0).KR.reindex(K50)


def counts(lab_kr, lab_for, y0, y1):
    """KSIC × cc 건수: KR은 lab_kr 라벨링, 해외는 lab_for 라벨링."""
    out = {}
    for cc in CCS:
        d = C[lab_kr if cc == "KR" else lab_for]
        out[cc] = d[(d.cc == cc) & d.y.between(y0, y1)].groupby("k").n.sum()
    return pd.DataFrame(out).reindex(K50).fillna(0)


def rta_ew(n):
    sh = n / n.sum()
    return sh.div(sh.mean(1), axis=0)


def abroad(lab, y0, y1):
    d = C[lab]; d = d[(d.cc != "KR") & d.y.between(y0, y1)]
    k = d[d.o].groupby("k").n.sum().reindex(K50).fillna(0); a = d.groupby("k").n.sum().reindex(K50)
    return (k / a) / (k.sum() / a.sum())


rng = np.random.default_rng(20261005); B = 2000


def sp(x, y):
    return spearmanr(x, y).statistic


def paired(xp, xa, y):
    """ρ(xp,y) − ρ(xa,y), KSIC 쌍 부트스트랩. 세 열이 모두 있는 KSIC만."""
    d = pd.concat([xp, xa, y], axis=1).dropna(); a, b, c = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
    rp, ra = sp(a, c), sp(b, c); n = len(d)
    I = rng.integers(0, n, (B, n))
    bp = np.array([sp(a[i], c[i]) for i in I]); ba = np.array([sp(b[i], c[i]) for i in I]); bd = bp - ba
    ci = lambda z: [round(float(np.nanpercentile(z, 2.5)), 3), round(float(np.nanpercentile(z, 97.5)), 3)]
    dd = rp - ra; lo, hi = ci(bd)
    verdict = "제공 우위" if (dd > 0 and lo > 0) else ("대체 우위" if (dd < 0 and hi < 0) else "같다")
    return dict(n=n, rho_prov=round(rp, 3), ci_prov=ci(bp), rho_alt=round(ra, 3), ci_alt=ci(ba), d=round(dd, 3), ci_d=[lo, hi], verdict=verdict,
                alt_keeps_alignment=bool(ci(ba)[0] > 0), rho_prov_vs_alt=round(sp(a, b), 3))


R = {"definitions": "PREREG.txt (2)", "alts": ["cweq", "txt", "free", "pkr(진단)"]}
# ---- 재현 ----
prof = pd.read_csv(os.path.join(MAP5, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")
ab_p3 = {lab: abroad(lab, 2017, 2019) for lab in C}; ab_p2 = {lab: abroad(lab, 2015, 2017) for lab in C}
R["repro_abroad_rta_maxabs_vs_map5"] = float((ab_p3["prov"] - prof.kr_origin_abroad_rta_P3.reindex(K50)).abs().max())
r0 = sp(*pd.concat([ab_p3["prov"], RCA], axis=1).dropna().values.T)
R["repro_rho_abroad_P3"] = round(r0, 4); assert abs(r0 - 0.545) <= 0.005, r0
n3_prov = counts("prov", "prov", 2017, 2019); rt_prov = rta_ew(n3_prov)
R["repro_rta_P3_maxabs_vs_map5"] = float(max((rt_prov[c] - prof[f"rta_{c}_P3"].reindex(K50)).abs().max() for c in CCS))
log("재현", R["repro_rho_abroad_P3"], R["repro_abroad_rta_maxabs_vs_map5"], R["repro_rta_P3_maxabs_vs_map5"])
# ---- (2) 주 지표 ----
R["main_abroadRTA_P3_vs_RCA"] = {lab: paired(ab_p3["prov"], ab_p3[lab], RCA) for lab in ["cweq", "txt", "free", "pkr"]}
R["sub_abroadRTA_P2_vs_RCA"] = {lab: paired(ab_p2["prov"], ab_p2[lab], RCA) for lab in ["cweq", "txt", "free", "pkr"]}
log("주 지표", {k: (v["rho_alt"], v["d"], v["ci_d"], v["verdict"]) for k, v in R["main_abroadRTA_P3_vs_RCA"].items()})
# KR 기원 해외 문서 수(분자) 보존
R["abroad_kr_docs_P3"] = {lab: round(float(C[lab][(C[lab].cc != "KR") & C[lab].o & C[lab].y.between(2017, 2019)].n.sum())) for lab in C}
# ---- 지형도 (P3 2017–19), F·H 시나리오 ----
KRLAB = {"cweq": "cweq", "txt": "txt", "free": "free", "pkr": "prov"}
lead_prov = rt_prov.idxmax(1); kr_lead_prov = set(lead_prov.index[lead_prov == "KR"])
land = {"prov": dict(kr_rta_gt1=int((rt_prov.KR > 1).sum()), kr_leader=int(len(kr_lead_prov)),
                     leaders={c: int((lead_prov == c).sum()) for c in CCS}, kr_leader_list=sorted(kr_lead_prov))}
tabs = {"prov": rt_prov.add_prefix("prov_")}
for lab in ["cweq", "txt", "free", "pkr"]:
    for scen, lk in (("F", KRLAB[lab]), ("H", "prov")):
        if lab == "pkr" and scen == "F": continue   # pkr F = H (KR = 제공)
        rt = rta_ew(counts(lk, lab, 2017, 2019)); ld = rt.idxmax(1); krl = set(ld.index[ld == "KR"])
        land[f"{lab}_{scen}"] = dict(kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=int(len(krl)), kr_leader_kept_of_prov7=int(len(krl & kr_lead_prov)),
                                    leaders={c: int((ld == c).sum()) for c in CCS},
                                    rho_vs_prov={c: round(sp(rt[c], rt_prov[c]), 3) for c in CCS},
                                    kr_rta_gt1_agree=float(((rt.KR > 1) == (rt_prov.KR > 1)).mean()),
                                    big_change=bool(len(krl & kr_lead_prov) < 4 or min(sp(rt[c], rt_prov[c]) for c in CCS) < 0.7))
        tabs[f"{lab}_{scen}"] = rt.add_prefix(f"{lab}_{scen}_")
R["landscape_P3"] = land
log("지형도", {k: (v["kr_rta_gt1"], v["kr_leader"], v.get("kr_leader_kept_of_prov7"), v.get("rho_vs_prov")) for k, v in land.items()})
# ---- 보조: 한국 특허 RTA(P2) ↔ RCA, F·H ----
rtp2 = rta_ew(counts("prov", "prov", 2015, 2017)).KR
sub = {}
for lab in ["cweq", "txt", "free", "pkr"]:
    for scen, lk in (("F", KRLAB[lab]), ("H", "prov")):
        if lab == "pkr" and scen == "F": continue
        sub[f"{lab}_{scen}"] = paired(rtp2, rta_ew(counts(lk, lab, 2015, 2017)).KR, RCA)
R["repro_rho_patRTA_P2"] = round(sp(*pd.concat([rtp2, RCA], axis=1).dropna().values.T), 4)
R["sub_patRTA_P2_vs_RCA"] = sub
# ---- 표 저장 ----
T = pd.concat([pd.DataFrame({"rca_P2": RCA}), pd.DataFrame({f"abroadP3_{k}": v for k, v in ab_p3.items()}),
               pd.DataFrame({f"abroadP2_{k}": v for k, v in ab_p2.items()})] + list(tabs.values()), axis=1)
T.index.name = "ksic"; T.round(4).to_csv(os.path.join(HERE, "c2_ksic_table.csv"), encoding="utf-8-sig")
json.dump(R, open(os.path.join(HERE, "c2_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float)[:9000])
log("end")
