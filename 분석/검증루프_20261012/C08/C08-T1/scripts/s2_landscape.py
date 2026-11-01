# -*- coding: utf-8 -*-
# 2단계: 3.1.1 RTA, 3.1.2 ρ·표 5, 3.1.3 분산 비중·C2620 — S0(재현)·S1·S1-KR·S2, 집중도 표
import sys, os, json, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group
KEY = sys.argv[1] if len(sys.argv) > 1 else "a2"   # a2 = N2 주 키, a0 = 원문 보조 키
t0 = time.time()
A = pd.read_parquet(os.path.join(CACHE, "s1_agg.parquet"))
A["ksic"] = A.ksic.astype(str)
names = pd.read_parquet(os.path.join(CACHE, "s1_names.parquet"))
IDC = "id_n2" if KEY == "a2" else "id_n0"
idname = names.sort_values("n", ascending=False).drop_duplicates(IDC).set_index(IDC).raw
K50 = sorted(A.ksic.unique()); assert len(K50) == 50


def win(years, inv_only, cnt):
    d = A[A.yr.isin(list(years))]
    if inv_only:
        d = d[d.is_inv]
    a = d.groupby(["ksic", "cc", KEY])[[cnt]].sum().rename(columns={cnt: "n"})
    b = d[d.kro].groupby(["ksic", "cc", KEY])[[cnt]].sum().rename(columns={cnt: "nk"})
    return a.join(b, how="left").fillna(0).reset_index()


def top1_ids(d):
    x = d[d[KEY] >= 0].sort_values(["ksic", "cc", "n", KEY], ascending=[True, True, False, True])
    return x.drop_duplicates(["ksic", "cc"]).set_index(["ksic", "cc"])[KEY]


def cap_weights(n, cap=0.2):
    """출원인별 문서 수 n(결측 제외) → 상한 λ (Σmin(n,λ)에서 λ=cap·Σmin(n,λ)). 출원인 <1/cap이면 None."""
    v = np.sort(np.asarray(n, float))[::-1]
    m = len(v)
    if m < int(round(1 / cap)):
        return None
    tot = v.sum()
    if v[0] <= cap * tot:
        return 1e18
    for k in range(1, m):
        if 1 - cap * k <= 0:
            break
        R = v[k:].sum()
        lam = cap * R / (1 - cap * k)
        if v[k - 1] >= lam >= v[k]:
            return lam
    return None


def apply_variant(d, var, top=None):
    """d: ksic,cc,KEY,n,nk → 변형 적용 후 (ksic,cc) 합계 n, nk"""
    d = d.copy()
    if var in ("S1", "S1KR"):
        tp = (top if top is not None else top1_ids(d)).rename("top").reset_index()
        if var == "S1KR":
            tp = tp[tp.cc == "KR"]
        d = d.merge(tp, on=["ksic", "cc"], how="left")
        d = d[~(d[KEY] == d.top)].drop(columns="top")
    elif var == "S2":
        out = []
        for (k, c), g in d.groupby(["ksic", "cc"]):
            gm = g[g[KEY] >= 0]; gx = g[g[KEY] < 0]
            lam = cap_weights(gm.n.values)
            if lam is None:   # 출원인 5개 미만: 서로 다른 출원인 수
                w = np.where(gm.n > 0, 1.0 / gm.n.clip(lower=1e-9), 0)
            else:
                w = np.minimum(gm.n, lam) / gm.n
            out.append((k, c, float((gm.n * w).sum() + gx.n.sum()), float((gm.nk * w).sum() + gx.nk.sum())))
        return pd.DataFrame(out, columns=["ksic", "cc", "n", "nk"]).set_index(["ksic", "cc"])
    elif var == "S2b":
        g = d[d[KEY] >= 0]
        r = g.groupby(["ksic", "cc"]).agg(n=("n", "size"))
        r["nk"] = g[g.nk > 0].groupby(["ksic", "cc"]).size().reindex(r.index).fillna(0)
        return r
    return d.groupby(["ksic", "cc"])[["n", "nk"]].sum()


def rta_from(tot):
    n = tot.n.unstack().reindex(index=K50, columns=CCS).fillna(0)
    s = n / n.sum()
    return s.div(s.mean(1), axis=0)


def conc(d, label, defn):
    rows = []
    for (k, c), g in d.groupby(["ksic", "cc"]):
        gm = g[g[KEY] >= 0].sort_values("n", ascending=False)
        N = g.n.sum(); Nm = gm.n.sum()
        sh = gm.n.values / Nm if Nm > 0 else np.array([])
        hhi = float((sh ** 2).sum()) if Nm > 0 else np.nan
        rows.append(dict(ksic=k, cc=c, period=label, defn=defn, n_docs=int(N), n_missing_app=int(N - Nm), n_app=int(len(gm)),
                         top1_name=idname.get(int(gm[KEY].iloc[0]), "") if len(gm) else "",
                         top1_share=float(sh[:1].sum()) if Nm else np.nan, top2_share=float(sh[:2].sum()) if Nm else np.nan,
                         top5_share=float(sh[:5].sum()) if Nm else np.nan, hhi=hhi, eff_n=1 / hhi if hhi and hhi > 0 else np.nan))
    return rows


C = []
PER_INV = {"P1": range(2008, 2011), "P2": range(2015, 2018), "P3": range(2017, 2020)}
PER_ALL = {"E0507": range(2005, 2008), "L1517": range(2015, 2018), "W0809": [2008, 2009], "W1617": [2016, 2017]}
W = {}
for p, ys in PER_INV.items():
    W[p] = win(ys, True, "n_rows"); C += conc(W[p], p, "발명특허 행(map5)")
for p, ys in PER_ALL.items():
    W[p] = win(ys, False, "n_uniq"); C += conc(W[p], p, "전 유형 고유 출원(split34)")
CB = pd.DataFrame(C)
CB.to_csv(os.path.join(WD if KEY == "a2" else CACHE, "concentration_by_cell.csv" if KEY == "a2" else f"concentration_by_cell_{KEY}.csv"), index=False, encoding="utf-8-sig")
print("conc done", round(time.time() - t0), flush=True)

res = {"key": KEY}
VARS = ["S0", "S1", "S1KR", "S2", "S2b"]
# ---------------- 3.1.1 RTA P3 ----------------
prof = pd.read_csv(os.path.join(MAP5, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")
orig_leaders = sorted(prof.index[prof.leader_P3 == "KR"])
res["orig_KR_leaders_P3"] = orig_leaders
rt = {v: rta_from(apply_variant(W["P3"], v)) for v in VARS}
for v in VARS:
    r = rt[v]; lead = r.idxmax(1)
    res[f"P3_{v}"] = dict(kr_rta_gt1=int((r.KR > 1).sum()), kr_leader=int((lead == "KR").sum()),
                         leaders=lead.value_counts().to_dict(), kr_leaders=sorted(r.index[lead == "KR"]),
                         kept_orig_leaders=int(sum(lead[k] == "KR" for k in orig_leaders)),
                         rta_26212=round(r.at["26212", "KR"], 3), rta_31312=round(r.at["31312", "KR"], 3), rta_20423=round(r.at["20423", "KR"], 3),
                         lead_26212=lead["26212"], lead_31312=lead["31312"], lead_20423=lead["20423"],
                         rho_KR_vs_S0=round(spearmanr(r.KR, rt["S0"].KR).statistic, 3),
                         max_abs_diff_vs_m2=round(float((r.KR - prof.rta_KR_P3).abs().max()), 6) if v == "S0" else None)
pd.concat({v: rt[v].KR for v in VARS}, axis=1).to_csv(os.path.join(CACHE, f"s2_rta_P3_KR_{KEY}.csv"), encoding="utf-8-sig")
pd.concat({v: rt[v] for v in VARS}, axis=1).to_csv(os.path.join(CACHE, f"s2_rta_P3_all_{KEY}.csv"), encoding="utf-8-sig")
print("rta done", round(time.time() - t0), flush=True)

# ---------------- 3.1.2 ρ·표 5 (P2) ----------------
ek = pd.read_csv(os.path.join(MAP5, "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack()
s = v / v.sum(); rca = s.div(s.mean(1), axis=0).KR


def sp_ci(a, b, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = pd.concat([a, b], axis=1).dropna(); x, y = d.iloc[:, 0].values, d.iloc[:, 1].values
    r = spearmanr(x, y).statistic
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return dict(rho=round(r, 3), ci=[round(np.nanpercentile(bs, 2.5), 3), round(np.nanpercentile(bs, 97.5), 3)], n=len(x))


EX = {"SS": ["20423", "20202", "26121"], "SW": ["31312", "20491", "26291"], "WS": ["29280", "26224", "26519"], "WW": ["21220", "26323"]}
qt = pd.read_csv(os.path.join(MAP5, "m4_quadrant_table.csv"), dtype={"ksic": str}).set_index("ksic")
r2s = {}
for vv in VARS:
    r2 = rta_from(apply_variant(W["P2"], vv)).KR; r2s[vv] = r2
    rc = rca.reindex(r2.index)
    q = pd.Series(np.where(rc.isna(), "NA", np.where(r2 >= 1, np.where(rc >= 1, "SS", "SW"), np.where(rc >= 1, "WS", "WW"))), index=r2.index)
    res[f"P2_{vv}"] = dict(rho=sp_ci(r2, rc), quad_counts=q.value_counts().to_dict(),
                          examples={g: {k: q[k] for k in ks} for g, ks in EX.items()},
                          examples_flipped=[k for g, ks in EX.items() for k in ks if q[k] != g],
                          sw_list=sorted(q.index[q == "SW"]), n_quad_changed_vs_m4=int((q != qt.quad_tech_export_P2.map(
                              lambda z: "NA" if z == "판정불가" else ("SS" if z.startswith("기술 강·수출 강") else "SW" if z.startswith("기술 강") else "WS" if "수출 강" in z else "WW")).reindex(q.index)).sum()),
                          max_abs_diff_vs_m4=round(float((r2 - qt.pat_rta_P2).abs().max()), 6) if vv == "S0" else None)
pd.concat(r2s, axis=1).assign(rca=rca.reindex(K50)).to_csv(os.path.join(CACHE, f"s2_rta_P2_KR_{KEY}.csv"), encoding="utf-8-sig")
print("rho done", round(time.time() - t0), flush=True)

# ---------------- 3.1.3 분산 비중·C2620 ----------------
G = pd.Series({k: ksic_to_group(k) for k in K50}); gs = G.value_counts()
K34 = sorted([k for k in K50 if gs[G[k]] > 1]); assert len(K34) == 34


def within_share(vals, grp):
    d = pd.DataFrame({"v": vals, "g": grp}).dropna()
    return float(((d.v - d.groupby("g").v.transform("mean")) ** 2).sum() / ((d.v - d.v.mean()) ** 2).sum())


def two_window(a_years, b_years, var):
    da = win(a_years, False, "n_uniq"); db = win(b_years, False, "n_uniq")
    tp = None
    if var in ("S1", "S1KR"):
        both = pd.concat([da, db]).groupby(["ksic", "cc", KEY])[["n", "nk"]].sum().reset_index()
        tp = top1_ids(both)
    return apply_variant(da, var, tp), apply_variant(db, var, tp)


m313 = {}
for vv in VARS:
    ta, tb = two_window([2005, 2006, 2007], [2015, 2016, 2017], vv)
    U = lambda t, c: t[c].unstack().reindex(index=K50, columns=CCS).fillna(0)
    na, nb, ka, kb = U(ta, "n"), U(tb, "n"), U(ta, "nk"), U(tb, "nk")
    lg = lambda x, y: (np.log(y / 3 + 1) - np.log(x / 3 + 1)) / 10
    g_KR = lg(na.KR, nb.KR); g_5 = lg(na.sum(1), nb.sum(1))
    sh_ch = 100 * (kb.sum(1) / nb.sum(1) - ka.sum(1) / na.sum(1))
    wa, wb = two_window([2008, 2009], [2016, 2017], vv)
    xa = U(wa, "n").KR; xb = U(wb, "n").KR
    gw = (np.log(xb / 2 + 1) - np.log(xa / 2 + 1)) / 8
    m313[vv] = pd.DataFrame({"g_KR": g_KR, "g_5": g_5, "kro_share_change_pp": sh_ch, "gw_KR_0809_1617": gw, "grp": G})
    res[f"S313_{vv}"] = dict(within_g_KR=round(within_share(g_KR[K34], G[K34]), 3),
                            within_krorigin_share_change=round(within_share(sh_ch[K34], G[K34]), 3),
                            within_g_5=round(within_share(g_5[K34], G[K34]), 3),
                            gw_KR_C2620={k: round(100 * gw[k], 1) for k in K50 if G[k] == "C2620"},
                            gw_KR_26111=round(100 * gw["26111"], 1), gw_KR_29280=round(100 * gw["29280"], 1),
                            gw_KR_C2610={k: round(100 * gw[k], 1) for k in K50 if G[k] == "C2610"})
pd.concat(m313, axis=1).to_csv(os.path.join(CACHE, f"s2_313_metrics_{KEY}.csv"), encoding="utf-8-sig")
print("313 done", round(time.time() - t0), flush=True)
json.dump(res, open(os.path.join(CACHE, f"s2_results_{KEY}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
