# -*- coding: utf-8 -*-
"""s5: 대안 A1(HS 버전 정합)·A2(가중 분할)·A3(전 품목 분모)·A4(조합) 재계산 → quadrant_sensitivity.csv, hs_revision_diag.csv."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

mp = load_map(); codes = set(mp.hs6)
ex = load_ex().drop_duplicates(["year", "reporter", "hs6"])
dl = pd.read_csv(os.path.join(CACHE, "s4_downloads_parsed.csv"), dtype={"hs6": str})
tot = pd.read_csv(os.path.join(CACHE, "comtrade_total_rep_year.csv")).set_index("rep"); tot.columns = tot.columns.astype(int)
c12 = pd.read_csv(os.path.join(CACHE, "unsd_corr_17_12.csv"), dtype=str)
c07 = pd.read_csv(os.path.join(CACHE, "unsd_corr_17_07.csv"), dtype=str)
rta = patent_rta()
prof = pd.read_csv(os.path.join(MAP5, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")
ko = pd.read_csv(os.path.join(ROOT, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
ko = ko[(ko.마스킹 == 0) & (ko.기준 == "10차") & (ko.항목코드 == "T04") & ko.연도.isin([2015, 2016, 2017])]
ship = ko.groupby("세세분류코드").값.mean()
YRS = [2008, 2009, 2010, 2015, 2016, 2017, 2021]
VER = {2008: "H3", 2009: "H3", 2010: "H3", 2015: "H4", 2016: "H4", 2017: "H5", 2021: "H5"}


def ksic_w(scheme):
    m = mp.drop_duplicates(["k5", "hs6"]).copy()
    if scheme == "equal":
        m["w"] = 1 / m.groupby("hs6").k5.transform("nunique")
    else:
        s = rta["nKR_P2"] if scheme == "patent" else ship
        m["s"] = m.k5.map(s).fillna(0)
        g = m.groupby("hs6").s.transform("sum")
        m["w"] = np.where(g > 0, m.s / g.replace(0, np.nan), 1 / m.groupby("hs6").k5.transform("nunique"))
    return m[["k5", "hs6", "w"]]


def old_to_new(corr, mode):
    """구 코드 → (신 코드, 몫). mode equal: 1/|T(o)|, lo: T(o)⊆대응표일 때만, hi: 대응표 안 대상으로 재정규화."""
    t = corr.copy(); t["nT"] = t.groupby("old").new.transform("nunique")
    if mode == "cont":  # 사후 변형: 같은 코드가 HS2017에 이어지면 그 코드로 100%, 폐지된 구 코드만 대상에 균등 분할
        cont = t.groupby("old").apply(lambda g: g.name in set(g.new), include_groups=False)
        t["cont"] = t.old.map(cont)
        z = pd.concat([t[t.cont & (t.new == t.old)].assign(f=1.0), t[~t.cont].assign(f=lambda d: 1 / d.nT)])
        return z[z.new.isin(codes)][["old", "new", "f"]]
    t["inmap"] = t.new.isin(codes)
    t["nT_in"] = t.groupby("old").inmap.transform("sum")
    t = t[t.inmap]
    if mode == "equal":
        t["f"] = 1 / t.nT
    elif mode == "lo":
        t = t[t.nT_in == t.nT]; t["f"] = 1 / t.nT
    else:
        t["f"] = 1 / t.nT_in
    return t[["old", "new", "f"]]


def rows_year(y, a1):
    """연도 y의 (rep, hs6[신코드], v) — a1=None 이면 원본."""
    e = ex[ex.year == y][["year", "rep", "hs6", "export_usd"]].rename(columns={"export_usd": "v"})
    if a1 is None or VER[y] == "H5":
        return e
    corr = c12 if VER[y] == "H4" else c07
    add = dl[(dl.year == y) & dl.key.str.startswith("OLD" + VER[y])][["year", "rep", "hs6", "v"]]
    o = pd.concat([e, add]).drop_duplicates(["year", "rep", "hs6"])
    o = o[o.hs6.isin(set(corr.old))]
    tr = old_to_new(corr, a1)
    z = o.merge(tr, left_on="hs6", right_on="old")
    z["v"] = z.v * z.f
    return z.groupby(["year", "rep", "new"]).v.sum().reset_index().rename(columns={"new": "hs6"})


def build(a1=None, scheme="equal"):
    rows = pd.concat([rows_year(y, a1) for y in YRS])
    return ksic_value(rows, ksic_w(scheme)), rows


def run(label, a1=None, scheme="equal", denom=False):
    ek, rows = build(a1, scheme)
    dn = tot if denom else None
    p2, v2 = rca_from(ek, [2015, 2016, 2017], dn); p1, _ = rca_from(ek, [2008, 2009, 2010], dn); p4, _ = rca_from(ek, [2021], dn)
    out, q = evaluate(rta, p2, p1, p4, label)
    return out, q, p2, ek


variants = [("B_original", None, "equal", False), ("A1_hs_align_equal", "equal", "equal", False), ("A1lo_hs_align_strict", "lo", "equal", False),
            ("A1hi_hs_align_renorm", "hi", "equal", False), ("A2a_weight_KRpatent", None, "patent", False), ("A2b_weight_KOSISship", None, "ship", False),
            ("A3_alltrade_denominator", None, "equal", True), ("A4_A1+A2a+A3", "equal", "patent", True), ("A4b_A1+A2b+A3", "equal", "ship", True),
            ("P_A1c_continuity(사후)", "cont", "equal", False), ("P_A1c+A3(사후)", "cont", "equal", True), ("P_A1c+A2a+A3(사후)", "cont", "patent", True)]
outs = []; Q = {}; P2 = {}; EK = {}
for lab, a1, sc, dn in variants:
    o, q, p2, ek = run(lab, a1, sc, dn); outs.append(o); Q[lab] = q; P2[lab] = p2; EK[lab] = ek
    print(lab, {k: o[k] for k in ["SS", "SW", "WS", "WW", "NA", "rho", "rho_lo", "rho_hi", "rho_chg", "list42_out", "list42_in", "persist"]}, flush=True)
S = pd.DataFrame(outs)
base = S.iloc[0]
S["dSW"] = S.SW - base.SW
for c in ["SS", "WS", "WW"]:
    S["d" + c] = S[c] - base[c]
S["SW_removed"] = [" ".join(sorted(set(base.SW_list.split()) - set(s.split()))) for s in S.SW_list]
S["SW_added"] = [" ".join(sorted(set(s.split()) - set(base.SW_list.split()))) for s in S.SW_list]


def verdict(r):
    if r.variant.startswith("B_"): return "기준"
    if abs(r.dSW) > 3 or (r.rho_lo <= 0 <= r.rho_hi) or max(r.list42_out, r.list42_in) > 3: return "수정 필요"
    if abs(r.dSW) >= 1 or max(r.list42_out, r.list42_in) >= 1 or max(abs(r.dSS), abs(r.dWS), abs(r.dWW)) > 3: return "단서 필요"
    return "유지"


S["prereg_verdict"] = S.apply(verdict, axis=1)
S["judged"] = S.variant.isin(["A1_hs_align_equal", "A2a_weight_KRpatent", "A2b_weight_KOSISship", "A3_alltrade_denominator"])
# 표 5 산업 예 소속
ex_names = {"31312": "무인항공기", "20491": "감광재료", "26291": "전자축전기", "29280": "산업용로봇", "26224": "실장기판", "26519": "영상기기",
            "20423": "화장품", "20202": "합성수지", "26121": "발광다이오드", "21220": "한의약품", "26323": "컴퓨터프린터"}
S["table5_examples"] = [" ".join(f"{n}:{Q[v].get(k, 'NA')}" for k, n in ex_names.items()) for v in S.variant]
# 경계 근처(취약 판정): 원본 SW 중 RCA∈[0.9,1) 또는 RTA∈[1,1.1)
p2b = P2["B_original"]
sw = base.SW_list.split()
S.loc[0, "note"] = "원본 SW 중 경계(RTA<1.1 또는 RCA≥0.9): " + " ".join(k for k in sw if rta["P2"][k] < 1.1 or p2b[k] >= 0.9)
S.to_csv(os.path.join(W, "quadrant_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ---------- 추가(판정 제외): IO 단위 수출 단위 수, Balassa 합산 분모 ----------
extra = {}
io_struct = pd.read_csv(os.path.join(CACHE, "s2c_io_structure.csv"), dtype=str)
k2io = mp.drop_duplicates(["k5", "io"]).groupby("k5").io.apply(lambda s: " ".join(sorted(s)))
extra["SW_distinct_export_units"] = int(k2io.reindex(sw).nunique())
extra["SW_export_unit_groups"] = k2io.reindex(sw).to_dict()
# 원본 사분면에서 IO 쌍이 같은 RCA인지 확인
pairs = [("26121", "26129"), ("26111", "26112"), ("26223", "26224"), ("26322", "26323"), ("27211", "27212"), ("29131", "29132"), ("33301", "33309")]
extra["io_twin_rca_P2"] = {f"{a}-{b}": [round(float(p2b[a]), 4), round(float(p2b[b]), 4), Q["B_original"][a], Q["B_original"][b]] for a, b in pairs}
# Balassa(5개국 합산) 전 품목
ek0 = EK["B_original"]
v = ek0[ek0.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack()
T = tot[[2015, 2016, 2017]].sum(1)
bal = (v.KR / T["KR"]) / (v.sum(1) / T[REPS].sum())
qb = quad(rta["P2"], bal.reindex(rta["P2"].index))
extra["balassa_pooled_alltrade"] = dict(counts=qb.value_counts().to_dict(), rho=sp_ci(rta["P2"], bal))
json.dump(extra, open(os.path.join(CACHE, "s5_extra.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(extra, ensure_ascii=False, indent=1, default=str)[:3000])

# ---------- hs_revision_diag.csv (KSIC별) ----------
ident = pd.read_csv(os.path.join(CACHE, "s3a_map_code_versions.csv"), dtype={"hs6": str})
id12 = set(ident[(ident.ver == "H4") & ident.same_code].hs6); id07 = set(ident[(ident.ver == "H3") & ident.same_code].hs6)
s3a = json.load(open(os.path.join(CACHE, "s3a_out.json"), encoding="utf-8"))
abs12 = set(s3a["requested_codes_absent_in_H4"]); abs07 = set(s3a["requested_codes_absent_in_H3"])
w0 = ksic_w("equal"); multi = set(w0[w0.w < 1].hs6)
e17 = ex[ex.year == 2017].merge(w0, on="hs6").assign(v=lambda d: d.export_usd * d.w)
eP2 = ex[ex.year.isin([2015, 2016, 2017])].merge(w0, on="hs6").assign(v=lambda d: d.export_usd * d.w)
diag = pd.DataFrame(index=sorted(set(mp.k5)))
diag["name"] = prof.name.reindex(diag.index)
diag["io"] = k2io.reindex(diag.index)
nt = {r.io: r.nontarget for r in io_struct.itertuples()}
tw = {r.io: r.target for r in io_struct.itertuples()}
diag["io_target_twins"] = [" ".join(sorted(set(" ".join(tw.get(i, "") for i in str(ios).split()).split()) - {k})) for k, ios in zip(diag.index, diag.io)]
diag["io_nontarget_ksic"] = [" ".join(sorted(set(" ".join(str(nt.get(i, "")) for i in str(ios).split() if isinstance(nt.get(i, ""), str)).split()))) for ios in diag.io]
diag["n_hs6"] = mp.drop_duplicates(["k5", "hs6"]).groupby("k5").hs6.nunique()
for nm, d in [("KR2017", e17[e17.rep == "KR"]), ("5rep_P2", eP2)]:
    g = d.groupby("k5").v.sum()
    diag[f"share_changed_vs_HS2012_{nm}"] = (d[~d.hs6.isin(id12)].groupby("k5").v.sum() / g).reindex(diag.index).fillna(0).round(4)
    diag[f"share_absent_in_HS2012_{nm}"] = (d[d.hs6.isin(abs12)].groupby("k5").v.sum() / g).reindex(diag.index).fillna(0).round(4)
    diag[f"share_changed_vs_HS2007_{nm}"] = (d[~d.hs6.isin(id07)].groupby("k5").v.sum() / g).reindex(diag.index).fillna(0).round(4)
    diag[f"share_equal_split_hs6_{nm}"] = (d[d.hs6.isin(multi)].groupby("k5").v.sum() / g).reindex(diag.index).fillna(0).round(4)
# KR 2016→2017 증가율: 원본 vs A1
for lab in ["B_original", "A1_hs_align_equal", "P_A1c_continuity(사후)"]:
    e = EK[lab]; kr = e[e.rep == "KR"].pivot_table(index="k5", columns="year", values="v")
    diag[f"KR_g16_17_{lab}"] = (kr[2017] / kr[2016] - 1).reindex(diag.index).round(4)
    diag[f"KR_P2usd_{lab}"] = kr[[2015, 2016, 2017]].mean(1).reindex(diag.index)
    diag[f"KR_P1usd_{lab}"] = kr[[2008, 2009, 2010]].mean(1).reindex(diag.index)
diag["KR_P2usd_A1_over_orig"] = (diag.KR_P2usd_A1_hs_align_equal / diag.KR_P2usd_B_original).round(4)
diag["KR_P2usd_A1c_over_orig"] = (diag["KR_P2usd_P_A1c_continuity(사후)"] / diag.KR_P2usd_B_original).round(4)
diag["KR_P1usd_A1c_over_orig"] = (diag["KR_P1usd_P_A1c_continuity(사후)"] / diag.KR_P1usd_B_original).round(4)
for lab in ["B_original", "A1_hs_align_equal", "P_A1c_continuity(사후)", "A2a_weight_KRpatent", "A3_alltrade_denominator", "A4_A1+A2a+A3"]:
    diag[f"rca_P2_{lab}"] = P2[lab].reindex(diag.index).round(4)
    diag[f"quad_{lab}"] = Q[lab].reindex(diag.index)
diag["pat_rta_P2"] = rta["P2"].reindex(diag.index).round(4)
diag.index.name = "k5"
diag.to_csv(os.path.join(W, "hs_revision_diag.csv"), encoding="utf-8-sig")
top = diag.sort_values("share_absent_in_HS2012_KR2017", ascending=False).head(12)
print(top[["name", "share_changed_vs_HS2012_KR2017", "share_absent_in_HS2012_KR2017", "KR_g16_17_B_original", "KR_g16_17_A1_hs_align_equal", "KR_P2usd_A1_over_orig", "quad_B_original", "quad_A1_hs_align_equal"]].to_string())
print(S[["variant", "SS", "SW", "WS", "WW", "NA", "rho", "rho_lo", "rho_hi", "rho_chg", "rho_chg_lo", "rho_chg_hi", "list42", "list42_out", "list42_in", "persist", "SW_removed", "SW_added", "prereg_verdict"]].to_string())
