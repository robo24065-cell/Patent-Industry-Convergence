# -*- coding: utf-8 -*-
"""s4: 내려받은 파일 파싱·검증 + D1 총량(coverage_reporter_year.csv) + D5 시점(재조회·다른 수집본 대조) + 개정 진단(hs_revision_diag.csv)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
CT = os.path.join(DL, "comtrade")
res = {}


def parse(key):
    j = json.load(open(os.path.join(CT, key + ".json"), encoding="utf-8"))
    rows = []
    for x in j.get("data", []):
        rows.append(dict(year=int(x["refYear"]), reporter=int(x["reporterCode"]), hs6=str(x["cmdCode"]).zfill(6) if x["cmdCode"] != "TOTAL" else "TOTAL",
                         v=x["primaryValue"], cls=x.get("classificationCode"), mot=x.get("motCode"), cust=x.get("customsCode"), p2=x.get("partner2Code"),
                         partner=x.get("partnerCode"), flow=x.get("flowCode"), isrep=x.get("isReported"), isagg=x.get("isAggregate")))
    return pd.DataFrame(rows), j


# ---------- 파싱·무결성 ----------
keys = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CT, "*.json")))
frames = []; integ = []
for k in keys:
    d, j = parse(k)
    d["key"] = k
    integ.append(dict(key=k, rows=len(d), error=j.get("error", ""), n_partner_ne0=int((d.partner != 0).sum()) if len(d) else 0,
                      n_flow_neX=int((d.flow != "X").sum()) if len(d) else 0, n_mot_ne0=int((d.mot != 0).sum()) if len(d) else 0,
                      n_cust_neC00=int((d.cust != "C00").sum()) if len(d) else 0, dup=int(d.duplicated(["year", "reporter", "hs6"]).sum()) if len(d) else 0,
                      neg_zero=int((d.v <= 0).sum()) if len(d) else 0, cls=",".join(sorted(d.cls.dropna().unique())) if len(d) else ""))
    frames.append(d)
allf = pd.concat(frames, ignore_index=True)
allf = allf[(allf.partner == 0) & (allf.flow == "X") & (allf.mot == 0) & (allf.cust == "C00") & (allf.p2 == 0)]
integ = pd.DataFrame(integ); integ.to_csv(os.path.join(CACHE, "s4_download_integrity.csv"), index=False)
res["download_integrity"] = dict(files=len(integ), any_error=int((integ.error != "").sum()), dup=int(integ.dup.sum()), neg_zero=int(integ.neg_zero.sum()),
                                 max_rows=int(integ.rows.max()), filtered_out=int(sum(integ[c].sum() for c in ["n_partner_ne0", "n_flow_neX", "n_mot_ne0", "n_cust_neC00"])))
allf["rep"] = allf.reporter.map(REP)
allf.to_csv(os.path.join(CACHE, "s4_downloads_parsed.csv"), index=False)
tot = allf[allf.hs6 == "TOTAL"].pivot_table(index="rep", columns="year", values="v", aggfunc="sum")
tot.to_csv(os.path.join(CACHE, "comtrade_total_rep_year.csv"))
# WDI(WTO) 총수출
wj = json.load(open(os.path.join(DL, "worldbank", "wdi_TX.VAL.MRCH.CD.WT.json"), encoding="utf-8"))
iso = {"KOR": "KR", "USA": "US", "JPN": "JP", "CHN": "CN", "DEU": "DE"}
wdi = pd.DataFrame([dict(rep=iso[x["countryiso3code"]], year=int(x["date"]), v=x["value"]) for x in wj[1]]).pivot(index="rep", columns="year", values="v")
res["wdi_meta"] = dict(page=wj[0], n=len(wj[1]), missing=int(wdi.isna().sum().sum()))
# ---------- D1 ----------
mp = load_map(); codes = sorted(mp.hs6.unique())
ex = load_ex()
ry = ex.groupby(["rep", "year"]).agg(n_codes=("hs6", "nunique"), map_sum=("export_usd", "sum"), hs_rev=("hs_rev", "first")).reset_index()
ry["comtrade_total"] = [tot.at[r, y] if y in tot.columns else np.nan for r, y in zip(ry.rep, ry.year)]
ry["wdi_wto_total"] = [wdi.at[r, y] for r, y in zip(ry.rep, ry.year)]
ry["comtrade_vs_wdi"] = ry.comtrade_total / ry.wdi_wto_total
ry["map_share_of_total"] = ry.map_sum / ry.comtrade_total
ry["map_share_rel_rep_mean"] = ry.map_share_of_total / ry.groupby("rep").map_share_of_total.transform("mean")
ry["flag_jump25"] = (ry.map_share_rel_rep_mean - 1).abs() > 0.25
res["kr_totals_vs_public"] = {int(y): dict(comtrade=float(tot.at["KR", y]), wdi=float(wdi.at["KR", y])) for y in [2008, 2009, 2010, 2015, 2016, 2017, 2021]}
res["comtrade_vs_wdi_range"] = [float(ry.comtrade_vs_wdi.min()), float(ry.comtrade_vs_wdi.max())]
res["comtrade_vs_wdi_by_rep"] = ry.groupby("rep").comtrade_vs_wdi.agg(["min", "max"]).round(4).to_dict("index")
# 공개 통계(한국): 산업통상자원부·한국무역협회 발표 연간 수출(통관 기준, 억 달러) — 기억값이 아니라 WDI(WTO) 값으로 독립 대조, 아래는 참고
res["map_share_KR"] = ry[ry.rep == "KR"].set_index("year").map_share_of_total.round(4).to_dict()
res["map_share_by_rep_P2"] = ry[ry.year.isin([2015, 2016, 2017])].groupby("rep").map_share_of_total.mean().round(4).to_dict()
res["map_share_by_rep_P1"] = ry[ry.year.isin([2008, 2009, 2010])].groupby("rep").map_share_of_total.mean().round(4).to_dict()
# 코드 수 급변
ry = ry.sort_values(["rep", "year"])
ry["n_codes_chg"] = ry.groupby("rep").n_codes.pct_change()
res["n_codes_drop_gt10pct"] = ry[ry.n_codes_chg < -0.10][["rep", "year", "n_codes", "n_codes_chg"]].values.tolist()
# ---------- D4 개정 단절: 2016(H4) vs 2017(H5) ----------
c12 = pd.read_csv(os.path.join(CACHE, "unsd_corr_17_12.csv"), dtype=str)
old12 = set(c12.old)
ident = pd.read_csv(os.path.join(CACHE, "s3a_map_code_versions.csv"), dtype={"hs6": str})
id12 = set(ident[(ident.ver == "H4") & ident.same_code].hs6)
id07 = set(ident[(ident.ver == "H3") & ident.same_code].hs6)
rev = []
for r in REPS:
    e = ex[ex.rep == r]
    a = e[e.year == 2016]; b = e[e.year == 2017]
    rev.append(dict(rep=r, codes_2016=a.hs6.nunique(), codes_2017=b.hs6.nunique(), usd_2016=a.export_usd.sum(), usd_2017=b.export_usd.sum(),
                    usd_2016_ident=a[a.hs6.isin(id12)].export_usd.sum(), usd_2017_ident=b[b.hs6.isin(id12)].export_usd.sum(),
                    tot_2016=tot.at[r, 2016], tot_2017=tot.at[r, 2017]))
rev = pd.DataFrame(rev)
rev["map_share_2016"] = rev.usd_2016 / rev.tot_2016; rev["map_share_2017"] = rev.usd_2017 / rev.tot_2017
rev["ident_share_2016"] = rev.usd_2016_ident / rev.tot_2016; rev["ident_share_2017"] = rev.usd_2017_ident / rev.tot_2017
rev["jump_all"] = rev.map_share_2017 / rev.map_share_2016 - 1; rev["jump_ident"] = rev.ident_share_2017 / rev.ident_share_2016 - 1
rev["jump_excess"] = rev.jump_all - rev.jump_ident
rev.to_csv(os.path.join(CACHE, "s4_rev_2016_2017.csv"), index=False)
res["rev_2016_2017"] = rev.round(4).to_dict("records")
# ---------- D5 시점 ----------
rf = allf[allf.key.str.startswith("REFETCH")][["year", "rep", "hs6", "v"]]
m = rf.merge(ex[["year", "rep", "hs6", "export_usd"]], on=["year", "rep", "hs6"], how="outer", indicator=True)
m["rel"] = (m.v - m.export_usd) / m.export_usd
d5 = {}
for y in [2015, 2016, 2017, 2021]:
    mm = m[m.year == y]
    both = mm[mm._merge == "both"]
    d5[y] = dict(n_both=int(len(both)), only_refetch=int((mm._merge == "left_only").sum()), only_orig=int((mm._merge == "right_only").sum()),
                 median_abs_rel=float(both.rel.abs().median()), share_gt1pct=float((both.rel.abs() > 0.01).mean()),
                 sum_rel=float(both.v.sum() / both.export_usd.sum() - 1))
res["d5_refetch_vs_orig_chunk0"] = d5
# 다른 수집본(KR)
kr = ex[ex.rep == "KR"][["year", "hs6", "export_usd"]]
others = {}
uv = pd.read_csv(os.path.join(EXT, "trade_kr_export_hs6_uv_2000_2021.csv"), dtype={"hs6": str}); uv["hs6"] = uv.hs6.str.zfill(6)
n1 = pd.read_csv(os.path.join(EXT, "N1final_kr_export_hs6_2019_2024.csv"), dtype={"hs6": str}); n1["hs6"] = n1.hs6.str.zfill(6)
gl = pd.read_csv(os.path.join(EXT, "R2global_kr_export_hs6.csv"), dtype={"hs6": str}); gl["hs6"] = gl.hs6.str.zfill(6)
for nm, d, vc in [("trade_kr_export_hs6_uv", uv, "export_usd"), ("N1final", n1[n1.year <= 2021], "value"), ("R2global", gl, "value")]:
    mm = kr.merge(d[["year", "hs6", vc]].rename(columns={vc: "v2"}), on=["year", "hs6"], how="outer", indicator=True)
    both = mm[mm._merge == "both"]; rel = (both.v2 - both.export_usd) / both.export_usd
    yrs = sorted(set(kr.year) & set(d.year))
    mm = mm[mm.year.isin(yrs)]
    others[nm] = dict(years=[int(min(yrs)), int(max(yrs))], n_both=int(len(both)), only_5c=int((mm._merge == "left_only").sum()), only_other=int((mm._merge == "right_only").sum()),
                      median_abs_rel=float(rel.abs().median()), share_gt1pct=float((rel.abs() > 0.01).mean()), max_abs_rel=float(rel.abs().max()),
                      by_year_gt1pct={int(y): float((rel[both.year == y].abs() > 0.01).mean()) for y in sorted(both.year.unique()) if y in (2008, 2009, 2010, 2015, 2016, 2017, 2021)})
res["d5_other_collections_KR"] = others
# ---------- 커버리지 표 저장 ----------
ry.to_csv(os.path.join(W, "coverage_reporter_year.csv"), index=False, encoding="utf-8-sig")
json.dump(res, open(os.path.join(CACHE, "s4_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float)[:9000])
