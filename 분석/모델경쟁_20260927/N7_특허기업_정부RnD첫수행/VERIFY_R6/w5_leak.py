# -*- coding: utf-8 -*-
r"""누수 재탐색(새 눈): 모형은 고치지 않고 영향만 잰다. 🔒 새 결과 구간 계산 없음(시험 레이블은 이미 채점된 것만).
L1 NTIS 수행 이력의 '이름-사업자번호' 불일치: 원점 이전에 같은 이름으로 수행기관에 등장했지만 사업자번호로는 이력이 없는 기업(모집단 오염 → '첫 수행' 레이블이 사실은 재등장)
L2 M2(NTIS 성과 연결) 피처의 자기 과제: 자사 특허가 연결된 원점 이전 과제의 수행기관이 자기 자신(이름 또는 사업자번호)인 경우
L3 레이블 과제의 시작 시점: 결과창 첫 과제의 총 연구기간 시작(total_start)이 원점 이전인 비율(연차 과제에 사업자번호가 늦게 붙은 경우)
L4 모집단 = 2026 CORP_APPLICANT 사업자번호 연결: 원점 시점 활동별 연결률과 역가중(IPW) skill
L5 출원→출원인코드 '상호 경로'(CORP_APPLICANT 2026 상호 일치) 비중: 상호 변경 기업의 출원 누락 여부"""
from w_common import *
import duckdb
from sklearn.metrics import roc_auc_score
D = pd.read_parquet(os.path.join(VC, "final_panel_orig.parquet"))
DA = pd.read_parquet(os.path.join(VC, "final_panel_alias.parquet"))
O = pd.read_parquet(os.path.join(VC, "preds_orig.parquet"))
te = D[D.t0 == 2022].reset_index(drop=True); assert (te.applicantCd.values == O.applicantCd.values).all()
hidden = (te.nps_present.values == 0) & (DA[DA.t0 == 2022].nps_present.values == 1)
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "aname", "bizno"]).drop_duplicates("applicantCd")
nm = dict(zip(ns.applicantCd, ns.aname.map(norm)))
D["nname"] = D.applicantCd.map(nm); D["bz"] = D.bizno.str.replace(r"[^0-9]", "", regex=True)
f = os.path.join(EXT, "ntis_project_raw.parquet").replace("\\", "/")
RAW = duckdb.sql(f"select pjt_no, year, research_agency, corp_rgst_no, total_start from read_parquet('{f}') where year between 2010 and 2024").df()
RAW["n"] = RAW.research_agency.map(norm); RAW["bz"] = RAW.corp_rgst_no.fillna("").str.replace(r"[^0-9]", "", regex=True)
RAW["ts"] = pd.to_numeric(RAW.total_start.astype(str).str[:4], errors="coerce")
R = {}
def skl(mask):
    y = te.y.values[mask]
    a1, a0 = roc_auc_score(y, O.p_champ.values[mask]), roc_auc_score(y, O.p_M0.values[mask])
    return dict(n=int(mask.sum()), pos=int(y.sum()), auc_champ=float(a1), auc_M0=float(a0), skill=float(skill(a1, a0)))

# ---- L1 ----
R["L1"] = {}
for t0, g in D.groupby("t0"):
    r = RAW[RAW.year <= t0]
    nf = r[r.n.str.len() >= 2].groupby("n").year.min()
    fl = g.nname.map(nf).notna() & (g.nname.str.len() >= 2)
    R["L1"][int(t0)] = dict(n=len(g), flagged=int(fl.sum()), share=float(fl.mean()), y_flag=float(g.y[fl].mean()) if fl.any() else None, y_rest=float(g.y[~fl].mean()))
fl_te = (te.applicantCd.map(nm).map(RAW[RAW.year <= 2022].groupby("n").year.min()).notna()).values
R["L1"]["test_excl_flagged"] = skl(~fl_te); R["L1"]["test_all"] = skl(np.ones(len(te), bool))
log("L1", R["L1"])

# ---- L2 ----
lk = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["appl_no", "Year", "ProjectYear", "ProjectID", "PerformAgency"])
lk["Year"] = pd.to_numeric(lk.Year, errors="coerce"); lk["ProjectYear"] = pd.to_numeric(lk.ProjectYear, errors="coerce")
lk["pn"] = lk.PerformAgency.map(norm); lk["pbz"] = lk.ProjectID.astype(str).map(dict(zip(RAW.pjt_no.astype(str), RAW.bz)))
AF = pd.read_parquet(os.path.join(VC, "app_firm.parquet")); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
R["L2"] = {}
for t0, g in D.groupby("t0"):
    end = t0 * 10000 + 1231
    af = AF[AF.applicantCd.isin(set(g.applicantCd)) & (AF.known_date <= end)][["applicantCd", "appl_no"]].drop_duplicates()
    a = af.merge(lk, on="appl_no"); a = a[(a.Year <= t0) & (a.ProjectYear <= t0)]
    a["self_name"] = a.pn == a.applicantCd.map(nm); a["self_bz"] = a.pbz == a.applicantCd.map(dict(zip(g.applicantCd, g.bz)))
    s = a.groupby("applicantCd")[["self_name", "self_bz"]].any()
    gp = g[g.g_part_any == 1]
    sn = gp.applicantCd.map(s.self_name).fillna(False).astype(bool); sb = gp.applicantCd.map(s.self_bz).fillna(False).astype(bool)
    R["L2"][int(t0)] = dict(g_part_any=len(gp), self_name=int(sn.sum()), self_bz=int(sb.sum()), y_gpart=float(gp.y.mean()),
                            y_self_name=float(gp.y[sn].mean()) if sn.any() else None, y_gpart_notself=float(gp.y[~sn].mean()))
    if t0 == 2022:
        selfm = te.applicantCd.isin(set(gp.applicantCd[sn | sb])).values
        R["L2"]["test_excl_self"] = skl(~selfm)
log("L2", R["L2"])

# ---- L3 ---- (결과창 첫 과제의 총 연구기간 시작연도)
R["L3"] = {}
for t0, g in D.groupby("t0"):
    pos = g[g.y == 1]
    r = RAW[RAW.year.between(t0 + 1, t0 + 2) & RAW.bz.isin(set(pos.bz))]
    fy = r.groupby("bz").year.min()
    r = r[r.year == r.bz.map(fy)]
    early = r.groupby("bz").ts.min() <= t0
    R["L3"][int(t0)] = dict(pos=len(pos), matched=int(len(early)), started_le_t0=int(early.sum()), share=float(early.mean()))
log("L3", R["L3"])

# ---- L4 ----
R["L4"] = {}
nsr = n7lib.load_nts()
for t0 in [2017, 2022]:
    end = t0 * 10000 + 1231
    kn = AF[AF.known_date <= end]
    pt = pd.read_parquet(os.path.join(VC, "reg_pt.parquet")); p = pt[pt.rgstDate <= end]
    firms = set(kn.applicantCd) & set(p.applicantCd)
    g = kn[kn.applicantCd.isin(firms)].groupby("applicantCd")
    fr = pd.DataFrame({"last": g.known_date.max(), "n": g.appl_no.nunique()})
    fr["yrs"] = t0 - fr["last"] // 10000
    live = p[~(p.ext_d <= end)].groupby("applicantCd").rgst_no.nunique()
    fr["has_live"] = fr.index.map(live).fillna(0) > 0
    fr["linked"] = fr.index.isin(set(nsr.applicantCd))
    fr["yb"] = pd.cut(fr.yrs, [-1, 0, 2, 5, 10, 100], labels=["0", "1-2", "3-5", "6-10", ">10"]).astype(str)
    fr["nb"] = pd.cut(fr.n, [0, 1, 3, 10, 1e9], labels=["1", "2-3", "4-10", ">10"]).astype(str)
    fr["cell"] = fr.yb + "|" + fr.has_live.astype(str) + "|" + fr.nb
    lr = fr.groupby("cell").linked.mean()
    R["L4"][t0] = dict(n_firms=len(fr), link_rate=float(fr.linked.mean()), by_yrs=fr.groupby("yb").linked.mean().round(4).to_dict(),
                       by_live=fr.groupby("has_live").linked.mean().round(4).to_dict())
    if t0 == 2022:
        w = te.applicantCd.map(fr.cell).map(lr).values
        w = 1 / np.clip(w, 0.05, 1)
        y = te.y.values
        a1 = roc_auc_score(y, O.p_champ, sample_weight=w); a0 = roc_auc_score(y, O.p_M0, sample_weight=w)
        R["L4"]["test_ipw"] = dict(auc_champ=float(a1), auc_M0=float(a0), skill=float(skill(a1, a0)), n_cells=int(len(lr)), w_max=float(w.max()))
log("L4", R["L4"])

# ---- L5 ----
end = 2022 * 10000 + 1231
kn = AF[(AF.known_date <= end) & AF.applicantCd.isin(set(te.applicantCd))].drop_duplicates(["applicantCd", "appl_no"])
sh = kn.assign(nm_=(kn.src == "name").astype(float)).groupby("applicantCd").nm_.mean()
s = te.applicantCd.map(sh).values
R["L5"] = dict(name_route_share_mean=float(np.nanmean(s)), hidden_mean=float(np.nanmean(s[hidden])), other_mean=float(np.nanmean(s[~hidden])),
               y_by_share={"0": float(te.y[s == 0].mean()), "(0,0.5]": float(te.y[(s > 0) & (s <= 0.5)].mean()), ">0.5": float(te.y[s > 0.5].mean())},
               n_by_share={"0": int((s == 0).sum()), "(0,0.5]": int(((s > 0) & (s <= 0.5)).sum()), ">0.5": int((s > 0.5).sum())})
log("L5", R["L5"])
json.dump(R, open(os.path.join(HERE, "w5_leak.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("saved")
