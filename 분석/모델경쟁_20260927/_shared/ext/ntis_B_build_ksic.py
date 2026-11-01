# 2단계: 과제 -> KSIC50 R&D 신호.
# (a) 성과특허 연결 기반 분류->KSIC 배분표(연도 t는 성과연도 <= t-LAG 연결만 사용, 시점별)
# (b) 공개 KSIC 정의문 텍스트 대응(과제명+키워드+과학기술표준분류명, t년 과제 정보만)
import sys, json, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ntis_B_common as c
sys.stdout.reconfigure(encoding="utf-8")
BASE = c.BASE; EXT = c.EXT; WORK = c.WORK
LAG = 2          # 출원 후 공개 18개월 -> t년 말에 확인 가능한 성과는 대략 t-2년까지
M_SHRINK = 20    # 계층 수축 강도(중분류 -> 대분류 -> 전체)
raw = pd.read_parquet(EXT + r"\ntis_project_raw.parquet")
raw = raw[raw.year.between(2002, 2024)].reset_index(drop=True)
raw["gov_funds"] = raw.gov_funds.astype("float64").fillna(0).to_numpy(); raw["total_funds"] = raw.total_funds.astype("float64").fillna(0).to_numpy()
raw["year"] = raw.year.astype(int)
def canon_L(nm):
    """구(2002-08)·신(2009-)·2018 개정 분류의 대분류명을 공통 대분류로 맞춘다(체계 간 배분표 공유용)."""
    if nm is None or pd.isna(nm): return "NA"
    for key, lab in [("수학", "수학"), ("물리", "물리"), ("화학공정", "화공"), ("화공", "화공"), ("화학", "화학"),
                     ("우주", "우주항공해양"), ("지구", "지구과학"), ("뇌", "생명과학"), ("생명", "생명과학"), ("농림", "농림수산"),
                     ("보건", "보건의료"), ("기계", "기계"), ("재료", "재료"), ("전기", "전기전자"), ("정보", "정보통신"),
                     ("통신", "정보통신"), ("에너지", "에너지자원"), ("원자력", "원자력"), ("환경", "환경"), ("건설", "건설교통")]:
        if key in nm: return lab
    return "인문사회정책기타"
raw["clsL_key"] = raw.cls_L_nm.map(canon_L).astype(str)
raw["clsM_key"] = np.where(raw.cls_M.notna(), raw.cls_sys.astype(str) + ":" + raw.cls_M.astype(str), "L:" + raw.clsL_key)
FIRM = {"대기업", "중소기업", "중견기업"}
raw["pg3"] = np.where(raw.perform_grp.isin(FIRM), "firm", np.where(raw.perform_grp.eq("대학"), "univ",
             np.where(raw.perform_grp.isin(["출연연", "국공립연"]), "gri", "other")))
ksic_defs = json.load(open(BASE + r"\공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json", encoding="utf-8"))
KSIC = [d["code"] for d in ksic_defs]
report = {"n_projects_2002_2024": int(len(raw))}

def add_rows(rows, sub, Am, y, name):
    d = pd.DataFrame({"ksic": KSIC, "year": y, "method": name, "n_proj": Am.sum(0),
                      "gov_funds": sub.gov_funds.values @ Am, "total_funds": sub.total_funds.values @ Am})
    for g in ("firm", "univ", "gri"):
        mm = sub.pg3.values == g
        d[f"gov_funds_{g}"] = sub.gov_funds.values[mm] @ Am[mm]
    d["year_total_gov"] = sub.gov_funds.sum(); d["year_n_proj"] = len(sub)
    rows.append(d)

# ---------------- (b) 공개 정의문 텍스트 대응 ----------------
def kdoc(d):
    s = " ".join(str(d.get(k) or "") for k in ["codenmKr", "codeIndex", "codeDefine", "codeExample"])
    s = re.sub(r"제조업|제조|산업활동|말한다|포함한다|경우도|각종|기타|및", " ", s)
    return re.sub(r"[·\(\),]", " ", s)
kd = [kdoc(d) for d in ksic_defs]
pd_all = (raw.title_ko.fillna("") + " " + raw.kw_ko.fillna("").str.replace(",", " ") + " " +
          raw.cls_S_nm.fillna("") + " " + raw.cls_M_nm.fillna("")).str.replace(r"<[^>]+>", " ", regex=True)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True, min_df=2, max_features=300000)
samp = pd_all.sample(min(200000, len(pd_all)), random_state=0).tolist()
vec.fit(kd + samp)
K = vec.transform(kd)
n = len(raw)
top = np.zeros(n, dtype=np.int16); s1 = np.zeros(n, dtype=np.float32); s2 = np.zeros(n, dtype=np.float32)
S_all = np.zeros((n, len(KSIC)), dtype=np.float32)
CH = 100000
for i in range(0, n, CH):
    P = vec.transform(pd_all.iloc[i:i + CH].tolist())
    S = (P @ K.T).toarray().astype(np.float32)
    o = np.argsort(-S, axis=1)
    top[i:i + CH] = o[:, 0]; s1[i:i + CH] = S[np.arange(len(S)), o[:, 0]]; s2[i:i + CH] = S[np.arange(len(S)), o[:, 1]]
    S_all[i:i + CH] = S
del top, s1, s2
link = pd.read_parquet(EXT + r"\ntis_patent_link.parquet")
link = link[link.ProjectID.notna() & (link.ProjectID.str.len() > 0)].copy()
lk = link.groupby("ProjectID").ksic.agg(set)
raw["lk_ksic"] = raw.pjt_no.map(lk)
brows = []

def do_b(tag, S_all, grid_vals):
    """S_all: (n,50) 점수. 임계값은 2014년 이하 과제의 성과특허 연결로만 보정(정밀도>=0.5, n>=30 중 최소 임계값)."""
    o1 = S_all.argmax(1); sc1 = S_all[np.arange(n), o1]
    topk = np.array(KSIC)[o1]
    raw[f"{tag}_top_ksic"] = topk; raw[f"{tag}_score"] = sc1
    m = raw.lk_ksic.notna().values
    val = pd.DataFrame({"year": raw.year.values[m], "sc": sc1[m],
                        "hit": [t in s for t, s in zip(topk[m], raw.lk_ksic.values[m])]})
    cal = val[val.year <= 2014]
    grid = []
    for th in grid_vals:
        x = cal[cal.sc >= th]
        grid.append((round(float(th), 3), int(len(x)), round(float(x.hit.mean()), 3) if len(x) else np.nan))
    grid = pd.DataFrame(grid, columns=["th", "n", "prec"])
    print(f"({tag}) 임계값 보정(연결 과제, 연도<=2014):\n", grid.to_string(index=False))
    ok = grid[(grid.prec >= 0.5) & (grid.n >= 30)]
    TH = float(ok.th.min()) if len(ok) else float(grid.loc[grid.prec.idxmax(), "th"])
    print(tag, "선택 임계값", TH)
    va = val[val.sc >= TH]
    report[f"{tag}_threshold"] = TH; report[f"{tag}_grid"] = grid.to_dict("records")
    report[f"{tag}_validation"] = {"n_linked_projects": int(len(val)), "n_linked_projects_le2014": int(len(cal)),
        "prec_top1_all_linked": round(float(val.hit.mean()), 4) if len(val) else None,
        "n_above_th": int(len(va)), "prec_above_th": round(float(va.hit.mean()), 4) if len(va) else None,
        "prec_above_th_le2014": round(float(va[va.year <= 2014].hit.mean()), 4) if (va.year <= 2014).any() else None,
        "prec_above_th_2015plus": round(float(va[va.year >= 2015].hit.mean()), 4) if (va.year >= 2015).any() else None,
        "share_linked_assigned": round(float((val.sc >= TH).mean()), 4) if len(val) else None}
    asg = sc1 >= TH
    raw[f"{tag}_assigned"] = asg
    W = np.where(S_all >= TH, S_all, 0).astype(np.float32); rs = W.sum(1, keepdims=True)
    W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    hard = np.zeros_like(W); hard[np.arange(n), o1] = asg.astype(np.float32)
    for y in range(2002, 2025):
        mm = (raw.year == y).values; sub = raw[mm]
        add_rows(brows, sub, hard[mm], y, f"{tag}_hard")
        add_rows(brows, sub, W[mm], y, f"{tag}_soft")
    report[f"{tag}_assign_rate_by_year"] = pd.Series(asg).groupby(raw.year.values).mean().round(4).to_dict()
    report[f"{tag}_assign_funds_share_by_year"] = (pd.Series(raw.gov_funds.values * asg).groupby(raw.year.values).sum() / raw.groupby("year").gov_funds.sum().values).round(4).to_dict()

do_b("b1_tfidf", S_all, np.arange(0.05, 0.60, 0.025))
del S_all
import ntis_B_ksic_dict as kdict
terms = kdict.build_terms(ksic_defs)
terms.to_csv(WORK + r"\ntis_B_ksic_terms.csv", index=False, encoding="utf-8-sig")
report["b2_dict_n_terms"] = int(len(terms))
S2 = np.load(WORK + r"\ntis_B_dict_scores.npy")   # ntis_B_score_dict.py 선행 실행
_ix = pd.read_parquet(WORK + r"\ntis_B_dict_scores_index.parquet").pjt_no.values
assert len(_ix) == n and (_ix == raw.pjt_no.values).all(), "사전 점수 캐시와 과제 순서 불일치 - ntis_B_score_dict.py 재실행"
do_b("b2_dict", S2, [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
del S2

# ---------------- (a) 성과특허 연결 기반 배분표 ----------------
rp = pd.read_parquet(EXT + r"\ntis_rpatent_raw.parquet")
rp = rp[rp.is_kr.astype(bool) & rp.IprType.eq("특허") & rp.ProjectID.notna() & (rp.ProjectID.str.len() > 0)].copy()
rp["ryear"] = pd.to_numeric(rp.Year, errors="coerce")
pcls = raw.drop_duplicates("pjt_no").set_index("pjt_no")[["clsL_key", "clsM_key"]]
rp = rp.join(pcls, on="ProjectID")
report["a_rpatent_kr_patent_with_pid"] = int(len(rp)); report["a_rpatent_pid_matched_to_project"] = round(float(rp.clsM_key.notna().mean()), 4)
rp = rp[rp.clsM_key.notna()].drop_duplicates(["ResultID", "ProjectID"]).copy()
rp["w"] = 1 / rp.groupby("ResultID").ProjectID.transform("nunique")   # 여러 과제 성과면 1/n
lnk = link.merge(rp[["ResultID", "ProjectID", "clsL_key", "clsM_key", "ryear", "w"]], on=["ResultID", "ProjectID"], how="inner")
lnk["wk"] = lnk.w / lnk.groupby(["ResultID", "ProjectID"]).ksic.transform("nunique")   # 여러 KSIC면 분할
report["a_link_rows_total"] = int(len(link)); report["a_link_rows_matched_to_project"] = int(len(lnk))
report["a_rpatent_year_range"] = [float(rp.ryear.min()), float(rp.ryear.max())] if len(rp) else None

def share_table(tmax):
    d = rp if tmax is None else rp[rp.ryear <= tmax]
    l = lnk if tmax is None else lnk[lnk.ryear <= tmax]
    if len(d) == 0: return None
    NM = d.groupby("clsM_key").w.sum(); NL = d.groupby("clsL_key").w.sum(); N = d.w.sum()
    nMk = l.groupby(["clsM_key", "ksic"]).wk.sum().unstack(fill_value=0).reindex(columns=KSIC, fill_value=0)
    nLk = l.groupby(["clsL_key", "ksic"]).wk.sum().unstack(fill_value=0).reindex(columns=KSIC, fill_value=0)
    p_all = l.groupby("ksic").wk.sum().reindex(KSIC, fill_value=0) / N
    sL = nLk.reindex(NL.index, fill_value=0).add(M_SHRINK * p_all, axis=1).div(NL + M_SHRINK, axis=0)
    m2L = d.drop_duplicates("clsM_key").set_index("clsM_key").clsL_key.reindex(NM.index)
    priorM = sL.reindex(m2L.values); priorM.index = NM.index
    priorM = priorM.fillna(pd.Series(p_all.values, index=KSIC))
    sM = (nMk.reindex(NM.index, fill_value=0) + M_SHRINK * priorM).div(NM + M_SHRINK, axis=0)
    return sM, sL, p_all

def alloc(sub, tab):
    sM, sL, p_all = tab
    A = sM.reindex(sub.clsM_key.values).values; AL = sL.reindex(sub.clsL_key.values).values
    Am = np.where(np.isnan(A), AL, A)
    return np.where(np.isnan(Am), p_all.values[None, :], Am)

arows = []; cover = {}
for y in range(2002, 2025):
    sub = raw[raw.year == y]
    tab = share_table(y - LAG)
    if tab is None:
        cover[y] = None; continue
    Am = alloc(sub, tab)
    inM = sub.clsM_key.isin(tab[0].index).values
    cover[y] = {"ref_kr_patents": round(float(rp[rp.ryear <= y - LAG].w.sum()), 1),
                "ref_links": round(float(lnk[lnk.ryear <= y - LAG].wk.sum()), 1),
                "proj_cls_in_table": round(float(inM.mean()), 4),
                "funds_cls_in_table": round(float(sub.gov_funds.values[inM].sum() / max(sub.gov_funds.sum(), 1)), 4)}
    add_rows(arows, sub, Am, y, "a_link_pit")
tab_all = share_table(None)
for y in range(2002, 2025):
    sub = raw[raw.year == y]
    add_rows(arows, sub, alloc(sub, tab_all), y, "a_link_static_NOPIT")
tab_all[0].to_parquet(WORK + r"\ntis_B_share_static_M.parquet")
report["a_cover_by_year"] = cover
lp = set(link.ProjectID)
report["a_proj_linked_rate_by_year"] = raw.assign(l=raw.pjt_no.isin(lp)).groupby("year").l.mean().round(4).to_dict()
report["a_funds_linked_rate_by_year"] = (raw[raw.pjt_no.isin(lp)].groupby("year").gov_funds.sum() / raw.groupby("year").gov_funds.sum()).round(4).to_dict()

out = pd.concat(arows + brows, ignore_index=True)
out["ksic"] = out.ksic.astype("string"); out["year"] = out.year.astype("int16"); out["method"] = out.method.astype("string")
out["avail_year"] = (out.year + 1).astype("int16")   # 권장 사용 시점: 과제연도 다음 해
out["gov_share_of_year"] = out.gov_funds / out.year_total_gov
out.to_parquet(EXT + r"\ntis_project_ksic_year.parquet", index=False)
print("저장 ntis_project_ksic_year.parquet", out.shape)
raw[["pjt_no", "year", "b1_tfidf_top_ksic", "b1_tfidf_score", "b1_tfidf_assigned", "b2_dict_top_ksic", "b2_dict_score", "b2_dict_assigned"]].to_parquet(EXT + r"\ntis_project_ksic_text_map.parquet", index=False)

from scipy.stats import spearmanr
for bm in ["b1_tfidf_soft", "b2_dict_soft"]:
    pv = out[out.method.isin(["a_link_pit", bm])].pivot_table(index=["ksic", "year"], columns="method", values="gov_funds").dropna()
    if len(pv) > 10 and pv.shape[1] == 2:
        cs = {}
        for y in sorted(set(pv.index.get_level_values("year"))):
            x = pv.xs(y, level="year")
            if len(x) >= 10 and x["a_link_pit"].nunique() > 1: cs[int(y)] = round(float(spearmanr(x["a_link_pit"], x[bm]).statistic), 3)
        report[f"spearman_KSIC_cross_section_a_pit_vs_{bm}"] = cs
json.dump(report, open(WORK + r"\ntis_B_ksic_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in report.items() if k != "b_grid"}, ensure_ascii=False, default=str)[:5000])
