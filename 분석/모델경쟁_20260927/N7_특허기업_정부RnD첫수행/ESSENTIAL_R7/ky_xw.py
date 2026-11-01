# -*- coding: utf-8 -*-
r"""NTIS 과제 → 산업 배분표(a_link_pit)를 _shared\ext\ntis_B_build_ksic.py (a) 부분과 같은 규칙으로 다시 만든다.
산업 키만 바꿀 수 있게: key='ksic'(제공 라벨, 재현 확인용) 또는 'xw'(공식 IPC-KSIC 연계표 배정). 읽기 전용 입력, 출력은 이 폴더."""
import os, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); MC = os.path.dirname(TR)
EXT = os.path.join(MC, "_shared", "ext")
sys.path.insert(0, os.path.join(MC, "T4_특허포기예측", "ESSENTIAL_R7", "common"))
from xwalk import load_app_xw
LAG, M_SHRINK = 2, 20


def canon_L(nm):
    if nm is None or pd.isna(nm): return "NA"
    for key, lab in [("수학", "수학"), ("물리", "물리"), ("화학공정", "화공"), ("화공", "화공"), ("화학", "화학"),
                     ("우주", "우주항공해양"), ("지구", "지구과학"), ("뇌", "생명과학"), ("생명", "생명과학"), ("농림", "농림수산"),
                     ("보건", "보건의료"), ("기계", "기계"), ("재료", "재료"), ("전기", "전기전자"), ("정보", "정보통신"),
                     ("통신", "정보통신"), ("에너지", "에너지자원"), ("원자력", "원자력"), ("환경", "환경"), ("건설", "건설교통")]:
        if key in nm: return lab
    return "인문사회정책기타"


def build(key, ymax=2022):
    fn = os.path.join(HERE, f"ky_{key}_le{ymax}.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    raw = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["pjt_no", "year", "gov_funds", "total_funds", "cls_L_nm", "cls_M", "cls_sys", "perform_grp"])
    raw = raw[raw.year.between(2002, ymax)].reset_index(drop=True)
    raw["gov_funds"] = raw.gov_funds.astype("float64").fillna(0).to_numpy(); raw["total_funds"] = raw.total_funds.astype("float64").fillna(0).to_numpy()
    raw["year"] = raw.year.astype(int)
    raw["clsL_key"] = raw.cls_L_nm.map(canon_L).astype(str)
    raw["clsM_key"] = np.where(raw.cls_M.notna(), raw.cls_sys.astype(str) + ":" + raw.cls_M.astype(str), "L:" + raw.clsL_key)
    FIRM = {"대기업", "중소기업", "중견기업"}
    raw["pg3"] = np.where(raw.perform_grp.isin(FIRM), "firm", np.where(raw.perform_grp.eq("대학"), "univ",
                 np.where(raw.perform_grp.isin(["출연연", "국공립연"]), "gri", "other")))
    link = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["cc", "appl_no", "ksic", "ResultID", "ProjectID"])
    link = link[link.ProjectID.notna() & (link.ProjectID.str.len() > 0)].copy()
    if key in ("xw", "ipc4"):
        ax = load_app_xw().set_index("appl_no")
        xw = ax.xw if key == "xw" else ax.main_ipc.fillna("").str.replace(" ", "").str[:4].replace("", np.nan)
        link["ksic"] = link.appl_no.astype(str).map(xw)
        link = link.dropna(subset=["ksic"]).drop_duplicates(["ResultID", "ProjectID", "appl_no", "ksic"])
    KS = sorted(link.ksic.astype(str).unique()) if key in ("xw", "ipc4") else sorted(pd.read_parquet(os.path.join(EXT, "ntis_project_ksic_year.parquet")).ksic.astype(str).unique())
    link["ksic"] = link.ksic.astype(str)
    rp = pd.read_parquet(os.path.join(EXT, "ntis_rpatent_raw.parquet"), columns=["is_kr", "IprType", "ProjectID", "Year", "ResultID"])
    rp = rp[rp.is_kr.astype(bool) & rp.IprType.eq("특허") & rp.ProjectID.notna() & (rp.ProjectID.str.len() > 0)].copy()
    rp["ryear"] = pd.to_numeric(rp.Year, errors="coerce")
    pcls = raw.drop_duplicates("pjt_no").set_index("pjt_no")[["clsL_key", "clsM_key"]]
    rp = rp.join(pcls, on="ProjectID")
    rp = rp[rp.clsM_key.notna()].drop_duplicates(["ResultID", "ProjectID"]).copy()
    rp["w"] = 1 / rp.groupby("ResultID").ProjectID.transform("nunique")
    lnk = link.merge(rp[["ResultID", "ProjectID", "clsL_key", "clsM_key", "ryear", "w"]], on=["ResultID", "ProjectID"], how="inner")
    lnk["wk"] = lnk.w / lnk.groupby(["ResultID", "ProjectID"]).ksic.transform("nunique")

    def share_table(tmax):
        d = rp[rp.ryear <= tmax]; l = lnk[lnk.ryear <= tmax]
        if len(d) == 0: return None
        NM = d.groupby("clsM_key").w.sum(); NL = d.groupby("clsL_key").w.sum(); N = d.w.sum()
        nMk = l.groupby(["clsM_key", "ksic"]).wk.sum().unstack(fill_value=0).reindex(columns=KS, fill_value=0)
        nLk = l.groupby(["clsL_key", "ksic"]).wk.sum().unstack(fill_value=0).reindex(columns=KS, fill_value=0)
        p_all = l.groupby("ksic").wk.sum().reindex(KS, fill_value=0) / N
        sL = nLk.reindex(NL.index, fill_value=0).add(M_SHRINK * p_all, axis=1).div(NL + M_SHRINK, axis=0)
        m2L = d.drop_duplicates("clsM_key").set_index("clsM_key").clsL_key.reindex(NM.index)
        priorM = sL.reindex(m2L.values); priorM.index = NM.index
        priorM = priorM.fillna(pd.Series(p_all.values, index=KS))
        sM = (nMk.reindex(NM.index, fill_value=0) + M_SHRINK * priorM).div(NM + M_SHRINK, axis=0)
        return sM, sL, p_all

    rows = []
    for y in range(2002, ymax + 1):
        sub = raw[raw.year == y]
        tab = share_table(y - LAG)
        if tab is None: continue
        sM, sL, p_all = tab
        A = sM.reindex(sub.clsM_key.values).values; AL = sL.reindex(sub.clsL_key.values).values
        Am = np.where(np.isnan(A), AL, A); Am = np.where(np.isnan(Am), p_all.values[None, :], Am)
        d = pd.DataFrame({"ksic": KS, "year": y, "n_proj": Am.sum(0), "gov_funds": sub.gov_funds.values @ Am, "total_funds": sub.total_funds.values @ Am})
        for g in ("firm", "univ", "gri"):
            mm = sub.pg3.values == g
            d[f"gov_funds_{g}"] = sub.gov_funds.values[mm] @ Am[mm]
        d["year_total_gov"] = sub.gov_funds.sum(); d["year_n_proj"] = len(sub)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["avail_year"] = out.year + 1
    out["gov_share_of_year"] = out.gov_funds / out.year_total_gov
    out["method"] = f"a_link_pit_{key}"
    out.to_parquet(fn, index=False)
    return out


if __name__ == "__main__":
    a = build("ksic")
    ref = pd.read_parquet(os.path.join(EXT, "ntis_project_ksic_year.parquet"))
    ref = ref[(ref.method == "a_link_pit") & (ref.year <= 2022)]
    m = a.merge(ref, on=["ksic", "year"], suffixes=("", "_ref"))
    print("재현 확인 rows", len(m), "/", len(ref), "max rel diff gov_funds", float((abs(m.gov_funds - m.gov_funds_ref) / m.gov_funds_ref.clip(lower=1)).max()))
    b = build("xw")
    print("xw 표", b.shape, b.ksic.nunique())
