# 1단계: 페이지 parquet → ntis_project_raw.parquet(슬림) + ntis_project_year_agg.parquet
import sys, glob, os, json, pandas as pd, numpy as np
import ntis_B_common as c
sys.stdout.reconfigure(encoding="utf-8")
files = sorted(glob.glob(c.WORK + r"\proj\ntis_proj_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
n0 = len(df)
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
df = df.drop_duplicates(["pjt_no", "year"]).reset_index(drop=True)
print("페이지", len(files), "행", n0, "고유(과제번호,연도)", len(df))
for col in ["gov_funds", "sbiz_funds", "total_funds"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
df["nongov_funds"] = (df["total_funds"] - df["gov_funds"]).clip(lower=0)
df["secret_flag"] = (df["perform_agent_cd"].eq("SECRET PROJECT") | df["sc_old1_L_cd"].eq("SECRET PROJECT")).fillna(False).astype(bool)
GRP = {"01": "국공립연", "02": "출연연", "03": "대학", "04": "대기업", "05": "중소기업", "06": "중견기업",
       "07": "정부부처", "08": "기타", "99": "기타"}
df["perform_grp"] = df["perform_agent_cd"].map(GRP).fillna(pd.Series(np.where(df["secret_flag"], "보안과제", "기타"), index=df.index))
def nz(s):
    s = s.astype("string").str.strip()
    return s.where(s.notna() & (s != "") & (s != "SECRET PROJECT"))
new_L, new_M, new_S = nz(df["sc_new1_L_cd"]), nz(df["sc_new1_M_cd"]), nz(df["sc_new1_S_cd"])
old_L, old_M = nz(df["sc_old1_L_cd"]), nz(df["sc_old1_M_cd"])
# 구분류 중분류 코드가 대분류 코드와 같으면(2002 등) 중분류 없음으로 간주. 'Z'는 미분류
old_M = old_M.where(old_M.str.len() > 1)
old_L = old_L.where(old_L != "Z")
df["cls_sys"] = np.where(new_L.notna(), "new", np.where(old_L.notna(), "old", "none"))
df["cls_L"] = new_L.fillna(old_L)
df["cls_M"] = np.where(new_L.notna(), new_M, old_M)
df["cls_S"] = np.where(new_L.notna(), new_S, None)
df["cls_L_nm"] = nz(df["sc_new1_L"]).fillna(nz(df["sc_old1_L"]))
df["cls_M_nm"] = np.where(new_L.notna(), nz(df["sc_new1_M"]), nz(df["sc_old1_M"]))
df["cls_S_nm"] = np.where(new_L.notna(), nz(df["sc_new1_S"]), None)
for col in ["cls_L", "cls_M", "cls_S", "cls_L_nm", "cls_M_nm", "cls_S_nm", "cls_sys", "perform_grp"]:
    df[col] = df[col].astype("string")
def canon_L(nm):
    if nm is None or pd.isna(nm): return "NA"
    for key, lab in [("수학", "수학"), ("물리", "물리"), ("화학공정", "화공"), ("화공", "화공"), ("화학", "화학"),
                     ("우주", "우주항공해양"), ("지구", "지구과학"), ("뇌", "생명과학"), ("생명", "생명과학"), ("농림", "농림수산"),
                     ("보건", "보건의료"), ("기계", "기계"), ("재료", "재료"), ("전기", "전기전자"), ("정보", "정보통신"),
                     ("통신", "정보통신"), ("에너지", "에너지자원"), ("원자력", "원자력"), ("환경", "환경"), ("건설", "건설교통")]:
        if key in nm: return lab
    return "인문사회정책기타"
df["cls_L_canon"] = df["cls_L_nm"].map(canon_L).astype("string")
keep = ["pjt_no", "year", "title_ko", "title_en", "kw_ko", "kw_en", "ministry", "order_agency", "manage_agency",
        "research_agency", "org_pno", "corp_rgst_no", "budget_project", "business_name", "perform_agent_cd",
        "perform_agent", "perform_grp", "secret_flag", "dev_phase_cd", "dev_phase", "tech_lifecycle",
        "econ_goal_cd", "econ_goal", "sixt_cd", "sixt", "apply1_cd", "apply1", "apply2_cd", "apply3_cd",
        "region_cd", "period_start", "period_end", "total_start", "total_end", "continuous", "policy_flag",
        "pre_pjt", "series", "gov_funds", "sbiz_funds", "nongov_funds", "total_funds",
        "cls_sys", "cls_L_canon", "cls_L", "cls_M", "cls_S", "cls_L_nm", "cls_M_nm", "cls_S_nm",
        "sc_new2_M_cd", "sc_new2_S_cd", "sc_new3_M_cd", "sc_new3_S_cd", "sc_old1_L", "msc_L_cd", "msc_M_cd"]
raw = df[keep]
raw.to_parquet(c.EXT + r"\ntis_project_raw.parquet", index=False)
print("저장 ntis_project_raw.parquet", raw.shape)
# 연도×분류(대·중)×수행주체 집계
g = raw.groupby(["year", "cls_sys", "cls_L_canon", "cls_L", "cls_L_nm", "cls_M", "cls_M_nm", "perform_grp"], dropna=False)
agg = g.agg(n_projects=("pjt_no", "size"), gov_funds=("gov_funds", "sum"), nongov_funds=("nongov_funds", "sum"),
            total_funds=("total_funds", "sum")).reset_index()
agg.to_parquet(c.EXT + r"\ntis_project_year_agg.parquet", index=False)
print("저장 ntis_project_year_agg.parquet", agg.shape)
yr = raw.groupby("year").agg(n=("pjt_no", "size"), gov=("gov_funds", "sum"), tot=("total_funds", "sum"),
     new=("cls_sys", lambda s: (s == "new").mean()), old=("cls_sys", lambda s: (s == "old").mean()),
     kw=("kw_ko", lambda s: s.notna().mean()), clsM=("cls_M", lambda s: s.notna().mean()),
     corp=("corp_rgst_no", lambda s: s.notna().mean()), apply=("apply1_cd", lambda s: s.notna().mean()))
yr["gov_조원"] = (yr.gov / 1e12).round(2); yr["tot_조원"] = (yr.tot / 1e12).round(2)
print(yr.drop(columns=["gov", "tot"]).round(3).to_string())
yr.to_csv(c.WORK + r"\ntis_B_year_summary.csv", encoding="utf-8-sig")
