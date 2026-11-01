# NTIS rpatent 원시 페이지 → ntis_rpatent_raw.parquet, 제공 KR 특허 연결 → ntis_patent_link.parquet / ntis_patent_flag.parquet / ntis_link_rates.csv
import glob, json, gzip, re, pathlib
import pandas as pd, duckdb
import ntis_common as n
EXT = n.EXT; WORK = n.WORK
META = EXT.parent / "meta"
rows = []
for f in sorted(glob.glob(str(WORK / "ntis_rp_*.jsonl.gz"))):
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        rows += [json.loads(l) for l in fh]
raw = pd.DataFrame(rows).astype("string")
n0 = len(raw)
raw = raw.drop_duplicates("ResultID").reset_index(drop=True)
print("원시 행", n0, "고유 ResultID", len(raw))
raw = raw.rename(columns={"RegistCountry_code": "ipc_codes"})
raw["num_digits"] = raw["RegistNumber"].str.replace(r"\D", "", regex=True)
raw["is_kr"] = raw["RegistCountry"].eq("대한민국")
raw.to_parquet(EXT / "ntis_rpatent_raw.parquet", index=False)

# 제공 KR 특허
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2")
kr = con.sql(f"""select appl_no, rgst_no, appl_year, rgst_date, ksic
  from read_parquet('{(META / '*.parquet').as_posix()}') where cc='KR'""").df()
kr["appl_no"] = kr["appl_no"].str.replace(r"\D", "", regex=True)
kr["rgst_no"] = kr["rgst_no"].str.replace(r"\D", "", regex=True)
print("제공 KR", len(kr), "고유 appl_no", kr.appl_no.nunique())

k = raw[raw.is_kr & raw.num_digits.str.len().eq(13)].copy()
keep = ["ResultID", "Year", "RegistNumber", "RegistType", "IprType", "ProjectYear", "ProjectID",
        "BudgetProjectNumber", "BudgetProject", "MinistryName", "MinistryName_code", "PerformAgency",
        "PerformAgency_code", "PerformAgent", "PerformAgent_code", "SixTechnology_code",
        "TechnologyRoadMap_code", "ScienceClass1_code", "ScienceClass1", "ScienceClass2_code",
        "ScienceClass3_code", "ipc_codes", "SourceFlag", "num_digits"]
k = k[keep]
a = kr[["appl_no", "ksic"]].merge(k, left_on="appl_no", right_on="num_digits").assign(match_route="appl_no")
r = kr[kr.rgst_no.notna()][["appl_no", "rgst_no", "ksic"]].merge(k, left_on="rgst_no", right_on="num_digits").drop(columns="rgst_no").assign(match_route="rgst_no")
link = pd.concat([a, r], ignore_index=True).drop(columns="num_digits")
link = link.drop_duplicates(["appl_no", "ResultID"])
link = link.merge(kr[["appl_no", "rgst_no", "appl_year"]], on="appl_no", how="left")
def agent_grp(c):
    return {"03": "대학", "02": "출연연", "01": "국공립연", "04": "기업", "05": "기업", "08": "기업", "06": "정부부처"}.get(c, "기타")
link["perform_grp"] = link["PerformAgent_code"].map(agent_grp)
link.insert(0, "cc", "KR")
link.to_parquet(EXT / "ntis_patent_link.parquet", index=False)
print("연결 행", len(link), "연결 특허", link.appl_no.nunique())
print(pd.crosstab(link.RegistType, link.match_route))

# 특허 단위 플래그
g = link.groupby("appl_no")
flag = pd.DataFrame({
    "gov_rnd": True,
    "n_results": g.ResultID.nunique(),
    "n_projects": g.ProjectID.nunique(),
    "project_ids": g.ProjectID.agg(lambda s: "|".join(sorted(set(s.dropna())))),
    "ministries": g.MinistryName.agg(lambda s: "|".join(sorted(set(s.dropna())))),
    "perform_grps": g.perform_grp.agg(lambda s: "|".join(sorted(set(s.dropna())))),
    "perform_agents": g.PerformAgent.agg(lambda s: "|".join(sorted(set(s.dropna())))),
    "min_project_year": pd.to_numeric(g.ProjectYear.min(), errors="coerce"),
    "max_project_year": pd.to_numeric(g.ProjectYear.max(), errors="coerce"),
}).reset_index()
flag["has_univ"] = flag.perform_grps.str.contains("대학")
flag["has_gri"] = flag.perform_grps.str.contains("출연연|국공립연")
flag["has_firm"] = flag.perform_grps.str.contains("기업")
full = kr[["appl_no", "rgst_no", "appl_year", "ksic"]].merge(flag, on="appl_no", how="left")
full["gov_rnd"] = full["gov_rnd"].fillna(False).astype(bool)
for c in ["has_univ", "has_gri", "has_firm"]:
    full[c] = full[c].fillna(False).astype(bool)
full.insert(0, "cc", "KR")
full.to_parquet(EXT / "ntis_patent_flag.parquet", index=False)

# 연계율
full["registered"] = full.rgst_no.notna()
def rates(df, keys):
    o = df.groupby(keys, dropna=False).agg(n_patents=("appl_no", "size"), n_linked=("gov_rnd", "sum"),
        n_registered=("registered", "sum"),
        n_linked_registered=("gov_rnd", lambda s: int((s & df.loc[s.index, "registered"]).sum())),
        n_univ=("has_univ", "sum"), n_gri=("has_gri", "sum"), n_firm=("has_firm", "sum")).reset_index()
    o["link_rate"] = (o.n_linked / o.n_patents).round(4)
    o["link_rate_registered"] = (o.n_linked_registered / o.n_registered.where(o.n_registered > 0)).round(4)
    return o
allr = rates(full.assign(level="ALL"), ["level"]).assign(ksic="ALL", appl_year=pd.NA)
kr_ = rates(full, ["ksic"]).assign(level="ksic", appl_year=pd.NA)
yr = rates(full, ["appl_year"]).assign(level="year", ksic="ALL")
ky = rates(full, ["ksic", "appl_year"]).assign(level="ksic_year")
out = pd.concat([allr, kr_, yr, ky], ignore_index=True)
cols = ["level", "ksic", "appl_year", "n_patents", "n_linked", "link_rate", "n_registered", "n_linked_registered", "link_rate_registered", "n_univ", "n_gri", "n_firm"]
out[cols].to_csv(EXT / "ntis_link_rates.csv", index=False, encoding="utf-8-sig")
print(allr[cols].to_string())
print(kr_.link_rate.describe())
print(yr[cols].to_string())
