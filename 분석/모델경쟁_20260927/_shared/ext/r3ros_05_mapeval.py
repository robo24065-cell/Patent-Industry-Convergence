# -*- coding: utf-8 -*-
"""IPC 메인그룹/셀 ↔ OpenAlex 분야 대응표의 (1) 시점 정합판 (2) 신뢰도(split-half) (3) 개발 구간 선행검정(T3형).
입력: r3ros_work/oa_sample.jsonl(무작위 표본 논문의 primary_topic), cit/fam_us/rec, r3ros_patent_sci(발행일).
외부 호출 없음.
🔒 선행검정 원점 t0 ∈ 2008..2016, 결과 KR 출원 t0+1..t0+3 ≤ 2019. 피처는 issue_year ≤ t0, first_rec_year ≤ t0, 논문 출판연도 ≤ t0.
출력: r3ros_oa_work_topic_sample.parquet, r3ros_cell_paper_topic_sample.parquet, r3ros_ipc_mg_openalex_subfield_le2016.parquet,
      r3ros_map_reliability.csv, r3ros_test_fieldmomentum.csv
"""
import json, numpy as np, pandas as pd, duckdb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
EXT = Path(__file__).resolve().parent; E = EXT.as_posix()
con = duckdb.connect(); con.execute("set memory_limit='3500MB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute(f"set temp_directory='{E}/r3ros_work/tmp'")
oa = pd.DataFrame([json.loads(l) for l in open(EXT / "r3ros_work/oa_sample.jsonl", encoding="utf-8")]).drop_duplicates("oaid")
oa = oa[oa.field.notna()].copy()
oa.to_parquet(EXT / "r3ros_oa_work_topic_sample.parquet", index=False)
print("표본 논문(primary_topic 보유)", len(oa), "subfield 수", oa.subfield_id.nunique(), "field 수", oa.field_id.nunique())
con.register("oa", oa)
con.execute(f"""create table cp as select r.ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year
   from read_parquet('{E}/r3ros_work/fam_us.parquet') f join read_parquet('{E}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic)
   where r.mg is not null group by all""")
con.execute(f"""create table t as select c.ksic, c.mg, c.us_pn, c.first_rec_year, p.issue_year, x.oaid, o.year paper_year,
      o.subfield_id, o.subfield, o.field_id, o.field, o.domain
   from (select distinct pn us_pn, oaid from read_parquet('{E}/r3ros_work/cit.parquet') where oaid in (select oaid from oa)) x
   join cp c using(us_pn) join read_parquet('{E}/r3ros_patent_sci.parquet') p using(us_pn) join oa o using(oaid)""")
T = con.sql("select * from t").df()
T.to_parquet(EXT / "r3ros_cell_paper_topic_sample.parquet", index=False)
print("표본 삼중쌍(셀×US특허×논문)", len(T), "셀", T.groupby(["ksic", "mg"]).ngroups, "메인그룹", T.mg.nunique(), "KSIC", T.ksic.nunique())
# ---- (1) 시점 정합판(≤2016) ----
d16 = T[(T.issue_year <= 2016) & (T.first_rec_year <= 2016) & (T.paper_year <= 2016)]
mg16 = d16.groupby(["mg", "subfield_id", "subfield", "field_id", "field"]).agg(n_pairs=("oaid", "size"), n_works=("oaid", "nunique")).reset_index()
mg16["share"] = mg16.n_pairs / mg16.groupby("mg").n_pairs.transform("sum")
nw = d16.groupby("mg").oaid.nunique().rename("mg_n_works"); mg16 = mg16.merge(nw, on="mg")
mg16.to_parquet(EXT / "r3ros_ipc_mg_openalex_subfield_le2016.parquet", index=False)
print("≤2016 대응표: 메인그룹", mg16.mg.nunique(), "서로 다른 논문 ≥30인 메인그룹", int((nw >= 30).sum()))
nwa = T.groupby("mg").oaid.nunique()
print("전체기간 메인그룹", len(nwa), "논문≥30", int((nwa >= 30).sum()), "논문≥100", int((nwa >= 100).sum()))
# 제공 KR 출원(2005-2016) 중 논문≥30 메인그룹에 속하는 비율
cov = con.sql(f"select mg, count(*) n from read_parquet('{E}/r3ros_work/rec.parquet') where cc='KR' and appl_year between 2005 and 2016 and mg is not null group by 1").df()
cov["ok30"] = cov.mg.map(nw).fillna(0) >= 30
print("KR 출원 2005-16 중 ≤2016판 논문≥30 메인그룹 소속 비율", round(cov.loc[cov.ok30, "n"].sum() / cov.n.sum(), 4))
# ---- (2) split-half 신뢰도 ----
rng = np.random.default_rng(7); half = pd.Series(rng.integers(0, 2, len(oa)), index=oa.oaid.values)
T["h"] = T.oaid.map(half)
rel = []
for lev, key, cat in (("maingroup", "mg", "subfield_id"), ("maingroup_field", "mg", "field_id"), ("ksic", "ksic", "subfield_id"), ("ksic_field", "ksic", "field_id")):
    g = T.groupby([key, "h", cat]).size().rename("n").reset_index()
    nwh = T.groupby([key, "h"]).oaid.nunique().unstack(fill_value=0)
    ok = nwh[(nwh.min(1) >= 30)].index
    P = g[g[key].isin(ok)].pivot_table(index=[key, "h"], columns=cat, values="n", fill_value=0)
    P = P.div(P.sum(1), axis=0)
    a = P.xs(0, level="h"); b = P.xs(1, level="h").reindex(a.index)
    cos = (a.values * b.values).sum(1) / (np.linalg.norm(a.values, axis=1) * np.linalg.norm(b.values, axis=1))
    top = (a.values.argmax(1) == b.values.argmax(1))
    rel.append(dict(level=lev, n_units=len(a), median_cos=round(float(np.median(cos)), 4), mean_cos=round(float(cos.mean()), 4), top1_agree=round(float(top.mean()), 4)))
    print(rel[-1])
pd.DataFrame(rel).to_csv(EXT / "r3ros_map_reliability.csv", index=False, encoding="utf-8-sig")
# ---- (3) 선행검정: 과학분야 모멘텀 → 셀 성장 ----
kr = con.sql(f"""select ksic, mg, appl_year y, count(*) n from read_parquet('{E}/r3ros_work/rec.parquet')
   where cc='KR' and mg is not null and appl_year between 2000 and 2019 group by all""").df()
W = kr.pivot_table(index=["ksic", "mg"], columns="y", values="n", aggfunc="sum", fill_value=0)
ci = con.sql(f"select * from read_parquet('{E}/r3ros_cell_issueyear.parquet')").df()
rows, D = [], {}
for t0 in range(2008, 2017):
    past = W[[t0 - 2, t0 - 1, t0]].sum(1); prev = W[[t0 - 5, t0 - 4, t0 - 3]].sum(1); fut = W[[t0 + 1, t0 + 2, t0 + 3]].sum(1)
    d = pd.DataFrame({"past": past, "prev": prev, "fut": fut}).reset_index(); d = d[d.past >= 5].copy()
    d["g"] = np.log1p(d.fut) - np.log1p(d.past); d["y"] = (d.g >= d.g.quantile(0.9)).astype(int)
    d["lpast"] = np.log1p(d.past); d["gpast"] = np.log1p(d.past) - np.log1p(d.prev)
    c = ci[(ci.issue_year <= t0) & (ci.issue_year >= t0 - 9) & (ci.first_rec_year <= t0)]
    c = c.groupby(["ksic", "mg"])[["n_us_pat", "n_us_pat_sci"]].sum().reset_index()
    d = d.merge(c, on=["ksic", "mg"], how="left").fillna({"n_us_pat": 0, "n_us_pat_sci": 0})
    d["lus"] = np.log1p(d.n_us_pat)
    d["sci_share"] = np.where(d.n_us_pat >= 5, d.n_us_pat_sci / d.n_us_pat.clip(lower=1), np.nan)
    d["sci_share"] = d.sci_share.fillna(d.groupby("ksic").sci_share.transform("mean")).fillna(np.nanmean(d.sci_share))
    # 분야 모멘텀: 연결 US 특허(발행 t0-2..t0 vs t0-5..t0-3)가 인용한 표본 논문의 subfield별 건수 증가율(전체 대비)
    a = T[(T.issue_year <= t0) & (T.first_rec_year <= t0) & (T.paper_year <= t0)]
    A = a[a.issue_year >= t0 - 2].groupby("subfield_id").size(); B = a[(a.issue_year >= t0 - 5) & (a.issue_year <= t0 - 3)].groupby("subfield_id").size()
    S = pd.concat([A.rename("A"), B.rename("B")], axis=1).fillna(0)
    mom = np.log1p(S.A) - np.log1p(S.B); mom = mom - (np.log1p(S.A.sum()) - np.log1p(S.B.sum()))
    # 셀 과학 프로필(발행 t0-9..t0) · 논문 최신성(발행연도-출판연도)
    w = a[a.issue_year >= t0 - 9].copy(); w["mom"] = w.subfield_id.map(mom); w["age"] = w.issue_year - w.paper_year
    cf = w.groupby(["ksic", "mg"]).agg(n_sp=("oaid", "size"), mom_cell=("mom", "mean"), age_cell=("age", "median"),
                                      n_sub=("subfield_id", "nunique")).reset_index()
    d = d.merge(cf, on=["ksic", "mg"], how="left"); d["n_sp"] = d.n_sp.fillna(0)
    d["has_sp"] = (d.n_sp >= 5).astype(int)
    for col in ["mom_cell", "age_cell"]:
        d.loc[d.has_sp == 0, col] = np.nan
        d[col] = d[col].fillna(d.groupby("ksic")[col].transform("mean")).fillna(np.nanmean(d[col]))
        d[col + "_w"] = d[col] - d.groupby("ksic")[col].transform("mean")
    d["sci_share_w"] = d.sci_share - d.groupby("ksic").sci_share.transform("mean")
    d["t0"] = t0; D[t0] = d
    s = d[d.has_sp == 1]
    X = np.column_stack([np.ones(len(s)), s.lpast, s.gpast, s.lus, s.sci_share, pd.get_dummies(s.ksic, drop_first=True).values.astype(float)])
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    r1, p1 = spearmanr(res(s.mom_cell.values), res(s.g.values)); r2, p2 = spearmanr(res(s.age_cell.values), res(s.g.values))
    rows.append(dict(kind="partial_spearman", t0=t0, n_cells=len(d), n_cells_sp5=len(s), cov_sp5=round(len(s) / len(d), 3),
                     rho_mom=round(r1, 4), p_mom=round(p1, 4), rho_age=round(r2, 4), p_age=round(p2, 4)))
    print(rows[-1], flush=True)
BASE = ["lpast", "gpast", "lus", "sci_share_w"]
for v in range(2011, 2017):
    tr = pd.concat([D[t] for t in range(2008, v - 2)]); te = D[v]
    r = dict(kind="auc_walkforward", t0=v, n_train=len(tr), n_cells=len(te))
    for name, F in (("base_sci", BASE), ("base_sci+mom", BASE + ["mom_cell_w"]), ("base_sci+mom+age", BASE + ["mom_cell_w", "age_cell_w"])):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[F], tr.y)
        r["auc_" + name] = round(roc_auc_score(te.y, m.predict_proba(te[F])[:, 1]), 4)
    r["d_mom"] = round(r["auc_base_sci+mom"] - r["auc_base_sci"], 4); r["d_mom_age"] = round(r["auc_base_sci+mom+age"] - r["auc_base_sci"], 4)
    rows.append(r); print(r, flush=True)
R = pd.DataFrame(rows); R.to_csv(EXT / "r3ros_test_fieldmomentum.csv", index=False, encoding="utf-8-sig")
ps = R[R.kind == "partial_spearman"]; aw = R[R.kind == "auc_walkforward"]
print("모멘텀 편상관 평균", round(ps.rho_mom.mean(), 4), "양", int((ps.rho_mom > 0).sum()), "/", len(ps), "p<0.05", int((ps.p_mom < 0.05).sum()))
print("최신성(나이) 편상관 평균", round(ps.rho_age.mean(), 4), "음", int((ps.rho_age < 0).sum()), "/", len(ps), "p<0.05", int((ps.p_age < 0.05).sum()))
print("AUC 증분 mom 평균", round(aw.d_mom.mean(), 4), "우세", int((aw.d_mom > 0).sum()), "/", len(aw), "| mom+age 평균", round(aw.d_mom_age.mean(), 4), "우세", int((aw.d_mom_age > 0).sum()))
