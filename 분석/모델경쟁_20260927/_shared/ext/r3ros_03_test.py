# -*- coding: utf-8 -*-
"""개발 구간 선행 검정(T3형): 셀(KSIC×IPC 메인그룹)의 '과학 의존도'(원점까지 발행된 연결 US 특허 중 논문 인용 특허 비율)가
이후 3년 KR 출원 성장을 선행하는가?
🔒 원점 t0 ∈ 2008..2016, 결과 KR 출원연도 t0+1..t0+3 ≤ 2019 (T3 개발 규칙 t0 ≤ 2016). 피처는 issue_year ≤ t0 & first_rec_year ≤ t0.
출력: r3ros_test_cell.csv
"""
import numpy as np, pandas as pd, duckdb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
EXT = Path(__file__).resolve().parent; E = EXT.as_posix()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
kr = con.sql(f"""select ksic, mg, appl_year y, count(*) n from read_parquet('{E}/r3ros_work/rec.parquet')
   where cc='KR' and mg is not null and appl_year between 2000 and 2019 group by all""").df()
ci = con.sql(f"select * from read_parquet('{E}/r3ros_cell_issueyear.parquet')").df()
W = kr.pivot_table(index=["ksic", "mg"], columns="y", values="n", aggfunc="sum", fill_value=0)
rows, D = [], {}
for t0 in range(2008, 2017):
    past = W[[t0 - 2, t0 - 1, t0]].sum(1); prev = W[[t0 - 5, t0 - 4, t0 - 3]].sum(1); fut = W[[t0 + 1, t0 + 2, t0 + 3]].sum(1)
    d = pd.DataFrame({"past": past, "prev": prev, "fut": fut}).reset_index()
    d = d[d.past >= 5].copy()
    d["g"] = np.log1p(d.fut) - np.log1p(d.past)
    d["y"] = (d.g >= d.g.quantile(0.9)).astype(int)
    d["lpast"] = np.log1p(d.past); d["gpast"] = np.log1p(d.past) - np.log1p(d.prev)
    c = ci[(ci.issue_year <= t0) & (ci.issue_year >= t0 - 9) & (ci.first_rec_year <= t0)]
    c = c.groupby(["ksic", "mg"])[["n_us_pat", "n_us_pat_sci", "n_paper_cit", "n_us_pat_sci_front"]].sum().reset_index()
    d = d.merge(c, on=["ksic", "mg"], how="left").fillna({"n_us_pat": 0, "n_us_pat_sci": 0, "n_paper_cit": 0, "n_us_pat_sci_front": 0})
    d["has_us"] = (d.n_us_pat >= 5).astype(int)
    ks = d[d.has_us == 1].groupby("ksic").apply(lambda x: x.n_us_pat_sci.sum() / x.n_us_pat.sum())
    d["sci_share"] = np.where(d.has_us == 1, d.n_us_pat_sci / d.n_us_pat.clip(lower=1), d.ksic.map(ks))
    d["sci_share"] = d.sci_share.fillna(d.sci_share.mean())
    d["lpaper_pp"] = np.where(d.has_us == 1, np.log1p(d.n_paper_cit / d.n_us_pat.clip(lower=1)), np.nan)
    d["lpaper_pp"] = d.lpaper_pp.fillna(d.lpaper_pp.mean())
    d["lus"] = np.log1p(d.n_us_pat)
    # KSIC 내 편차(산업 고정효과 제거)
    for col in ["sci_share", "lpaper_pp"]:
        d[col + "_w"] = d[col] - d.groupby("ksic")[col].transform("mean")
    d["t0"] = t0; D[t0] = d
    # 편상관: 통제(lpast, gpast, lus, KSIC FE) 잔차끼리 Spearman
    s = d[d.has_us == 1]
    X = np.column_stack([np.ones(len(s)), s.lpast, s.gpast, s.lus, pd.get_dummies(s.ksic, drop_first=True).values.astype(float)])
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    rho, p = spearmanr(res(s.sci_share.values), res(s.g.values))
    rho2, p2 = spearmanr(res(s.lpaper_pp.values), res(s.g.values))
    rows.append(dict(kind="partial_spearman", t0=t0, n_cells=len(d), n_cells_us5=len(s), cov_us5=round(len(s) / len(d), 3),
                     rho_sci_share=round(rho, 4), p_sci_share=round(p, 4), rho_paper_pp=round(rho2, 4), p_paper_pp=round(p2, 4)))
    print(rows[-1], flush=True)
BASEF = ["lpast", "gpast", "lus"]
for v in range(2011, 2017):
    tr = pd.concat([D[t] for t in range(2008, v - 2)]); te = D[v]
    r = dict(kind="auc_walkforward", t0=v, n_train=len(tr), n_cells=len(te))
    for name, F in (("base", BASEF), ("base+sci", BASEF + ["sci_share_w", "lpaper_pp_w"])):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[F], tr.y)
        r["auc_" + name] = round(roc_auc_score(te.y, m.predict_proba(te[F])[:, 1]), 4)
    r["delta"] = round(r["auc_base+sci"] - r["auc_base"], 4)
    rows.append(r); print(r, flush=True)
R = pd.DataFrame(rows); R.to_csv(EXT / "r3ros_test_cell.csv", index=False, encoding="utf-8-sig")
ps = R[R.kind == "partial_spearman"]; aw = R[R.kind == "auc_walkforward"]
print("편상관 sci_share 평균", round(ps.rho_sci_share.mean(), 4), "양(+)", int((ps.rho_sci_share > 0).sum()), "/", len(ps), "p<0.05", int((ps.p_sci_share < 0.05).sum()))
print("편상관 paper/pat 평균", round(ps.rho_paper_pp.mean(), 4), "양(+)", int((ps.rho_paper_pp > 0).sum()), "/", len(ps))
print("AUC 증분 평균", round(aw.delta.mean(), 4), "우세", int((aw.delta > 0).sum()), "/", len(aw))
