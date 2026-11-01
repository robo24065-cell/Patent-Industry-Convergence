# -*- coding: utf-8 -*-
r"""C01-T1 s4: (c)(iii) MAP-SYNC CN 보정 민감도, (d) JP 2021 공개 결손이 P4 RTA에 주는 영향(m2 분야계수 적용 시나리오),
(e) T3 공개연도 정의(least(pub_date, rgst_date))에서 CN 계열의 구성 변화(진단만, 재학습 없음).
산출: sync_sensitivity.csv, jp2021_p4.json, t3_cn_pubyear_diag.csv, s4_results.json
"""
import os, json
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
import wlib
OUT = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(OUT, "cache")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
M5 = os.path.join(MC, r"_shared_report\fusion_advantage\map5")
CCS = ["KR", "US", "JP", "CN", "EP"]
res = {}
ag = pd.read_parquet(os.path.join(C, "agg_ksic.parquet")); ag = ag[ag.is_inv].copy(); ag["ksic"] = ag.ksic.astype(str)
W, T = wlib.build_tables()


# ---------- (iii) MAP-SYNC ----------
def sync(panel, col, years, c="CN", nperm=1000, seed=7):
    rng = np.random.default_rng(seed)
    pa = panel[panel.year.between(*years)].copy()
    pa["x"] = np.log((pa[col] + 0.5) / pa.groupby(["cc", "year"])[col].transform("sum"))
    X = pa.pivot_table(index=["cc", "ksic"], columns="year", values="x"); D = X.diff(axis=1).iloc[:, 1:]
    D = D - D.groupby(level=0).transform("mean")
    a = D.loc["KR"].values; b = D.loc[c].values
    mc = lambda bb: np.nanmean([np.corrcoef(a[i], bb[i])[0, 1] for i in range(len(a))])
    obs = mc(b); null = np.array([mc(b[rng.permutation(len(b))]) for _ in range(nperm)])
    return round(obs, 4), round((np.sum(null >= obs) + 1) / (nperm + 1), 4)


p0 = pd.read_csv(os.path.join(M5, "m2_panel_appl.csv"), dtype={"ksic": str})
res["sync_repro_2004_2018_raw_CN"] = sync(p0, "n_inv", (2004, 2018))
d = wlib.weights(ag[ag.appl_year.between(2010, 2018)], W, T)
pan = d.groupby(["ksic", "cc", "appl_year"])[["n", "nA", "nB"]].sum().reset_index().rename(columns={"appl_year": "year"})
full = pd.MultiIndex.from_product([sorted(pan.ksic.unique()), CCS, range(2010, 2019)], names=["ksic", "cc", "year"])
pan = pan.set_index(["ksic", "cc", "year"]).reindex(full, fill_value=0).reset_index()
rows = []
for c in ["CN", "US", "JP", "EP"]:
    for col in ["n", "nA", "nB"]:
        mcor, p = sync(pan, col, (2011, 2018), c=c)
        rows.append(dict(partner=c, window="2011-2018", series={"n": "raw", "nA": "A", "nB": "B"}[col], mean_corr=mcor, p_perm=p))
sy = pd.DataFrame(rows); sy.to_csv(os.path.join(OUT, "sync_sensitivity.csv"), index=False, encoding="utf-8-sig")
cn = sy[sy.partner == "CN"].set_index("series")
res["sync_CN_2011_2018"] = cn[["mean_corr", "p_perm"]].to_dict("index")
res["sync_CN_robust"] = bool(all(abs(cn.at[s, "mean_corr"] - cn.at["raw", "mean_corr"]) <= 0.05 and cn.at[s, "p_perm"] < 0.05 for s in ["A", "B"]))

# ---------- (d) JP 2021: m2 분야계수를 JP 공개 2021 문서에만 적용한 P4 RTA ----------
fm2 = pd.read_csv(os.path.join(M5, "m2_coverage_factors_pub.csv"))
fj = fm2[(fm2.cc == "JP") & (fm2.year == 2021)][["wipo_field", "f"]]
p4 = ag[ag.appl_year.between(2020, 2022)].merge(fj, on="wipo_field", how="left")
jp21 = (p4.cc == "JP") & (p4.pub_year == 2021)
p4["f"] = np.where(jp21, p4.f.fillna(1.293), 1.0)
p4["n_jp"] = p4.n * p4.f


def rta(n):
    x = n.unstack("cc")[CCS].fillna(0); s = x / x.sum(0); return s.div(s.mean(1), axis=0)


R0 = rta(p4.groupby(["ksic", "cc"]).n.sum()); R1 = rta(p4.groupby(["ksic", "cc"]).n_jp.sum())
res["jp2021_p4"] = dict(jp_docs_pub2021_share_of_JP_P4=round(float(p4[jp21].n.sum() / p4[p4.cc == "JP"].n.sum()), 4),
                        leaders_raw=R0.idxmax(1).value_counts().to_dict(), leaders_m2f=R1.idxmax(1).value_counts().to_dict(),
                        rho_JP=round(spearmanr(R0.JP, R1.JP).statistic, 4), rho_KR=round(spearmanr(R0.KR, R1.KR).statistic, 4),
                        kr_gt1=[int((R0.KR > 1).sum()), int((R1.KR > 1).sum())], leader_changed=int((R0.idxmax(1) != R1.idxmax(1)).sum()),
                        m2_factor_range_JP2021=[round(float(fj.f.min()), 3), round(float(fj.f.max()), 3)])
# P4 JP 문서의 공개연도 분포
res["jp_P4_pubyear_dist"] = p4[p4.cc == "JP"].groupby(p4.pub_year.fillna(0).astype(int)).n.sum().to_dict()

# ---------- (e) T3 공개연도 정의에서 CN 계열 ----------
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
t3 = con.sql(f"""select ksic, py::int y, src, typ, count(*) n from (
  select ksic, left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4) py,
   case when nullif(pub_date,'') is null then 'rgst_only' when nullif(rgst_date,'') is null then 'pub_only'
        when nullif(pub_date,'') <= nullif(rgst_date,'') then 'pub_first' else 'rgst_first' end src,
   case when length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') then 'inv' when substr(appl_no,5,1) in ('2','9') then 'um' else 'other' end typ
  from read_parquet('{META}') where cc='CN' and appl_year>=2000) where py is not null and py::int between 2004 and 2021 group by all""").df()
tt = t3.groupby(["y", "typ", "src"]).n.sum().unstack(["typ", "src"]).fillna(0).astype(int)
tt.columns = [f"{a}_{b}" for a, b in tt.columns]
tt["total"] = tt.sum(1)
tt["inv_rgst_only_share"] = (tt.filter(like="inv_rgst_only").sum(1) / tt.total).round(4)
tt["um_share"] = (tt.filter(like="um_").sum(1) / tt.total).round(4)
tt.to_csv(os.path.join(OUT, "t3_cn_pubyear_diag.csv"), encoding="utf-8-sig")
# KSIC 간 구성 비균일성: 연도별 KSIC별 'inv_rgst_only' 비중의 CV
k = t3.assign(ro=(t3.typ == "inv") & (t3.src == "rgst_only")).groupby(["y", "ksic"]).apply(lambda g: g[g.ro].n.sum() / g.n.sum(), include_groups=False)
cv = k.groupby(level=0).agg(lambda s: s.std() / s.mean() if s.mean() > 0 else np.nan)
res["t3_cn_inv_rgst_only_share_by_year"] = tt.inv_rgst_only_share.to_dict()
res["t3_cn_um_share_by_year"] = tt.um_share.to_dict()
res["t3_cn_rgst_only_ksic_cv_by_year"] = cv.round(3).to_dict()
res["t3_cn_rgst_only_ksic_share_range_2012"] = [round(float(k.loc[2012].min()), 3), round(float(k.loc[2012].max()), 3)]
res["t3_cn_rgst_only_ksic_share_range_2019"] = [round(float(k.loc[2019].min()), 3), round(float(k.loc[2019].max()), 3)]
json.dump(res, open(os.path.join(OUT, "s4_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 250)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str)); print(sy.to_string()); print(tt.to_string())
