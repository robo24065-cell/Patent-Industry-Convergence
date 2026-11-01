# E-C01-05 독립 재계산: MAP-SYNC (m3c 정의) 원판 창별 + 커버리지 재가중 A/B (2011–18)
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = B + r"\모델경쟁_20260927"; OUT = B + r"\검증루프_20261012\C02\C02-T3"
CCS = ["KR", "US", "JP", "CN", "EP"]; FOR = ["US", "JP", "CN", "EP"]
d = pd.read_parquet(OUT + r"\cache\doc_agg.parquet"); d["ksic"] = d.ksic.astype(str)
cw = pd.read_csv(MC + r"\_shared\ext\quality_wipo_ipc_field_concordance.csv", dtype=str)
g = cw[cw.main_group.notna() & (cw.main_group != "")].set_index(["subclass", "main_group"]).wipo_field.to_dict()
s = cw[cw.main_group.isna() | (cw.main_group == "")].drop_duplicates("subclass").set_index("subclass").wipo_field.to_dict()
d["f"] = pd.to_numeric(pd.Series(list(zip(d["sub"], d["grp"]))).map(g).values, errors="coerce")
d["f"] = d.f.fillna(pd.to_numeric(d["sub"].map(s), errors="coerce")).fillna(0).astype(int)
wp = pd.read_csv(MC + r"\_shared\ext\quality_raw\quality_wipo_pi\dc_indicator_patent_4_publication_by_technology.csv")
wp = wp[wp.office.isin(CCS) & (wp.tec_id > 0)].groupby(["office", "year", "tec_id"])["count"].sum().reset_index(); wp.columns = ["cc", "year", "f", "wipo"]
pf = d[d.pub_year.notna() & (d.f > 0)].groupby(["cc", "pub_year", "f"]).n.sum().reset_index(); pf.columns = ["cc", "year", "f", "prov"]
m = wp.merge(pf, on=["cc", "year", "f"], how="left").fillna({"prov": 0}); m["r"] = m.prov / m.wipo; m["small"] = (m.wipo < 300) | (m.prov < 30)
tot = m.groupby(["cc", "year"])[["prov", "wipo"]].sum(); tot["r_tot"] = tot.prov / tot.wipo; tot = tot.reset_index()
gt = tot[tot.year >= 2010].groupby("year").r_tot.apply(lambda x: np.exp(np.log(x).mean())).rename("g_tot")
tot = tot.merge(gt, on="year", how="left"); tot["q_tot"] = tot.r_tot / tot.g_tot
m = m.merge(tot[["cc", "year", "r_tot", "q_tot"]], on=["cc", "year"]); m["rA"] = np.where(m.small | (m.r <= 0), m.r_tot, m.r)
gg = m.groupby(["year", "f"]).rA.apply(lambda x: np.exp(np.log(x.clip(lower=1e-6)).mean())).rename("g")
m = m.merge(gg, on=["year", "f"]); m["qB"] = np.where(m.small | (m.r <= 0), m.q_tot, m.r / m.g)
m["wA"] = 1 / m.rA.clip(0.02, 1.5); m["wB"] = 1 / m.qB.clip(0.25, 4); tot["wA_tot"] = 1 / tot.r_tot.clip(0.02, 1.5); tot["wB_tot"] = 1 / tot.q_tot.clip(0.25, 4)
wf = m.set_index(["cc", "year", "f"])[["wA", "wB"]]; wt = tot.set_index(["cc", "year"])[["wA_tot", "wB_tot"]]
x = d[d.appl_year.between(2004, 2018)].copy()
miss = x.pub_year.isna() | (x.pub_year < 2003)
x["wy"] = np.where(miss, (x.appl_year + 1).clip(2003, 2022), x.pub_year.clip(upper=2022)).astype(int)
A = wf.reindex(list(zip(x.cc, x.wy, x.f))); T = wt.reindex(list(zip(x.cc, x.wy)))
use_tot = miss.values | (x.f.values == 0) | A.wA.isna().values
x["nA"] = x.n * np.where(use_tot, T.wA_tot.values, A.wA.values); x["nB"] = x.n * np.where(use_tot, T.wB_tot.values, A.wB.values)
x["n_"] = x.n.astype(float)
res = {"fallback_share_by_cc_2004_10": x[x.appl_year <= 2010].assign(u=use_tot[x.appl_year.values <= 2010] * x[x.appl_year <= 2010].n).groupby("cc").apply(lambda z: round(z.u.sum() / z.n.sum(), 3)).to_dict()}
def sync(col, y0, y1, nperm=1000, seed=7):
    rng = np.random.default_rng(seed)
    p = x[x.appl_year.between(y0, y1)].groupby(["cc", "ksic", "appl_year"])[col].sum()
    full = pd.MultiIndex.from_product([CCS, sorted(x.ksic.unique()), range(y0, y1 + 1)], names=["cc", "ksic", "appl_year"])
    p = p.reindex(full, fill_value=0).rename("v").reset_index()
    p["xx"] = np.log((p.v + 0.5) / p.groupby(["cc", "appl_year"]).v.transform("sum"))
    X = p.pivot_table(index=["cc", "ksic"], columns="appl_year", values="xx"); D = X.diff(axis=1).iloc[:, 1:]
    D = D - D.groupby(level=0).transform("mean")
    out = {}
    for c in FOR:
        a = D.loc["KR"].values; b = D.loc[c].values
        mc = lambda bb: np.nanmean([np.corrcoef(a[i], bb[i])[0, 1] for i in range(len(a))])
        obs = mc(b); null = np.array([mc(b[rng.permutation(len(b))]) for _ in range(nperm)])
        out[c] = (round(obs, 4), round((np.sum(null >= obs) + 1) / (nperm + 1), 4))
    return out
for (col, y0, y1) in [("n_", 2004, 2018), ("n_", 2011, 2018), ("n_", 2004, 2010), ("nA", 2011, 2018), ("nB", 2011, 2018)]:
    res[f"{col}_{y0}_{y1}"] = sync(col, y0, y1); print(col, y0, y1, res[f"{col}_{y0}_{y1}"], flush=True)
json.dump(res, open(OUT + r"\s3_sync_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res["fallback_share_by_cc_2004_10"])
