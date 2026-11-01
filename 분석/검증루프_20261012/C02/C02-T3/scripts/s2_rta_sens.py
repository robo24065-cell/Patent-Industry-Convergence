# E-C01-02 독립 재생성: 원천 meta + WIPO 원파일에서 직접 (C01 cache 미사용)
import sys, os, time, json, duckdb, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = B + r"\모델경쟁_20260927"; OUT = B + r"\검증루프_20261012\C02\C02-T3"; CA = OUT + r"\cache"; os.makedirs(CA, exist_ok=True)
META = (MC + r"\_shared\meta\*.parquet").replace("\\", "/")
CCS = ["KR", "US", "JP", "CN", "EP"]
t = time.time()
f_doc = CA + r"\doc_agg.parquet"
if not os.path.exists(f_doc):
    con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
    inv = """case when cc='KR' then substr(appl_no,1,2)='10'
     when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
     when cc='JP' then length(appl_no) in (8,17) else true end"""
    # 문서(cc, appl_no, ksic) 단위로 subclass·main group·연도만 뽑아 집계(행=출원×KSIC)
    con.sql(f"""copy (select ksic, cc, appl_year, try_cast(substr(pub_date,1,4) as int) pub_year,
        substr(main_ipc,1,4) sub, trim(split_part(substr(main_ipc,5),'/',1)) grp, count(*) n,
        count(distinct appl_no) n_doc
        from read_parquet('{META}') where ({inv}) group by all) to '{f_doc.replace(chr(92),'/')}' (format parquet)""")
    print("agg built", round(time.time() - t), flush=True)
d = pd.read_parquet(f_doc); d["ksic"] = d.ksic.astype(str)
cw = pd.read_csv(MC + r"\_shared\ext\quality_wipo_ipc_field_concordance.csv", dtype=str)
g = cw[cw.main_group.notna() & (cw.main_group != "")].set_index(["subclass", "main_group"]).wipo_field
s = cw[cw.main_group.isna() | (cw.main_group == "")].drop_duplicates("subclass").set_index("subclass").wipo_field
key = list(zip(d["sub"], d["grp"]))
fg = pd.Series(key).map(g.to_dict())
fs = d["sub"].map(s.to_dict())
d["f"] = pd.to_numeric(fg.values, errors="coerce")
d["f"] = d["f"].fillna(pd.to_numeric(fs, errors="coerce")).fillna(0).astype(int)
# 1) m1_counts_appl 대조(원천 일치 확인)
m1 = pd.read_parquet(MC + r"\_shared_report\fusion_advantage\map5\m1_counts_appl.parquet"); m1["ksic"] = m1.ksic.astype(str)
mine = d.groupby(["ksic", "cc", "appl_year"]).n.sum()
cmp_ = m1.set_index(["ksic", "cc", "appl_year"]).n_inv.reindex(mine.index)
print("m1 n_inv vs mine maxabs:", float((cmp_ - mine).abs().max()), "rows", len(mine), flush=True)
# 2) WIPO 공개 분야 건수
wp = pd.read_csv(MC + r"\_shared\ext\quality_raw\quality_wipo_pi\dc_indicator_patent_4_publication_by_technology.csv")
wp = wp[wp.office.isin(CCS) & (wp.tec_id > 0)].groupby(["office", "year", "tec_id"])["count"].sum().reset_index()
wp.columns = ["cc", "year", "f", "wipo"]
# 제공 공개 분야 건수: 행 기준(n)과 고유문서 기준(n_doc) 둘 다
res = {}
for unit in ["n", "n_doc"]:
    pf = d[d.pub_year.notna() & (d.f > 0)].groupby(["cc", "pub_year", "f"])[unit].sum().reset_index()
    pf.columns = ["cc", "year", "f", "prov"]
    m = wp.merge(pf, on=["cc", "year", "f"], how="left").fillna({"prov": 0})
    m["r"] = m.prov / m.wipo; m["small"] = (m.wipo < 300) | (m.prov < 30)
    tot = m.groupby(["cc", "year"])[["prov", "wipo"]].sum(); tot["r_tot"] = tot.prov / tot.wipo; tot = tot.reset_index()
    gt = tot[tot.year >= 2010].groupby("year").r_tot.apply(lambda x: np.exp(np.log(x).mean())).rename("g_tot")
    tot = tot.merge(gt, on="year", how="left"); tot["q_tot"] = tot.r_tot / tot.g_tot
    m = m.merge(tot[["cc", "year", "r_tot", "q_tot"]], on=["cc", "year"])
    m["rA"] = np.where(m.small | (m.r <= 0), m.r_tot, m.r)
    gg = m.groupby(["year", "f"]).rA.apply(lambda x: np.exp(np.log(x.clip(lower=1e-6)).mean())).rename("g")
    m = m.merge(gg, on=["year", "f"]); m["qB"] = np.where(m.small | (m.r <= 0), m.q_tot, m.r / m.g)
    m["wA"] = 1 / m.rA.clip(0.02, 1.5); m["wB"] = 1 / m.qB.clip(0.25, 4)
    tot["wA_tot"] = 1 / tot.r_tot.clip(0.02, 1.5); tot["wB_tot"] = 1 / tot.q_tot.clip(0.25, 4)
    wf = m.set_index(["cc", "year", "f"])[["wA", "wB"]]; wt = tot.set_index(["cc", "year"])[["wA_tot", "wB_tot"]]
    for p, (y0, y1) in {"P3": (2017, 2019), "P4": (2020, 2022)}.items():
        x = d[d.appl_year.between(y0, y1)].copy()
        miss = x.pub_year.isna() | (x.pub_year < 2003)
        x["wy"] = np.where(miss, (x.appl_year + 1).clip(2003, 2022), x.pub_year.clip(upper=2022)).astype(int)
        A = wf.reindex(list(zip(x.cc, x.wy, x.f)))
        T = wt.reindex(list(zip(x.cc, x.wy)))
        use_tot = miss.values | (x.f.values == 0) | A.wA.isna().values
        x["wA"] = np.where(use_tot, T.wA_tot.values, A.wA.values); x["wB"] = np.where(use_tot, T.wB_tot.values, A.wB.values)
        out = {}
        R = {}
        for tag, w in [("raw", None), ("A", "wA"), ("B", "wB")]:
            v = x.n if w is None else x.n * x[w]
            nn = v.groupby([x.ksic, x.cc]).sum().unstack()[CCS].fillna(0)
            sh = nn / nn.sum(0); R[tag] = sh.div(sh.mean(1), axis=0)
        for tag in ["raw", "A", "B"]:
            r = R[tag]; lead = r.idxmax(1)
            out[tag] = dict(leaders={c: int((lead == c).sum()) for c in CCS}, kr_gt1=int((r.KR > 1).sum()),
                            kr_rho_vs_raw=round(spearmanr(R["raw"].KR, r.KR).statistic, 4),
                            kr_leaders=sorted(lead[lead == "KR"].index),
                            cosmetics_20423={c: round(float(r.loc["20423", c]), 3) for c in CCS}, cosmetics_leader=lead["20423"],
                            kr_top_vals={k: round(float(r.loc[k, "KR"]), 3) for k in ["20423", "26212", "31312", "24191", "26121", "25922", "28511"]},
                            share_total_fallback=round(float((x.n * use_tot).sum() / x.n.sum()), 4))
        res[f"{unit}_{p}"] = out
json.dump(res, open(OUT + r"\s2_rta_sens_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in res.items():
    print(k)
    for tag, o in v.items():
        print(" ", tag, o["leaders"], "KR>1", o["kr_gt1"], "rho", o["kr_rho_vs_raw"], "cos", o["cosmetics_20423"]["KR"], o["cosmetics_leader"], "fallback", o["share_total_fallback"])
print("sec", round(time.time() - t))
