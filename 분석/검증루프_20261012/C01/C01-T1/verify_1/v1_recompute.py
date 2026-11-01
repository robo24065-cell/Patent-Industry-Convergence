# -*- coding: utf-8 -*-
r"""C01-T1 verify_1: F7(WIPO35 RTA 제공판 vs WIPO판 ρ) 독립 재계산. 원천 meta + WIPO csv 직접(캐시 미사용). 읽기만, 쓰기는 이 폴더."""
import os, json, duckdb, numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = os.path.dirname(os.path.abspath(__file__))
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
CW = os.path.join(MC, r"_shared\ext\quality_wipo_ipc_field_concordance.csv")
WF = os.path.join(MC, r"_shared\ext\quality_raw\quality_wipo_pi\dc_indicator_patent_4_publication_by_technology.csv")
CCS = ["KR", "US", "JP", "CN", "EP"]
cw = pd.read_csv(CW, dtype=str, encoding="utf-8-sig")
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
con.register("cw", cw)
inv = """case when cc='KR' then substr(appl_no,1,2)='10'
 when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
 when cc='JP' then length(appl_no) in (8,17) else true end"""
# 독립 구현: 문서 단위로 먼저 고유화(min pub, first main_ipc) 후 매핑
q = f"""
with d as (select cc, appl_no, min(try_cast(substr(pub_date,1,4) as int)) py, min(main_ipc) ipc, count(*) nrow,
             bool_or({inv}) is_inv
           from read_parquet('{META}') group by cc, appl_no),
 x as (select *, upper(substr(replace(ipc,' ',''),1,4)) sub,
          try_cast(trim(split_part(substr(ipc,5),'/',1)) as int) grp from d)
select x.cc, x.py, coalesce(try_cast(g.wipo_field as int), try_cast(s.wipo_field as int), 0) f, x.is_inv, count(*) n, sum(nrow) nrows
from x left join (select * from cw where main_group is not null) g on x.sub=g.subclass and x.grp=try_cast(g.main_group as int)
 left join (select * from cw where main_group is null) s on x.sub=s.subclass
where x.py between 2010 and 2022 group by all"""
a = con.sql(q).df()
a.to_csv(os.path.join(OUT, "v1_prov_cc_py_f.csv"), index=False, encoding="utf-8-sig")
print("rows==docs check", int((a.n != a.nrows).sum()))
w = pd.read_csv(WF, keep_default_na=False)
res = {"wipo_origins_nonISO": sorted({o for o in w.origin.unique() if len(o) != 2})}
w5 = w[w.office.isin(CCS) & (w.tec_id > 0)].groupby(["office", "year", "tec_id"])["count"].sum()
wworld = w[(w.tec_id > 0)].groupby(["office", "year", "tec_id"])["count"].sum()


def rta(x):  # x: DataFrame f x cc
    s = x / x.sum(0); return s.div(s.mean(1), axis=0)


def comp(y0, y1, inv_only=True, drop=None, base="5cc"):
    p = a[(a.f > 0) & a.py.between(y0, y1) & (a.is_inv if inv_only else True)].groupby(["f", "cc"]).n.sum().unstack()[CCS]
    ww = w5.loc[(slice(None), slice(y0, y1), slice(None))].groupby(["tec_id", "office"]).sum().unstack()[CCS]
    if drop: p = p.drop(drop); ww = ww.drop(drop)
    Rp, Rw = rta(p), rta(ww)
    out = {c: round(spearmanr(Rp[c], Rw[c]).statistic, 4) for c in CCS}
    pear = {c: round(float(np.corrcoef(np.log(Rp[c]), np.log(Rw[c]))[0, 1]), 4) for c in CCS}
    return Rp, Rw, out, pear, p, ww


Rp, Rw, rho, pear, p, ww = comp(2017, 2019)
res["P3_rho"] = rho; res["P3_pearson_log"] = pear
res["P3_kr_gt1_prov_wipo"] = [int((Rp.KR > 1).sum()), int((Rw.KR > 1).sum())]
res["P3_leader_agree"] = int((Rp.idxmax(1) == Rw.idxmax(1)).sum())
res["P3_leaders_prov"] = Rp.idxmax(1).value_counts().to_dict(); res["P3_leaders_wipo"] = Rw.idxmax(1).value_counts().to_dict()
old = pd.read_csv(os.path.join(os.path.dirname(OUT), "rta_wipo35_P3.csv"), encoding="utf-8-sig").set_index("f")
res["maxabs_vs_orig_csv"] = float(max(np.abs(old[[f"prov_{c}" for c in CCS]].values - Rp.values).max(), np.abs(old[[f"wipo_{c}" for c in CCS]].values - Rw.values).max()))
# 창 민감도
win = {}
for y0, y1 in [(2014, 2016), (2015, 2017), (2016, 2018), (2017, 2019), (2018, 2020), (2019, 2021), (2020, 2022)]:
    _, _, r_, pe_, _, _ = comp(y0, y1); win[f"{y0}-{y1}"] = {"spearman": r_, "pearson_log": pe_, "KR_rank_lowest": min(r_, key=r_.get) == "KR"}
res["window"] = win
# 단년
res["single_year"] = {y: comp(y, y)[2] for y in range(2015, 2023)}
# 작은 칸 제외: 어느 국가든 prov<100 또는 wipo<300인 분야 제외
small = ((p < 100) | (ww < 300)).any(axis=1); res["small_fields"] = p.index[small].tolist()
res["P3_rho_excl_small"] = comp(2017, 2019, drop=p.index[small].tolist())[2]
res["P3_rho_excl16"] = comp(2017, 2019, drop=[16])[2]
# 부트스트랩(분야 재표집) KR ρ와 (나머지 4개국 중앙값 − KR)
rng = np.random.default_rng(12); B = 2000; kr, gap, low = [], [], 0
for _ in range(B):
    i = rng.integers(0, 35, 35); P = Rp.iloc[i]; W_ = Rw.iloc[i]
    r = {c: spearmanr(P[c], W_[c]).statistic for c in CCS}
    kr.append(r["KR"]); gap.append(np.median([r[c] for c in CCS if c != "KR"]) - r["KR"]); low += min(r, key=r.get) == "KR"
res["boot_KR_rho_CI95"] = [round(float(np.quantile(kr, .025)), 3), round(float(np.quantile(kr, .975)), 3)]
res["boot_gap_CI95"] = [round(float(np.quantile(gap, .025)), 3), round(float(np.quantile(gap, .975)), 3)]
res["boot_P_KR_lowest"] = low / B
# 실용신안 포함(발명 정의 미적용) 민감도
res["P3_rho_alldocs"] = comp(2017, 2019, inv_only=False)[2]
# KR 불일치 상위 분야
dif = pd.DataFrame({"prov_KR": Rp.KR, "wipo_KR": Rw.KR, "rank_prov": Rp.KR.rank(), "rank_wipo": Rw.KR.rank()}); dif["d"] = dif.rank_prov - dif.rank_wipo
dif["r_KR"] = p.KR / ww.KR
dif.sort_values("d").round(3).to_csv(os.path.join(OUT, "v1_kr_field_rank_diff.csv"), encoding="utf-8-sig")
json.dump(res, open(os.path.join(OUT, "v1_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(dif.sort_values("d").round(3).to_string())
