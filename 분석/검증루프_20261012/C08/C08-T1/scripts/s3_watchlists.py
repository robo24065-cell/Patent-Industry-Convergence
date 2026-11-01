# -*- coding: utf-8 -*-
# 3단계: 3.2.3 T3 감시 목록, 3.3.3 T1 대기실 지수, 4.3 T4 포기·N8 분쟁 산업 순위 — S0(재현)·S1 (모형 재학습 없음, 미래 구간 점수만 재집계)
# 원자료·트랙 파일 읽기만, sqlite mode=ro. 쓰기: C08-T1\cache, C08-T1 루트.
import sys, os, json, time, sqlite3
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd, duckdb
t0 = time.time()
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false")
res = {}


def topk_set(s, k=10):
    """값 내림차순 상위 k (경계 동점은 포함하지 않고 안정 정렬 순서로 자름)"""
    return list(s.sort_values(ascending=False, kind="mergesort").index[:k])


def overlap(a, b):
    return len(set(a) & set(b))


# ======================= T3 (3.2.3) =======================
T3 = os.path.join(MC, r"T3_기술셀부상예측\R3")
ff = pd.read_csv(os.path.join(T3, "future_forecast_r3.csv"), dtype={"ksic": str})
BR = os.path.join(T3, r"cache_pub_final\base_rows.parquet").replace("\\", "/")
con.sql(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
kr = con.sql(f"""select ksic, mg, app1, count(*) n from (
   select ksic, mgn(ipc_raw) mg, appl_no, any_value(app1) app1 from read_parquet('{BR}')
   where cc='KR' and year between 2022 and 2024 and mgn(ipc_raw) is not null group by 1,2,3) group by 1,2,3""").df()
kr["a"] = [norm_n2(x) if isinstance(x, str) and x.strip() else None for x in kr.app1]
krc = kr.groupby(["ksic", "mg"]).n.sum()
chk = ff.set_index(["ksic", "mg"]).kr_W
m = chk.to_frame().join(krc.rename("n"), how="left").fillna(0)
res["T3_reproduce_krW"] = dict(cells=int(len(m)), exact=int((m.kr_W == m.n).sum()), max_abs=float((m.kr_W - m.n).abs().max()))
ka = kr[kr.a.notna()].groupby(["ksic", "a"]).n.sum().reset_index()
top_ks = ka.sort_values(["ksic", "n", "a"], ascending=[True, False, True]).drop_duplicates("ksic").set_index("ksic")
ksic_tot = kr.groupby("ksic").n.sum()
top_ks["share_ksic"] = top_ks.n / ksic_tot.reindex(top_ks.index)
ca = kr[kr.a.notna()].groupby(["ksic", "mg", "a"]).n.sum().reset_index()
ca["is_top"] = ca.a.values == top_ks.a.reindex(ca.ksic).values
cell_top = ca[ca.is_top].groupby(["ksic", "mg"]).n.sum()
cell_any = ca.groupby(["ksic", "mg"]).n.max()   # 보조: 셀 안 최다 출원인(어느 출원인이든)
F = ff.set_index(["ksic", "mg"])[["kr_W", "pred_breakout_top10"]].copy()
F["top_n"] = cell_top.reindex(F.index).fillna(0).values
F["any_n"] = cell_any.reindex(F.index).fillna(0).values
F["top_sh"] = np.where(F.kr_W > 0, F.top_n / F.kr_W.clip(lower=1), 0)
F["any_sh"] = np.where(F.kr_W > 0, F.any_n / F.kr_W.clip(lower=1), 0)
F["drop_S1"] = F.top_sh >= 0.5
F["drop_S1b"] = (F.any_sh >= 0.5) & (F.kr_W >= 1)
F = F.reset_index()


def t3_rates(d):
    g = d.groupby("ksic").agg(n_cells=("mg", "size"), n_pred=("pred_breakout_top10", "sum"))
    g["rate"] = g.n_pred / g.n_cells
    return g


r0 = t3_rates(F); r1 = t3_rates(F[~F.drop_S1]); r1b = t3_rates(F[~F.drop_S1b])
fig7 = pd.read_csv(os.path.join(ROOT, r"분석\보고서_20261012\figs\fig07_t3_future_breakout_by_ksic.csv"), dtype={"ksic": str}).set_index("ksic")
named = ["26112", "28902", "31312", "27211"]
t10_0 = topk_set(r0.rate); t10_1 = topk_set(r1.rate); t10_1b = topk_set(r1b.rate)
res["T3"] = dict(reproduce_fig07_max_abs_rate=float((r0.rate - fig7.pred_rate.reindex(r0.index)).abs().max()),
                 n_cells=int(len(F)), n_pred=int(F.pred_breakout_top10.sum()),
                 S1_dropped_cells=int(F.drop_S1.sum()), S1_dropped_pred=int((F.drop_S1 & (F.pred_breakout_top10 == 1)).sum()),
                 S1b_dropped_cells=int(F.drop_S1b.sum()), S1b_dropped_pred=int((F.drop_S1b & (F.pred_breakout_top10 == 1)).sum()),
                 pred_cells_with_krW_gt0=int(((F.kr_W > 0) & (F.pred_breakout_top10 == 1)).sum()),
                 top10_S0=t10_0, top10_S1=t10_1, top10_S1b=t10_1b, overlap_S1=overlap(t10_0, t10_1), overlap_S1b=overlap(t10_0, t10_1b),
                 named_rates={k: dict(S0=round(r0.rate[k] * 100, 1), S1=round(r1.rate.get(k, np.nan) * 100, 1), S1b=round(r1b.rate.get(k, np.nan) * 100, 1),
                                      rank_S0=int(r0.rate.rank(ascending=False, method="min")[k]), rank_S1=int(r1.rate.rank(ascending=False, method="min")[k]),
                                      rank_S1b=int(r1b.rate.rank(ascending=False, method="min")[k])) for k in named},
                 named_in_top4_S1=sum(k in topk_set(r1.rate, 4) for k in named),
                 ksic_top_applicant_share_median=round(float(top_ks.share_ksic.median()), 3),
                 ksic_top_applicant_share_max=round(float(top_ks.share_ksic.max()), 3))
top_ks.assign(name=top_ks.index.map(lambda k: "")).to_csv(os.path.join(CACHE, "s3_t3_ksic_top_applicant.csv"), encoding="utf-8-sig")
pd.concat({"S0": r0, "S1": r1, "S1b": r1b}, axis=1).to_csv(os.path.join(CACHE, "s3_t3_rates.csv"), encoding="utf-8-sig")
print("T3 done", round(time.time() - t0), json.dumps(res["T3_reproduce_krW"]), flush=True)

# ======================= T1 (3.3.3) =======================
T1 = os.path.join(MC, r"T1_산업대기실\R3")
o = pd.read_parquet(os.path.join(T1, r"runs\FROZEN_future\o2024.parquet"))
te = o[~o.prior_any.values].copy()
st = pd.read_parquet(os.path.join(MC, r"_shared\ext\r3nts_applicant_status.parquet"), columns=["applicantCd", "end_dt"]).dropna(subset=["end_dt"])
closed = set(st[st.end_dt <= pd.Timestamp("2024-12-31")].applicantCd.astype(str))
te = te[~te.firm.astype(str).isin(closed)]
K = sorted(ff.ksic.unique()) if False else None
f1 = pd.read_csv(os.path.join(T1, "future_forecast_r3.csv"), dtype={"ksic": str})
K = sorted(f1.ksic.astype(str).unique()); assert len(K) == 50
te["ksic"] = [K[i] for i in te.k.values]
g0 = te.groupby("ksic").agg(at_risk=("px", "size"), idx=("px", "sum"))
g0["share"] = g0.idx / g0.idx.sum()
f1i = f1.set_index("ksic")
res["T1_reproduce"] = dict(at_risk_exact=int((g0.at_risk == f1i.at_risk.reindex(g0.index)).sum()),
                           idx_max_abs=float((g0.idx - f1i.waiting_room_index.reindex(g0.index)).abs().max()),
                           total_pairs=int(len(te)))
# 집중도: KSIC 안 법인별 px 몫
fx = te.groupby(["ksic", "firm"]).px.sum().reset_index()
fx["sh"] = fx.px / fx.groupby("ksic").px.transform("sum")
fx = fx.sort_values(["ksic", "px"], ascending=[True, False])
c1 = fx.groupby("ksic").agg(top1_sh=("sh", "max"), hhi=("sh", lambda s: float((s ** 2).sum())), top5_sh=("sh", lambda s: float(s.iloc[:5].sum())))
top1_firm = fx.drop_duplicates("ksic").set_index("ksic").firm
te1 = te[~(te.firm.values == top1_firm.reindex(te.ksic).values)]
g1 = te1.groupby("ksic").px.sum(); s1 = g1 / g1.sum()
# 보조 S1g: 전체 지수 합 1위 법인 1곳(모든 KSIC에서) 제거
gfirm = te.groupby("firm").px.sum().sort_values(ascending=False)
te1g = te[te.firm != gfirm.index[0]]; s1g = te1g.groupby("ksic").px.sum(); s1g = s1g / s1g.sum()
named1 = ["27212", "29280", "27199", "26429", "27301"]
t10_0 = topk_set(g0.share); t10_1 = topk_set(s1); t10_1g = topk_set(s1g)
res["T1"] = dict(top10_S0=t10_0, top10_S1=t10_1, overlap_S1=overlap(t10_0, t10_1), top5_S1=topk_set(s1, 5), top5_S0=topk_set(g0.share, 5),
                 top5_names_kept_in_top5_S1=sum(k in topk_set(s1, 5) for k in named1),
                 top5_share_sum_S0=round(float(g0.share[named1].sum() * 100), 1), top5_share_sum_S1=round(float(s1[named1].sum() * 100), 1),
                 named_share={k: dict(S0=round(g0.share[k] * 100, 1), S1=round(s1[k] * 100, 1), S1g=round(s1g[k] * 100, 1)) for k in named1},
                 top1_firm_share_of_ksic_index={k: round(float(c1.top1_sh[k]), 4) for k in named1},
                 top1_firm_share_median=round(float(c1.top1_sh.median()), 4), top1_firm_share_max=round(float(c1.top1_sh.max()), 4),
                 eff_firms_median=round(float((1 / c1.hhi).median()), 1),
                 global_top_firm_index_share=round(float(gfirm.iloc[0] / gfirm.sum()), 4), overlap_S1g=overlap(t10_0, t10_1g))
c1.assign(eff_n=1 / c1.hhi).to_csv(os.path.join(CACHE, "s3_t1_ksic_concentration.csv"), encoding="utf-8-sig")
print("T1 done", round(time.time() - t0), json.dumps(res["T1_reproduce"]), flush=True)
del o, te, te1, te1g, fx

# ======================= T4 (4.3 포기 위험) =======================
out43 = []
for rnd in ["R2", "R3"]:
    T4 = os.path.join(MC, rf"T4_특허포기예측\{rnd}")
    p = pd.read_parquet(os.path.join(T4, r"runs_future\FUT\pred_20250701.parquet"), columns=["pid", "ksic", "acd", "atype", "p"])
    p["ksic"] = p.ksic.astype(str); p["acd"] = p.acd.astype(str)
    fr = pd.read_csv(os.path.join(T4, f"future_forecast_{rnd.lower()}.csv"), dtype={"ksic": str})
    fl = fr[fr.L == "2025-07-01"].set_index("ksic")
    m0 = p.groupby("ksic").agg(n=("p", "size"), risk=("p", "mean"))
    cnt = p.groupby(["ksic", "acd"]).size().rename("c").reset_index().sort_values(["ksic", "c", "acd"], ascending=[True, False, True])
    top = cnt.drop_duplicates("ksic").set_index("ksic")
    top["sh"] = top.c / m0.n.reindex(top.index)
    p1 = p[p.acd.values != top.acd.reindex(p.ksic).values]
    m1 = p1.groupby("ksic").p.mean()
    t10_0 = topk_set(m0.risk); t10_1 = topk_set(m1)
    res[f"T4_{rnd}"] = dict(reproduce_n_exact=int((m0.n == fl.n_alive.reindex(m0.index)).sum()),
                            reproduce_risk_max_abs=float((m0.risk - fl.pred_risk.reindex(m0.index)).abs().max()),
                            top10_S0=t10_0, top10_S1=t10_1, overlap_S1=overlap(t10_0, t10_1),
                            rank_spearman_S0_S1=round(float(m0.risk.rank().corr(m1.reindex(m0.index).rank())), 3),
                            top1_holder_share_median=round(float(top.sh.median()), 3), top1_holder_share_max=round(float(top.sh.max()), 3),
                            top1_holder_share_ge20pct=sorted(top.index[top.sh >= 0.2]),
                            risk_26211={"S0": round(float(m0.risk["26211"]), 3), "S1": round(float(m1["26211"]), 3),
                                        "rank_S0": int(m0.risk.rank(ascending=False)["26211"]), "rank_S1": int(m1.rank(ascending=False)["26211"])})
    out43.append(pd.DataFrame({"track": f"T4_{rnd}", "ksic": m0.index, "S0": m0.risk.values, "S1": m1.reindex(m0.index).values,
                               "top1_share": top.sh.reindex(m0.index).values, "n": m0.n.values}))
print("T4 done", round(time.time() - t0), flush=True)

# ======================= N8 (4.3 분쟁 위험) =======================
N8 = os.path.join(MC, "N8_특허분쟁표적레이더")
sc = pd.read_parquet(os.path.join(N8, "future_scores_r6.parquet")); sc["ksic"] = sc.ksic.astype(str); sc["rgst_no"] = sc.rgst_no.astype(str)
fk = pd.read_csv(os.path.join(N8, "future_ksic_r6.csv"), dtype={"ksic": str}).set_index("ksic")
db = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
ap = pd.read_sql("select rgst_no, seq, applicantCd from applicant", db)
db.close()
ap["rgst_no"] = ap.rgst_no.astype(str)
ap = ap.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no").set_index("rgst_no").applicantCd.astype(str)
sc["acd"] = sc.rgst_no.map(ap)
n0 = sc.groupby("ksic").agg(n=("p", "size"), exp=("p", "sum"), mean_p=("p", "mean"))
cnt = sc[sc.acd.notna()].groupby(["ksic", "acd"]).size().rename("c").reset_index().sort_values(["ksic", "c", "acd"], ascending=[True, False, True])
top = cnt.drop_duplicates("ksic").set_index("ksic"); top["sh"] = top.c / n0.n.reindex(top.index)
# 기대 분쟁 몫도
psum = sc[sc.acd.notna()].groupby(["ksic", "acd"]).p.sum().rename("ps").reset_index()
top["p_sh"] = [float(psum[(psum.ksic == k) & (psum.acd == a)].ps.sum() / n0.exp[k]) for k, a in zip(top.index, top.acd)]
s1n = sc[sc.acd.values != top.acd.reindex(sc.ksic).values]
n1 = s1n.groupby("ksic").agg(exp=("p", "sum"), mean_p=("p", "mean"))
t10_0 = topk_set(n0.exp); t10_1 = topk_set(n1.exp); t10m0 = topk_set(n0.mean_p); t10m1 = topk_set(n1.mean_p)
res["N8"] = dict(reproduce_n_exact=int((n0.n == fk.n.reindex(n0.index)).sum()), reproduce_exp_max_abs=float((n0.exp - fk.exp_disputes.reindex(n0.index)).abs().max()),
                 acd_link_rate=round(float(sc.acd.notna().mean()), 4),
                 top10_S0=t10_0, top10_S1=t10_1, overlap_S1=overlap(t10_0, t10_1),
                 meanp_top10_S0=t10m0, meanp_top10_S1=t10m1, overlap_meanp_S1=overlap(t10m0, t10m1),
                 top1_holder_count_share_median=round(float(top.sh.median()), 3), top1_holder_count_share_max=round(float(top.sh.max()), 3),
                 top1_holder_expdispute_share_median=round(float(top.p_sh.median()), 3), top1_holder_expdispute_share_max=round(float(top.p_sh.max()), 3),
                 ksic_top1_expdispute_share_ge30pct=sorted(top.index[top.p_sh >= 0.3]))
out43.append(pd.DataFrame({"track": "N8_exp", "ksic": n0.index, "S0": n0.exp.values, "S1": n1.exp.reindex(n0.index).values,
                           "top1_share": top.p_sh.reindex(n0.index).values, "n": n0.n.values}))
pd.concat(out43).to_csv(os.path.join(CACHE, "s3_43_ksic.csv"), index=False, encoding="utf-8-sig")
print("N8 done", round(time.time() - t0), flush=True)
json.dump(res, open(os.path.join(CACHE, "s3_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
