# -*- coding: utf-8 -*-
r"""verify_0 — 독립 재계산(작업자 코드 b_lib/c10_common/z_feat를 import하지 않음).
(1) sklearn roc_auc_score로 공식 seed 예측의 원점별 AUC와 P50−HSC1_50, P50−FREEIPC50, P50−FREE50 5원점 평균 Δ 재계산
(2) IPC 서브클래스 군집 '다항 재표집'(군집을 복원 추출, Poisson 아님) B=300, AUC는 dense rank + bincount 방식(작업자 cumsum 커널과 다른 코드)
(3) H-sc-1 라벨을 duckdb SQL로 다시 만들어(원점 2010, 표 = alt_KR 출원 ≤2010) HSC1_qmain 피처를 독립 재구성해 Z_o2010과 대조
(4) 반박 시도: (a) 표에 미래 문서(≤2019)를 쓰면 라벨이 얼마나 바뀌나(시점 민감도) (b) Δ가 한 원점에 의존하는지(원점 1개 제외 평균)
사용: python -B v_check.py → verify_0/v_result.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
NC8 = os.path.join(MC, r"T1_산업대기실\NONCIRC_R8\cache")
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
ZT = "FREEIPC50-HSC1_50-P50HSC1"
CS = [2008, 2009, 2010, 2011, 2012]
out = {}
data = {}
for c in CS:
    a = pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["applicantCd", "sc", "first_ky"])
    p1 = pd.read_parquet(os.path.join(W, "cache", f"pred_c{c}_k0_P50-FREE50-TXT50_p012.parquet"), columns=["P50_p0", "FREE50_p0"])
    p2 = pd.read_parquet(os.path.join(W, "cache", f"pred_c{c}_k0_{ZT}_p0.parquet"), columns=["HSC1_50_p0", "FREEIPC50_p0"])
    d = pd.concat([a, p1, p2], axis=1); d["y"] = d.first_ky.notna().astype(int)
    data[c] = d
# (1)
A = {c: {arm: roc_auc_score(d.y, d[arm + "_p0"].astype(np.float64)) for arm in ("P50", "FREE50", "HSC1_50", "FREEIPC50")} for c, d in data.items()}
out["sklearn_auc"] = A
for b in ("FREE50", "HSC1_50", "FREEIPC50"):
    dd = [A[c]["P50"] - A[c][b] for c in CS]
    out[f"P50-{b}"] = {"mean_d": float(np.mean(dd)), "per_origin": dd, "leave_one_out_means": [float(np.mean([x for j, x in enumerate(dd) if j != i])) for i in range(5)]}
log("sklearn", {k: v["mean_d"] for k, v in out.items() if k.startswith("P50-")})


# (2) dense rank AUC
def rank_prep(y, s, g):
    u, r = np.unique(s, return_inverse=True)
    return r, len(u), y.astype(bool), g


def wauc(pr, wg):
    r, nu, yb, g = pr; w = wg[g]
    wp = np.bincount(r[yb], weights=w[yb], minlength=nu); wn = np.bincount(r[~yb], weights=w[~yb], minlength=nu)
    below = np.concatenate([[0.0], np.cumsum(wn)[:-1]])
    return (wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


scs = np.array(sorted(set().union(*[set(d.sc) for d in data.values()]))); si = {s: i for i, s in enumerate(scs)}
PR = {(c, arm): rank_prep(d.y.values, d[arm + "_p0"].values.astype(np.float64), d.sc.map(si).values) for c, d in data.items() for arm in ("P50", "FREE50", "HSC1_50", "FREEIPC50")}
one = np.ones(len(scs))
out["rank_auc_unit_vs_sklearn_maxdiff"] = float(max(abs(wauc(PR[(c, arm)], one) - A[c][arm]) for c in CS for arm in ("P50", "FREE50", "HSC1_50", "FREEIPC50")))
rng = np.random.default_rng(99); B = 300
bd = {b: [] for b in ("FREE50", "HSC1_50", "FREEIPC50")}
for r in range(B):
    wg = np.bincount(rng.integers(0, len(scs), len(scs)), minlength=len(scs)).astype(float)
    au = {(c, arm): wauc(PR[(c, arm)], wg) for c in CS for arm in ("P50", "FREE50", "HSC1_50", "FREEIPC50")}
    for b in bd:
        bd[b].append(np.mean([au[(c, "P50")] - au[(c, b)] for c in CS]))
    if r % 50 == 0: log("boot", r)
out["sc_multinomial_boot"] = {f"P50-{b}": {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "p_gt0": float(np.mean(np.array(v) > 0)), "B": B, "n_sc": len(scs)} for b, v in bd.items()}
log("sc boot", out["sc_multinomial_boot"])

# (3) H-sc-1 독립 재구성(원점 2010)
f = lambda *p: os.path.join(MC, *p).replace("\\", "/")
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
con.execute(f"""create table lab as with docs as (
   select a.appl_no, a.appl_year, f.k_free, left(r.sg,4) sc4 from read_parquet('{f("_shared_report","fusion_advantage","c5_R10","alt_assign","alt_KR.parquet")}') a
   join read_parquet('{f("T1_산업대기실","TXT_R9","cache","kr_txtfree.parquet")}') f using(appl_no)
   left join (select appl_no, any_value(sg) sg from read_parquet('{f("T3_기술셀부상예측","ESSENTIAL_R7","rec_conc.parquet")}') where cc='KR' group by 1) r using(appl_no)),
 tab as (select sc4, k_free, count(*) n from docs where appl_year<=2010 and sc4 is not null and sc4<>'' and k_free is not null group by 1,2),
 best as (select sc4, arg_min(k_free, -n*100000 + cast(k_free as int)) k from tab group by 1)
 select d.appl_no, b.k from docs d left join best b using(sc4)""")
lab = con.sql("select * from lab").df()
P = pd.read_parquet(os.path.join(NC8, "patents.parquet"), columns=["applicantCd", "appl_no", "sc", "ky"]).merge(lab, on="appl_no", how="left")
V = P[(P.ky <= 2010) & P.k.notna()]
KS = np.sort(V.k.unique())
cnt = V.groupby(["applicantCd", "k"]).size().unstack(fill_value=0).reindex(columns=KS, fill_value=0)
firm_main = pd.Series(KS[cnt.values.argmax(1)], index=cnt.index)
scmode = V.groupby(["sc", "k"]).size().reset_index(name="n").sort_values(["sc", "n", "k"], ascending=[True, False, True]).drop_duplicates("sc").set_index("sc").k
z = pd.read_parquet(os.path.join(W, "cache", "Z_o2010.parquet"), columns=["applicantCd", "sc", "HSC1_qmain"])
# Q[sc, main] = sc 안에서 main 라벨 비율(H-sc-1이면 sc 안 라벨이 하나라 0/1이 기대되지만, sc 키는 rec_conc, Q는 T1 sc → 같으면 one-hot)
qsh = V.groupby(["sc", "k"]).size() / V.groupby("sc").size()
key = pd.MultiIndex.from_arrays([z.sc.values, z.applicantCd.map(firm_main).values])
mine = qsh.reindex(key).fillna(0).values
out["hsc1_qmain_2010"] = {"rows": len(z), "maxabs": float(np.max(np.abs(mine - z.HSC1_qmain.values))), "share_eq": float(np.mean(np.abs(mine - z.HSC1_qmain.values) < 1e-6)),
                          "share_q_onehot_sc": float((V.groupby("sc").k.nunique() == 1).mean()), "share_qmain_1": float((z.HSC1_qmain.values > 0.999).mean())}
# (4a) 시점 민감도: 표 ≤2019(미래 포함) 라벨과 ≤2010 라벨 차이(가시 문서 기준)
con.execute(f"""create table lab19 as with docs as (
   select a.appl_no, a.appl_year, f.k_free, left(r.sg,4) sc4 from read_parquet('{f("_shared_report","fusion_advantage","c5_R10","alt_assign","alt_KR.parquet")}') a
   join read_parquet('{f("T1_산업대기실","TXT_R9","cache","kr_txtfree.parquet")}') f using(appl_no)
   left join (select appl_no, any_value(sg) sg from read_parquet('{f("T3_기술셀부상예측","ESSENTIAL_R7","rec_conc.parquet")}') where cc='KR' group by 1) r using(appl_no)),
 tab as (select sc4, k_free, count(*) n from docs where appl_year<=2019 and sc4 is not null and sc4<>'' group by 1,2),
 best as (select sc4, arg_min(k_free, -n*100000 + cast(k_free as int)) k from tab group by 1)
 select d.appl_no, b.k k19 from docs d left join best b using(sc4)""")
l19 = con.sql("select * from lab19").df()
VV = V.merge(l19, on="appl_no", how="left")
out["time_sensitivity_2010_vs_2019_table"] = {"vis_docs_label_changed": float((VV.k != VV.k19).mean())}
json.dump(out, open(os.path.join(HERE, "v_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done", json.dumps({k: v for k, v in out.items() if k != "sklearn_auc"}, ensure_ascii=False, default=float)[:3000])
