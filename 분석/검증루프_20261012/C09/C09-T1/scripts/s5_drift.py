# -*- coding: utf-8 -*-
# 5단계: 배정 규칙 점검 — 같은 IPC 메인그룹의 KSIC 분포 연도 간 JS 거리, 출원인 결정성
import sys, os, time, json
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
t0 = time.time(); res = {}
kr = pd.read_parquet(os.path.join(CACHE, "kr_meta.parquet"), columns=["appl_no", "appl_year", "main_ipc", "ksic", "applicant"])
k = kr[(kr.appl_no.str[:2] == "10") & kr.appl_year.between(2010, 2022) & kr.main_ipc.notna()].copy()
k["mg"] = k.main_ipc.str.split("/").str[0].str.replace(" ", "", regex=False)
top = k[k.appl_year <= 2019].mg.value_counts().head(200).index.tolist()
k = k[k.mg.isin(top)]
res["top200_doc_share_of_kr10_2010_22"] = float(len(k) / ((kr.appl_no.str[:2] == "10") & kr.appl_year.between(2010, 2022)).sum())
ks = sorted(k.ksic.unique()); kidx = {x: i for i, x in enumerate(ks)}
k["ki"] = k.ksic.map(kidx)
YRS = list(range(2010, 2023))
C = {}   # (mg, y) -> counts vector
for (mg, y), s in k.groupby(["mg", "appl_year"]).ki:
    C[(mg, y)] = np.bincount(s.values, minlength=len(ks))


def js(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    def kl(a, b):
        msk = a > 0
        return float((a[msk] * np.log2(a[msk] / b[msk])).sum())
    return np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m))   # JS 거리(밑 2, 제곱근)


rng = np.random.default_rng(20261013)
rows = []
for mg in top:
    for i, y1 in enumerate(YRS):
        for y2 in YRS[i + 1:]:
            a, b = C.get((mg, y1)), C.get((mg, y2))
            if a is None or b is None or a.sum() < 20 or b.sum() < 20:
                continue
            d = js(a.astype(float), b.astype(float))
            row = dict(main_group=mg, y1=y1, y2=y2, n1=int(a.sum()), n2=int(b.sum()), js=d)
            if y2 == y1 + 1 or (y1, y2) in [(2014, 2018), (2015, 2017), (2016, 2017), (2013, 2019)]:
                pool = (a + b) / (a + b).sum()
                nul = [js(rng.multinomial(a.sum(), pool).astype(float) + 0.0, rng.multinomial(b.sum(), pool).astype(float) + 0.0) for _ in range(200)]
                row["null95"] = float(np.percentile(nul, 95)); row["null_mean"] = float(np.mean(nul)); row["exceed"] = bool(d > row["null95"])
            rows.append(row)
    if len(rows) % 5000 < 80:
        pass
D = pd.DataFrame(rows)
D.to_csv(os.path.join(WD, "ipc_ksic_drift.csv"), index=False, encoding="utf-8-sig")
adj = D[D.y2 == D.y1 + 1]
byy = adj.groupby("y1").agg(n_groups=("js", "size"), js_med=("js", "median"), null_med=("null_mean", "median"), exceed_rate=("exceed", "mean"))
res["adjacent_year_drift"] = byy.round(4).reset_index().to_dict("records")
res["adjacent_exceed_rate_overall"] = float(adj.exceed.mean())
# 거리와 연도 간격: 간격이 커질수록 JS가 커지는지(점진 이동) 대 특정 해에 계단(체제)
D["gap"] = D.y2 - D.y1
res["js_by_gap_median"] = D.groupby("gap").js.median().round(4).to_dict()
# 2016 기준: 앞뒤 대칭 비교 — 같은 간격(1년)에서 2015→16, 16→17, 17→18
res["exceed_2015_16_16_17_17_18"] = [float(adj[adj.y1 == y].exceed.mean()) for y in (2015, 2016, 2017)]
print("drift done", round(time.time() - t0), flush=True)
# 출원인 결정성: 같은 해, 같은 메인그룹·같은 출원인의 다른 문서 최빈 KSIC로 맞힌 비율 vs 같은 메인그룹 최빈 KSIC(LOO)
k2 = k[k.applicant.notna()].copy()
det = []
for y in YRS:
    s = k2[k2.appl_year == y]
    ga = s.groupby(["mg", "applicant", "ki"]).size().rename("c").reset_index()
    tot_a = ga.groupby(["mg", "applicant"]).c.transform("sum")
    gg = s.groupby(["mg", "ki"]).size().rename("c").reset_index()
    # LOO: 문서 자신의 KSIC 개수에서 1을 뺀 뒤 최빈과 비교 — 셀 단위로 계산
    # 출원인 기준
    ga["tot"] = tot_a
    ga = ga[ga.tot >= 2]
    hits_a = 0; n_a = 0; hits_g = 0
    gmap = {mg: grp.set_index("ki").c for mg, grp in gg.groupby("mg")}
    for (mg, ap), grp in ga.groupby(["mg", "applicant"]):
        cnt = grp.set_index("ki").c
        gc = gmap[mg]
        for ki, cc in cnt.items():
            # 이 KSIC 문서 cc개 각각: 자신을 뺀 출원인 분포 최빈이 ki인가(동률이면 자기 KSIC 우선하지 않음: 엄격)
            ca = cnt.copy(); ca[ki] -= 1
            mx = ca.max(); top_a = ca[ca == mx].index
            hit_a = (ki in top_a) and len(top_a) == 1
            cg = gc.copy(); cg[ki] -= 1
            mxg = cg.max(); top_g = cg[cg == mxg].index
            hit_g = (ki in top_g) and len(top_g) == 1
            hits_a += cc * hit_a; hits_g += cc * hit_g; n_a += cc
    det.append(dict(year=y, n_docs=int(n_a), acc_applicant_loo=hits_a / n_a, acc_group_loo=hits_g / n_a))
DET = pd.DataFrame(det)
res["applicant_determinism"] = DET.round(4).to_dict("records")
print("det done", round(time.time() - t0), flush=True)
# 출원인 안 KSIC 다양도(상위 50 출원인, 연도별) — 전체 KR 범위 안 문서 기준
k3 = kr[(kr.appl_no.str[:2] == "10") & kr.appl_year.between(2010, 2022) & kr.applicant.notna()]
top50 = k3[k3.appl_year <= 2016].applicant.value_counts().head(50).index
div = []
for y in YRS:
    s = k3[(k3.appl_year == y) & k3.applicant.isin(top50)]
    for ap, grp in s.groupby("applicant"):
        if len(grp) < 30:
            continue
        p = grp.ksic.value_counts(normalize=True).values
        div.append(dict(year=y, applicant=ap, n=len(grp), n_ksic=int(len(p)), eff_n=float(np.exp(-(p * np.log(p)).sum())), top_share=float(p[0])))
DV = pd.DataFrame(div)
res["within_applicant_diversity_by_year"] = DV.groupby("year").agg(n_app=("applicant", "size"), eff_n_med=("eff_n", "median"), top_share_med=("top_share", "median")).round(3).reset_index().to_dict("records")
DV.to_csv(os.path.join(CACHE, "s5_within_applicant_diversity.csv"), index=False, encoding="utf-8-sig")
DET.to_csv(os.path.join(CACHE, "s5_applicant_determinism.csv"), index=False, encoding="utf-8-sig")
res["sec"] = round(time.time() - t0)
json.dump(res, open(os.path.join(CACHE, "s5_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
for kk, v in res.items():
    print(kk, json.dumps(v, ensure_ascii=False, default=float)[:1800])
