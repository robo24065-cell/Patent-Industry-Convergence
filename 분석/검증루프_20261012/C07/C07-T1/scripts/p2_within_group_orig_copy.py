# -*- coding: utf-8 -*-
"""P2 연계표 그룹 내 세세분류 분화 — 외부 결과(KOSIS 광업제조업조사 5자리)로 보는 비순환 필수성 파일럿.
단위 (t, k), k ∈ 연계표가 5자리로 지목 못 하는 34개 KSIC. 결과 = k의 KOSIS 3년 성장이 같은 그룹 형제 합계보다 큰가.
점수(적합 없음): 형제 합계 중 k의 특허 몫 변화. PROV-KR / PROV-5C / TXT(해설서 배정) / CW(형제 동일).
🔒 KOSIS 2022 이후 값, 공표연도 2022 이후 특허는 읽자마자 버린다.
"""
import sys
sys.dont_write_bytecode = True
from r8_common import *
import duckdb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

YMAX = 2021
ORIG = [2009, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
# ---- KOSIS ----
d = pd.read_csv(KOSIS50, dtype=str, encoding="utf-8-sig")
d.columns = ["ver", "k", "nm", "ic", "item", "yr", "v", "mask"]
d["yr"] = d.yr.astype(int); d["v"] = pd.to_numeric(d.v, errors="coerce")
d = d[d.yr <= YMAX].copy()
assert d.yr.max() <= YMAX
d.loc[d["mask"] == "1", "v"] = np.nan


def series(ic):
    a = d[(d.ic == ic) & (d.ver == "10차")].pivot_table(index="k", columns="yr", values="v", aggfunc="first")
    b = d[(d.ic == ic) & (d.ver == "11차")].pivot_table(index="k", columns="yr", values="v", aggfunc="first")
    s = a.copy()
    s[2021] = a[2020] * (b[2021] / b[2020])  # 2020 겹침 연도로 연쇄
    return s.reindex(KSIC50)


KOS = {"ship": series("T04"), "emp": series("T02")}
# ---- 특허 ----
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
R, A = p('cache', 'rec.parquet').replace("\\", "/"), p('cache', 'kr_assign.parquet').replace("\\", "/")
pk = con.sql(f"SELECT ksic k, pyear, count(*) n FROM '{R}' WHERE cc='KR' AND pyear<={YMAX} GROUP BY ALL").df()
p5 = con.sql(f"SELECT ksic k, pyear, count(*) n FROM '{R}' WHERE pyear<={YMAX} GROUP BY ALL").df()
pt = con.sql(f"""SELECT a.k_txt k, r.pyear, count(*) n FROM '{R}' r JOIN '{A}' a ON r.cc='KR' AND a.appl_no=r.appl_no
   WHERE r.cc='KR' AND r.pyear<={YMAX} AND a.k_txt IS NOT NULL GROUP BY ALL""").df()
for x in (pk, p5, pt):
    assert x.pyear.max() <= YMAX
PAT = {nm: x.pivot_table(index="k", columns="pyear", values="n", aggfunc="sum", fill_value=0).reindex(KSIC50).fillna(0)
       for nm, x in (("PROV_KR", pk), ("PROV_5C", p5), ("TXT", pt))}


def wsum(w, y0, y1):
    cs = [c for c in w.columns if y0 <= c <= y1]
    return w[cs].sum(1)


def sib_share(v):
    """v: KSIC50 인덱스 Series → 형제(같은 연계표 그룹, 제공 50개 안) 합계 중 몫."""
    g = v.index.map(K2G)
    return v / v.groupby(g).transform("sum").replace(0, np.nan)


def kosis_base(t, cols):
    for b in (t - 3, t - 2, t - 4, t - 1):
        if b in cols:
            return b
    return None


rows = []
for t in ORIG:
    rec = pd.DataFrame(index=MULTI_K); rec["t"] = t; rec["g"] = rec.index.map(K2G)
    for nm, w in PAT.items():
        sP = sib_share(wsum(w, t - 2, t)); sPP = sib_share(wsum(w, t - 5, t - 3))
        rec[f"sig_{nm}"] = (sP - sPP).reindex(MULTI_K).values
        rec[f"sh_{nm}"] = sP.reindex(MULTI_K).values
    rec["sig_CW"] = 0.0  # 연계표는 형제를 구분하지 못한다(같은 값)
    for o, S in KOS.items():
        cols = set(S.columns)
        b = kosis_base(t, cols)
        sh_t, sh_b = sib_share(S[t]), sib_share(S[b])
        rec[f"mom_{o}"] = np.log(sh_t / sh_b).reindex(MULTI_K).values
        if t + 3 <= YMAX:
            sh_f = sib_share(S[t + 3])
            rel = np.log(sh_f / sh_t).reindex(MULTI_K)
            rec[f"rel_{o}"] = rel.values
            rec[f"y_{o}"] = np.where(rel.isna(), np.nan, (rel > 0).astype(float))
    rows.append(rec.reset_index().rename(columns={"index": "k"}))
D = pd.concat(rows, ignore_index=True)
D.to_csv(p("p2_panel.csv"), index=False)
SC = ["sig_PROV_KR", "sig_PROV_5C", "sig_TXT", "sig_CW", "mom_ship", "mom_emp"]
res = {}
per = []
for o in ("ship", "emp"):
    y = f"y_{o}"
    for t in ORIG:
        x = D[(D.t == t) & D[y].notna()]
        r = {"o": o, "t": t, "n": len(x), "pos": int(x[y].sum())}
        for s in SC:
            ok = x[s].notna()
            r[s] = auc(x[y][ok], x[s][ok])
            r[s + "_rho"] = spearmanr(x[s][ok], x[f"rel_{o}"][ok]).statistic if x[s][ok].nunique() > 1 else np.nan
        # 결합: 로지스틱(모멘텀 + 특허 신호), 결과가 완전 관측된 과거 원점만 학습
        tr = D[(D.t + 3 <= t) & D[y].notna()]
        for s in ("sig_PROV_KR", "sig_PROV_5C", "sig_TXT"):
            if len(tr) >= 20 and t >= 2012:
                X = [f"mom_{o}", s]
                trx = tr.dropna(subset=X); tex = x.dropna(subset=X)
                m = LogisticRegression(max_iter=1000).fit(trx[X], trx[y])
                r["MOM+" + s] = auc(tex[y], m.predict_proba(tex[X])[:, 1])
            else:
                r["MOM+" + s] = np.nan
        per.append(r)
PER = pd.DataFrame(per)
PER.to_csv(p("p2_by_origin.csv"), index=False)
for o in ("ship", "emp"):
    q = PER[PER.o == o]
    res[o] = {"auc_mean": {s: float(q[s].mean()) for s in SC + ["MOM+sig_PROV_KR", "MOM+sig_PROV_5C", "MOM+sig_TXT"]},
              "rho_mean": {s: float(q[s + "_rho"].mean()) for s in SC if s != "sig_CW"}}
    cmp = {}
    for a, b in (("sig_PROV_KR", "sig_TXT"), ("sig_PROV_5C", "sig_TXT"), ("sig_PROV_5C", "sig_PROV_KR"),
                 ("sig_PROV_KR", "sig_CW"), ("sig_TXT", "sig_CW"), ("MOM+sig_PROV_KR", "MOM+sig_TXT"), ("MOM+sig_PROV_5C", "MOM+sig_TXT"),
                 ("MOM+sig_PROV_KR", "mom_" + o)):
        dd = (q[a] - q[b]).values
        cmp[f"{a}-{b}"] = {"mean": float(np.nanmean(dd)), "pos": f"{int((dd > 0).sum())}/{int((~np.isnan(dd)).sum())}", "ci": boot_ci(dd)}
    res[o]["compare"] = cmp
    # 원점 통합 AUC(점수 척도가 원점 간 비교 가능: 몫 변화)
    x = D[D[f"y_{o}"].notna()]
    res[o]["pooled_auc"] = {s: float(auc(x[f"y_{o}"][x[s].notna()], x[s].dropna())) for s in SC}
    res[o]["n_units"] = int(len(x)); res[o]["base"] = float(x[f"y_{o}"].mean())
# 기술통계: 34개 KSIC 3년 KOSIS 성장의 그룹 내 분산 비중(연계표 그룹이 잃는 변동)
vs = {}
for o, S in KOS.items():
    g3 = []
    for t in ORIG:
        gr = np.log(S[t + 3] / S[t]).reindex(MULTI_K)
        g3.append(pd.DataFrame({"t": t, "k": MULTI_K, "gr": gr.values}))
    G3 = pd.concat(g3).dropna()
    G3["g"] = G3.k.map(K2G)
    tot = G3.groupby("t").gr.transform(lambda z: z - z.mean())
    within = G3.gr - G3.groupby(["t", "g"]).gr.transform("mean")
    vs[o] = {"within_group_var_share": float((within ** 2).sum() / (tot ** 2).sum()), "n": int(len(G3))}
res["within_group_variance_share_34"] = vs
# TXT 배정이 제공 신호를 얼마나 재현하나(같은 단위에서 신호 상관)
res["sig_corr_PROVKR_TXT"] = float(spearmanr(D.sig_PROV_KR, D.sig_TXT, nan_policy="omit").statistic)
json.dump(res, open(p("p2_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
log("end")
