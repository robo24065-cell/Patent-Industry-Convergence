# -*- coding: utf-8 -*-
"""N9 최종시험 전후 감사·사후 점검 (보고 모형은 바꾸지 않는다). 출처: NTIS Open API 활용.
사용: python n9_final_audit.py funds_dev      — (final 전) 연구비 4열 제거 민감도, dev 원점만
      python n9_final_audit.py verify         — (final 후) 레이블·지표 독립 재계산 대조
      python n9_final_audit.py placebo [n]    — (final 후) 시험 레이블 층 내 섞기 위약 n회(기본 1000)
      python n9_final_audit.py subci          — (final 후) 사전 등록 하위집단 skill + 수행기관 군집 CI, 수행주체별
      python n9_final_audit.py funds_final    — (final 후) 연구비 4열 제거 민감도(보고 모형 아님)
      python n9_final_audit.py complete       — (final 후) 출원연도 2022·2023 NTIS 연결률(보고 지연 점검)
산출: final_audit_r6.json (키별 갱신)"""
import sys, json
import numpy as np, pandas as pd
from n9_lib import *
import final_run_r6 as FR

AJ = os.path.join(OUT, "final_audit_r6.json")
FUNDS = ["log_gov", "log_tot", "nongov_sh", "sbiz_sh"]


def save(key, val):
    d = json.load(open(AJ, encoding="utf-8")) if os.path.exists(AJ) else {}
    d[key] = val
    json.dump(d, open(AJ, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("saved", key)


def funds_dev():
    D = pd.read_parquet(f"{CACHE}/panel_dev.parquet").reset_index(drop=True)
    fc = [f for f in FR.FEATS if f not in FUNDS]; f0 = [f for f in FR.BASE_SETS["M0_L2"] if f not in FUNDS]
    rows = []
    for v in DEV_ORIGINS:
        X = FR.with_txt(D, "dev", v); y = X[X.t0 == v].y.values
        a_c = auc(y, FR.predict(X, v, fc, 2010, v - GAP)); a_0 = auc(y, FR.predict(X, v, f0, 2010, v - GAP))
        a_0full = auc(y, FR.predict(X, v, FR.BASE_SETS["M0_L2"], 2010, v - GAP))
        rows.append(dict(v=v, auc_champ_nofunds=a_c, auc_M0L2_nofunds=a_0, auc_M0L2=a_0full, skill_nofunds=skill(a_c, a_0), skill_nofunds_vs_M0L2full=skill(a_c, a_0full)))
        log(rows[-1])
    df = pd.DataFrame(rows)
    save("funds_dev", dict(rows=rows, mean=df.mean().to_dict(), win=float((df.skill_nofunds > 0).mean())))


def load_final():
    P = pd.read_parquet(os.path.join(OUT, "final_preds_r6.parquet"))
    D = pd.read_parquet(f"{CACHE}/panel_final.parquet")
    te = D[D.t0 == FINAL_T].reset_index(drop=True)
    assert (te.pjt_no.values == P.pjt_no.values).all()
    return P, te, D


def verify():
    """독립 재계산: 원시 parquet에서 t=2022 레이블을 n9_build 없이 다시 만들고, sklearn으로 지표를 다시 계산."""
    from sklearn.metrics import roc_auc_score
    P, te, _ = load_final()
    raw = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "year", "secret_flag"])
    raw = raw[(raw.year.astype(int) == 2022) & (raw.secret_flag.astype(float).fillna(0) == 0)].drop_duplicates("pjt_no")
    L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["ProjectID", "appl_no", "appl_year"])
    L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce")
    pos = set(L[L.appl_year.isin([2022, 2023])].ProjectID.astype(str))
    y2 = raw.pjt_no.astype(str).isin(pos).astype(int)
    m = dict(zip(raw.pjt_no.astype(str), y2))
    yy = P.pjt_no.astype(str).map(m)
    r = dict(n_raw_2022=int(len(raw)), n_preds=int(len(P)), pos_raw=int(y2.sum()), pos_preds=int(P.y.sum()),
             label_match=bool((yy.values == P.y.values).all()), set_match=bool(set(raw.pjt_no.astype(str)) == set(P.pjt_no.astype(str))))
    y = P.y.values
    a = roc_auc_score(y, P.p_champ); a0 = roc_auc_score(y, P.p_M0_L2); a_s = roc_auc_score(y, P.p_strong_L2)
    o = np.argsort(-P.p_champ.values); k1 = int(round(0.01 * len(y)))
    r.update(auc=a, auc_M0L2=a0, skill=(a - a0) / (1 - a0), skill_vs_strong_L2=(a - a_s) / (1 - a_s), top1=float(y[o[:k1]].mean()), hit100=int(y[o[:100]].sum()))
    fj = json.load(open(os.path.join(OUT, "final_result_r6.json"), encoding="utf-8"))
    r["metrics_match_final_json"] = bool(abs(a - fj["auc"]) < 1e-9 and abs(a0 - fj["auc_M0L2"]) < 1e-9 and abs(r["top1"] - fj["top1"]) < 1e-9 and r["hit100"] == fj["hit100"])
    # 레이블에 쓰인 성과 출원연도 분포(보고용)
    Lp = L[L.ProjectID.astype(str).isin(set(raw.pjt_no.astype(str))) & L.appl_year.isin([2022, 2023])].drop_duplicates(["ProjectID", "appl_no"])
    r["linked_patents_by_appl_year"] = {str(int(k)): int(v) for k, v in Lp.groupby("appl_year").size().items()}
    log(r); save("verify", r)


def strata(te, kind):
    if kind == "perform_grp":
        return te.perform_grp.astype(str).values
    if kind == "perform_grp_x_ministry":
        return (te.perform_grp.astype(str) + "|" + te.ministry_c.astype(str)).values
    raise ValueError


def placebo(n=1000):
    P, te, _ = load_final()
    y = P.y.values; pc = P.p_champ.values; p0 = P.p_M0_L2.values; ps = P.p_strong_L2.values
    rng = np.random.default_rng(20261005)
    out = {}
    for kind in ("perform_grp", "perform_grp_x_ministry"):
        s = strata(te, kind); idx = [np.where(s == u)[0] for u in np.unique(s)]
        sk, sks, a1s = [], [], []
        for _ in range(n):
            yp = y.copy()
            for ix in idx:
                yp[ix] = y[rng.permutation(ix)]
            a1 = auc(yp, pc); a0 = auc(yp, p0); a2 = auc(yp, ps)
            sk.append(skill(a1, a0)); sks.append(skill(a1, a2)); a1s.append(a1)
        sk = np.array(sk); real = json.load(open(os.path.join(OUT, "final_result_r6.json"), encoding="utf-8"))
        out[kind] = dict(n=n, n_strata=len(idx), skill_mean=float(sk.mean()), skill_sd=float(sk.std()), skill_p95=float(np.percentile(sk, 95)), skill_p99=float(np.percentile(sk, 99)),
                         skill_max=float(sk.max()), auc_champ_mean=float(np.mean(a1s)), skill_vs_strong_mean=float(np.mean(sks)),
                         real_skill=real["skill"], p_value=float((1 + (sk >= real["skill"]).sum()) / (n + 1)))
        log(kind, out[kind])
    save("placebo_test_label_shuffle", out)


def subci():
    P, te, _ = load_final()
    y = P.y.values; g = P.org.values
    groups = {"전체": np.ones(len(te), bool), "기업": te.firm.values == 1, "중소기업": te.perform_grp.astype(str).values == "중소기업",
              "대기업·중견": te.perform_grp.astype(str).isin(["대기업", "중견기업"]).values, "대학": te.perform_grp.astype(str).values == "대학",
              "출연연·국공립연": te.perform_grp.astype(str).isin(["출연연", "국공립연"]).values, "신규수행기관": te.org_nproj.values == 0,
              "선행과제없음": te.has_pre.values == 0, "선행과제있음": te.has_pre.values == 1}
    out = {}
    for name, m in groups.items():
        if y[m].sum() < 10:
            out[name] = dict(n=int(m.sum()), pos=int(y[m].sum()), note="양성 <10"); continue
        a1 = auc(y[m], P.p_champ.values[m]); a0 = auc(y[m], P.p_M0_L2.values[m]); a2 = auc(y[m], P.p_strong_L2.values[m])
        ci = boot_ci(y[m], P.p_champ.values[m], P.p_M0_L2.values[m], groups=g[m], n=1000, seed=7)
        out[name] = dict(n=int(m.sum()), pos=int(y[m].sum()), base_rate=float(y[m].mean()), auc=a1, auc_M0L2=a0, skill=skill(a1, a0), skill_ci95=ci["skill_ci"],
                         skill_vs_strong_L2=skill(a1, a2), top1=topk(y[m], P.p_champ.values[m], 0.01)[0], top1_M0L2=topk(y[m], P.p_M0_L2.values[m], 0.01)[0])
        log(name, round(a1, 4), round(a0, 4), round(out[name]["skill"], 4), [round(x, 3) for x in ci["skill_ci"]])
    # 수행주체별 기저율(보고 지연 편향 점검용) — 개발 원점 대비
    D = pd.read_parquet(f"{CACHE}/panel_final.parquet", columns=["t0", "y", "perform_grp"])
    br = D[D.t0.between(2014, 2020)].groupby("perform_grp").y.mean(); br22 = D[D.t0 == FINAL_T].groupby("perform_grp").y.mean()
    out["_base_rate_by_perform_grp"] = {str(k): dict(dev_2014_2020=float(br.get(k, np.nan)), final_2022=float(br22.get(k, np.nan))) for k in br22.index}
    save("subgroups_ci", out)


def funds_final():
    _, _, D = load_final()
    D = D.reset_index(drop=True); v = FINAL_T
    X = FR.with_txt(D, "final", v); y = X[X.t0 == v].y.values
    fc = [f for f in FR.FEATS if f not in FUNDS]; f0 = [f for f in FR.BASE_SETS["M0_L2"] if f not in FUNDS]
    a_c = auc(y, FR.predict(X, v, fc, 2010, 2020)); a_0 = auc(y, FR.predict(X, v, f0, 2010, 2020))
    fj = json.load(open(os.path.join(OUT, "final_result_r6.json"), encoding="utf-8"))
    r = dict(auc_champ_nofunds=a_c, auc_M0L2_nofunds=a_0, skill_nofunds=skill(a_c, a_0), skill_nofunds_vs_M0L2full=skill(a_c, fj["auc_M0L2"]))
    log(r); save("funds_final", r)


def complete():
    """출원연도별 NTIS 연결률(제공 KR 출원 중 성과 연결 비율). final 이후라 2022·2023 포함."""
    import duckdb
    kr = duckdb.sql(f"select appl_year, count(distinct appl_no) n from read_parquet('{META}') where cc='KR' and appl_year between 2014 and 2023 group by appl_year").df()
    L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "appl_year"])
    L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce")
    lk = L[L.appl_year.between(2014, 2023)].drop_duplicates("appl_no").groupby("appl_year").size()
    kr = kr.set_index("appl_year").n
    rate = {int(a): float(lk.get(a, 0) / kr[a]) for a in kr.index}
    base = np.mean([rate[a] for a in (2014, 2015, 2016)])
    r = {str(a): dict(rate=rate[a], rel_2014_16=rate[a] / base) for a in sorted(rate)}
    log(r); save("completeness_incl_2022_23", r)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd != "funds_dev":
        assert os.path.exists(os.path.join(OUT, "final_result_r6.json")), "final 이후에만"
    {"funds_dev": funds_dev, "verify": verify, "placebo": lambda: placebo(int(sys.argv[2]) if len(sys.argv) > 2 else 1000),
     "subci": subci, "funds_final": funds_final, "complete": complete}[cmd]()
