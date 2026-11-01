# -*- coding: utf-8 -*-
"""N9 사후 점검(채택 판단에 쓰지 않음). 사용: python n9_robust.py byorigin | sub | variants | placebo | complete
산출: champion_by_origin.csv, robust_checks.csv(행 추가), completeness.csv"""
import sys, json
import numpy as np, pandas as pd
import n9_exp as E
from n9_lib import *

D = E.D
ST = json.load(open(E.STATE, encoding="utf-8"))
CH = ST["tag"]; FEATS = ST["feats"]
RC = f"{OUT}/robust_checks.csv"


def yv(v):
    return D.y.values[D.t0.values == v]


def add_row(row):
    df = pd.read_csv(RC, encoding="utf-8-sig") if os.path.exists(RC) else pd.DataFrame()
    if len(df):
        df = df[df.check != row["check"]]
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(RC, index=False, encoding="utf-8-sig")


def compare(name, pa, pb, sub=None, note=""):
    sks, wins, aa, ab = [], [], [], []
    for v in DEV_ORIGINS:
        te = D[D.t0 == v]; m = np.ones(len(te), bool) if sub is None else sub(te).values
        y = te.y.values[m]
        if y.sum() < 10 or y.sum() == len(y):
            continue
        a1 = auc(y, pa[v][m]); a0 = auc(y, pb[v][m]); aa.append(a1); ab.append(a0); sks.append(skill(a1, a0)); wins.append(a1 > a0)
    row = dict(check=name, n_origins=len(sks), auc_new=round(np.mean(aa), 4), auc_ref=round(np.mean(ab), 4), mean_skill=round(np.mean(sks), 4),
               win=round(np.mean(wins), 3), by_origin=" ".join(f"{s:.3f}" for s in sks), note=note)
    add_row(row); log(row)
    return row


def byorigin():
    pc = E.load_preds(CH); refs = {"M0L2": "R04", "M0orig": "R01", "strongL2": "R06", "strongOrig": "R07", "pilotChampL2": "R05", "pilotChampOrig": "R03"}
    P = {k: E.load_preds(e) for k, e in refs.items()}
    rows = []
    for v in DEV_ORIGINS:
        te = D[D.t0 == v]; y = te.y.values; g = te.org.values
        r = dict(v=v, **metrics(y, pc[v], P["M0L2"][v], prefix_ref="M0L2"))
        ci = boot_ci(y, pc[v], P["M0L2"][v], groups=g, n=200, seed=v)
        r.update(skill_ci_lo=ci["skill_ci"][0], skill_ci_hi=ci["skill_ci"][1], auc_ci_lo=ci["auc_ci"][0], auc_ci_hi=ci["auc_ci"][1])
        for k in ["M0orig", "strongL2", "strongOrig", "pilotChampL2", "pilotChampOrig"]:
            a0 = auc(y, P[k][v]); r[f"auc_{k}"] = a0; r[f"skill_vs_{k}"] = skill(r["auc"], a0)
        ci2 = boot_ci(y, pc[v], P["strongL2"][v], groups=g, n=200, seed=100 + v)
        r.update(skill_vs_strongL2_ci_lo=ci2["skill_ci"][0], skill_vs_strongL2_ci_hi=ci2["skill_ci"][1])
        r["top1_M0orig"] = topk(y, P["M0orig"][v], 0.01)[0]; r["top1_pilotChampL2"] = topk(y, P["pilotChampL2"][v], 0.01)[0]
        rows.append(r); log(v, round(r["auc"], 4), round(r["skill"], 4), [round(x, 3) for x in ci["skill_ci"]])
    df = pd.DataFrame(rows); df.insert(0, "champ", CH)
    df.to_csv(f"{OUT}/champion_by_origin.csv", index=False, encoding="utf-8-sig")
    print(df[["v", "auc", "auc_M0L2", "skill", "skill_ci_lo", "skill_ci_hi", "skill_vs_M0orig", "skill_vs_strongL2", "top1", "top1_M0L2", "top5", "hit100"]].round(4).to_string())


def sub():
    pc = E.load_preds(CH); p0 = E.load_preds("R04")
    pg = lambda te: te.perform_grp.astype(str)
    compare("챔피언 vs M0_L2 — 기업 수행 과제만", pc, p0, sub=lambda te: te.firm == 1)
    compare("챔피언 vs M0_L2 — 대학 과제만", pc, p0, sub=lambda te: pg(te) == "대학")
    compare("챔피언 vs M0_L2 — 출연연·국공립연", pc, p0, sub=lambda te: pg(te).isin(["출연연", "국공립연"]))
    compare("챔피언 vs M0_L2 — 중소기업", pc, p0, sub=lambda te: pg(te) == "중소기업")
    compare("챔피언 vs M0_L2 — 대기업·중견기업", pc, p0, sub=lambda te: pg(te).isin(["대기업", "중견기업"]))
    compare("챔피언 vs M0_L2 — 과거 3년 과제 없는 신규 수행기관", pc, p0, sub=lambda te: te.org_nproj == 0)
    compare("챔피언 vs M0_L2 — 선행과제 없음", pc, p0, sub=lambda te: te.has_pre == 0)
    compare("챔피언 vs M0_L2 — 선행과제 있음", pc, p0, sub=lambda te: te.has_pre == 1)
    c7 = E.load_preds("C07")
    compare("C10(텍스트 분류기) vs C07 — 선행과제 없음", pc, c7, sub=lambda te: te.has_pre == 0)
    compare("C10(텍스트 분류기) vs C07 — 선행과제 있음", pc, c7, sub=lambda te: te.has_pre == 1)
    compare("C10(텍스트 분류기) vs C07 — 기관 과거 50산업 성과 0(지연판)", pc, c7, sub=lambda te: te.org_p50_L2 == 0)


def variants():
    pc = E.load_preds(CH); p0 = E.load_preds("R04"); p0o = E.load_preds("R01")
    # 원판(이력 출원 <= t-1) 챔피언 vs 원판 M0
    orig = [f[:-3] if f.endswith("_L2") and f[:-3] in LAGGED else f for f in FEATS]
    po = E.score(orig, ST["cfg"]); E.save_preds("V_orig_champ", po)
    compare("원판 챔피언(이력 출원 <= t-1) vs 원판 M0", po, p0o, note="사슬·KSIC 과거율·기관-KSIC는 지연판 그대로")
    compare("원판 챔피언 vs 지연판 챔피언", po, pc)
    # 계획 기간 변수 제거(스냅샷 수정 가능성)
    noplan = [f for f in FEATS if f not in ("total_len", "end_in_year", "series_yr")]
    pn = E.score(noplan, ST["cfg"]); E.save_preds("V_noplan", pn)
    pn0 = E.score([f for f in M0_L2 if f not in ("total_len", "end_in_year", "series_yr")], {}); E.save_preds("V_noplan_M0", pn0)
    compare("계획기간 변수 3개 제거: 챔피언 vs M0_L2(둘 다 제거)", pn, pn0)
    compare("계획기간 변수 3개 제거 챔피언 vs 챔피언", pn, pc)
    # 로지스틱 보조 모형족
    pl = E.score(FEATS, {"model": "logit"}); E.save_preds("V_logit_champ", pl)
    compare("로지스틱 챔피언 vs 로지스틱 M0_L2", pl, E.load_preds("R08"))
    compare("로지스틱 챔피언 vs XGB M0_L2", pl, p0)
    # 텍스트 분류기 단독
    tc = {}
    for v in DEV_ORIGINS:
        q = pd.read_parquet(f"{CACHE}/txtclf_dev_{v}.parquet"); s = pd.Series(q.txt_clf.values, index=q.row_id.values)
        tc[v] = s.reindex(D.index[D.t0 == v]).values
    compare("텍스트 분류기 점수 단독 vs M0_L2", tc, p0)
    compare("챔피언 vs 강한 기준선 지연판(M0_L2+M2_L2)", pc, E.load_preds("R06"))
    compare("챔피언 vs 원판 M0(정보 1년 더 씀)", pc, p0o)
    compare("챔피언 vs 파일럿 챔피언 원판(R03)", pc, E.load_preds("R03"))
    # 음성 표본추출 시드 1
    rng = np.random.default_rng(1); keep_old = E.KEEP.copy()
    E.KEEP = (D.y.values == 1) | (rng.random(len(D)) < NEG_FRAC)
    ps1 = E.score(FEATS, ST["cfg"]); p01 = E.score(M0_L2, {})
    E.KEEP = keep_old
    compare("음성 표본 시드 1: 챔피언 vs M0_L2", ps1, p01)


def placebo():
    """학습 레이블을 (과제연도×수행주체) 층 안에서 섞고 텍스트 분류기까지 다시 적합 → 진짜 시험 레이블로 AUC. 0.5 근처(층 효과만)여야 정상."""
    import n9_textclf as T
    rng = np.random.default_rng(7)
    Dp = D[["pjt_no", "t0", "y", "perform_grp"]].copy()
    Dp["y"] = Dp.groupby(["t0", "perform_grp"], observed=True).y.transform(lambda s: rng.permutation(s.values))
    T.run("placebo", DEV_ORIGINS, D=Dp)
    ytrue = D.y.values.copy()
    E.D["y"] = Dp.y.values; E.KEEP = (Dp.y.values == 1) | (np.random.default_rng(0).random(len(D)) < NEG_FRAC); E.W = np.where(Dp.y.values == 1, 1.0, 1 / NEG_FRAC)
    import shutil
    for v in DEV_ORIGINS:   # 섞인 레이블 텍스트 점수를 dev 이름으로 잠시 쓰지 않도록 score를 직접 구성
        pass
    preds, preds0 = {}, {}
    for v in DEV_ORIGINS:
        tc = pd.read_parquet(f"{CACHE}/txtclf_placebo_{v}.parquet")
        X = E.D.copy(); X["txt_clf"] = np.nan; X.loc[tc.row_id.values, "txt_clf"] = tc.txt_clf.values
        msk = (X.t0 <= v - GAP).values & E.KEEP; tr = X[msk]; te = X[X.t0 == v]
        preds[v] = fit_pred(tr[FEATS].astype(float), tr.y, te[FEATS].astype(float), w=E.W[msk])
        preds0[v] = fit_pred(tr[M0_L2].astype(float), tr.y, te[M0_L2].astype(float), w=E.W[msk])
    E.D["y"] = ytrue
    rows = []
    for v in DEV_ORIGINS:
        y = ytrue[D.t0.values == v]; rows.append((auc(y, preds[v]), auc(y, preds0[v])))
    a = np.array(rows)
    add_row(dict(check="위약: 학습 레이블을 연도×수행주체 층 안에서 섞음(텍스트 분류기 재적합) — 진짜 시험 레이블 AUC", n_origins=7,
                 auc_new=round(a[:, 0].mean(), 4), auc_ref=round(a[:, 1].mean(), 4), mean_skill=round(np.mean([skill(x, z) for x, z in a]), 4),
                 by_origin=" ".join(f"{x:.3f}" for x in a[:, 0]), note="auc_new=챔피언 피처, auc_ref=M0_L2 피처, 둘 다 섞인 레이블로 학습"))
    log("placebo", a.mean(0))


def complete():
    """보고 완결도: 출원연도별 제공 KR 특허 중 NTIS 성과 연결 비율(출원 <= 2021만 읽음)."""
    import duckdb
    kr = duckdb.sql(f"select appl_year, count(distinct appl_no) n from read_parquet('{META}') where cc='KR' and appl_year between 2007 and {DEV_MAX_APPL_YEAR} group by 1 order by 1").df()
    L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "appl_year", "perform_grp", "ProjectYear"])
    L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce"); L = L[L.appl_year.between(2007, DEV_MAX_APPL_YEAR)]
    assert L.appl_year.max() <= DEV_MAX_APPL_YEAR
    lk = L.groupby("appl_year").appl_no.nunique().rename("n_linked")
    g = L.drop_duplicates(["appl_no", "perform_grp"]).groupby(["appl_year", "perform_grp"]).appl_no.nunique().unstack(fill_value=0)
    out = kr.set_index("appl_year").join(lk).join(g)
    out["link_rate"] = out.n_linked / out.n
    ref = out.loc[2014:2016, "link_rate"].mean()
    out["rel_to_2014_16"] = out.link_rate / ref
    for c in g.columns:
        out[f"{c}_rel"] = out[c] / out.loc[2014:2016, c].mean()
    # 과제연도 대비 출원연도 시차(연결 성과 중, 과제연도 2010–2017: 관측 기간 충분)
    L["ProjectYear"] = pd.to_numeric(L.ProjectYear, errors="coerce")
    lag = (L[L.ProjectYear.between(2010, 2017)].appl_year - L[L.ProjectYear.between(2010, 2017)].ProjectYear).value_counts(normalize=True).sort_index()
    out.to_csv(f"{OUT}/completeness.csv", encoding="utf-8-sig")
    lag.rename("share").to_csv(f"{OUT}/completeness_lag.csv", encoding="utf-8-sig")
    print(out.round(4).to_string()); print(lag.round(4).to_string())


if __name__ == "__main__":
    {"byorigin": byorigin, "sub": sub, "variants": variants, "placebo": placebo, "complete": complete}[sys.argv[1]]()
