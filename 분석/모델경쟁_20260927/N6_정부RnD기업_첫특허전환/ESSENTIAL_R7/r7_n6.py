# -*- coding: utf-8 -*-
"""N6 R7 제공 특허-산업 융합데이터 필수성 시험 (개발 원점 2014–2019, 동결 챔피언 C11 설정 그대로: XGB+로지스틱 순위 평균 0.5/0.5, 시드 0, 학습 t0 2012..v-2).
사용:
  python r7_n6.py build   : 동결 n6_features.build('dev')로 패널을 새로 만들어(감사 수정 후 코드) ESSENTIAL_R7/panel_dev_r7.parquet 저장,
                            연계표 대체 중심점 피처·연계표 결과(y_alt) 생성 → alt_n6_dev.parquet, n6_replication.json
  python r7_n6.py run VAR.. : 변형 실행 → results_dev.csv
제공 데이터 유래 피처(챔피언 안): P2 = pt_mean, pt_max, pt_margin, pt_all (과제 텍스트 ↔ 제공 KR 특허 제목의 KSIC별 중심점; 제공 KSIC 라벨이 다리)
KSIC50 정의문 대응(외부 KSIC 해설, 제공 데이터의 50개 산업 범위만 사용): DEF = k50_assigned, k50_score, b1_score
연계표 대체: 같은 제목 집합(제공 KR 출원 <=2010, 2011까지 공개·등록)을 메인 IPC 연계표 배정(free, 25개 소분류 안만)별 중심점으로.
결과 대체(정의 시험): y_alt = 기업 NTIS 과제 성과 중 KR 특허 출원 레코드(출원연도 t0+1..t0+2 <= 2021)의 첫 IPC 연계표 배정이 25개 소분류 안 >= 1."""
import os, sys, json
import numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); N6 = os.path.dirname(HERE)
N9E = os.path.join(os.path.dirname(N6), "N9_정부RnD과제_특허성과예측", "ESSENTIAL_R7")
sys.path.insert(0, N6); sys.path.insert(0, N9E)
from r7_concord import assign, first_ipc, G50          # N9 ESSENTIAL_R7의 공용 연계표 배정기(읽기 전용 import)
from n6_lib import *

SPEC = json.load(open(os.path.join(N6, "final_spec_r5.json"), encoding="utf-8"))
FEATS = SPEC["feats"]; MSPEC = SPEC["model"]; M0F = SPEC["baseline"]["feats"]
P2 = ["pt_mean", "pt_max", "pt_margin", "pt_all"]; DEF = ["k50_assigned", "k50_score", "b1_score"]
drop = lambda L: [f for f in FEATS if f not in L]
VARS = {"FULL": dict(feats=FEATS), "R_P2": dict(feats=drop(P2)), "R_DEF": dict(feats=drop(DEF)), "R_ALLKSIC": dict(feats=drop(P2 + DEF)),
        "S_P2": dict(feats=FEATS, sub=P2),
        "D_YALT_altfeats": dict(feats=FEATS, sub=P2, ylab="y_alt"), "D_YALT_provfeats": dict(feats=FEATS, ylab="y_alt")}


def centroid_sims(tag):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    import scipy.sparse as sp
    k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "공개일자", "등록일자", "발명명칭", "세세분류코드", "메인IPC"])
    k.columns = ["appl_no", "ay", "pub", "rg", "title", "ksic", "ipc"]
    for c in ["pub", "rg"]:
        k[c] = pd.to_numeric(k[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    k["known"] = k[["pub", "rg"]].min(axis=1); k["ay"] = pd.to_numeric(k.ay, errors="coerce")
    k = k[(k.ay <= 2010) & (k.known <= 20111231)].dropna(subset=["title"])
    if tag == "rep":
        k["lab"] = k.ksic.astype(str)
    else:
        g, _ = assign(k.ipc.values, "free"); k["lab"] = g; k = k[np.isin(g, G50)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=150000, min_df=3, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(k.title.values)
    ks = sorted(k.lab.unique()); ki = k.lab.map({c: i for i, c in enumerate(ks)}).values
    M = sp.csr_matrix((np.ones(len(ki), dtype=np.float32), (ki, np.arange(len(ki)))), shape=(len(ks), len(ki)))
    C = normalize(np.asarray((M @ X).todense())); call = normalize(np.asarray(X.sum(axis=0)))
    nt = duckdb.sql(f"""select pjt_no, year, coalesce(title_ko,'') || ' ' || coalesce(kw_ko,'') txt from read_parquet('{EXTF}/ntis_project_raw.parquet')
      where year between 2008 and 2019 and perform_grp in ('중소기업','중견기업','대기업')""").df()
    P = vec.transform(nt.txt.values); S = np.asarray(P @ C.T); srt = np.sort(S, axis=1)
    nt["pt_sim_max"] = srt[:, -1]; nt["pt_margin"] = srt[:, -1] - srt[:, -2]; nt["pt_sim_all"] = np.asarray(P @ call.T).ravel()
    log("centroids", tag, len(ks), len(k))
    return nt.drop(columns="txt"), dict(n_titles=int(len(k)), n_groups=len(ks))


def agg_pt(D, pts):
    NT = load_ntis(2008, 2019).merge(pts, on=["pjt_no", "year"], how="left")
    out = []
    for t0 in sorted(D.t0.unique()):
        g = NT[NT.year.between(t0 - 2, t0)].groupby("bz")
        f = pd.DataFrame({"pt_mean": g.pt_sim_max.mean(), "pt_max": g.pt_sim_max.max(), "pt_margin": g.pt_margin.mean(), "pt_all": g.pt_sim_all.mean()})
        f["t0"] = t0; out.append(f.reset_index().rename(columns={"index": "bz"}))
    A = pd.concat(out, ignore_index=True)
    return D[["bz", "t0"]].merge(A, on=["bz", "t0"], how="left")


def build_step():
    from n6_features import build
    D = build("dev"); D = D.reset_index(drop=True)
    D.to_parquet(os.path.join(HERE, "panel_dev_r7.parquet"), index=False)
    rep = {}
    res = {}
    for tag in ("rep", "alt"):
        pts, info = centroid_sims(tag); res[tag] = info
        A = agg_pt(D, pts)
        for c in P2:
            D[f"{c}__{tag}"] = A[c].values
    for c in P2:
        a = D[c].astype(float).values; b = D[f"{c}__rep"].astype(float).values
        rep[c] = dict(max_abs_diff=float(np.nanmax(np.abs(a - b))), corr_alt=float(pd.Series(a).corr(pd.Series(D[f"{c}__alt"].astype(float).values))))
    # y_alt
    RP = duckdb.sql(f"""select cast(r.Year as int) yr, r.ipc_codes, regexp_replace(p.corp_rgst_no,'[^0-9]','','g') bz
        from read_parquet('{EXTF}/ntis_rpatent_raw.parquet') r join (select distinct pjt_no, corp_rgst_no from read_parquet('{EXTF}/ntis_project_raw.parquet')
        where perform_grp in ('중소기업','중견기업','대기업')) p on r.ProjectID = p.pjt_no
        where r.is_kr and r.IprType='특허' and r.RegistType='출원' and cast(r.Year as int) <= 2021""").df()
    g, _ = assign(RP.ipc_codes.map(first_ipc).values, "free"); RP = RP[np.isin(g, G50)]
    ya = np.zeros(len(D), int)
    for t0 in sorted(D.t0.unique()):
        s = set(RP[RP.yr.between(t0 + 1, t0 + 2)].bz); m = D.t0.values == t0
        ya[m] = D.bz[m].isin(s).astype(int).values
    D["y_alt"] = ya
    ev = D[D.t0.isin(DEV_ORIGINS)]; tp = int(((ev.y == 1) & (ev.y_alt == 1)).sum())
    res["y_alt_overlap"] = dict(n=int(len(ev)), base_y=float(ev.y.mean()), base_y_alt=float(ev.y_alt.mean()), jaccard=float(tp / max(1, ((ev.y == 1) | (ev.y_alt == 1)).sum())),
                                precision_alt_vs_y=float(tp / max(1, ev.y_alt.sum())), recall_alt_vs_y=float(tp / max(1, ev.y.sum())), auc_yalt_as_score=float(auc(ev.y, ev.y_alt)))
    res["replication"] = rep
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(HERE, "n6_replication.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    D[["bz", "t0", "y_alt"] + [f"{c}__{t}" for c in P2 for t in ("rep", "alt")]].to_parquet(os.path.join(HERE, "alt_n6_dev.parquet"), index=False)


def run_step(names):
    D = pd.read_parquet(os.path.join(HERE, "panel_dev_r7.parquet")); A = pd.read_parquet(os.path.join(HERE, "alt_n6_dev.parquet"))
    assert (A.bz.values == D.bz.values).all() and (A.t0.values == D.t0.values).all()
    for c in A.columns[2:]:
        D[c] = A[c].values
    D = D[D.t0 >= MIN_TRAIN].reset_index(drop=True)
    fp = os.path.join(HERE, "results_dev.csv"); old = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()
    rows = []
    for nm in names:
        sp_ = VARS[nm]; X = D.copy()
        for c in sp_.get("sub", []):
            X[c] = X[f"{c}__alt"].values
        yl = sp_.get("ylab", "y")
        for v in DEV_ORIGINS:
            tr = X[X.t0 <= v - 2]; te = X[X.t0 == v]
            p = predict(tr, te, sp_["feats"], ycol=yl, spec=MSPEC); pb = predict(tr, te, M0F, spec={"model": "xgb", "seeds": [0]})
            y = te.y.values; a, a0 = auc(y, p), auc(y, pb)
            rows.append(dict(variant=nm, origin=v, n_feats=len(sp_["feats"]), auc=a, auc_M0=a0, skill=skill(a, a0), top1=topk(y, p, 0.01), top10=topk(y, p, 0.10)))
            log(nm, v, round(a, 4), round(a0, 4), round(skill(a, a0), 4))
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.variant.isin(names)]
    pd.concat([old, new], ignore_index=True).to_csv(fp, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build_step()
    else:
        run_step(sys.argv[2:])
