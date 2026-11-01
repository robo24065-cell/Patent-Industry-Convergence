# -*- coding: utf-8 -*-
"""N6 단계 A: 패널 2종 + P2 중심점 4판.
- panel prov : 동결 위험집합(제공범위 출원 없음). 결과 y(동결, 순환 참고), y_ntis(라벨 무관 = NTIS KR 특허 출원 성과, 분야 무관)
- panel ntis : 라벨 무관 위험집합(성과연도 <= t0 NTIS KR 특허 성과 없음). 결과 y_ntis = 첫 NTIS KR 특허 출원 성과(분야 무관)
- P2(pt_mean, pt_max, pt_margin, pt_all) 판: rep(동결 방식 재계산, 대조용) / alt(연계표, R7 방식) / txt(R8 해설서 텍스트 5자리, 제목 풀 전체)
산출: cache/n6_panel_{prov,ntis}.parquet, n6_build_check.json"""
import sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from r9_common import log as rlog, CACHE as RC, OUT as ROUT, N6E, N9E, R8, BASE as RBASE
sys.path.insert(1, N9E)
import n6_features_r9 as F
from n6_lib import load_ntis, EXTF, auc
from r7_concord import assign, G50
import numpy as np, pandas as pd, duckdb, json

P2 = ["pt_mean", "pt_max", "pt_margin", "pt_all"]
SPEC = json.load(open(os.path.join(os.path.dirname(N6E), "final_spec_r5.json"), encoding="utf-8")); FEATS = SPEC["feats"]


def centroid_sims(tag):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    import scipy.sparse as sp
    k = pd.read_parquet(os.path.join(RBASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "공개일자", "등록일자", "발명명칭", "세세분류코드", "메인IPC"])
    k.columns = ["appl_no", "ay", "pub", "rg", "title", "ksic", "ipc"]
    for c in ["pub", "rg"]:
        k[c] = pd.to_numeric(k[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    k["known"] = k[["pub", "rg"]].min(axis=1); k["ay"] = pd.to_numeric(k.ay, errors="coerce")
    k = k[(k.ay <= 2010) & (k.known <= 20111231)].dropna(subset=["title"])
    if tag == "rep":
        k["lab"] = k.ksic.astype(str)
    elif tag == "alt":
        g, _ = assign(k.ipc.values, "free"); k["lab"] = g; k = k[np.isin(g, G50)]
    else:
        A = pd.read_parquet(os.path.join(R8, "cache", "kr_assign.parquet"), columns=["appl_no", "k_txt"]).dropna(subset=["k_txt"])
        k["lab"] = k.appl_no.astype(str).map(A.set_index("appl_no").k_txt); k = k.dropna(subset=["lab"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=150000, min_df=3, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(k.title.values)
    ks = sorted(k.lab.unique()); ki = k.lab.map({c: i for i, c in enumerate(ks)}).values
    M = sp.csr_matrix((np.ones(len(ki), dtype=np.float32), (ki, np.arange(len(ki)))), shape=(len(ks), len(ki)))
    C = normalize(np.asarray((M @ X).todense())); call = normalize(np.asarray(X.sum(axis=0)))
    nt = duckdb.sql(f"""select pjt_no, year, coalesce(title_ko,'') || ' ' || coalesce(kw_ko,'') txt from read_parquet('{EXTF}/ntis_project_raw.parquet')
      where year between 2008 and 2019 and perform_grp in ('중소기업','중견기업','대기업')""").df()
    P = vec.transform(nt.txt.values); S = np.asarray(P @ C.T); srt = np.sort(S, axis=1)
    nt["pt_sim_max"] = srt[:, -1]; nt["pt_margin"] = srt[:, -1] - srt[:, -2]; nt["pt_sim_all"] = np.asarray(P @ call.T).ravel()
    rlog("centroids", tag, len(ks), len(k))
    return nt.drop(columns="txt"), dict(n_titles=int(len(k)), n_groups=len(ks))


def agg_pt(D, pts, NT):
    NT = NT[["pjt_no", "year", "bz"]].merge(pts, on=["pjt_no", "year"], how="left")
    out = []
    for t0 in sorted(D.t0.unique()):
        g = NT[NT.year.between(t0 - 2, t0)].groupby("bz")
        f = pd.DataFrame({"pt_mean": g.pt_sim_max.mean(), "pt_max": g.pt_sim_max.max(), "pt_margin": g.pt_margin.mean(), "pt_all": g.pt_sim_all.mean()})
        f["t0"] = t0; out.append(f.reset_index().rename(columns={"index": "bz"}))
    A = pd.concat(out, ignore_index=True)
    return D[["bz", "t0"]].merge(A, on=["bz", "t0"], how="left")


chk = {}
panels = {}
for risk in ("prov", "ntis"):
    D = F.build("dev", risk=risk).reset_index(drop=True)
    panels[risk] = D
    rlog("panel", risk, D.shape)
# 재현: prov 패널 = R7 패널(동결 빌더)
R7 = pd.read_parquet(os.path.join(N6E, "panel_dev_r7.parquet"), columns=["bz", "t0", "y"] + FEATS)
Dp = panels["prov"]
same_rows = len(R7) == len(Dp) and (R7.bz.values == Dp.bz.values).all() and (R7.t0.values == Dp.t0.values).all()
chk["prov_panel_rows_equal_R7"] = bool(same_rows)
if same_rows:
    chk["prov_y_equal"] = bool((R7.y.values == Dp.y.values).all())
    chk["prov_feats_maxdiff"] = float(max(np.nanmax(np.abs(R7[c].astype(float).values - Dp[c].astype(float).values)) if R7[c].notna().any() else 0 for c in FEATS))
NT = load_ntis(2008, 2019)
for tag in ("rep", "alt", "txt"):
    pts, info = centroid_sims(tag); chk[f"centroid_{tag}"] = info
    for risk, D in panels.items():
        A = agg_pt(D, pts, NT)
        for c in P2:
            D[f"{c}__{tag}"] = A[c].values
for risk, D in panels.items():
    rep = {}
    for c in P2:
        a = D[c].astype(float).values; b = D[f"{c}__rep"].astype(float).values
        rep[c] = dict(max_abs_diff=float(np.nanmax(np.abs(a - b))), corr_alt=float(pd.Series(a).corr(pd.Series(D[f"{c}__alt"].astype(float).values))),
                      corr_txt=float(pd.Series(a).corr(pd.Series(D[f"{c}__txt"].astype(float).values))))
    chk[f"replication_{risk}"] = rep
    ev = D[D.t0.between(2014, 2019)]
    st = dict(n=int(len(ev)), base_y_ntis=float(ev.y_ntis.mean()), pos_y_ntis=int(ev.y_ntis.sum()), base_y_frozen=float(ev.y.mean()),
              by_origin={int(t): [int(len(g)), float(g.y_ntis.mean())] for t, g in ev.groupby("t0")})
    if risk == "prov":
        tp = int(((ev.y == 1) & (ev.y_ntis == 1)).sum())
        st["jaccard_y_vs_y_ntis"] = float(tp / ((ev.y == 1) | (ev.y_ntis == 1)).sum()); st["share_frozen_pos_in_y_ntis"] = float(tp / ev.y.sum())
    else:
        st["overlap_with_prov_risk_rows"] = float(ev.merge(panels["prov"][["bz", "t0"]], on=["bz", "t0"]).shape[0] / len(ev))
    chk[f"stats_{risk}"] = st
    D.to_parquet(os.path.join(RC, f"n6_panel_{risk}.parquet"), index=False)
json.dump(chk, open(os.path.join(ROUT, "n6_build_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
rlog(json.dumps(chk, ensure_ascii=False)[:3000])
rlog("end")
