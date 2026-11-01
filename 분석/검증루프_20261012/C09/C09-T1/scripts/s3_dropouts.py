# -*- coding: utf-8 -*-
# 3단계: 급감 출원인(코드 기준, PREREG 2.3)과 원인 분류 + 이름 기준 급감(PREREG 2.5)
import sys, os, time, json, re
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
t0 = time.time(); res = {}
ga = pd.read_parquet(os.path.join(CACHE, "s2_applicant_year.parquet"))
top300 = pd.read_csv(os.path.join(CACHE, "s2_top300.csv"), dtype=str).applicant_cd.tolist()
c = ro(BR)
names = dict(c.execute("select applicant_cd, applicant_name from applicant_query where status='OK'").fetchall())
kr = pd.read_parquet(os.path.join(CACHE, "kr_meta.parquet"))
kr["sub"] = kr.main_ipc.str[:4]
U = pd.read_parquet(os.path.join(CACHE, "U_raw.parquet"), columns=["applicant_cd", "rgst_no", "appl_no", "appl_date", "title", "ainfo"])
U["ay"] = pd.to_numeric(U.appl_date.str[:4], errors="coerce")
inset = set(kr.appl_no)
U["inscope"] = U.appl_no.isin(inset)
U = U.merge(kr[["appl_no", "sub"]], on="appl_no", how="left")
# ---- 코드 기준 급감 ----
drops = []
for cd in top300:
    d = ga[ga.applicant_cd == cd].set_index("ay")
    for t in range(2011, 2023):
        prev = [t - 3, t - 2, t - 1]
        if not all(p in d.index for p in prev):
            continue
        pm = d.loc[prev, "share"].mean(); nin_prev = int(d.loc[prev, "n_in"].sum()); den_prev = d.loc[prev, "n_den"].mean()
        pt = float(d.share.get(t, 0.0)); den_t = int(d.n_den.get(t, 0)); nin_t = int(d.n_in.get(t, 0))
        if pt < 0.25 * pm and nin_prev >= 50:
            drops.append(dict(applicant_cd=cd, name=names.get(cd), year=t, share_prev3=round(pm, 4), share_t=round(pt, 4), n_in_prev3=nin_prev,
                              n_in_t=nin_t, n_den_prev3_mean=round(den_prev, 1), n_den_t=den_t, portfolio_collapse=bool(den_t < 0.5 * den_prev)))
            break   # 출원인마다 첫 급감 연도
D = pd.DataFrame(drops)
res["n_dropouts_code"] = len(D)
print("code dropouts", len(D), flush=True)
# 원인 분류용 제목 유사도
vec = None


def title_sims(cd, t):
    u = U[(U.applicant_cd == cd)]
    pre_in = u[u.ay.between(t - 3, t - 1) & u.inscope].title.dropna()
    pre_out = u[u.ay.between(t - 3, t - 1) & ~u.inscope].title.dropna()
    post_out = u[u.ay.between(t, t + 2) & ~u.inscope].title.dropna()
    if len(pre_in) < 10 or len(post_out) < 5:
        return dict(sim_post_out=np.nan, base_in=np.nan, base_out=np.nan, n_pre_in=len(pre_in), n_post_out=len(post_out))
    v = TfidfVectorizer(analyzer="char", ngram_range=(2, 2), min_df=1)
    v.fit(pd.concat([pre_in, pre_out, post_out]))
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(pre_in)); h = len(idx) // 2
    A = v.transform(pre_in.iloc[idx[:h]]); Bm = v.transform(pre_in.iloc[idx[h:]])
    cen = np.asarray(A.mean(axis=0)); cen_all = np.asarray(v.transform(pre_in).mean(axis=0))

    def msim(X, cvec):
        num = X @ cvec.T
        return float(np.mean(np.asarray(num).ravel() / (np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) * np.linalg.norm(cvec) + 1e-12)))
    base_in = msim(Bm, cen)
    base_out = msim(v.transform(pre_out), cen_all) if len(pre_out) >= 5 else np.nan
    sim_post = msim(v.transform(post_out), cen_all)
    return dict(sim_post_out=sim_post, base_in=base_in, base_out=base_out, n_pre_in=len(pre_in), n_post_out=len(post_out))


def other_sub_stability(cd, t):
    u = U[(U.applicant_cd == cd) & U.ay.between(t - 3, t - 1) & U.inscope]
    subs = u["sub"].value_counts().head(3).index.tolist()
    o = U[(U.applicant_cd != cd) & U.inscope & U["sub"].isin(subs)].drop_duplicates("rgst_no")
    pre = o[o.ay.between(t - 3, t - 1)].groupby("ay").size().mean()
    post = o[o.ay == t].shape[0]
    return subs, (post / pre if pre else np.nan)


cls = []
for r in D.itertuples():
    ts = title_sims(r.applicant_cd, r.year)
    subs, stab = other_sub_stability(r.applicant_cd, r.year)
    if r.portfolio_collapse:
        cause = "개명·합병으로 코드 변경(후보: 분모 급감)"
    elif not np.isnan(ts["sim_post_out"]) and ts["sim_post_out"] >= 0.8 * ts["base_in"] and stab >= 0.5:
        cause = "진짜 결손(의심)"
    elif not np.isnan(ts["sim_post_out"]) and not np.isnan(ts["base_out"]) and ts["sim_post_out"] <= ts["base_out"]:
        cause = "KSIC 배정 변경(포트폴리오 이동)"
    else:
        cause = "분류 불가"
    cls.append(dict(**ts, top_subs="|".join(subs), other_same_sub_ratio=stab, cause=cause))
if len(D):
    D = pd.concat([D, pd.DataFrame(cls)], axis=1)
# 분모 급감 후보: 후계 코드 탐색(같은 해 이후 applicantInfo에 옛 이름 어간이 나오는 다른 코드)
succ = []
for r in D.itertuples() if len(D) else []:
    if not r.portfolio_collapse:
        succ.append(""); continue
    stem = re.sub(r"(주식회사|\(주\)|㈜|\s)", "", r.name or "")[:6]
    if len(stem) < 2:
        succ.append(""); continue
    o = U[(U.applicant_cd != r.applicant_cd) & U.ay.between(r.year - 1, r.year + 2) & U.ainfo.fillna("").str.replace(" ", "").str.contains(re.escape(stem))]
    vc = o.applicant_cd.value_counts().head(3)
    succ.append("; ".join(f"{names.get(k, k)}({k}):{v}" for k, v in vc.items()))
if len(D):
    D["successor_candidates"] = succ
# 출원 연도별 KR 범위 안 등록 대비 비중(PREREG 2.3 1% 규칙)
P = U.drop_duplicates("rgst_no")
inyr = P[P.inscope].groupby("ay").size()
if len(D):
    D["prev3_in_share_of_year_inscope"] = [r.n_in_prev3 / 3 / inyr.get(r.year, np.nan) for r in D.itertuples()]
D.to_csv(os.path.join(CACHE, "s3_code_dropouts.csv"), index=False, encoding="utf-8-sig")
res["code_dropout_causes"] = D.cause.value_counts().to_dict() if len(D) else {}
real = D[D.cause == "진짜 결손(의심)"] if len(D) else D
res["real_missing_n"] = len(real)
res["real_missing_max_share"] = float(real.prev3_in_share_of_year_inscope.max()) if len(real) else 0.0
res["applicant_specific_missing"] = bool(len(real) and real.prev3_in_share_of_year_inscope.max() >= 0.01)
# 현대 코드 기준 비율(이 단위에서는 공동출원 기아 표기가 문제 되지 않음)
res["hyundai_code_share"] = ga[ga.applicant_cd == HY].set_index("ay")[["n_in", "n_den", "share"]].loc[2008:2023].round(4).reset_index().to_dict("records")
res["kia_code_share"] = ga[ga.applicant_cd == KIA].set_index("ay")[["n_in", "n_den", "share"]].loc[2008:2023].round(4).reset_index().to_dict("records") if KIA in set(ga.applicant_cd) else None
print("classified", round(time.time() - t0), flush=True)

# ---- 이름 기준 급감(PREREG 2.5) ----
kr10 = kr[kr.appl_no.str[:2] == "10"]
nb = kr10[kr10.appl_year.between(2010, 2016)].groupby("applicant").size().sort_values(ascending=False)
topn = nb.head(300).index.tolist()
cnt = kr10[kr10.applicant.isin(topn) & kr10.appl_year.between(2005, 2023)].groupby(["applicant", "appl_year"]).size().unstack(fill_value=0)
rfa = pd.read_parquet(os.path.join(CACHE, "rf_applicant.parquet"))
rfp = pd.read_parquet(os.path.join(CACHE, "rf_patent.parquet"), columns=["rgst_no", "applNo"])
rfa = rfa.merge(rfp, on="rgst_no", how="left")
k_reg = kr10[kr10.rgst_no.notna()][["appl_no", "applicant", "appl_year"]]
nd = []
for nm in topn:
    s = cnt.loc[nm]
    for t in range(2011, 2023):
        prev = s.loc[t - 3:t - 1]
        if prev.sum() >= 50 and s.get(t, 0) < 0.25 * prev.mean():
            # 이 이름의 대표 코드(2010-16 등록 문서에서 가장 많이 같이 나온 코드)
            ap = k_reg[(k_reg.applicant == nm) & k_reg.appl_year.between(t - 3, t - 1)].appl_no
            codes = rfa[rfa.applNo.isin(set(ap))].applicantCd.value_counts()
            code = codes.index[0] if len(codes) else None
            if code:
                docs = rfa[rfa.applicantCd == code].applNo.unique()
                kk = kr10[kr10.appl_no.isin(set(docs))]
                bycode = kk.groupby("appl_year").size()
                after_names = kk[kk.appl_year.between(t, t + 2)].applicant.value_counts().head(3)
                code_prev = bycode.loc[t - 3:t - 1].mean() if len(bycode) else np.nan
                code_t = bycode.get(t, 0)
                ucd = U[U.applicant_cd == code]
                den_prev = ucd[ucd.ay.between(t - 3, t - 1)].drop_duplicates("rgst_no").groupby("ay").size().mean()
                den_t = ucd[ucd.ay == t].rgst_no.nunique()
            else:
                code_prev = code_t = den_prev = den_t = np.nan; after_names = pd.Series(dtype=int)
            if code and code_prev and code_t >= 0.5 * code_prev:
                kind = "이름 표기 이동(같은 코드 문서는 범위 안에 계속)"
            elif code and den_prev and den_t < 0.5 * den_prev:
                kind = "문서 감소(전체 포트폴리오도 감소)"
            elif code:
                kind = "문서 감소(범위 안만 감소)"
            else:
                kind = "코드 연결 불가"
            nd.append(dict(name=nm, year=t, n_prev3=int(prev.sum()), n_t=int(s.get(t, 0)), code=code, code_inscope_reg_prev3_mean=code_prev,
                           code_inscope_reg_t=int(code_t) if code_t == code_t else None, code_U_den_prev3_mean=den_prev, code_U_den_t=den_t,
                           names_after="; ".join(f"{a}:{b}" for a, b in after_names.items()), kind=kind))
            break
ND = pd.DataFrame(nd)
ND.to_csv(os.path.join(CACHE, "s3_name_dropouts.csv"), index=False, encoding="utf-8-sig")
res["n_dropouts_name"] = len(ND)
res["name_dropout_kinds"] = ND.kind.value_counts().to_dict() if len(ND) else {}
# applicant_dropouts.csv (두 기준 합본)
A1 = D.assign(basis="code(business_rights holder)") if len(D) else pd.DataFrame()
A2 = ND.rename(columns={"kind": "cause", "n_prev3": "n_in_prev3", "n_t": "n_in_t"}).assign(basis="provided_name") if len(ND) else pd.DataFrame()
pd.concat([A1, A2], ignore_index=True).to_csv(os.path.join(WD, "applicant_dropouts.csv"), index=False, encoding="utf-8-sig")
res["sec"] = round(time.time() - t0)
json.dump(res, open(os.path.join(CACHE, "s3_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if "share" not in k}, ensure_ascii=False, default=float))
pd.set_option("display.width", 250)
if len(D):
    print(D[["name", "year", "share_prev3", "share_t", "n_in_prev3", "n_in_t", "n_den_prev3_mean", "n_den_t", "cause", "sim_post_out", "base_in", "base_out", "other_same_sub_ratio"]].to_string())
if len(ND):
    print(ND[["name", "year", "n_prev3", "n_t", "code", "code_inscope_reg_prev3_mean", "code_inscope_reg_t", "code_U_den_prev3_mean", "code_U_den_t", "kind", "names_after"]].to_string())
