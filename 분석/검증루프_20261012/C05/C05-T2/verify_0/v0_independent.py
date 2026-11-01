# -*- coding: utf-8 -*-
"""verify_0: 독립 재계산. app_firm 캐시·lib5를 쓰지 않고 등록원부 sqlite(mode=ro)·kr_patents·CORP_APPLICANT 원천에서
출원인별 '기준일 후 활동(M)'을 다시 만들고, sklearn roc_auc_score로 N7(t0=2022)·N4(t0=2023) 층 AUC·skill을 대조한다.
반박: (ii) M을 '출원일 > t0' 정의로 바꿈, (iv) M 층 안에서 공식 기준선 대비 skill 부호, (v) N6 상호 보조 경로의 사업자번호 모순."""
import os, re, sys, json, sqlite3
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + r"\분석\모델경쟁_20260927"
HERE = os.path.dirname(os.path.abspath(__file__))
SUFX = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nm(s):
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUFX.sub("", str(s or "")).upper())
def d8(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce").values

# 1) 제공범위 KR 문서(kr_patents)
k = pd.read_parquet(BASE + r"\공모전데이터\작업데이터\kr_patents.parquet", columns=["출원번호", "출원일", "공개일자", "등록일자", "출원인명"])
k.columns = ["appl_no", "ad", "pub", "rg", "names"]
k["appl_no"] = k.appl_no.astype(str).str.replace(r"\D", "", regex=True)
for c in ["ad", "pub", "rg"]:
    k[c] = d8(k[c])
k["known"] = k[["pub", "rg"]].min(axis=1)
# 2) 등록원부: 출원번호 → 출원인코드
con = sqlite3.connect(f"file:{BASE}/수집파이프라인/registry_full.sqlite?mode=ro", uri=True)
ra = pd.read_sql("select p.applNo appl_no, a.applicantCd from patent p join applicant a on a.rgst_no=p.rgst_no", con)
ra["appl_no"] = ra.appl_no.astype(str).str.replace(r"\D", "", regex=True)
ra = ra.dropna(subset=["applicantCd"]).drop_duplicates()
reg_docs = k.merge(ra, on="appl_no", how="inner")[["applicantCd", "appl_no", "ad", "known"]]
# 3) 등록원부에 없는 문서: 출원인명 → CORP_APPLICANT 유일 상호
rows = []
with open(BASE + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 2:
            rows.append((p[0].strip(), nm(p[1])))
ca = pd.DataFrame(rows, columns=["applicantCd", "n"]).drop_duplicates()
cnt = ca.groupby("n").applicantCd.nunique()
uq = ca[ca.n.map(cnt) == 1].set_index("n").applicantCd
rest = k[~k.appl_no.isin(set(ra.appl_no))].copy()
rest["nm"] = rest.names.fillna("").str.split("|"); rest = rest.explode("nm"); rest["n"] = rest.nm.map(nm)
rest = rest[rest.n.str.len() > 0]
rest["applicantCd"] = rest.n.map(uq)
name_docs = rest.dropna(subset=["applicantCd"])[["applicantCd", "appl_no", "ad", "known"]]
DOCS = pd.concat([reg_docs, name_docs], ignore_index=True).drop_duplicates(["applicantCd", "appl_no"])
DOCS["applicantCd"] = DOCS.applicantCd.astype(str)
print("docs", len(reg_docs), len(name_docs), len(DOCS), flush=True)

def skill(a, b):
    return (a - b) / (1 - b)

OUTR, ROWS = {}, []
s3 = pd.read_csv(os.path.join(os.path.dirname(HERE), "cache", "rescore_N4N7.csv"))
for tr, fp, t0 in [("N7", MC + r"\N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet", 2022),
                   ("N4", MC + r"\N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet", 2023)]:
    P = pd.read_parquet(fp); P["applicantCd"] = P.applicantCd.astype(str)
    end = t0 * 10000 + 1231; m_end = (t0 + 2) * 10000 + 831
    dd = DOCS[DOCS.applicantCd.isin(set(P.applicantCd))]
    Mk = set(dd[(dd.known > end) & (dd.known <= m_end)].applicantCd)
    Ma = set(dd[dd.ad > end].applicantCd)
    P["M_v"] = P.applicantCd.isin(Mk).astype(int); P["M_a"] = P.applicantCd.isin(Ma).astype(int)
    # 공식 재현
    a1, a0 = roc_auc_score(P.y, P.p_champ), roc_auc_score(P.y, P.p_M0)
    OUTR[tr] = {"official": [len(P), int(P.y.sum()), round(a1, 6), round(a0, 6), round(skill(a1, a0), 6)]}
    for col, lab in [("M_v", "M(known, 20개월)"), ("M_a", "M(출원일>t0, 반박 ii)")]:
        for v in [0, 1]:
            m = P[col] == v
            a1, a0 = roc_auc_score(P.y[m], P.p_champ[m]), roc_auc_score(P.y[m], P.p_M0[m])
            ROWS.append(dict(track=tr, stratum=f"{lab}={v}", n=int(m.sum()), pos=int(P.y[m].sum()), rate=round(float(P.y[m].mean()), 4),
                             auc=round(a1, 4), auc_base=round(a0, 4), skill=round(skill(a1, a0), 4)))
    # 원 계산(s3, app_firm 캐시)과 층 소속 일치
    U = pd.read_parquet(os.path.join(os.path.dirname(HERE), "cache", f"U_{tr}.parquet"), columns=["applicantCd", "M"])
    U["applicantCd"] = U.applicantCd.astype(str)
    X = P.merge(U, on="applicantCd", how="left")
    OUTR[tr]["M_agree"] = round(float((X.M == X.M_v).mean()), 4)
    OUTR[tr]["M_disagree_n"] = int((X.M != X.M_v).sum())
    OUTR[tr]["M_v_share"] = round(float(P.M_v.mean()), 4)
    for v in [0, 1]:
        mine = [r for r in ROWS if r["track"] == tr and r["stratum"] == f"M(known, 20개월)={v}"][0]
        orig = s3[(s3.track == tr) & s3.stratum.str.startswith(f"S1 M={v}")].iloc[0]
        OUTR[tr][f"M={v}_skill_mine_vs_s3"] = [mine["skill"], round(float(orig.skill), 4), round(mine["skill"] - float(orig.skill), 4)]
        OUTR[tr][f"M={v}_rate_mine_vs_s3"] = [mine["rate"], round(float(orig.rate), 4)]
# (v) N6 상호 보조 경로: NTIS 사업자번호와 연결된 출원인코드의 CA 사업자번호가 다른가
ns = pd.read_parquet(MC + r"\_shared\ext\r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
P6 = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet")
nt = pd.read_parquet(MC + r"\_shared\ext\ntis_project_raw.parquet", columns=["year", "research_agency", "corp_rgst_no", "perform_grp"])
nt["bz"] = nt.corp_rgst_no.fillna("").str.replace(r"\D", "", regex=True)
nt = nt[(nt.bz.str.len() == 10) & nt.year.between(2010, 2021) & nt.perform_grp.isin(["중소기업", "중견기업", "대기업"])]
fn = nt.groupby("bz").research_agency.agg(lambda s: s.value_counts().index[0])
ns["n"] = ns.aname.map(nm)
c2 = ns.groupby("n").applicantCd.nunique(); uq2 = ns[ns.n.map(c2) == 1].set_index("n")
F = P6[["bz", "y"]].copy(); F["has_b"] = F.bz.isin(set(ns.bizno)); F["n"] = F.bz.map(fn).map(nm)
fb = F[~F.has_b & F.n.isin(uq2.index)].copy()
fb["code_bizno"] = fb.n.map(uq2.bizno)
OUTR["N6_name_fallback"] = dict(n=len(fb), pos=int(fb.y.sum()),
                                code_has_other_bizno=int((fb.code_bizno.notna() & (fb.code_bizno != fb.bz)).sum()),
                                pos_with_other_bizno=int(((fb.code_bizno.notna()) & (fb.code_bizno != fb.bz) & (fb.y == 1)).sum()))
pd.DataFrame(ROWS).to_csv(os.path.join(HERE, "v0_strata.csv"), index=False, encoding="utf-8-sig")
json.dump(OUTR, open(os.path.join(HERE, "v0_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(OUTR, ensure_ascii=False, indent=1)); print(pd.DataFrame(ROWS).to_string())
