# -*- coding: utf-8 -*-
"""I8 정부R&D 수행기업의 '50개 산업 첫 특허' 전환 예측 (NTIS 과제 -> 제공 KSIC 특허).
단위 = (사업자번호, t0): t0-2..t0 과제연도에 NTIS 과제를 수행한 기업(중소·중견·대기업) 중,
       t0 말까지 공개·등록으로 알려진 제공범위(50 KSIC) KR 출원이 없는 기업(실시간 위험집합).
결과 y = 출원일이 t0+1..t0+2 인 제공범위 KR 출원 1건 이상(사업자번호 -> 출원인코드 연결, 이름 보조 연결).
[봉인] 최종시험 t0=2021 -> 출원 2022–2023 (F=2022). 출원일 >= 2022-01-01 레이블은 만들지 않는다. 개발 원점 t0 <= 2019.
주의(누수 방지): business_rights(제공범위 보유 법인만 조회) 존재 여부·CORP_APPLICANT 코드 보유 여부는 피처로 쓰지 않는다(결과 연결에만 사용).
피처: M0 = NTIS 과제 이력(규모·연구비·경력·주체), M1 = +과제 내용(개발단계·6T·분류·KSIC50 텍스트 대응),
      M2 = +국민연금(2015-12~; 이전 원점은 결측), M3 = +KSIC 수준 정부연구비 성장(교훈 재확인)."""
import re, time, numpy as np, pandas as pd, duckdb
from r4_common import *
T = time.time()


def log(*a):
    print(*a, round(time.time() - T), flush=True)


SEAL_APPL = 20220101
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


EXTF = EXT.replace("\\", "/")
NT = duckdb.sql(f"""select pjt_no, year, research_agency nm, regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz, perform_grp, dev_phase, sixt,
 cls_L_canon, gov_funds, sbiz_funds, total_funds, continuous
 from read_parquet('{EXTF}/ntis_project_raw.parquet') where year between 2010 and 2019 and perform_grp in ('중소기업','중견기업','대기업')""").df()
TM = pd.read_parquet(f"{EXT}/ntis_project_ksic_text_map.parquet", columns=["pjt_no", "year", "b1_tfidf_score", "b2_dict_top_ksic", "b2_dict_score", "b2_dict_assigned"])
NT = NT.merge(TM, on=["pjt_no", "year"], how="left")
NT = NT[NT.bz.str.len() == 10]
log("NTIS firm projects", len(NT), NT.bz.nunique())
# 사업자번호 -> 출원인코드 (국세청 대상표 = CORP_APPLICANT 사업자번호) + 이름 보조(유일 상호)
ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(norm); uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
firm = NT.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()
firm["codes"] = firm.bz.map(b2a)
nm_code = firm.nm.map(norm).map(uq)
firm["codes"] = [c if isinstance(c, list) else ([n] if isinstance(n, str) else []) for c, n in zip(firm.codes, nm_code)]
log("firms", len(firm), "code linked", round(np.mean([len(c) > 0 for c in firm.codes]), 3))
AF = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "appl_no", "appl_date", "known_date"])
ex = firm.explode("codes").dropna(subset=["codes"])
AF = AF[AF.applicantCd.isin(set(ex.codes))].merge(ex[["bz", "codes"]], left_on="applicantCd", right_on="codes")
AF_out = AF[AF.appl_date < SEAL_APPL]   # 결과용: 봉인 이전 출원일만
first_known = AF.groupby("bz").known_date.min()
log("firm apps linked", len(AF))
KY = pd.read_parquet(f"{EXT}/ntis_project_ksic_year.parquet"); KY = KY[KY.method == "a_link_pit"]
NPS = {}
for sn in ["2015-12", "2016-12", "2017-12", "2018-12", "2019-12"]:
    n = pd.read_parquet(f"{R3I}/cache/nps_firm_{sn}.parquet", columns=["key", "n_emp", "active", "adpt", "n_new", "n_loss"])
    NPS[sn] = n[n.active].drop_duplicates("key").set_index("key")
firm["key"] = firm.bz.str[:6] + "|" + firm.nm.map(norm)
k2 = dict(zip(firm.bz, firm.key))
SIXT = ["IT", "BT", "NT", "ET", "ST", "CT"]
CLS = ["기계", "전기전자", "정보통신", "화공", "재료", "보건의료", "농림수산", "생명과학"]
rows = []
for t0 in range(2012, 2020):
    end = t0 * 10000 + 1231; lo = (t0 + 1) * 10000 + 101; hi = (t0 + 2) * 10000 + 1231; assert hi < SEAL_APPL
    w = NT[NT.year.between(t0 - 2, t0)]
    g = w.groupby("bz")
    f = pd.DataFrame({"n_proj": np.log1p(g.pjt_no.nunique()), "funds": np.log1p(g.gov_funds.sum()), "n_yrs": g.year.nunique(),
                      "rec": t0 - g.year.max(), "sbiz_sh": g.sbiz_funds.sum() / (g.total_funds.sum() + 1),
                      "cont_sh": g.continuous.apply(lambda s: (pd.to_numeric(s, errors="coerce") == 1).mean()),
                      "sme": g.perform_grp.apply(lambda s: (s == "중소기업").mean())})
    allp = NT[NT.year <= t0].groupby("bz"); f["career"] = t0 - allp.year.min().reindex(f.index); f["n_proj_all"] = np.log1p(allp.pjt_no.nunique().reindex(f.index))
    f["corp_type"] = pd.to_numeric(pd.Series(f.index, index=f.index).str[3:5], errors="coerce")
    # M1 과제 내용
    f["ph_dev"] = g.dev_phase.apply(lambda s: (s == "개발연구").mean()); f["ph_app"] = g.dev_phase.apply(lambda s: (s == "응용연구").mean())
    for s6 in SIXT:
        f["t6_" + s6] = g.sixt.apply(lambda s, s6=s6: s.fillna("").str.contains(s6).mean())
    for cl in CLS:
        f["cl_" + cl] = g.cls_L_canon.apply(lambda s, cl=cl: (s == cl).mean())
    f["k50_assigned"] = g.b2_dict_assigned.apply(lambda s: s.fillna(False).astype(float).mean())
    f["k50_score"] = g.b2_dict_score.mean(); f["b1_score"] = g.b1_tfidf_score.mean()
    # 위험집합: t0말까지 알려진 제공범위 출원 없음
    f = f[~(f.index.map(first_known) <= end)].copy()
    # M2 NPS
    sn = f"{t0}-12"
    if sn in NPS:
        cur = NPS[sn]; prv = NPS.get(f"{t0 - 1}-12")
        keys = f.index.map(k2)
        emp = pd.Series([cur.n_emp.get(k, np.nan) for k in keys], index=f.index)
        f["nps_present"] = emp.notna().astype(float); f["log_emp"] = np.log1p(emp)
        f["g_emp1"] = np.log((emp + 1) / (pd.Series([prv.n_emp.get(k, np.nan) for k in keys], index=f.index) + 1)) if prv is not None else np.nan
        f["nps_age"] = t0 - pd.to_numeric(pd.Series([cur.adpt.get(k, None) for k in keys], index=f.index).astype(str).str[:4], errors="coerce")
    else:
        for c in ["nps_present", "log_emp", "g_emp1", "nps_age"]:
            f[c] = np.nan
    # M3 KSIC 수준 정부연구비 성장(과제 텍스트 대응 1순위 KSIC 기준)
    ky = KY[KY.avail_year <= t0]
    pv = ky.pivot_table(index="ksic", columns="year", values="gov_funds", aggfunc="sum")
    a_ = pv[[y for y in pv.columns if t0 - 2 <= y <= t0 - 1]].sum(1); b_ = pv[[y for y in pv.columns if t0 - 4 <= y <= t0 - 3]].sum(1)
    gr = np.log((a_ + 1) / (b_ + 1)); gr[b_ <= 0] = np.nan
    ww = w[w.b2_dict_assigned.fillna(False).astype(bool)].copy(); ww["gr"] = ww.b2_dict_top_ksic.astype(str).map(gr)
    f["k_rnd_g"] = ww.groupby("bz").gr.mean().reindex(f.index)
    # 결과
    o = AF_out[(AF_out.appl_date >= lo) & (AF_out.appl_date <= hi)]
    f["y"] = f.index.isin(set(o.bz)).astype(int); f["t0"] = t0
    f["linked"] = f.index.map(lambda b: len(firm.set_index("bz").codes.get(b, [])) > 0) if False else np.nan
    rows.append(f.reset_index().rename(columns={"index": "bz"}))
    log(t0, len(f), "y", int(f.y.sum()), round(f.y.mean(), 4), "nps", round(f.nps_present.mean(), 3) if f.nps_present.notna().any() else None)
D = pd.concat(rows, ignore_index=True)
D.to_parquet(f"{CACHE}/I8_panel.parquet", index=False)
log("done", D.shape)
