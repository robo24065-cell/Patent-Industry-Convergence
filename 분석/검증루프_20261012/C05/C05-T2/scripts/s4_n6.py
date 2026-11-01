# -*- coding: utf-8 -*-
"""s4: N6 사업자번호 경로(위험집합 제외·레이블) 감사. 동결 예측(final_preds_r5) 재채점만.
N6-a 레이블 놓침(독립 탐지: kr_patents 출원인명 = NTIS 상호, CA 사업자번호 미사용), N6-b 위험집합 오염, N6-c 연결 출처 층."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *
import duckdb
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
T0 = 2021; END = 20211231; W0, W1 = 20220101, 20231231
EXTF = EXT.replace("\\", "/")
P = pd.read_parquet(os.path.join(N6, "final_preds_r5.parquet"))
a1, a0 = wauc(P.y, P.p_champ), wauc(P.y, P.p_m0)
log("N6 재현", len(P), int(P.y.sum()), round(a1, 6), round(a0, 6), round(skill(a1, a0), 6))
S_off = skill(a1, a0)
# ---- NTIS 기업·상호 (n6_lib.load_ntis/firm_link 규칙 재현: 2008..t0, 중소·중견·대기업, 10자리)
nt = duckdb.sql(f"""select pjt_no, year, research_agency nm, regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz, perform_grp
  from read_parquet('{EXTF}/ntis_project_raw.parquet') where year between 2008 and {T0} and perform_grp in ('중소기업','중견기업','대기업')""").df()
nt = nt[nt.bz.str.len() == 10]
src = nt[nt.year >= 2010]
firm = src.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()
rest = nt[~nt.bz.isin(set(firm.bz))]
if len(rest):
    firm = pd.concat([firm, rest.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()], ignore_index=True)
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "aname", "src"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
b2src = ns.groupby("bizno").src.agg(lambda s: "CA" if (s == "CA").any() else "RL").to_dict()
ns["nn"] = ns.aname.map(norm)
uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
nn2biz = ns.groupby("nn").bizno.agg(lambda s: set(s)).to_dict()
firm["nn"] = firm.nm.map(norm)
firm["codes_b"] = firm.bz.map(b2a)
firm["code_n"] = firm.nn.map(uq)
firm["route"] = np.where(firm.codes_b.notna(), firm.bz.map(b2src).fillna("CA") + "_bizno", np.where(firm.code_n.notna(), "name_fallback", "unlinked"))
F = P.merge(firm, on="bz", how="left")
log("route", F.route.value_counts(dropna=False).to_dict(), "test pop route NA", int(F.route.isna().sum()))
# ---- 독립 탐지 D1: kr_patents 출원인명 정규화 = NTIS 상호 정규화(NTIS 기업 사이 유일, 길이≥3, CA가 같은 상호에 다른 사업자번호를 주면 제외)
cnt_nn = firm.groupby("nn").bz.nunique()
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원일", "공개일자", "등록일자", "출원인명"])
k.columns = ["appl_no", "ad", "pub", "rg", "names"]
for c in ["ad", "pub", "rg"]:
    k[c] = pd.to_numeric(k[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
k["known"] = k[["pub", "rg"]].min(axis=1)
k["nm"] = k.names.fillna("").str.split("|"); k = k.explode("nm"); k["nn"] = k.nm.map(norm)
tp = F[["bz", "nn"]].copy()
tp = tp[tp.nn.str.len() >= 3]
tp = tp[tp.nn.map(cnt_nn) == 1]
tp["ca_conflict"] = [isinstance(nn2biz.get(n), set) and (b not in nn2biz.get(n)) for b, n in zip(tp.bz, tp.nn)]
log("D1 후보 기업", len(tp), "CA 상호 모순 제외", int(tp.ca_conflict.sum()))
tp = tp[~tp.ca_conflict]
kk = k[k.nn.isin(set(tp.nn))].merge(tp[["bz", "nn"]], on="nn")
D1_win = set(kk[(kk.ad >= W0) & (kk.ad <= W1)].bz)
D1_prior = set(kk[kk.known <= END].bz)
# ---- 독립 탐지 D2: NTIS 성과 특허(제공 KR) — 과제 수행기관 사업자번호 = 기업, 출원일 창 안, 출원인명 = 기업 상호(정규화) 또는 성과 수행기관명
pl = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["cc", "appl_no", "ProjectID", "PerformAgency", "Year"])
pl = pl[pl.cc == "KR"]
pj = duckdb.sql(f"select pjt_no, regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz from read_parquet('{EXTF}/ntis_project_raw.parquet')").df()
pl = pl.merge(pj, left_on=pl.ProjectID.astype(str), right_on=pj.pjt_no.astype(str), how="inner")
pl = pl[pl.bz.isin(set(F.bz))]
kn = k[["appl_no", "ad", "known", "nn"]].copy(); kn["appl_no"] = kn.appl_no.astype(str)
pl["appl_no"] = pl.appl_no.astype(str)
m2 = pl.merge(kn, on="appl_no", how="inner").merge(F[["bz", "nn"]].rename(columns={"nn": "fnn"}), on="bz")
m2["pa"] = m2.PerformAgency.map(norm)
m2 = m2[(m2.nn == m2.fnn) | (m2.nn == m2.pa)]
D2_win = set(m2[(m2.ad >= W0) & (m2.ad <= W1)].bz)
D2_prior = set(m2[m2.known <= END].bz)
F["ind_win"] = F.bz.isin(D1_win | D2_win).astype(int)
F["ind_prior"] = F.bz.isin(D1_prior | D2_prior).astype(int)
R = dict(n=len(F), pos=int(F.y.sum()), route=F.route.value_counts().to_dict(),
         D1_win=len(D1_win & set(F.bz)), D2_win=len(D2_win & set(F.bz)), ind_win=int(F.ind_win.sum()),
         ind_win_captured=int(((F.ind_win == 1) & (F.y == 1)).sum()),
         ind_prior=int(F.ind_prior.sum()))
R["capture_rate"] = R["ind_win_captured"] / max(R["ind_win"], 1)
missed = F[(F.ind_win == 1) & (F.y == 0)]
R["missed"] = len(missed); R["missed_route"] = missed.route.value_counts().to_dict()
R["captured_route"] = F[(F.ind_win == 1) & (F.y == 1)].route.value_counts().to_dict()
R["ind_prior_route"] = F[F.ind_prior == 1].route.value_counts().to_dict()
R["ind_prior_posrate"] = float(F[F.ind_prior == 1].y.mean()) if R["ind_prior"] else None
R["pos_route"] = F[F.y == 1].route.value_counts().to_dict()
log(json.dumps(R, ensure_ascii=False))
RS = []
def row(name, y, mask=None, note=""):
    m = np.ones(len(F), bool) if mask is None else mask
    r = score(y[m], F.p_champ.values[m], F.p_m0.values[m], F.bz.values[m], None, NB)
    r.update(track="N6", stratum=name, d_skill=r["skill"] - S_off, note=note); RS.append(r)
    log(name, {kk_: (round(v, 4) if isinstance(v, float) else v) for kk_, v in r.items() if kk_ in ("n", "pos", "rate", "auc", "auc_base", "skill", "skill_ci")})
y = F.y.values.astype(float)
row("공식(재현)", y)
row("N6-a 대체 레이블 y'=y OR 독립 탐지(창 안)", np.maximum(y, F.ind_win.values), note=f"추가 양성 {R['missed']}")
row("N6-b 위험집합 오염 제외(독립 탐지로 2021년 말까지 알려진 출원 있음)", y, F.ind_prior.values == 0, note=f"제외 {R['ind_prior']}")
row("N6-a+b 둘 다", np.maximum(y, F.ind_win.values), F.ind_prior.values == 0)
for rt in ["CA_bizno", "RL_bizno", "name_fallback", "unlinked"]:
    m = (F.route == rt).values
    if m.sum() == 0:
        continue
    if F.y[m].sum() >= 5:
        row(f"N6-c 연결 출처 {rt}", y, m)
    else:
        RS.append(dict(track="N6", stratum=f"N6-c 연결 출처 {rt}", n=int(m.sum()), pos=int(F.y[m].sum()), rate=float(F.y[m].mean()), note="양성 부족(구조상 y=0이면 미연결)"))
row("N6-c 연결 기업만(CA·RL·상호)", y, (F.route != "unlinked").values, note="final_result subgroup_linked_only 대조")
# 미연결 기업 중 독립 탐지 양성 비율(진짜 첫 특허율 근사) 대 연결 기업
for rt in ["CA_bizno", "name_fallback", "unlinked"]:
    m = F.route == rt
    R[f"ind_win_rate_{rt}"] = float(F[m].ind_win.mean()); R[f"y_rate_{rt}"] = float(F[m].y.mean())
pd.DataFrame(RS).to_csv(os.path.join(CACHE, "rescore_N6.csv"), index=False, encoding="utf-8-sig")
json.dump(R, open(os.path.join(CACHE, "s4_n6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
missed[["bz", "nm", "route"]].head(30).to_csv(os.path.join(CACHE, "n6_missed_sample.csv"), index=False, encoding="utf-8-sig")
log("done")
