# -*- coding: utf-8 -*-
"""s4c: N6 상호 보조 경로(name_fallback) — NTIS 사업자번호와 다른 사업자번호를 가진 출원인코드가 붙는 문제의 크기와 동결 예측 재채점.
verify_0 (v)에서 362사 전부 '연결 코드의 사업자번호 ≠ NTIS 사업자번호'로 확인됨. 지점(85)·본점 관계인지, 다른 법인인지 사업자번호 구조로 구분."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
P = pd.read_parquet(os.path.join(N6, "final_preds_r5.parquet"))
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "aname", "corpno"] if False else ["applicantCd", "bizno", "aname"])
nt = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["year", "research_agency", "corp_rgst_no", "perform_grp"])
nt["bz"] = nt.corp_rgst_no.fillna("").str.replace(r"\D", "", regex=True)
nt = nt[(nt.bz.str.len() == 10) & nt.year.between(2010, 2021) & nt.perform_grp.isin(["중소기업", "중견기업", "대기업"])]
fn = nt.groupby("bz").research_agency.agg(lambda s: s.value_counts().index[0])
ns["n"] = ns.aname.map(norm)
c2 = ns.groupby("n").applicantCd.nunique(); uq2 = ns[ns.n.map(c2) == 1].set_index("n")
F = P.copy(); F["has_b"] = F.bz.isin(set(ns.bizno)); F["n"] = F.bz.map(fn).map(norm)
F["fb"] = (~F.has_b) & F.n.isin(uq2.index)
F["code_bizno"] = np.where(F.fb, F.n.map(uq2.bizno), None)
fb = F[F.fb].copy()
fb["bt_ntis"] = fb.bz.str[3:5]; fb["bt_code"] = fb.code_bizno.astype(str).str[3:5]
fb["same_office"] = fb.bz.str[:3] == fb.code_bizno.astype(str).str[:3]
R = dict(n=len(fb), pos=int(fb.y.sum()),
         ntis_branch85=int((fb.bt_ntis == "85").sum()), code_branch85=int((fb.bt_code == "85").sum()),
         ntis_type=fb.bt_ntis.value_counts().head(8).to_dict(), code_type=fb.bt_code.value_counts().head(8).to_dict(),
         same_tax_office=int(fb.same_office.sum()),
         pos_ntis_branch85=int(((fb.bt_ntis == "85") & (fb.y == 1)).sum()))
S_off = skill(wauc(P.y, P.p_champ), wauc(P.y, P.p_m0))
RS = []
def row(name, y, m, note=""):
    r = score(y[m], F.p_champ.values[m], F.p_m0.values[m], F.bz.values[m], None, NB)
    r.update(track="N6", stratum=name, d_skill=r["skill"] - S_off, note=note); RS.append(r)
    log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("n", "pos", "skill", "skill_ci", "d_skill")})
y = F.y.values.astype(float); allm = np.ones(len(F), bool)
row("N6-d 상호 보조 경로 기업 제외", y, ~F.fb.values, note=f"제외 {len(fb)}사(양성 {R['pos']})")
row("N6-d 상호 보조 경로 양성을 0으로(다른 사업자번호 문서)", np.where(F.fb.values, 0.0, y), allm)
pd.DataFrame(RS).to_csv(os.path.join(CACHE, "rescore_N6_fallback.csv"), index=False, encoding="utf-8-sig")
json.dump(R, open(os.path.join(CACHE, "s4c_n6_fallback.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log(json.dumps(R, ensure_ascii=False))
