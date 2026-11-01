# -*- coding: utf-8 -*-
"""s4b: N6 놓친 양성(CA_bizno 연결인데 y=0) 원인 진단. s4_n6.py와 같은 탐지를 다시 만들어 출원번호 단위로 app_firm 매핑을 본다."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
exec(open(__file__.replace("s4b_n6_diag.py", "s4_n6.py"), encoding="utf-8").read().split("# ---- 독립 탐지 D2")[0])
af = pd.read_parquet(os.path.join(MC, r"_ideas_R3\cache\app_firm.parquet"), columns=["applicantCd", "appl_no", "src"])
af["appl_no"] = af.appl_no.astype(str)
code2name = ns.drop_duplicates("applicantCd").set_index("applicantCd").aname
w = kk[(kk.ad >= W0) & (kk.ad <= W1)].copy(); w["appl_no"] = w.appl_no.astype(str)
w = w.merge(F[["bz", "y", "route", "codes_b", "nm"]], on="bz")
mis = w[(w.y == 0) & (w.route == "CA_bizno")]
mis = mis.merge(af, on="appl_no", how="left")
mis["code_in_firm"] = [isinstance(c, list) and a in c for c, a in zip(mis.codes_b, mis.applicantCd)]
g = mis.groupby("bz").agg(any_af=("applicantCd", lambda s: s.notna().any()), any_in=("code_in_firm", "any"))
out = {"missed_CA_firms": int(mis.bz.nunique()), "appl_not_in_app_firm(firms)": int((~g.any_af).sum()),
       "appl_mapped_other_code(firms)": int((g.any_af & ~g.any_in).sum()), "appl_mapped_same_code(firms)": int(g.any_in.sum())}
# 다른 코드로 매핑된 경우 그 코드의 사업자번호
oth = mis[mis.applicantCd.notna() & ~mis.code_in_firm].drop_duplicates("bz")
a2b = ns.groupby("applicantCd").bizno.first()
oth["other_bizno"] = oth.applicantCd.map(a2b)
out["other_code_has_diff_bizno"] = int((oth.other_bizno.notna() & (oth.other_bizno != oth.bz)).sum())
out["other_code_no_bizno"] = int(oth.other_bizno.isna().sum())
# app_firm에 없는 출원: 등록 여부·공개 여부
na = mis[mis.applicantCd.isna()].drop_duplicates("appl_no")
out["notin_af_docs"] = len(na); out["notin_af_unregistered_share"] = float(na.rg.isna().mean())
# 같은 상호가 CA에 있는가(유일/모호/없음)
cnt_ca = ns.groupby("nn").applicantCd.nunique()
na_f = mis[mis.applicantCd.isna()].drop_duplicates("bz")
out["notin_af_firm_name_in_CA_unique"] = int((na_f.nn.map(cnt_ca) == 1).sum())
out["notin_af_firm_name_in_CA_ambig"] = int((na_f.nn.map(cnt_ca) > 1).sum())
out["notin_af_firm_name_not_in_CA"] = int(na_f.nn.map(cnt_ca).isna().sum())
# 연결 코드의 CA 상호와 NTIS 상호 일치 여부
na_f = na_f.assign(code_names=[[norm(code2name.get(c, "")) for c in cs] for cs in na_f.codes_b])
out["notin_af_ntis_name_eq_code_name"] = int(sum(n in cn for n, cn in zip(na_f.nn, na_f.code_names)))
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(CACHE, "s4b_n6_diag.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
