# -*- coding: utf-8 -*-
"""(a) 체크섬·DART·권리목록·국세청 대조, (b) 다대다·공란. 출력 cache/s02_out.json, cache/ca.parquet"""
import os, sys, re, json, difflib
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
import pandas as pd, numpy as np

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?\s*LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return np.nan
    return difflib.SequenceMatcher(None, a, b).ratio()


def jur_ok(d):
    if not re.fullmatch(r"\d{13}", d):
        return None
    s = sum(int(d[i]) * (1 if i % 2 == 0 else 2) for i in range(12))
    return (10 - s % 10) % 10 == int(d[12])


def biz_ok(d):
    if not re.fullmatch(r"\d{10}", d):
        return None
    w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    s = sum(int(d[i]) * w[i] for i in range(9)) + (int(d[8]) * 5) // 10
    return (10 - s % 10) % 10 == int(d[9])


hdr, rows = read_ca()
ca = pd.DataFrame(rows, columns=["cd", "name", "ename", "jur_raw", "biz_raw"])
for c in ca.columns:
    ca[c] = ca[c].str.strip()
ca["jur"] = ca.jur_raw.str.replace(r"\D", "", regex=True)
ca["biz"] = ca.biz_raw.str.replace(r"\D", "", regex=True)
O = {}
O["rows"] = len(ca)
O["cd_unique"] = int(ca.cd.nunique())
O["cd_dup_rows"] = int(ca.cd.duplicated(keep=False).sum())
O["cd_len"] = ca.cd.str.len().value_counts().to_dict()
O["cd_first_digit"] = ca.cd.str[0].value_counts().to_dict()
O["cd_issue_year_digits2_5"] = ca.cd.str[1:5].value_counts().sort_index().to_dict()
O["blank"] = {c: int((ca[c] == "").sum()) for c in ["name", "ename", "jur_raw", "biz_raw"]}
O["jur_raw_format"] = ca.jur_raw.str.replace(r"\d", "9", regex=True).value_counts().head(8).to_dict()
O["biz_raw_format"] = ca.biz_raw.str.replace(r"\d", "9", regex=True).value_counts().head(8).to_dict()
ca["jur_ok"] = ca.jur.map(lambda d: jur_ok(d) if d else np.nan)
ca["biz_ok"] = ca.biz.map(lambda d: biz_ok(d) if d else np.nan)


def rate(col, raw):
    nz = ca[raw] != ""
    v = ca.loc[nz, col]
    bad_fmt = int(v.isna().sum())
    fail = int((v == False).sum())
    ok = int((v == True).sum())
    return dict(nonblank=int(nz.sum()), ok=ok, fail_checksum=fail, bad_format=bad_fmt,
                fail_rate_incl_fmt=(fail + bad_fmt) / int(nz.sum()), fail_rate_valid_fmt=fail / max(1, ok + fail))


O["a1_jur_checksum"] = rate("jur_ok", "jur_raw")
O["a2_biz_checksum"] = rate("biz_ok", "biz_raw")
# 실패 사례 패턴(마스킹)
O["a1_fail_examples_masked"] = ca.loc[ca.jur_ok == False, "jur"].str[:6].value_counts().head(10).to_dict()
O["a2_fail_examples_masked"] = ca.loc[ca.biz_ok == False, "biz"].str[3:5].value_counts().head(10).to_dict()
# 법인 종류(법인번호 5-6자리) / 사업자번호 4-5자리(구분코드)
ca["jkind"] = np.where(ca.jur.str.len() == 13, ca.jur.str[4:6], "")
ca["jreg"] = np.where(ca.jur.str.len() == 13, ca.jur.str[:4], "")
ca["bkind"] = np.where(ca.biz.str.len() == 10, ca.biz.str[3:5], "")
O["jkind_dist"] = ca.jkind.value_counts().head(20).to_dict()
O["bkind_dist"] = ca.bkind.value_counts().head(25).to_dict()
# 개인형 사업자 구분코드(01–79)가 법인번호 행에 붙은 비율
indiv = ca.bkind.str.fullmatch(r"(0[1-9]|[1-7]\d)") & (ca.jur != "")
O["biz_individual_code_with_jur"] = int(indiv.sum())
# ---------- (b) 다대다 ----------
J = ca[ca.jur != ""]; B = ca[ca.biz != ""]
O["b_jur_rows"] = len(J); O["b_biz_rows"] = len(B)
jcnt = J.groupby("jur").cd.nunique(); bcnt = B.groupby("biz").cd.nunique()
O["b_jur_multi_cd_rows"] = int(J.jur.map(jcnt).gt(1).sum())
O["b_jur_multi_cd_entities"] = int((jcnt > 1).sum()); O["b_jur_unique"] = int(len(jcnt))
O["b_jur_cd_count_dist"] = jcnt.value_counts().sort_index().head(10).to_dict()
O["b_biz_multi_cd_rows"] = int(B.biz.map(bcnt).gt(1).sum()); O["b_biz_unique"] = int(len(bcnt))
bj = B[B.jur != ""].groupby("biz").jur.nunique()
O["b_biz_multi_jur_bizno"] = int((bj > 1).sum())
conf1 = B.biz.isin(bj[bj > 1].index) & (B.jur != "")
jb = J[J.biz != ""].groupby("jur").biz.nunique()
O["b_jur_multi_biz_entities"] = int((jb > 1).sum())
# 같은 법인번호 복수 코드 중 이름 유사도<0.5
Jm = J[J.jur.map(jcnt) > 1].copy()
Jm["nn"] = Jm.name.map(norm)
lowpair = set()
for j, g in Jm.groupby("jur"):
    ref = g.nn.value_counts().index[0]
    for i, r in g.iterrows():
        s = difflib.SequenceMatcher(None, r.nn, ref).ratio() if r.nn and ref else 0
        if s < 0.5:
            lowpair.add(i)
conf2 = pd.Series(False, index=ca.index); conf2.loc[list(lowpair)] = True
conf = (conf1.reindex(ca.index, fill_value=False) | conf2) & (ca.jur != "")
O["b_conflict_biz_multi_jur_rows"] = int(conf1.sum())
O["b_conflict_samejur_lowname_rows"] = int(conf2.sum())
O["b_conflict_union_rows"] = int(conf.sum())
O["b_conflict_rate_of_jur_rows"] = float(conf.sum() / len(J))
O["b_simple_many_to_one_rate"] = float(O["b_jur_multi_cd_rows"] / len(J))
ca["conflict"] = conf
# ---------- (a3) DART ----------
dc = ro(DART_DB)
D = pd.read_sql("select corp_code, corp_name, jurir_no, bizr_no, corp_cls, status from company", dc)
D["jur"] = D.jurir_no.fillna("").str.replace(r"\D", "", regex=True)
D["biz"] = D.bizr_no.fillna("").str.replace(r"\D", "", regex=True)
O["dart_rows"] = len(D); O["dart_jur_nonblank"] = int((D.jur.str.len() == 13).sum()); O["dart_biz_nonblank"] = int((D.biz.str.len() == 10).sum())
O["dart_jur_checksum_fail"] = int(sum(jur_ok(x) is False for x in D.jur)); O["dart_biz_checksum_fail"] = int(sum(biz_ok(x) is False for x in D.biz))
cj = ca[ca.jur != ""].drop_duplicates("jur")[["jur", "biz", "name", "cd"]]
m = D[D.jur.str.len() == 13].merge(cj, on="jur", how="left", suffixes=("_d", "_c"))
inca = m.cd.notna()
both = inca & (m.biz_c.fillna("") != "") & (m.biz_d.str.len() == 10)
O["a3_dart_jur_in_ca"] = int(inca.sum())
O["a3_dart_jur_in_ca_biz_blank_ca"] = int((inca & (m.biz_c.fillna("") == "")).sum())
O["a3_pairs_both_biz"] = int(both.sum())
O["a3_pair_agree"] = int((both & (m.biz_c == m.biz_d)).sum())
O["a3_pair_agree_rate"] = float(O["a3_pair_agree"] / max(1, both.sum()))
# 불일치 사례: 앞 3자리 같음(같은 세무서권) 여부
dis = m[both & (m.biz_c != m.biz_d)]
O["a3_disagree_same_first3"] = int((dis.biz_c.str[:3] == dis.biz_d.str[:3]).sum())
# 반대 방향: DART biz로 CA 찾아 jur 일치
cb = ca[ca.biz != ""][["biz", "jur", "cd"]]
m2 = D[D.biz.str.len() == 10].merge(cb, on="biz", how="inner", suffixes=("_d", "_c"))
m2b = m2[(m2.jur_c != "") & (m2.jur_d.str.len() == 13)]
O["a3_rev_pairs"] = int(len(m2b)); O["a3_rev_agree"] = int((m2b.jur_c == m2b.jur_d).sum())
O["a3_rev_agree_rate"] = float(O["a3_rev_agree"] / max(1, len(m2b)))
# 이름 유사도 CA ↔ DART
mm = m[inca].copy(); mm["s"] = [sim(a, b) for a, b in zip(mm.name, mm.corp_name)]
O["a3_name_sim_CA_DART"] = dict(n=int(mm.s.notna().sum()), lt05=int((mm.s < 0.5).sum()), median=float(mm.s.median()))
mm[["corp_code", "cd", "name", "corp_name", "s"]].to_parquet(guard_path(os.path.join(CACHE, "s02_ca_dart_names.parquet")))
# 파생본 확인
dp = pd.read_parquet(DART_CA)
O["dart_corp_applicant_parquet_rows"] = len(dp)
same = (dp["출원인코드"].astype(str).str.strip().values == ca.cd.values)
O["dart_corp_applicant_parquet_same_cd_order"] = float(same.mean())
O["dart_corp_applicant_parquet_jur_equal"] = float((dp["jurir"].fillna("").astype(str).values == ca.jur.values).mean())
O["dart_corp_applicant_parquet_biz_equal"] = float((dp["bizr"].fillna("").astype(str).values == ca.biz.values).mean())
# ---------- (a4) 권리목록 ----------
bc = ro(BR_DB)
R = pd.read_sql("select applicant_cd, business_no, corp_rgstno, count(*) n from right_list group by 1,2,3", bc)
for c in ["business_no", "corp_rgstno"]:
    R[c] = R[c].fillna("").astype(str).str.replace(r"\D", "", regex=True)
O["a4_rl_codes"] = int(R.applicant_cd.nunique())
def modal(col):
    t = R[R[col] != ""].sort_values("n", ascending=False).drop_duplicates("applicant_cd")
    nval = R[R[col] != ""].groupby("applicant_cd")[col].nunique()
    return t.set_index("applicant_cd")[col], nval
rb, rbn = modal("business_no"); rj, rjn = modal("corp_rgstno")
O["a4_rl_codes_multi_biz"] = int((rbn > 1).sum()); O["a4_rl_codes_multi_jur"] = int((rjn > 1).sum())
cai = ca.set_index("cd")
for lab, rs, col in [("biz", rb, "biz"), ("jur", rj, "jur")]:
    k = rs.index.intersection(cai.index)
    cv = cai.loc[k, col]; rv = rs.loc[k]
    nb = cv != ""
    O[f"a4_{lab}_codes_in_both"] = int(len(k)); O[f"a4_{lab}_ca_blank"] = int((~nb).sum())
    O[f"a4_{lab}_agree"] = int((cv[nb] == rv[nb]).sum()); O[f"a4_{lab}_compared"] = int(nb.sum())
    O[f"a4_{lab}_agree_rate"] = float((cv[nb] == rv[nb]).mean())
    O[f"a4_{lab}_rl_checksum_fail"] = int(sum((jur_ok if lab == "jur" else biz_ok)(x) is False for x in rv))
O["a4_rl_codes_not_in_ca"] = int(len(set(R.applicant_cd) - set(ca.cd)))
# ---------- (a5) 국세청 ----------
S = pd.read_parquet(os.path.join(EXT, "r3nts_status.parquet"))
O["a5_cols"] = list(S.columns)
bcol = [c for c in S.columns if c in ("b_no", "bizno", "biz")][0]
S["biz"] = S[bcol].astype(str).str.replace(r"\D", "", regex=True)
cabiz = set(ca.biz[ca.biz != ""])
Sc = S[S.biz.isin(cabiz)]
O["a5_nts_rows_ca_biz"] = len(Sc); O["a5_ca_biz_unique"] = len(cabiz)
stt = Sc.b_stt.fillna("").astype(str)
O["a5_status_dist"] = stt.value_counts().to_dict()
unreg = Sc.get("tax_type", pd.Series("", index=Sc.index)).fillna("").astype(str).str.contains("등록되지 않은")
O["a5_unregistered"] = int(unreg.sum()); O["a5_unregistered_rate"] = float(unreg.mean())
O["a5_unreg_checksum_fail"] = int(Sc.loc[unreg, "biz"].map(lambda d: biz_ok(d) is False).sum())
ca.drop(columns=["jur_raw", "biz_raw"]).to_parquet(guard_path(os.path.join(CACHE, "ca.parquet")))
json.dump(O, open(guard_path(os.path.join(CACHE, "s02_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, default=str, indent=0)[:6000])
