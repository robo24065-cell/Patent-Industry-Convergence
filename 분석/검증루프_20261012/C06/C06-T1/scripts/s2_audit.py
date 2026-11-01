# -*- coding: utf-8 -*-
r"""C06-T1 s2: 데이터 검증 (b)(c)(d). 입력 cache/firm_U.parquet, docs_U.parquet, nps_hist_U.parquet, _shared\ext 연계표.
쓰기: bias_dart.csv, out/s2_*.json|csv, cache/ksic_sets.json, cache/firm_flags.parquet"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(HERE, "cache"); OUT = os.path.join(HERE, "out")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"; EXT = os.path.join(MC, r"_shared\ext")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
S50 = set(KSIC50); J = {}
U = pd.read_parquet(os.path.join(CACHE, "firm_U.parquet")); X = pd.read_parquet(os.path.join(CACHE, "docs_U.parquet"))
H = pd.read_parquet(os.path.join(CACHE, "nps_hist_U.parquet"))
U["dart5"] = U.dart.where(U.dart.str.len() == 5)
U["G"] = np.where(U.dart5.isin(S50), "G1", np.where(U.dart_j.notna(), "G2", "G3"))
J["groups"] = U.G.value_counts().to_dict(); J["G1_share_U"] = round(float((U.G == "G1").mean()), 4); J["DART_linked_share_U"] = round(float((U.G != "G3").mean()), 4)
J["G1_cls"] = U[U.G == "G1"].cls_j.value_counts().to_dict(); J["G2_cls"] = U[U.G == "G2"].cls_j.value_counts().to_dict()
J["G2_code_len"] = U[U.G == "G2"].dart_j.str.len().value_counts().to_dict()
J["G2_5digit_not50"] = int(((U.G == "G2") & (U.dart_j.str.len() == 5)).sum())

# ---------------- (b) 편향
def sizebin(x):
    return pd.cut(x, [-1, 9, 49, 299, 1e12], labels=["<10", "10-49", "50-299", ">=300"]).astype(object).fillna("미상")
U["size"] = sizebin(U.n_emp); U["lemp"] = np.log1p(U.n_emp)
U["l_eval"] = np.log(U.n_pat_eval); U["l_all"] = np.log(U.n_pat_all)
U["prov2"] = U.prov_mode.str[:2]
GRP = {"26": "26", "27": "27", "28": "28", "29": "29-30", "30": "29-30", "20": "20-21", "21": "20-21"}
U["kgrp"] = U.prov2.map(lambda s: GRP.get(s, "기타") if isinstance(s, str) else "미상")
def smd(a, b):
    a, b = a.dropna(), b.dropna(); s = np.sqrt((a.var() + b.var()) / 2); return round(float((a.mean() - b.mean()) / s), 3) if s > 0 else np.nan
def tvd(a, b):
    pa = a.fillna("NA").value_counts(normalize=True); pb = b.fillna("NA").value_counts(normalize=True)
    return round(float(0.5 * pa.subtract(pb, fill_value=0).abs().sum()), 3)
rows = []
comps = {"G1 vs U\\G1": (U.G == "G1", U.G != "G1"), "G1 vs G3(DART 미연결)": (U.G == "G1", U.G == "G3"),
         "G2 vs G3": (U.G == "G2", U.G == "G3"), "DART연결(G1+G2) vs G3": (U.G != "G3", U.G == "G3")}
for nm, (ma, mb) in comps.items():
    A, B = U[ma], U[mb]
    r = {"비교": nm, "기업A": len(A), "기업B": len(B)}
    for v, lab in (("lemp", "SMD log(1+NPS가입자)"), ("age_proxy", "SMD 업력대리(2019-첫출원)"), ("l_eval", "SMD log 평가특허수"), ("l_all", "SMD log 전체KR특허수")):
        r[lab] = smd(A[v], B[v])
    r["A NPS가입자 중앙값"] = float(A.n_emp.median()); r["B NPS가입자 중앙값"] = float(B.n_emp.median())
    r["A 규모미상%"] = round(float((A["size"] == "미상").mean() * 100), 1); r["B 규모미상%"] = round(float((B["size"] == "미상").mean() * 100), 1)
    r["A ≥300%"] = round(float((A["size"] == ">=300").mean() * 100), 1); r["B ≥300%"] = round(float((B["size"] == ">=300").mean() * 100), 1)
    r["A 평가특허 평균"] = round(float(A.n_pat_eval.mean()), 2); r["B 평가특허 평균"] = round(float(B.n_pat_eval.mean()), 2)
    r["A 업력대리 평균"] = round(float(A.age_proxy.mean()), 2); r["B 업력대리 평균"] = round(float(B.age_proxy.mean()), 2)
    for v, lab in (("size", "TVD 규모5구간"), ("prov_mode", "TVD 제공KSIC50 최빈"), ("prov2", "TVD 제공KSIC 2자리"), ("kgrp", "TVD KSIC그룹6"),
                   ("sido", "TVD 시도(제공 주소)"), ("jur2", "TVD 법인번호 관할2자리")):
        r[lab] = tvd(A[v], B[v])
    r["A 서울%"] = round(float((A.sido == "서울").mean() * 100), 1); r["B 서울%"] = round(float((B.sido == "서울").mean() * 100), 1)
    r["A 경기%"] = round(float((A.sido == "경기").mean() * 100), 1); r["B 경기%"] = round(float((B.sido == "경기").mean() * 100), 1)
    rows.append(r)
BIAS = pd.DataFrame(rows)
BIAS.to_csv(os.path.join(HERE, "bias_dart.csv"), index=False, encoding="utf-8-sig")
# 분포 표(규모·KSIC그룹·시도) G1 vs U
dist = []
for v in ("size", "kgrp", "sido", "prov2"):
    for gname, m in (("U", U.G.notna()), ("G1", U.G == "G1"), ("G3", U.G == "G3")):
        vc = U.loc[m, v].fillna("NA").value_counts(normalize=True)
        for k, p in vc.items(): dist.append({"변수": v, "집단": gname, "값": k, "비율": round(float(p), 4)})
pd.DataFrame(dist).pivot_table(index=["변수", "값"], columns="집단", values="비율").reset_index().to_csv(os.path.join(OUT, "s2_dist.csv"), index=False, encoding="utf-8-sig")
# G1 상장 구분
g1 = U[U.G == "G1"]
J["G1_listed_share"] = round(float(g1.cls_j.isin(["Y", "K", "N"]).mean()), 4)

# ---------------- (c) 연결 방식
lr = {}
for k in ("dart_j", "dart_b", "dart_n"): lr[k] = round(float(U[k].notna().mean()), 4)
J["link_rate_U"] = lr
both = U[U.dart_j.notna() & U.dart_b.notna()]; J["j_b_same_corp"] = round(float((both.cc_j == both.cc_b).mean()), 4); J["j_b_n"] = len(both)
J["j_b_same_code"] = round(float((both.dart_j == both.dart_b).mean()), 4)
both = U[U.dart_j.notna() & U.dart_n.notna()]; J["j_n_same_corp"] = round(float((both.cc_j == both.cc_n).mean()), 4); J["j_n_n"] = len(both)
J["j_n_same_code"] = round(float((both.dart_j == both.dart_n).mean()), 4)
J["only_b"] = int((U.dart_j.isna() & U.dart_b.notna()).sum()); J["only_n"] = int((U.dart_j.isna() & U.dart_b.isna() & U.dart_n.notna()).sum())
J["jurir_missing_U"] = int((U.jurir.fillna("").str.len() != 13).sum())

# ---------------- (d) KSIC 차수
L = pd.read_csv(os.path.join(EXT, "r7_data_ksic_ksco_revision_link_raw.csv"), encoding="cp949", dtype=str)
L.columns = ["kind", "rev", "lvl", "new", "new_nm", "old", "old_nm"]
L = L[(L.kind == "KSIC") & (L.lvl == "5")].copy()
for c in ("new", "old"): L[c] = L[c].fillna("").str.replace(r"\D", "", regex=True)
L11 = L[L.rev == "11"]; L10 = L[L.rev == "10"]
V11 = set(L11.new) - {""}; V10a = set(L11.old) - {""}; V10b = set(L10.new) - {""}; V9 = set(L10.old) - {""}
V10 = V10a | V10b
J["ksic_counts"] = {"11차(차순11 신)": len(V11), "10차(차순11 구)": len(V10a), "10차(차순10 신)": len(V10b), "10차 두 목록 대칭차": len(V10a ^ V10b), "9차(차순10 구)": len(V9),
                    "L11_rows": len(L11), "L10_rows": len(L10)}
J["ksic50_in_V10"] = int(sum(k in V10 for k in KSIC50)); J["ksic50_in_V11"] = int(sum(k in V11 for k in KSIC50))
m10to11 = L11[L11.old != ""].groupby("old").new.apply(lambda s: sorted(set(s) - {""})).to_dict()
m11to10 = L11[L11.new != ""].groupby("new").old.apply(lambda s: sorted(set(s) - {""})).to_dict()
m9to10 = L10[L10.old != ""].groupby("old").new.apply(lambda s: sorted(set(s) - {""})).to_dict()
ch50 = []
for k in KSIC50:
    to = m10to11.get(k, []); back = sorted({o for n in to for o in m11to10.get(n, [])})
    ch50.append({"ksic10": k, "11차": "|".join(to), "11차→10차 역": "|".join(back), "동일": to == [k] and back == [k]})
CH = pd.DataFrame(ch50); CH.to_csv(os.path.join(OUT, "s2_ksic50_10to11.csv"), index=False, encoding="utf-8-sig")
J["ksic50_identity_10_11"] = int(CH["동일"].sum())
def edition(c):
    a, b, d = c in V9, c in V10, c in V11
    return {(False, True, True): "10∩11", (False, True, False): "10만", (False, False, True): "11만", (True, True, True): "9∩10∩11",
            (True, False, False): "9만", (True, True, False): "9∩10", (True, False, True): "9∩11", (False, False, False): "어디에도없음"}[(a, b, d)]
import sqlite3
c = sqlite3.connect("file:" + r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\외부데이터\DART\dart_company.sqlite" + "?mode=ro", uri=True)
DA = pd.read_sql("select induty_code ind, corp_cls from company", c); c.close()
d5 = DA[DA.ind.str.len() == 5].ind
J["dart_all5_edition"] = d5.map(edition).value_counts().to_dict()
J["G1_edition"] = g1.dart5.map(edition).value_counts().to_dict()
J["G2_5digit_edition"] = U[(U.G == "G2") & (U.dart_j.str.len() == 5)].dart_j.map(edition).value_counts().to_dict()
# 11차만 유효 코드 예시(어떤 10차에서 왔나) — 50개 관련
only11 = d5[d5.map(edition) == "11만"].value_counts()
J["dart_11only_codes_top"] = {k: [int(v), m11to10.get(k, [])] for k, v in only11.head(15).items()}
only10 = d5[d5.map(edition) == "10만"].value_counts()
J["dart_10only_codes_top"] = {k: [int(v), m10to11.get(k, [])] for k, v in only10.head(15).items()}
# 11차 코드 중 10차 50개로 역연결되는 DART 코드(현재 G1에 못 든 기업)
U["S10_11"] = U.dart_j.map(lambda d: m11to10.get(d) if isinstance(d, str) and len(d) == 5 and d in V11 else None)
J["G2_back_to_50"] = int(U[(U.G == "G2")].S10_11.map(lambda s: bool(s) and bool(set(s) & S50)).sum())
# NPS 매핑 코드 차수
nps19 = H[H.snap == "201912"]
J["nps2019_k5src"] = nps19.k5src.value_counts().head(8).to_dict()
J["nps_G_edition"] = U[U.nps.isin(S50)].nps.map(edition).value_counts().to_dict()
J["est_after_2019_G1"] = round(float((g1.est_j.fillna("") > "20191231").mean()), 4)
J["est_after_2014_G1"] = round(float((g1.est_j.fillna("") > "20141231").mean()), 4)
J["est_missing_G1"] = int((g1.est_j.fillna("").str.len() != 8).sum())
json.dump({"m10to11": m10to11, "m11to10": m11to10, "m9to10": m9to10, "V9": sorted(V9), "V10": sorted(V10), "V11": sorted(V11)},
          open(os.path.join(CACHE, "ksic_sets.json"), "w", encoding="utf-8"))

# ---------------- 업종 변경(NPS 스냅샷)
def changes(h, snaps):
    hh = h[h.snap.isin(snaps) & h.k5.notna()]
    return hh.groupby("applicantCd").k5.nunique()
post = ["201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
allS = sorted(H.snap.unique())
nch = changes(H, post); nch_all = changes(H, allS)
icc = H[H.snap.isin(post) & H.indcd_c.notna()].groupby("applicantCd").indcd_c.nunique()
U["nps_k5_n_post"] = U.applicantCd.map(nch); U["nps_k5_n_all"] = U.applicantCd.map(nch_all); U["nps_ic_n_post"] = U.applicantCd.map(icc)
U["chg_post"] = U.nps_k5_n_post >= 2; U["chg_all"] = U.nps_k5_n_all >= 2
for gname, m in (("U", U.G.notna()), ("G1", U.G == "G1")):
    sub = U[m]; known = sub.nps_k5_n_post.notna()
    J[f"chg_{gname}"] = {"NPS이력있음": int(known.sum()), "k5변경(2019-2026)": int(sub.chg_post.sum()), "비율(이력있음 중)": round(float(sub.chg_post[known].mean()), 4),
                         "k5변경(2015-2026)": int(sub.chg_all.sum()), "indcd변경(2019-2026)": int((sub.nps_ic_n_post >= 2).sum())}
# 연도쌍별 변경률(G1 연결 기업)
pair = []
g1k = set(U[U.G == "G1"].applicantCd); Uk = set(U.applicantCd)
for a, b in zip(allS[:-1], allS[1:]):
    A = H[(H.snap == a)].set_index("applicantCd").k5; B = H[(H.snap == b)].set_index("applicantCd").k5
    com = A.dropna().index.intersection(B.dropna().index)
    pair.append({"쌍": f"{a}->{b}", "U 공통": len(com), "U 변경률": round(float((A[com] != B[com]).mean()), 4),
                 "G1 공통": int(len([x for x in com if x in g1k])), "G1 변경률": round(float((A[[x for x in com if x in g1k]] != B[[x for x in com if x in g1k]]).mean()), 4) if any(x in g1k for x in com) else None})
pd.DataFrame(pair).to_csv(os.path.join(OUT, "s2_nps_change_pairs.csv"), index=False, encoding="utf-8-sig")
# DART(2026) vs NPS(2019) 5자리 일치(둘 다 있는 G1)
gg = U[(U.G == "G1") & U.nps.notna()]
J["G1_dart_eq_nps2019_5"] = [len(gg), round(float((gg.dart5 == gg.nps).mean()), 4)]
J["G1_dart_eq_nps2019_3"] = round(float((gg.dart5.str[:3] == gg.nps.str[:3]).mean()), 4)
U[["applicantCd", "G", "size", "kgrp", "sido", "chg_post", "chg_all", "nps_k5_n_post", "S10_11", "dart5", "cls_j", "est_j", "dart_b", "dart_n"]].to_parquet(os.path.join(CACHE, "firm_flags.parquet"), index=False)
json.dump(J, open(os.path.join(OUT, "s2_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 250); print(BIAS.T.to_string().encode("ascii", "replace").decode()[:200]); print("done")
