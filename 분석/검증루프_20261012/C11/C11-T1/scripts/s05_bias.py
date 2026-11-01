# -*- coding: utf-8 -*-
"""(b) 분포별 연결률, (c) 연결 선택 편향 SMD·TVD, 특허 단위 연결 비중. 출력 bias_smd.csv, cache/s05_out.json, cache/linkrate_tables.csv"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np
SIDO = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충청북", "충청남", "충북", "충남", "전라북", "전북", "전라남", "전남", "경상북", "경북", "경상남", "경남", "제주"]
SM = {"충청북": "충북", "충청남": "충남", "전라북": "전북", "전라남": "전남", "경상북": "경북", "경상남": "경남"}
def sido(a):
    a = str(a or "").strip()
    for k in SIDO:
        if a.startswith(k):
            return SM.get(k, k)
    return "미상/기타"
ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet")).set_index("cd")
pan = pd.read_parquet(os.path.join(CACHE, "reg_panel.parquet")).set_index("cd")
A = pd.read_parquet(os.path.join(CACHE, "reg_app.parquet"))
O = {}; ROWS = []; LR = []
pan["in_ca"] = pan.index.isin(ca.index)
pan["biz"] = ca.biz.reindex(pan.index).fillna("")
pan["jur"] = ca.jur.reindex(pan.index).fillna("")
pan["L"] = (pan.biz != "").astype(int)
pan["Lj"] = (pan.jur != "").astype(int)
pan["sido"] = pan.addr.map(sido)
pan["log_pat"] = np.log1p(pan.n_pat)
for d in ["1", "2", "3", "4", "5", "6"]:
    s = pan[pan.index.str[:1] == d]
    O[f"prefix{d}"] = dict(codes=len(s), in_ca=float(s.in_ca.mean()), biz=float(s.L.mean()), jur=float(s.Lj.mean()), example_names=s.reg_name.head(0).tolist())
# 국내 법인 = 코드 첫 자리 1 & 국적 대한민국 (PREREG 고정)
U = pan[(pan.index.str[:1] == "1") & (pan.natl == "대한민국")].copy()
O["U_codes"] = len(U); O["U_in_ca"] = float(U.in_ca.mean()); O["U_biz"] = float(U.L.mean()); O["U_jur"] = float(U.Lj.mean())
O["U_not_in_ca"] = int((~U.in_ca).sum())
for lab, col in [("사업자번호 연결(L)", "L"), ("법인번호 연결(Lj)", "Lj")]:
    a, b = U[U[col] == 1], U[U[col] == 0]
    for v, nm in [("first_ay", "첫 출원연도"), ("last_ay", "마지막 출원연도"), ("log_pat", "log(1+제공범위 등록특허 수)"), ("co_share", "공동출원 비중")]:
        ROWS.append(dict(comparison=f"국내 법인 출원인(코드1·KR) {lab} 1 vs 0", metric="SMD " + nm, value=round(smd(a[v], b[v]), 4),
                         mean_linked=round(float(a[v].mean()), 4), mean_unlinked=round(float(b[v].mean()), 4), n_linked=len(a), n_unlinked=len(b),
                         flag="편향(|SMD|>0.25)" if abs(smd(a[v], b[v])) > 0.25 else ""))
    t = tvd(a.ksic, b.ksic)
    ROWS.append(dict(comparison=f"국내 법인 출원인(코드1·KR) {lab} 1 vs 0", metric="TVD 최빈 제공 KSIC 5자리", value=round(t, 4), n_linked=len(a), n_unlinked=len(b), flag="업종 편향(>0.10)" if t > 0.10 else ""))
    t2 = tvd(a.sido, b.sido)
    ROWS.append(dict(comparison=f"국내 법인 출원인(코드1·KR) {lab} 1 vs 0", metric="TVD 시도(최근 출원 주소)", value=round(t2, 4), n_linked=len(a), n_unlinked=len(b), flag="(>0.10)" if t2 > 0.10 else ""))
# 특허 가중 SMD (특허 수 가중으로 보면?)
a, b = U[U.L == 1], U[U.L == 0]
for v in ["first_ay", "last_ay"]:
    wa, wb = a.n_pat, b.n_pat
    ma, mb = np.average(a[v], weights=wa), np.average(b[v], weights=wb)
    va = np.average((a[v] - ma) ** 2, weights=wa); vb = np.average((b[v] - mb) ** 2, weights=wb)
    ROWS.append(dict(comparison="국내 법인 출원인 L 1 vs 0 (특허 수 가중)", metric="SMD " + v, value=round((ma - mb) / np.sqrt((va + vb) / 2), 4), mean_linked=round(ma, 3), mean_unlinked=round(mb, 3), n_linked=len(a), n_unlinked=len(b)))
# 연결률 표(b)
def lr(name, key):
    t = U.groupby(key, observed=True).agg(codes=("L", "size"), biz_rate=("L", "mean"), jur_rate=("Lj", "mean"), in_ca=("in_ca", "mean"), patents=("n_pat", "sum"))
    for k, r in t.iterrows():
        LR.append(dict(table=name, level=str(k), **{c: (round(float(r[c]), 4) if c not in ("codes", "patents") else int(r[c])) for c in t.columns}))
lr("첫 출원연도", pd.cut(U.first_ay, [1900, 2004, 2008, 2012, 2016, 2020, 2026]))
lr("마지막 출원연도", pd.cut(U.last_ay, [1900, 2004, 2008, 2012, 2016, 2020, 2026]))
U["dec"] = pd.qcut(U.n_pat.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
lr("특허 수 십분위", U.dec)
lr("최빈 제공 KSIC", U.ksic)
lr("시도", U.sido)
# 법인번호 앞 4자리·법인 종류(CA 전체, 법인번호 보유 행) → 사업자번호 보유율
C = ca[ca.jur != ""].copy()
C["Lb"] = (C.biz != "").astype(int)
# 등기관서 4자리 → 시도: 등록원부 주소 최빈 시도(CA 연결 코드)로 경험적 대응
m = U[U.Lj == 1].assign(jreg=lambda d: d.jur.str[:4])
reg2sido = m.groupby("jreg").sido.agg(lambda s: s.value_counts().index[0])
reg2n = m.groupby("jreg").size()
reg2share = m.groupby("jreg").sido.agg(lambda s: s.value_counts(normalize=True).iloc[0])
C["jsido"] = C.jreg.map(reg2sido).fillna("미대응")
t = C.groupby("jsido").Lb.agg(["size", "mean"])
for k, r in t.iterrows():
    LR.append(dict(table="CA 법인번호 등기관서(앞4자리→경험적 시도)", level=k, codes=int(r["size"]), biz_rate=round(float(r["mean"]), 4)))
t = C.groupby("jkind").Lb.agg(["size", "mean"]).query("size>=50")
for k, r in t.iterrows():
    LR.append(dict(table="CA 법인 종류(법인번호 5–6자리)", level=k, codes=int(r["size"]), biz_rate=round(float(r["mean"]), 4)))
C["iy"] = C.index.str[1:5]
t = C.groupby("iy").Lb.agg(["size", "mean"])
for k, r in t.iterrows():
    LR.append(dict(table="CA 출원인코드 발급연도(2–5자리)", level=k, codes=int(r["size"]), biz_rate=round(float(r["mean"]), 4)))
O["jreg_mapping_purity_median"] = float(reg2share[reg2n >= 30].median())
O["jreg_top"] = {k: [reg2sido[k], int(reg2n[k]), round(float(reg2share[k]), 3)] for k in reg2n.sort_values(ascending=False).index[:12]}
# 지역 정합: 법인번호 등기관서 시도 ≠ 최근 출원 주소 시도 (이전 가능, 참고)
m["jsido"] = m.jreg.map(reg2sido)
O["jreg_sido_mismatch_rate"] = float((m.jsido != m.sido).mean())
# 특허 단위: 제공범위 KR 등록특허 중 연결된 출원인이 있는 특허
A["L"] = A.cd.map(ca.biz).fillna("").ne("").astype(int)
A["Lj"] = A.cd.map(ca.jur).fillna("").ne("").astype(int)
A["dom"] = A.cd.str[:1].eq("1") & A.natl.eq("대한민국")
first = A.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no")
O["pat_any_biz"] = float(A.groupby("rgst_no").L.max().mean())
O["pat_first_biz"] = float(first.L.mean())
O["pat_first_jur"] = float(first.Lj.mean())
O["pat_first_dom_corp"] = float(first.dom.mean())
O["pat_first_dom_corp_biz"] = float(first.loc[first.dom, "L"].mean())
O["pat_any_dom_corp"] = float(A.groupby("rgst_no").dom.max().mean())
O["pat_first_biz_by_ay"] = first.groupby(pd.cut(first.ay, [1990, 2004, 2008, 2012, 2016, 2020, 2026]), observed=True).L.mean().round(4).rename(index=str).to_dict()
O["pat_domcorp_first_biz_by_ay"] = first[first.dom].groupby(pd.cut(first[first.dom].ay, [1990, 2004, 2008, 2012, 2016, 2020, 2026]), observed=True).L.mean().round(4).rename(index=str).to_dict()
# 최근 발급 코드(2025–26)의 등록원부 출현과 DART 대조
rec = ca[ca.index.str[1:5].isin(["2025", "2026"])]
O["recent_codes_in_registry"] = int(rec.index.isin(pan.index).sum())
O["recent_codes_jurfail_in_registry"] = int(rec[rec.jur_ok == False].index.isin(pan.index).sum())
dc = ro(DART_DB)
D = pd.read_sql("select jurir_no, bizr_no from company", dc)
D["jur"] = D.jurir_no.fillna("").str.replace(r"\D", "", regex=True); D["biz"] = D.bizr_no.fillna("").str.replace(r"\D", "", regex=True)
mm = rec[rec.biz != ""].reset_index().merge(D, on="biz", suffixes=("_c", "_d"))
O["recent_codes_dart_by_biz"] = len(mm)
O["recent_codes_dart_jur_agree"] = int((mm.jur_c == mm.jur_d).sum())
O["recent_codes_dart_jur_agree_fail_rows"] = int(((mm.jur_c == mm.jur_d) & (mm.jur_ok == False)).sum())
O["recent_codes_dart_jur_disagree_fail_rows"] = int(((mm.jur_c != mm.jur_d) & (mm.jur_ok == False)).sum())
# 시험 모집단 관련: 공식 트랙 코드 중 최근 발급 코드 수(N4·N7 예측 파일)
for tr, fp in [("N4", os.path.join(MC, r"N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet")), ("N7", os.path.join(MC, r"N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet"))]:
    try:
        Pp = pd.read_parquet(fp)
        cds = Pp.applicantCd.astype(str)
        O[f"{tr}_codes"] = int(cds.nunique())
        O[f"{tr}_codes_issued_2025_26"] = int(cds.str[1:5].isin(["2025", "2026"]).sum())
        O[f"{tr}_codes_jur_checksum_fail"] = int(cds.map(ca.jur_ok).eq(False).sum())
    except Exception as e:
        O[f"{tr}_err"] = str(e)
pd.DataFrame(ROWS).to_csv(guard_path(os.path.join(W, "bias_smd.csv")), index=False, encoding="utf-8-sig")
pd.DataFrame(LR).to_csv(guard_path(os.path.join(CACHE, "linkrate_tables.csv")), index=False, encoding="utf-8-sig")
U.reset_index()[["cd", "first_ay", "last_ay", "n_pat", "co_share", "ksic", "reg_name", "n_names", "sido", "in_ca", "L", "Lj"]].to_parquet(guard_path(os.path.join(CACHE, "U.parquet")))
json.dump(O, open(guard_path(os.path.join(CACHE, "s05_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
print(pd.DataFrame(ROWS).to_string())
