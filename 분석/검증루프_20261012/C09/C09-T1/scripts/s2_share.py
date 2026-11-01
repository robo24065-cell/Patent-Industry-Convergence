# -*- coding: utf-8 -*-
# 2단계: 연도별 제공 범위 안 비율 (i) 전체 (ii-a) 서브클래스 (ii-b) WIPO 35분야 (iii) 상위 300 출원인코드 + 단절 검정
import sys, os, time, json
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from breaks import break_test, YEARS
import numpy as np, pandas as pd
from collections import Counter
t0 = time.time(); res = {}
kr = pd.read_parquet(os.path.join(CACHE, "kr_meta.parquet"))
kr["sub"] = kr.main_ipc.str[:4]
U = pd.read_parquet(os.path.join(CACHE, "U_raw.parquet"), columns=["applicant_cd", "rgst_no", "rgst_date", "appl_no", "appl_date", "ainfo"])
U["ay"] = pd.to_numeric(U.appl_date.str[:4], errors="coerce")
U["ry"] = pd.to_numeric(U.rgst_date.str[:4], errors="coerce")
res["U_pairs"] = len(U); res["U_patents"] = int(U.rgst_no.nunique())
res["U_appl_no_len13_share"] = float((U.appl_no.str.len() == 13).mean())
inset = set(kr.appl_no)
rf = pd.read_parquet(os.path.join(CACHE, "rf_patent.parquet"), columns=["rgst_no"])
rfset = set(rf.rgst_no)
U["inscope"] = U.appl_no.isin(inset); U["in_rf"] = U.rgst_no.isin(rfset)
P = U.drop_duplicates("rgst_no").copy()     # 특허 단위
res["U_patent_dup_pairs"] = int(len(U) - len(P))
# 제공 KR 등록(10) 중 U에 잡히는 비율 (U의 포괄률)
kr10 = kr[kr.appl_no.str[:2] == "10"]
kr10r = kr10[kr10.rgst_no.notna()]
res["provided_kr10_registered"] = len(kr10r)
Pappl = set(P.appl_no)
res["provided_kr10_registered_in_U_by_applno"] = float(kr10r.appl_no.isin(Pappl).mean())
cov_y = kr10r.assign(inU=kr10r.appl_no.isin(Pappl)).groupby("appl_year").inU.mean()
res["provided_kr10_reg_in_U_by_ay"] = {int(k): round(float(v), 4) for k, v in cov_y.loc[2008:2023].items()}
# (i) 출원연도별
g = P[P.ay.between(2005, 2024)].groupby("ay").agg(n_den=("rgst_no", "size"), n_in=("inscope", "sum"), n_in_rf=("in_rf", "sum"))
g["share"] = g.n_in / g.n_den; g["share_rf"] = g.n_in_rf / g.n_den
rows = [dict(level="all", key="ALL", year=int(y), n_in=int(r.n_in), n_den=int(r.n_den), share=r.share, share_rf=r.share_rf) for y, r in g.iterrows()]
bt = {}
bt["all"] = break_test(g.share.to_dict())
bt["all_rf"] = break_test(g.share_rf.to_dict())
print("(i)", bt["all"], flush=True)
# 등록연도 기준 포괄률: U 대 KOSIS 특허 등록
KO = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\download\kosis")
a6 = pd.DataFrame(json.load(open(os.path.join(KO, "data_TX_13801_A006.json"), encoding="utf-8")))
a6p = a6[a6.C1_NM == "특허"]
kos = pd.Series(a6p.DT.astype(int).values, index=a6p.PRD_DE.astype(int).values)
gr = P.groupby("ry").agg(U_reg=("rgst_no", "size"), U_reg_in=("inscope", "sum"))
rfp = pd.read_parquet(os.path.join(CACHE, "rf_patent.parquet"), columns=["rgst_no", "rgstDate"])
rfp["ry"] = pd.to_numeric(rfp.rgstDate.str[:4], errors="coerce")
gr["rf_reg"] = rfp.groupby("ry").size()
gr["kosis_patent_reg"] = kos
gr = gr.loc[2010:2025]
gr["U_cov"] = gr.U_reg / gr.kosis_patent_reg; gr["rf_over_kosis"] = gr.rf_reg / gr.kosis_patent_reg; gr["U_in_over_kosis"] = gr.U_reg_in / gr.kosis_patent_reg
res["by_rgst_year"] = gr.round(4).reset_index().to_dict("records")
# (ii-a) 서브클래스: 범위 안 U 특허(서브클래스) / U 전체(연도) x 1000
Pin = P[P.inscope].merge(kr[["appl_no", "sub", "main_ipc", "ksic", "applicant"]], on="appl_no", how="left")
base = Pin[Pin.ay.between(2010, 2016)].groupby("sub").size().sort_values(ascending=False)
top100 = base.head(100).index.tolist()
sc = Pin[Pin.ay.between(2008, 2023)].groupby(["sub", "ay"]).size().unstack(fill_value=0)
den = g.n_den
bt_sub = {}
for s in top100:
    ser = (sc.loc[s] / den.loc[sc.columns] * 1000)
    for y in range(2008, 2024):
        rows.append(dict(level="subclass_per1000U", key=s, year=y, n_in=int(sc.loc[s].get(y, 0)), n_den=int(den.get(y, 0)), share=float(ser.get(y, np.nan))))
    bt_sub[s] = break_test(ser.to_dict())
bt["subclass"] = bt_sub
print("(ii-a) done", round(time.time() - t0), flush=True)
# (ii-b) WIPO 35분야: 제공 KR 특허(10) 공개연도별 / WIPO KR 관청 공개
cc = pd.read_csv(os.path.join(MC, r"_shared\ext\quality_wipo_ipc_field_concordance.csv"), dtype=str)
cc["wipo_field"] = cc.wipo_field.astype(int)
k = kr10.copy(); k["py"] = pd.to_numeric(k.pub_date.str[:4], errors="coerce")
k["grp"] = k.main_ipc.str[4:].str.split("/").str[0].str.strip()
g1 = cc[cc.main_group.notna() & (cc.main_group != "")][["subclass", "main_group", "wipo_field"]].rename(columns={"subclass": "sub", "main_group": "grp", "wipo_field": "f1"})
g2 = cc[cc.main_group.isna() | (cc.main_group == "")][["subclass", "wipo_field"]].rename(columns={"subclass": "sub", "wipo_field": "f2"}).drop_duplicates("sub")
k = k.merge(g1.drop_duplicates(["sub", "grp"]), on=["sub", "grp"], how="left").merge(g2, on="sub", how="left")
k["f"] = k.f1.fillna(k.f2).fillna(0).astype(int)
res["kr10_pub_missing_share_by_ay"] = {int(a): round(float(v), 4) for a, v in k.groupby("appl_year").py.apply(lambda s: s.isna().mean()).loc[2008:2023].items()}
res["kr10_unmapped_field_share"] = float((k.f == 0).mean())
W = pd.read_csv(os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\download\wipo\pi\dc_indicator_patent_4_publication_by_technology.csv"))
wk = W[W.office == "KR"].groupby(["year", "tec_id"])["count"].sum()
pk = k[k.py.between(2005, 2022)].groupby(["py", "f"]).size()
bt_f = {}
for f in range(1, 36):
    ser = {}
    for y in range(2008, 2023):
        num = int(pk.get((y, f), 0)); dd = float(wk.get((y, f), np.nan))
        ser[y] = num / dd if dd and dd > 0 else np.nan
        rows.append(dict(level="wipo_field_pub", key=str(f), year=y, n_in=num, n_den=dd, share=ser[y]))
    bt_f[str(f)] = break_test(ser)
tot = {y: pk.loc[y].sum() / wk.loc[y].loc[1:35].sum() for y in range(2008, 2023)}
for y, v in tot.items():
    rows.append(dict(level="wipo_field_pub", key="ALL", year=y, n_in=int(pk.loc[y].sum()), n_den=float(wk.loc[y].loc[1:35].sum()), share=v))
bt["wipo_all"] = break_test(tot); bt["wipo_field"] = bt_f
print("(ii-b)", bt["wipo_all"], flush=True)
# (iii) 상위 300 출원인코드
Up = U[U.ay.between(2005, 2024)]
base_a = Up[Up.inscope & Up.ay.between(2010, 2016)].groupby("applicant_cd").size().sort_values(ascending=False)
top300 = base_a.head(300).index.tolist()
ga = Up[Up.applicant_cd.isin(top300)].groupby(["applicant_cd", "ay"]).agg(n_den=("rgst_no", "size"), n_in=("inscope", "sum")).reset_index()
ga["share"] = ga.n_in / ga.n_den
for r in ga.itertuples():
    rows.append(dict(level="applicant_cd", key=r.applicant_cd, year=int(r.ay), n_in=int(r.n_in), n_den=int(r.n_den), share=r.share))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(WD, "inscope_share_by_year.csv"), index=False, encoding="utf-8-sig")
ga.to_parquet(os.path.join(CACHE, "s2_applicant_year.parquet"), index=False)
pd.Series(top300, name="applicant_cd").to_csv(os.path.join(CACHE, "s2_top300.csv"), index=False)
Pin.to_parquet(os.path.join(CACHE, "s2_Pin.parquet"), index=False)
# 판정 요약
ball = bt["all"]


def near(b, ref):
    return b is not None and ref is not None and abs(b["b"] - ref["b"]) <= 1 and np.sign(b["delta"]) == np.sign(ref["delta"]) and b["sig"]


res["n_sub_sig_any"] = sum(1 for v in bt_sub.values() if v and v["sig"])
res["n_sub_sig_near_all"] = sum(1 for v in bt_sub.values() if near(v, ball))
res["n_field_sig_any"] = sum(1 for v in bt_f.values() if v and v["sig"])
res["n_field_sig_near_all"] = sum(1 for v in bt_f.values() if near(v, ball))
res["sub_sig_break_years"] = {int(k2): v for k2, v in Counter(v["b"] for v in bt_sub.values() if v and v["sig"]).items()}
res["field_sig_break_years"] = {int(k2): v for k2, v in Counter(v["b"] for v in bt_f.values() if v and v["sig"]).items()}
res["regime_change"] = bool(ball and ball["sig"] and (res["n_sub_sig_near_all"] >= 20 or res["n_field_sig_near_all"] >= 7))
res["sec"] = round(time.time() - t0)
json.dump(dict(res=res, breaks=bt), open(os.path.join(CACHE, "s2_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, default=float)[:4000])
print(g.loc[2008:2023].round(4).to_string())
