# -*- coding: utf-8 -*-
# 7단계: 원고 3.1.1·3.1.3 민감도판(급감 출원인·단절 서브클래스 제외) + T3·T4·N8 진단(공식 성적 불변)
import sys, os, time, json
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group
t0 = time.time(); res = {}
CCS = ["KR", "US", "JP", "CN", "EP"]
con = duck()
inv = """case when cc='KR' then substr(appl_no,1,2)='10'
 when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
 when cc='JP' then length(appl_no) in (8,17) else true end"""
FIRST = "nullif(trim(split_part(coalesce(applicant,''),'|',1)),'')"
m = con.sql(f"""select cc, appl_no, appl_year yr, ksic, substr(main_ipc,1,4) sub, {FIRST} app, ({inv}) is_inv
 from read_parquet('{META}') where appl_year between 2005 and 2019""").df()
print("loaded", len(m), round(time.time() - t0), flush=True)
K50 = sorted(m.ksic.unique()); assert len(K50) == 50
# 제외 집합
ND = pd.read_csv(os.path.join(CACHE, "s3_name_dropouts.csv"))
CD = pd.read_csv(os.path.join(CACHE, "s3_code_dropouts.csv"), dtype={"applicant_cd": str})
s2 = json.load(open(os.path.join(CACHE, "s2_results.json"), encoding="utf-8"))
sig_subs = sorted(k for k, v in s2["breaks"]["subclass"].items() if v and v["sig"])
U = pd.read_parquet(os.path.join(CACHE, "U_raw.parquet"), columns=["applicant_cd", "appl_no"])
code_appl = set(U[U.applicant_cd.isin(set(CD.applicant_cd))].appl_no)
names_drop = set(ND.name)
HYK = {"현대자동차주식회사", "기아 주식회사"}
res["n_sig_subs"] = len(sig_subs); res["sig_subs"] = sig_subs
VARS = {
    "V0_재현": pd.Series(np.zeros(len(m), bool), index=m.index),
    "V1_이름급감76_KR제외": (m.cc == "KR") & m.app.isin(names_drop),
    "V2_코드급감10_KR제외": (m.cc == "KR") & m.appl_no.isin(code_appl),
    "V3_현대기아이름_KR제외": (m.cc == "KR") & m.app.isin(HYK),
    "V4_단절서브클래스31_5개국제외": m["sub"].isin(set(sig_subs)),
}
VARS["V5_V1+V2+V4"] = VARS["V1_이름급감76_KR제외"] | VARS["V2_코드급감10_KR제외"] | VARS["V4_단절서브클래스31_5개국제외"]
G = pd.Series({k: ksic_to_group(k) for k in K50}); gs = G.value_counts()
K34 = sorted([k for k in K50 if gs[G[k]] > 1]); assert len(K34) == 34


def within_share(vals, grp):
    d = pd.DataFrame({"v": vals, "g": grp}).dropna()
    return float(((d.v - d.groupby("g").v.transform("mean")) ** 2).sum() / ((d.v - d.v.mean()) ** 2).sum())


def tab(d, years, inv_only, uniq):
    x = d[d.yr.isin(years)]
    if inv_only:
        x = x[x.is_inv]
    if uniq:
        x = x.drop_duplicates(["cc", "appl_no", "ksic"])
    return x.groupby(["ksic", "cc"]).size().unstack().reindex(index=K50, columns=CCS).fillna(0)


rows = []
LEAD0 = None
for vn, mask in VARS.items():
    d = m[~mask.values]
    n = tab(d, range(2017, 2020), True, False)
    s = n / n.sum(); r = s.div(s.mean(1), axis=0); lead = r.idxmax(1)
    kr_lead = sorted(r.index[lead == "KR"])
    if LEAD0 is None:
        LEAD0 = kr_lead
    na = tab(d, [2005, 2006, 2007], False, True); nb = tab(d, [2015, 2016, 2017], False, True)
    g_KR = (np.log(nb.KR / 3 + 1) - np.log(na.KR / 3 + 1)) / 10
    xa = tab(d, [2008, 2009], False, True).KR; xb = tab(d, [2016, 2017], False, True).KR
    gw = 100 * (np.log(xb / 2 + 1) - np.log(xa / 2 + 1)) / 8
    row = dict(variant=vn, removed_KR_docs_P3=int(((mask.values) & (m.cc == "KR").values & m.yr.between(2017, 2019).values & m.is_inv.values).sum()),
               kr_leaders=len(kr_lead), kr_rta_gt1=int((r.KR > 1).sum()), lost_leaders="|".join(sorted(set(LEAD0) - set(kr_lead))),
               new_leaders="|".join(sorted(set(kr_lead) - set(LEAD0))), rta_26212_KR=round(r.at["26212", "KR"], 3), lead_26212=lead["26212"],
               rta_31312_KR=round(r.at["31312", "KR"], 3), lead_31312=lead["31312"],
               within_g_KR_pct=round(100 * within_share(g_KR[K34], G[K34]), 1),
               gw_26211=round(gw["26211"], 1), gw_26212=round(gw["26212"], 1), gw_26111=round(gw["26111"], 1))
    rows.append(row); print(row, flush=True)
S = pd.DataFrame(rows)
# 원고 값 대비 판정
ref = dict(kr_leaders=7, kr_rta_gt1=25, within_g_KR_pct=56.1, gw_26211=-6.6, gw_26212=6.2)
S["vs_manuscript"] = ""
for i, r in S.iterrows():
    flips = []
    if r.lead_26212 != "KR": flips.append("OLED 1위 상실")
    if r.lead_31312 != "KR": flips.append("무인항공기 1위 상실")
    if np.sign(r.gw_26211) != np.sign(ref["gw_26211"]) or np.sign(r.gw_26212) != np.sign(ref["gw_26212"]): flips.append("C2620 부호 바뀜")
    if abs(r.within_g_KR_pct - ref["within_g_KR_pct"]) > 10: flips.append("코드 안 비중 |Δ|>10%p")
    S.at[i, "vs_manuscript"] = "; ".join(flips) if flips else "뒤집힘 없음"
res["sensitivity"] = S.to_dict("records")
print("sens done", round(time.time() - t0), flush=True)

# ---- T3 진단: 최종시험 적중 셀 ----
T3 = pd.read_parquet(os.path.join(MC, r"T3_기술셀부상예측\R3\preds\FINAL_R3_t0_2021.parquet"))
k = int(round(0.1 * len(T3)))
T3 = T3.sort_values("p", ascending=False).reset_index(drop=True); T3["top"] = T3.index < k
hit = T3[T3.top & (T3.yb == 1)].copy()
hit["sub"] = hit.mg.str[:4]
res["t3_hits"] = len(hit); res["t3_top"] = k
res["t3_hit_share_sig_sub"] = float(hit["sub"].isin(set(sig_subs)).mean())
res["t3_all_share_sig_sub"] = float(T3.mg.str[:4].isin(set(sig_subs)).mean())
kr = pd.read_parquet(os.path.join(CACHE, "kr_meta.parquet"), columns=["appl_no", "appl_year", "pub_date", "main_ipc", "ksic", "applicant"])
kr["mg"] = kr.main_ipc.str.split("/").str[0].str.strip()
kr["py"] = pd.to_numeric(kr.pub_date.str[:4], errors="coerce")
w = kr[kr.py.between(2019, 2024)]
w = w.assign(drop_nm=w.applicant.isin(names_drop), hyk=w.applicant.isin(HYK), drop_cd=w.appl_no.isin(code_appl))
cell = w.groupby(["ksic", "mg"]).agg(n=("appl_no", "size"), n_dn=("drop_nm", "sum"), n_hk=("hyk", "sum"), n_dc=("drop_cd", "sum")).reset_index()
hit = hit.merge(cell, on=["ksic", "mg"], how="left").fillna({"n": 0, "n_dn": 0, "n_hk": 0, "n_dc": 0})
res["t3_hit_share_any_dropname_doc"] = float((hit.n_dn > 0).mean())
res["t3_hit_share_dropname_majority"] = float((hit.n_dn >= 0.5 * hit.n.clip(lower=1)).mean() * 1) if len(hit) else 0.0
res["t3_hit_share_any_hyundai_kia_doc"] = float((hit.n_hk > 0).mean())
res["t3_hit_share_any_dropcode_doc"] = float((hit.n_dc > 0).mean())
res["t3_hit_dropname_docs_share_of_hit_docs"] = float(hit.n_dn.sum() / max(hit.n.sum(), 1))
# ---- T4·N8 진단: 공식 모집단 중 급감 출원인 문서 비중 ----
pt = pd.read_parquet(os.path.join(MC, r"T4_특허포기예측\cache\patent.parquet"), columns=["rgst_no", "rgstDate", "applNo"])
pt = pt[pt.rgstDate.notna() & (pt.rgstDate.astype(str).str.len() > 0) & (pt.rgst_no.str[:2] == "10")].reset_index(drop=True)
ev = pd.read_parquet(os.path.join(MC, r"T4_특허포기예측\R2\runs_final\F22\ev.parquet"), columns=["pid"])
res["t4_pid_max_lt_len"] = bool(ev.pid.max() < len(pt))
t4 = pt.iloc[ev.pid.values]
nm_map = kr.set_index("appl_no").applicant
res["t4_n"] = len(t4)
res["t4_share_dropname"] = float(t4.applNo.map(nm_map).isin(names_drop).mean())
res["t4_share_dropcode"] = float(t4.applNo.isin(code_appl).mean())
res["t4_share_hyundai_kia_name"] = float(t4.applNo.map(nm_map).isin(HYK).mean())
n8 = pd.read_parquet(os.path.join(MC, r"N8_특허분쟁표적레이더\final_preds_r6.parquet"), columns=["rgst_no", "y"])
rf = pd.read_parquet(os.path.join(CACHE, "rf_patent.parquet"), columns=["rgst_no", "applNo"]).set_index("rgst_no").applNo
n8["appl_no"] = n8.rgst_no.map(rf)
res["n8_n"] = len(n8); res["n8_appl_map_rate"] = float(n8.appl_no.notna().mean())
res["n8_share_dropname"] = float(n8.appl_no.map(nm_map).isin(names_drop).mean())
res["n8_share_dropcode"] = float(n8.appl_no.isin(code_appl).mean())
res["n8_pos_share_dropname"] = float(n8[n8.y == 1].appl_no.map(nm_map).isin(names_drop).mean())
S.to_csv(os.path.join(WD, "sensitivity.csv"), index=False, encoding="utf-8-sig")
res["sec"] = round(time.time() - t0)
json.dump(res, open(os.path.join(CACHE, "s7_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "sensitivity"}, ensure_ascii=False, default=float))
