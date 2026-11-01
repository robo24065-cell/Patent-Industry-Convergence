# -*- coding: utf-8 -*-
"""C12-T1 s4: (b) 분포·중복·결측·2011 이관 몰림, (d) 시점 정합(지연·창 끝 절단·재선언·갱신일). 입력: s1·s2 캐시. 출력: cache/s4_out.json"""
import os, sys, glob, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(W, "cache", "etsi")
O = {}
ag = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(C, "agg_*.parquet")))], ignore_index=True)
dg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(C, "dipg_*.parquet")))], ignore_index=True)
dg = dg.groupby("DIPG_ID").agg(cc=("cc", "first"), fam=("fam", "first"), comp=("comp", "first"), refl_min=("refl_min", "min"), refl_max=("refl_max", "max"),
                               n_refl=("n_refl", "max"), sig_min=("sig_min", "min"), n_decl=("n_decl", "max"), n_rows=("n_rows", "sum")).reset_index()
done = json.load(open(os.path.join(C, "done.json")))
O["source"] = dict(rows=done["rows"], csv_bytes=done["csv_bytes"], zip_time=done["zip_time"], n_dipg=int(len(dg)), n_family=int(dg.fam.nunique()),
                   max_refl=str(ag.refl_y.max()), refl_blank=int(ag.refl_blank.sum()), sig_blank=int(ag.sig_blank.sum()))
# (b) 국가·KR 비중
cc = ag.groupby("cc").rows.sum().sort_values(ascending=False)
O["rows_by_cc_top10"] = cc.head(10).astype(int).to_dict(); O["kr_row_share"] = float(cc.get("KR", 0) / cc.sum())
krfam = set(dg.loc[dg.cc == "KR", "fam"])
O["kr_family_share"] = float(len(krfam) / dg.fam.nunique())
# 반영연도·서명연도(DIPG 최초)
dg["ry"] = dg.refl_min.str[:4]; dg["sy"] = dg.sig_min.str[:4]
O["dipg_by_refl_year"] = dg.ry.value_counts().sort_index().astype(int).to_dict()
O["dipg_by_sig_year"] = dg.sy.value_counts().sort_index().astype(int).to_dict()
O["kr_dipg_by_refl_year"] = dg[dg.cc == "KR"].ry.value_counts().sort_index().astype(int).to_dict()
# 2011 일괄 이관
d11 = dg[dg.ry == "2011"]
O["bulk2011"] = dict(dipg=int(len(d11)), sig_before_2011=int((d11.sy < "2011").sum()), share_of_all_dipg=float(len(d11) / len(dg)),
                     top_refl_dates=d11.refl_min.str[:10].value_counts().head(5).astype(int).to_dict(),
                     top_date_share=float(d11.refl_min.str[:10].value_counts().iloc[0] / len(d11)))
# 연도별 최대 단일 반영일 몰림(2012–2026)
spk = {}
for y in range(2012, 2027):
    q = dg[dg.ry == str(y)]
    if len(q):
        vc = q.refl_min.str[:10].value_counts()
        spk[y] = dict(n=int(len(q)), top_date=vc.index[0], top_share=round(float(vc.iloc[0] / len(q)), 4))
O["max_single_day_share_by_year"] = spk
# 프로젝트·세대
pj = ag.groupby("proj").rows.sum().sort_values(ascending=False)
O["rows_by_project_top12"] = pj.head(12).astype(int).to_dict()
cell = ["UMTS", "LTE", "3GPP 5G NR", "5G", "GSM", "3GPP", "GERAN", "AMRWB", "TD-SCDMA", "New Core(NC)", "LTE-A", "IMT-2000", "UTRA", "E-UTRA", "NB-IOT", "3GPP LTE"]
O["cellular_named_share_rows"] = float(pj[pj.index.isin(cell)].sum() / pj.sum())
kag = ag[ag.cc == "KR"]; pjk = kag.groupby("proj").rows.sum()
O["cellular_named_share_kr_rows"] = float(pjk[pjk.index.isin(cell)].sum() / pjk.sum())
g = ag.groupby("gen").rows.sum(); gk = kag.groupby("gen").rows.sum()
O["gen_rows_share"] = {"5G 포함": float(g[g.index.str[0] == "1"].sum() / g.sum()), "4G 포함": float(g[g.index.str[1] == "1"].sum() / g.sum()),
                       "세대 표시 없음": float(g.get("0000", 0) / g.sum())}
O["gen_kr_rows_share"] = {"5G 포함": float(gk[gk.index.str[0] == "1"].sum() / gk.sum()), "4G 포함": float(gk[gk.index.str[1] == "1"].sum() / gk.sum()),
                          "세대 표시 없음": float(gk.get("0000", 0) / gk.sum())}
# 기업(패밀리 기준)
fm = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(C, "fam_*.parquet")))], ignore_index=True)
fm = fm.groupby(["DIPG_PATF_ID", "COMP_LEGAL_NAME"]).agg(refl=("refl", "min"), refl5=("refl5", "min"), refl4=("refl4", "min"), sig=("sig", "min"), kr=("kr", "max")).reset_index()
fm.to_parquet(os.path.join(W, "cache", "fam_company.parquet"), index=False)
cf = fm.groupby("COMP_LEGAL_NAME").DIPG_PATF_ID.nunique().sort_values(ascending=False)
O["families_by_company_top10"] = cf.head(10).astype(int).to_dict(); O["top5_family_share"] = float(cf.head(5).sum() / fm.DIPG_PATF_ID.nunique())
# 중복: 같은 패밀리를 여러 기업이 선언, 같은 DIPG 여러 선언·여러 반영일
O["dup"] = dict(families_multi_company=int((fm.groupby("DIPG_PATF_ID").size() > 1).sum()),
                dipg_multi_decl_share=float((dg.n_decl > 1).mean()), dipg_multi_refl_share=float((dg.n_refl > 1).mean()),
                kr_dipg_multi_refl_share=float((dg[dg.cc == "KR"].n_refl > 1).mean()))
# 서명 > 반영 이상치
O["sig_after_refl_dipg"] = int((dg.sig_min.str[:10] > dg.refl_min.str[:10]).sum())
# ---------------- (d) 26429 연결 특허 시점(완전 키)
M = pd.read_parquet(os.path.join(W, "cache", "link_matches.parquet")); M = M[M.lvl <= 2]
P = M.groupby("appl_no").agg(refl=("refl", "min"), sig=("sig", "min"), ksic=("ksic", "first"), ay=("ay", "first"), n_iprd=("iprd", "nunique"),
                            n_refl_dates=("refl", "nunique")).reset_index()
P26 = P[P.ksic == "26429"].copy()
def days(a, b):
    return (pd.to_datetime(a.astype(int).astype(str)) - pd.to_datetime(b.astype(int).astype(str))).dt.days
P26["lag"] = days(P26.refl, P26.sig)
O["kr26429_lag"] = {}
for lab, q in [("반영 2012–2022", P26[P26.refl.between(20120101, 20221231)]), ("반영 2023–2024(최종창)", P26[P26.refl.between(20230101, 20241231)]),
               ("반영 2011", P26[P26.refl.between(20110101, 20111231)])]:
    O["kr26429_lag"][lab] = dict(n=int(len(q)), median=float(q.lag.median()), p90=float(q.lag.quantile(0.9)), p99=float(q.lag.quantile(0.99)), max=float(q.lag.max()),
                                 neg=int((q.lag < 0).sum()))
# 창 끝 절단: 2024 서명 중 2025+ 반영, 2022 서명 중 2023+ 반영(최종시험 단위 기준에서 '이미 서명했지만 반영 전' 몫)
sigP = M.groupby("appl_no").agg(sig=("sig", "min"), refl=("refl", "min"), ksic=("ksic", "first")).reset_index()
s26 = sigP[sigP.ksic == "26429"]
O["truncation"] = dict(sig2024_n=int(s26.sig.between(20240101, 20241231).sum()),
                       sig2024_refl2025plus=int((s26.sig.between(20240101, 20241231) & (s26.refl >= 20250101)).sum()),
                       sig2022_refl2023plus=int((s26.sig.between(20220101, 20221231) & (s26.refl >= 20230101)).sum()),
                       sig2022_n=int(s26.sig.between(20220101, 20221231).sum()))
O["truncation"]["sig2024_late_share"] = O["truncation"]["sig2024_refl2025plus"] / max(1, O["truncation"]["sig2024_n"])
# 재선언: 26429 특허 중 선언(IPRD) 2개 이상, 반영일 2개 이상
O["kr26429_redecl"] = dict(n=int(len(P26)), multi_iprd_share=float((P26.n_iprd > 1).mean()), multi_refl_share=float((P26.n_refl_dates > 1).mean()))
# 2011 이관분만으로 선언된 26429 특허(S4 대상): 최초 반영 2011이고 최초 서명 < 2011
O["kr26429_bulk2011_only"] = int(((P26.refl // 10000) == 2011).sum()); O["kr26429_bulk2011_sig_lt2011"] = int((((P26.refl // 10000) == 2011) & (P26.sig < 20110101)).sum())
O["kr26429_refl_by_year"] = (P26.refl // 10000).value_counts().sort_index().astype(int).to_dict()
O["secs"] = round(time.time() - t0, 1)
json.dump(O, open(os.path.join(W, "cache", "s4_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))
