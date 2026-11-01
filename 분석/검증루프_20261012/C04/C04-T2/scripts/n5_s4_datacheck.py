# -*- coding: utf-8 -*-
"""C04-T2 N5 선언 원천(ETSI ISLD 추출본) 범위 점검 (a)~(d). 입력: n5_s1·n5_s2 산출, kr_patents(전체 KSIC, 로컬). 출력 cache/n5_s4_out.json"""
import os, re, glob, json, time, zipfile
import numpy as np, pandas as pd
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
N5 = os.path.join(BASE, r"분석\모델경쟁_20260927\N5_표준특허선언레이더")
E = os.path.join(W, "cache", "etsi")
R = {}
zi = zipfile.ZipFile(os.path.join(EXT, "trade_etsi_ISLD-export.zip")).getinfo("ISLD-export.csv")
R["a_source"] = dict(file="_shared\\ext\\trade_etsi_ISLD-export.zip (docbox.etsi.org/IPR/Open/ISLD-export.zip, 로그인 불필요)", csv_bytes=zi.file_size,
                     zip_internal_timestamp="%04d-%02d-%02d %02d:%02d" % zi.date_time[:5], rows=json.load(open(os.path.join(E, "done.json")))["rows"],
                     sources_md_rows=4984299)
ag = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(E, "agg_part_*.parquet")))], ignore_index=True)
dp = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(E, "dipg_part_*.parquet")))], ignore_index=True)
dp = dp.groupby("DIPG_ID").agg(cc=("cc", "first"), fam=("fam", "first"), comp=("comp", "first"), refl=("refl", "min"), sig=("sig", "min")).reset_index()
R["a_source"]["n_dipg"] = int(len(dp)); R["a_source"]["n_family"] = int(dp.fam.replace("", np.nan).nunique())
# (b) 분포: 국가, 표준 유형(WI_Type), 프로젝트, 반영연도, 기업 집중
by_cc = ag.groupby("cc").rows.sum().sort_values(ascending=False)
R["b_rows_by_cc_top10"] = {k: int(v) for k, v in by_cc.head(10).items()}
R["b_note_dipg"] = "DIPG_ID는 선언 안의 특허 묶음(패밀리 구성원 여러 국가 행)이라 국가별 DIPG 수는 KR 구성원이 있는 묶음 수로만 센다"
wi = ag.groupby("wi").rows.sum().sort_values(ascending=False)
R["b_rows_by_WI_Type"] = {k: int(v) for k, v in wi.head(10).items()}
pj = ag.groupby("proj").rows.sum().sort_values(ascending=False)
R["b_rows_by_project_top10"] = {k: int(v) for k, v in pj.head(10).items()}
R["b_project_share_3GPP"] = float(pj[[p for p in pj.index if "3GPP" in p.upper()]].sum() / pj.sum())
dp["refl_y"] = dp.refl.str[:4]; dp["sig_y"] = dp.sig.str[:4]
R["b_dipg_by_refl_year"] = {k: int(v) for k, v in dp.refl_y.value_counts().sort_index().items()}
R["b_missing"] = dict(refl_blank=int((dp.refl == "").sum()), sig_blank=int((dp.sig == "").sum()), comp_blank=int((dp.comp == "").sum()))
gc = dp.comp.value_counts()
R["b_company_top5_share_all_dipg"] = dict(share=float(gc.head(5).sum() / gc.sum()), top5={k: int(v) for k, v in gc.head(5).items()}, n_companies=int(len(gc)))
# KR
kr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(E, "kr_part_*.parquet")))], ignore_index=True)
kd = kr.groupby("DIPG_ID").agg(comp=("COMP_LEGAL_NAME", "first"), refl=("Reflected_Date", "min"), sig=("IPRD_SIGNATURE_DATE", "min"), app=("PATT_APPLICATION_NUMBER", "first"),
                               g4=("4G", lambda x: (x == "1").any()), g5=("5G", lambda x: (x == "1").any())).reset_index()
kc = kd.comp.value_counts()
R["b_KR_company_top5_share"] = dict(share=float(kc.head(5).sum() / kc.sum()), top5={k: int(v) for k, v in kc.head(5).items()}, n_companies=int(len(kc)))
R["b_KR_dup"] = dict(rows=int(len(kr)), dipg=int(kd.shape[0]), distinct_app_pub=int(kr.drop_duplicates(["PATT_APPLICATION_NUMBER", "PUBL_NUMBER"]).shape[0]), trade06_log_KR_patents=34040)
# 26429 매칭·최종창 양성의 선언 기업 집중
M = pd.read_parquet(os.path.join(W, "cache", "n5_kr26429_etsi_matches.parquet"))
M = M[M.strict == 1]
U = pd.read_parquet(os.path.join(W, "cache", "n5_units_indep.parquet"))
pos = U[U.y == 1].appl_no
mpos = M[M.appl_no.isin(pos) & (M.refl > 20221231) & (M.refl <= 20241231)]
first_comp = mpos.sort_values("refl").drop_duplicates("appl_no").comp.value_counts()
R["b_final_positive_declarant_top5"] = dict(share=float(first_comp.head(5).sum() / first_comp.sum()), top5={k: int(v) for k, v in first_comp.head(5).items()}, n_pos=int(len(pos)), n_companies=int(len(first_comp)))
F = pd.read_parquet(os.path.join(N5, "final_preds_r4.parquet"))
pa = F[F.y == 1].app.value_counts()
R["b_final_positive_applicant_top5_share"] = float(pa.head(5).sum() / pa.sum())
# (c) 연결 선택 편향: KR ETSI 특허(출원번호 키) 중 제공 KR 전체(1,087,511)와 매칭된 것 대 안 된 것
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "등록번호", "세세분류코드"])
k["a"] = k.출원번호.astype(str).str.strip(); k["r"] = k.등록번호.astype(str).str.replace(r"\D", "", regex=True)


def akey(s):
    raw = str(s or "").upper().strip(); a = re.sub(r"[^0-9A-Z]", "", raw); b = a[2:] if a.startswith("KR") else a
    if re.fullmatch(r"\d{11}", b): return "10" + b
    d = re.sub(r"\D", "", raw)
    if len(d) == 13 and d.startswith("10"): return d
    if re.search(r"[UY]\d?$", a) or (len(d) == 13 and d.startswith("20")): return "UM"
    return None


kd["akey"] = kd.app.map(akey)
kd["kind"] = np.where(kd.akey == "UM", "실용신안", np.where(kd.akey.isna(), "키 없음", "특허 출원번호"))
ks = set(k.a); kk = k.set_index("a")
kd["matched"] = kd.akey.isin(ks)
kd["ay"] = pd.to_numeric(kd.akey.str[2:6], errors="coerce")
kd["refl_y"] = kd.refl.str[:4]
pk = kd[kd.kind == "특허 출원번호"]
R["c_link"] = dict(KR_dipg=int(len(kd)), kind=kd.kind.value_counts().to_dict(), patent_key_matched_share=float(pk.matched.mean()),
                   matched_by_app_year={str(int(a)): float(v) for a, v in pk.groupby("ay").matched.mean().items() if a >= 2000 and pk[pk.ay == a].shape[0] >= 50},
                   provided_kr_app_year_range=[int(k.출원연도.min()), int(k.출원연도.max())])
top = pk.comp.value_counts().head(8).index
R["c_link"]["matched_share_by_company_top8"] = {c: dict(n=int((pk.comp == c).sum()), matched=float(pk[pk.comp == c].matched.mean())) for c in top}
R["c_link"]["matched_share_by_gen"] = {"5G": float(pk[pk.g5].matched.mean()), "4G_only": float(pk[pk.g4 & ~pk.g5].matched.mean()), "2G3G_or_none": float(pk[~pk.g4 & ~pk.g5].matched.mean())}
uq = pk.drop_duplicates("akey")
R["c_link"]["unique_patent_keys"] = dict(n=int(len(uq)), matched=int(uq.matched.sum()), matched_share=float(uq.matched.mean()), matched_share_ay2017_2022=float(uq[uq.ay.between(2017, 2022)].matched.mean()))
mk = pk[pk.matched].merge(k[["a", "세세분류코드"]], left_on="akey", right_on="a")
R["c_link"]["matched_ksic_top6"] = {str(a): int(b) for a, b in mk.drop_duplicates("akey").세세분류코드.value_counts().head(6).items()}
# 매칭 안 된 특허 출원번호의 원인: 제공 데이터 연도 범위 밖 vs 범위 안
um = pk[~pk.matched]
R["c_link"]["unmatched_patent_keys"] = dict(n=int(len(um)), app_year_lt_2000=int((um.ay < 2000).sum()), app_year_ge_2000=int((um.ay >= 2000).sum()))
# (d) 시점: 서명→반영 지연, 최종창, 창 끝 이후 반영
kd["refl_d"] = pd.to_datetime(kd.refl.str[:10], errors="coerce"); kd["sig_d"] = pd.to_datetime(kd.sig.str[:10], errors="coerce")
dp["refl_d"] = pd.to_datetime(dp.refl.str[:10], errors="coerce"); dp["sig_d"] = pd.to_datetime(dp.sig.str[:10], errors="coerce")
lag = (dp.refl_d - dp.sig_d).dt.days
m12 = dp.refl_d.dt.year >= 2012
R["d_lag_days_all_refl_ge2012"] = dict(median=float(lag[m12].median()), p90=float(lag[m12].quantile(0.9)), p99=float(lag[m12].quantile(0.99)))
mw = (dp.refl_d > "2022-12-31") & (dp.refl_d <= "2024-12-31")
R["d_lag_days_refl_2023_24"] = dict(median=float(lag[mw].median()), p90=float(lag[mw].quantile(0.9)), n=int(mw.sum()))
lagk = (kd.refl_d - kd.sig_d).dt.days; mwk = (kd.refl_d > "2022-12-31") & (kd.refl_d <= "2024-12-31")
R["d_KR_lag_refl_2023_24"] = dict(median=float(lagk[mwk].median()), p90=float(lagk[mwk].quantile(0.9)), n=int(mwk.sum()))
s24 = (dp.sig_d.dt.year == 2024)
R["d_sig2024_reflected_after_2024"] = dict(share=float((dp.refl_d[s24] > "2024-12-31").mean()), n_sig2024=int(s24.sum()))
r25 = dp.refl_d.dt.year == 2025
R["d_refl2025_signed_2024_or_earlier"] = dict(share=float((dp.sig_d[r25] <= "2024-12-31").mean()), n_refl2025=int(r25.sum()))
R["d_bulk_2011"] = dict(dipg_refl_2011=int((dp.refl_y == "2011").sum()), of_which_sig_before_2011=int(((dp.refl_y == "2011") & (dp.sig_d.dt.year < 2011)).sum()))
# 26429 최종창 양성의 서명→반영 지연과 반영 날짜 집중(일괄 반영 여부)
mp = mpos.sort_values("refl").drop_duplicates("appl_no")
lag_p = (pd.to_datetime(mp.refl.astype(int).astype(str)) - pd.to_datetime(mp.sig.astype(int).astype(str), errors="coerce")).dt.days
R["d_final_pos_lag"] = dict(median=float(lag_p.median()), p90=float(lag_p.quantile(0.9)), max=float(lag_p.max()), n=int(len(mp)),
                            top_refl_days={str(int(a)): int(b) for a, b in mp.refl.value_counts().head(5).items()})
# 최종창 이후(2025-01~2026-09) 반영된 26429 단위: 레이블상 음성
lateU = M[M.appl_no.isin(U.appl_no) & (M.refl > 20241231)].drop_duplicates("appl_no")
R["d_units_reflected_after_window"] = dict(n=int(len(lateU)), signed_by_2024=int((lateU.sig <= 20241231).sum()), share_of_units=float(len(lateU) / len(U)))
R["secs"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(W, "cache", "n5_s4_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
