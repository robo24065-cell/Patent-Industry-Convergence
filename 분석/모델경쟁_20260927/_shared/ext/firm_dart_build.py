# -*- coding: utf-8 -*-
"""OpenDART fnlttMultiAcnt 원시행(이미 수집된 R2firm·R2N2 파일, 사업보고서 2015–2023) → 기업×회계연도 패널.
새 API 호출 없음. 🔒 2024·2025 보고서 파일(R2N2_dart_fnltt_2024/2025)은 최종시험 봉인이라 읽지 않는다.
시점정합: 같은 (기업,회계연도,계정)이 여러 보고서에 있으면 '가장 먼저 공시된 보고서' 값을 채택하고
src_report(보고서 사업연도)와 avail_date(보고서 제출 기한 ≈ 회계연도말+90일, 12월 결산 가정 → fy+1-03-31)를 남긴다.
출력: firm_dart_fin_panel.parquet, firm_dart_link.csv(연계율)"""
import os, json, glob, sqlite3
import pandas as pd, numpy as np

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
FILES = [os.path.join(EXT, "R2firm_dart_fnltt_raw.jsonl")] + [os.path.join(EXT, f"R2N2_dart_fnltt_{y}.jsonl") for y in (2016, 2017, 2019, 2020, 2022)]
ACC = {"매출액": "rev", "영업이익": "op", "법인세차감전 순이익": "ebt", "당기순이익(손실)": "ni", "자산총계": "assets",
       "부채총계": "liab", "자본총계": "equity", "자본금": "capital", "유동자산": "cur_assets", "유동부채": "cur_liab",
       "비유동자산": "noncur_assets", "이익잉여금": "re"}
rows = []
for fp in FILES:
    for line in open(fp, encoding="utf-8"):
        x = json.loads(line)
        a = ACC.get(x["account_nm"])
        if a is None:
            continue
        by = int(x["bsns_year"])
        if by > 2023:
            continue
        for k, off in (("thstrm_amount", 0), ("frmtrm_amount", 1), ("bfefrmtrm_amount", 2)):
            v = (x.get(k) or "").replace(",", "").strip()
            if v in ("", "-"):
                continue
            try:
                v = float(v)
            except ValueError:
                continue
            rows.append((x["corp_code"], by - off, x["fs_div"], a, v, by, x["rcept_no"]))
D = pd.DataFrame(rows, columns=["corp_code", "fy", "fs", "acc", "v", "src_report", "rcept_no"])
D = D.sort_values(["src_report"]).drop_duplicates(["corp_code", "fy", "fs", "acc"])  # 최초 공시 보고서 우선
W = D.pivot_table(index=["corp_code", "fy", "fs"], columns="acc", values="v").reset_index()
S = D.groupby(["corp_code", "fy", "fs"]).src_report.min().rename("src_report").reset_index()
W = W.merge(S, on=["corp_code", "fy", "fs"])
W["ofs_first"] = (W.fs != "OFS").astype(int)  # 별도(OFS) 우선
W = W.sort_values("ofs_first").drop_duplicates(["corp_code", "fy"]).drop(columns="ofs_first")
W["avail_date"] = pd.to_datetime((W.src_report + 1).astype(str) + "-03-31")
W["pit_own_report"] = (W.src_report == W.fy)  # 해당 연도 보고서 당기값(실시간 값)
dc = sqlite3.connect(f"file:{os.path.join(BASE, r'공모전데이터/외부데이터/DART/dart_company.sqlite')}?mode=ro", uri=True)
firm = pd.read_sql("select corp_code, corp_name, jurir_no, bizr_no, induty_code, est_dt, corp_cls from company where status='OK'", dc)
W = W.merge(firm, on="corp_code", how="left")
# 파생: 성장·수익성·레버리지 (전년 값은 같은 패널 내 전년 행)
W = W.sort_values(["corp_code", "fy"])
g = W.groupby("corp_code")
for c in ("rev", "assets"):
    prev = g[c].shift(1)
    ok = (g.fy.shift(1) == W.fy - 1) & (prev > 0)
    W[f"g_{c}"] = np.where(ok, W[c] / prev - 1, np.nan)
W["opm"] = W.op / W.rev.where(W.rev > 0)
W["lev"] = W.liab / W.assets.where(W.assets > 0)
W.to_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"), index=False)
print(W.groupby("fy").agg(firms=("corp_code", "size"), rev=("rev", lambda s: s.notna().sum()), pit=("pit_own_report", "mean")).to_string())

# ---- 연계율 ----
pf = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\pat_firm.parquet"))  # 등록특허→출원인코드→법인번호→corp_code (f00_link.py)
fin_firms = set(W.corp_code)
out = []
tot = pf.rgst_no.nunique()
def add(k, v, note=""):
    out.append((k, v, note)); print(k, v, note)
add("KR등록특허(제공범위, 등록원부)", tot)
add("특허 중 법인번호 연결 출원인 보유", pf.loc[pf.jurir.notna(), "rgst_no"].nunique(), f"{pf.loc[pf.jurir.notna(),'rgst_no'].nunique()/tot:.3f}")
add("특허 중 DART 기업개황 연결", pf.loc[pf.corp_code.notna(), "rgst_no"].nunique(), f"{pf.loc[pf.corp_code.notna(),'rgst_no'].nunique()/tot:.3f}")
m = pf.corp_code.isin(fin_firms)
add("특허 중 DART 재무(2013-2023) 보유 기업 연결", pf.loc[m, "rgst_no"].nunique(), f"{pf.loc[m,'rgst_no'].nunique()/tot:.3f}")
add("재무 보유 기업 수", len(fin_firms))
add("특허 보유 DART 기업 수", pf.corp_code.nunique())
# 출원연도 2015-2022 특허로 한정 (재무 기간과 겹침)
pf["ay"] = pf.applDate.astype(str).str[:4].astype(int)
r = pf[pf.ay.between(2015, 2022)]
add("2015-22 출원 특허 중 재무기업 연결", r.loc[r.corp_code.isin(fin_firms), "rgst_no"].nunique(), f"{r.loc[r.corp_code.isin(fin_firms),'rgst_no'].nunique()/r.rgst_no.nunique():.3f}")
k = pf.groupby("ksic").apply(lambda d: d.loc[d.corp_code.isin(fin_firms), "rgst_no"].nunique() / d.rgst_no.nunique())
add("50 KSIC별 재무기업 연결 특허비중 최소/중앙/최대", f"{k.min():.3f}/{k.median():.3f}/{k.max():.3f}", f"KSIC {k.shape[0]}개")
# 권리목록(법인 53,967) ↔ DART
bc = sqlite3.connect(f"file:{os.path.join(BASE, r'수집파이프라인/business_rights.sqlite')}?mode=ro", uri=True)
rl = pd.read_sql("select distinct applicant_cd, corp_rgstno, business_no from right_list", bc)
rl["jur"] = rl.corp_rgstno.fillna("").str.replace("-", "")
jset = set(firm.jurir_no.dropna())
add("권리목록 출원인(법인) 수", rl.applicant_cd.nunique())
add("권리목록 출원인 중 DART 법인번호 일치", rl.loc[rl.jur.isin(jset), "applicant_cd"].nunique(), f"{rl.loc[rl.jur.isin(jset),'applicant_cd'].nunique()/rl.applicant_cd.nunique():.3f}")
fj = set(W.jurir_no.dropna())
add("권리목록 출원인 중 DART 재무 보유", rl.loc[rl.jur.isin(fj), "applicant_cd"].nunique(), f"{rl.loc[rl.jur.isin(fj),'applicant_cd'].nunique()/rl.applicant_cd.nunique():.3f}")
pd.DataFrame(out, columns=["항목", "값", "비율/비고"]).to_csv(os.path.join(EXT, "firm_dart_link.csv"), index=False, encoding="utf-8-sig")
