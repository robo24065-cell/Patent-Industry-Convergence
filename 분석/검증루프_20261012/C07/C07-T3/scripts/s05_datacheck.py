# -*- coding: utf-8 -*-
"""C07-T3 s05: 3단계 데이터 점검. s04a 캐시(사건·레이블) + sqlite(mode=ro, 사건 '전체 유형'의 월·일별 건수: 게시 지연 확인용).
→ ../cache/s05_out.json"""
import os, sys, json, sqlite3
import numpy as np, pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
EV = pd.read_parquet(os.path.join(W, "cache", "s04a_events.parquet"))
Lb = pd.read_parquet(os.path.join(W, "cache", "s04a_labels.parquet"))
S = pd.read_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"), columns=["rgst_no", "owner_v2", "ksic"])
popset = set(Lb.rgst_no)
out = {}
EV["ym"] = EV.d // 100; EV["inpop"] = EV.rgst_no.isin(popset)
out["monthly_all"] = {str(k): int(v) for k, v in EV.groupby("ym").size().items()}
out["monthly_inpop"] = {str(k): int(v) for k, v in EV[EV.inpop].groupby("ym").size().items()}
out["by_type_year"] = {f"{k[0]}|{k[1]}": int(v) for k, v in EV.groupby([EV.d // 10000, "rgstCsName"]).size().items()}
out["dup_same_patent_date_type"] = int(EV.duplicated(["rgst_no", "d", "rgstCsName"]).sum())
out["same_patent_same_date_multi"] = int((EV.groupby(["rgst_no", "d"]).size() > 1).sum())
out["patents_multi_events"] = int((EV.groupby("rgst_no").size() > 1).sum())
# 같은 권리자·같은 날 묶음(모집단 양성, 주창)
E2 = EV[EV.inpop & (EV.d <= 20260630)].merge(S, on="rgst_no")
fe = E2.groupby("rgst_no").agg(first=("d", "min"), owner=("owner_v2", "first"), ksic=("ksic", "first")).reset_index()
fe["owner"] = np.where(fe.owner.fillna("") == "", "R" + fe.rgst_no, fe.owner)
sd = fe.groupby(["owner", "first"]).size()
out["same_owner_same_day_bundles_ge2"] = int((sd >= 2).sum()); out["pos_in_bundles_ge2"] = int(sd[sd >= 2].sum())
oc = fe.owner.value_counts()
out["pos_owner_clusters"] = int(len(oc)); out["top1_owner_pos"] = int(oc.iloc[0]); out["top5_owner_pos"] = int(oc.head(5).sum())
out["pos_ksic_n"] = int(fe.ksic.nunique()); out["pos_ksic_top3_share"] = float(fe.ksic.value_counts().head(3).sum() / len(fe))
# 모집단 밖 사건 특허(원인)
outp = sorted(set(EV.rgst_no) - popset)
if outp:
    q = "select rgst_no, rgstDate, cndrtExptnDate from patent where rgst_no in (%s)" % ",".join("'%s'" % r for r in outp)
    O = pd.read_sql(q, c); O["rg"] = pd.to_numeric(O.rgstDate, errors="coerce")
    out["event_patents_outside_pop"] = dict(n=len(outp), reg_after_2024=int((O.rg > 20241231).sum()), reg_le_2024=int((O.rg <= 20241231).sum()))
# 게시 지연: 전체 사건 유형의 2026-05~08 주별 건수
ALL = pd.read_sql("select rgstCsDate d, rgstCsName cs from right where rgstCsDate >= '20250101' and rgstCsDate <= '20260831'", c)
ALL["d"] = pd.to_datetime(ALL.d, format="%Y%m%d", errors="coerce")
wk = ALL[ALL.d >= "2026-05-01"].groupby(ALL.d.dt.to_period("W").astype(str)).size()
out["all_types_weekly_2026_05_08"] = {k: int(v) for k, v in wk.items()}
mo = ALL.groupby(ALL.d.dt.to_period("M").astype(str)).size()
out["all_types_monthly_2025_2026"] = {k: int(v) for k, v in mo.items()}
out["all_types_max_date"] = str(ALL.d.max().date())
# 2025 vs 2026H1 월평균, 끝 두 달 포아송 검정
m25 = EV[(EV.d >= 20250101) & (EV.d <= 20251231)].shape[0] / 12
m26 = EV[(EV.d >= 20260101) & (EV.d <= 20260630)].shape[0] / 6
last2 = EV[(EV.d >= 20260501) & (EV.d <= 20260630)].shape[0]
from scipy.stats import poisson
out["rate_per_month"] = dict(y2025=m25, y2026H1=m26, last2_2026_05_06=int(last2), expected_last2_at_2025_rate=2 * m25,
                             p_le_last2=float(poisson.cdf(last2, 2 * m25)))
# 공식 기저율 대비
FP = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["y"])
out["base_rate"] = dict(official_permil_24m=float(1000 * FP.y.mean()), main_permil_18m=float(1000 * Lb.y_main.mean()),
                        main_permil_per730d=float(1000 * Lb.y_main.mean() * 730 / 546), auxA_permil_per730d=float(1000 * Lb.y_auxA.mean() * 730 / 592))
# KR 등록 결손 추정(등록월 모집단 수)
pt = pd.read_sql("select rgst_no, rgstDate from patent", c); pt["m"] = pd.to_numeric(pt.rgstDate, errors="coerce") // 100
P24 = pt[pt.rgst_no.isin(popset)]
mc = P24.groupby("m").size()
ref = mc[(mc.index >= 202301) & (mc.index <= 202311)].mean()
gap_months = [m for m in mc.index if 202312 <= m <= 202408]
deficit = float(sum(max(ref - mc[m], 0) for m in gap_months))
young = P24[(P24.m >= 202309) & (P24.m <= 202412)].rgst_no
yr = float(Lb[Lb.rgst_no.isin(set(young))].y_main.mean())
out["kr_gap"] = dict(ref_monthly_2023_01_11=float(ref), deficit_2023_12_to_2024_08=deficit, share_of_pop=deficit / (len(popset) + deficit),
                     young_cohort_n=int(len(young)), young_cohort_pos_main=int(Lb[Lb.rgst_no.isin(set(young))].y_main.sum()), young_rate=yr,
                     expected_missing_events=deficit * yr, C02T2_estimate="7,300–8,400건(2.1–2.4%), 기대 누락 사건 약 2건")
json.dump(out, open(os.path.join(W, "cache", "s05_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
