# -*- coding: utf-8 -*-
"""s6: 업종별 법인 폐업 공식표(KOSIS 9.8.15, download/) 검증 (a)(b)(d) + N4 레이블 폐업 업태 구성 대조 (c).
업태 대응: NPS 사업장 업종코드(국세청 6자리 업종코드) 앞 2자리 → 국세통계 업태 11군(근사). 99(번호 미상)·NPS 미연결 = 미상(대조에서 제외, 비율 별도).
출력: cache/s6_industry.json, cache/s6_industry_by_year.csv"""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *

OFF = pd.read_csv(os.path.join(OUT, "download", "kosis_DT_133N_9815A_closures_by_type_industry.csv"), sep=";", comment="#")
OFF = OFF[OFF.biz == "법인사업자"].set_index("ind")
YRS = [str(y) for y in range(2016, 2026)]
# 공식 업태 → 비교군
G_OFF = {"농·임·어업": "농림어", "광업": "광업", "제조업": "제조", "전기·가스·수도업": "전기가스수도", "건설업": "건설",
         "도매업": "도매", "소매업": "소매", "음식업": "음식숙박", "숙박업": "음식숙박", "운수·창고·통신업": "운수통신",
         "부동산매매업": "부동산", "부동산임대업": "부동산", "서비스업": "서비스기타", "대리·중개·도급업": "서비스기타"}


def g_nps(c):
    try:
        k = int(str(c)[:2])
    except Exception:
        return "미상"
    if k <= 5: return "농림어"
    if 10 <= k <= 14: return "광업"
    if 15 <= k <= 37: return "제조"
    if k in (40, 41): return "전기가스수도"
    if k == 45: return "건설"
    if k in (50, 51): return "도매"
    if k == 52: return "소매"
    if k == 55: return "음식숙박"
    if 60 <= k <= 64: return "운수통신"
    if k == 70: return "부동산"
    if k == 99: return "미상"
    return "서비스기타"


R = {}
# ---- (a) 총량 대조
tot = OFF.loc["합계", YRS].astype(float)
idx2825 = {"2016": 69, "2017": 70, "2018": 70, "2019": 70, "2020": 67, "2021": 65, "2022": 67, "2023": 76, "2024": 83, "2025": 85}
R["a_total_vs_2825"] = {y: [int(tot[y]), idx2825[y], round(tot[y] / 1000 / idx2825[y] - 1, 4)] for y in YRS}
sub = OFF.drop(index="합계")[YRS].replace("-", 0).astype(float)
R["a_sum_industries_eq_total"] = bool((sub.sum() == tot).all())
R["a_2015_definition_break"] = "2015는 12.31 가동사업자 기준(주석) → 2015 50,630은 2016 이후와 비교 불가, 사용 안 함"
# ---- 공식 분포
offg = sub.groupby(sub.index.map(G_OFF)).sum()
# ---- N4 레이블·전체 폐업
NS = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "registered", "end_dt"])
NS = NS[NS.registered.fillna(False).astype(bool)].copy()
NS["bt"] = NS.bizno.astype(str).str[3:5]
NS["ey"] = NS.end_dt.dt.year
corp = NS[NS.bt.isin(["81", "85", "86", "87", "88"])]
PN = pd.read_parquet(os.path.join(EXT, "firm_nps_firm_panel.parquet"), columns=["applicantCd", "snap", "indcd", "n_emp"])
PN = PN.dropna(subset=["indcd"])


def ind_asof(codes, snap_max):
    p = PN[PN.applicantCd.isin(codes) & (PN.snap <= snap_max)].sort_values("snap")
    return p.groupby("applicantCd").indcd.last().map(g_nps)


rows = []
for y in YRS:
    c = corp[corp.ey == int(y)].drop_duplicates("bizno")
    ind = ind_asof(set(c.applicantCd), f"{int(y) - 1}-12")       # 폐업 전년 말 스냅샷 기준 업태
    c = c.assign(g=c.applicantCd.map(ind).fillna("미상"))
    known = c[c.g != "미상"]
    pd_ = known.g.value_counts(normalize=True); po = offg[y] / offg[y].sum()
    ix = po.index.union(pd_.index)
    tv = 0.5 * (po.reindex(ix, fill_value=0) - pd_.reindex(ix, fill_value=0)).abs().sum()
    cap = (known.g.value_counts() / offg[y]).reindex(offg.index)
    rows.append(dict(year=int(y), r3nts_corp_closed=len(c), unknown_share=round(1 - len(known) / max(len(c), 1), 3), tvd=round(float(tv), 3),
                     manuf_share_r3nts=round(float(pd_.get("제조", 0)), 3), manuf_share_off=round(float(po.get("제조", 0)), 3),
                     capture_total=round(len(c) / float(tot[y]), 4),
                     **{f"cap_{k}": (round(float(v), 4) if v == v else None) for k, v in cap.items()}))
BY = pd.DataFrame(rows)
BY.to_csv(os.path.join(CACHE, "s6_industry_by_year.csv"), index=False, encoding="utf-8-sig")
log(BY[["year", "r3nts_corp_closed", "unknown_share", "tvd", "manuf_share_r3nts", "manuf_share_off", "capture_total"]].to_string())
# ---- N4 시험 양성(2024–25) 대 공식 2024–25
P = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"), columns=["applicantCd", "bizno", "y"])
P["applicantCd"] = P.applicantCd.astype(str)
ind = ind_asof(set(P.applicantCd), "2023-12")
P["g"] = P.applicantCd.map(ind).fillna("미상")
pos = P[P.y == 1]; kp = pos[pos.g != "미상"]
o2 = offg[["2024", "2025"]].sum(axis=1); o2 = o2 / o2.sum()
d2 = kp.g.value_counts(normalize=True)
ix = o2.index.union(d2.index)
R["N4_test_pos"] = dict(n=len(pos), unknown=int((pos.g == "미상").sum()), unknown_share=round(float((pos.g == "미상").mean()), 3),
                        tvd_vs_off_2024_25=round(float(0.5 * (o2.reindex(ix, fill_value=0) - d2.reindex(ix, fill_value=0)).abs().sum()), 3),
                        dist_N4=d2.round(3).to_dict(), dist_off=o2.round(3).to_dict(),
                        capture_by_group=(kp.g.value_counts() / offg[["2024", "2025"]].sum(axis=1)).round(4).dropna().to_dict(),
                        capture_total=round(len(pos) / float(tot[["2024", "2025"]].sum()), 4))
R["N4_test_pop_dist"] = P[P.g != "미상"].g.value_counts(normalize=True).round(3).to_dict()
R["N4_test_pop_unknown_share"] = round(float((P.g == "미상").mean()), 3)
# 업태별 양성률(시험 모집단 안) — 미상 층 포함
R["N4_test_rate_by_group"] = P.groupby("g").y.agg(["size", "mean"]).round(4).to_dict("index")
# 미상(NPS 없음) 층 = 사후 선택과 겹치는 층
json.dump(R, open(os.path.join(CACHE, "s6_industry.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log(json.dumps({k: R[k] for k in ["a_total_vs_2825", "a_sum_industries_eq_total", "N4_test_pos", "N4_test_pop_unknown_share"]}, ensure_ascii=False))
