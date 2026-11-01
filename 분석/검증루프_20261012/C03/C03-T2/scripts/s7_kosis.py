# -*- coding: utf-8 -*-
"""s7: KOSIS 광업·제조업조사 50개 산업 — 마스킹·10/11차 코드 대응·단절, SP-CASE 성장률(로그·복리) 재계산 → kosis_mask.csv."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
ko = pd.read_csv(os.path.join(ROOT, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
res = {}
res["shape"] = list(ko.shape); res["bases"] = ko.기준.value_counts().to_dict()
res["years_by_base"] = {b: sorted(map(int, g.연도.unique())) for b, g in ko.groupby("기준")}
res["items"] = ko.drop_duplicates("항목코드").set_index("항목코드").항목명.to_dict()
res["codes_by_base"] = {b: int(g.세세분류코드.nunique()) for b, g in ko.groupby("기준")}
c10 = set(ko[ko.기준 == "10차"].세세분류코드); c11 = set(ko[ko.기준 == "11차"].세세분류코드)
res["codes_only_10"] = sorted(c10 - c11); res["codes_only_11"] = sorted(c11 - c10)
res["dup_rows"] = int(ko.duplicated(["기준", "세세분류코드", "항목코드", "연도"]).sum())
res["masking_values"] = ko.마스킹.value_counts().to_dict()
res["masked_value_nonnull"] = int(ko[ko.마스킹 != 0].값.notna().sum())
res["neg_or_zero_unmasked"] = int((ko[ko.마스킹 == 0].값 <= 0).sum())
# 이름 변화(10차 vs 11차 같은 코드)
nm = ko.groupby(["세세분류코드", "기준"]).산업명.first().unstack()
res["name_changed_same_code"] = nm[(nm.get("10차").notna()) & (nm.get("11차").notna()) & (nm.get("10차") != nm.get("11차"))].to_dict("index") if "11차" in nm else {}
# 마스킹 산업×연도 (전 항목 기준 + 주요 항목)
full = ko.pivot_table(index=["기준", "세세분류코드", "연도"], columns="항목코드", values="마스킹", aggfunc="max")
mrows = ko[ko.마스킹 != 0]
mk = mrows.groupby(["기준", "세세분류코드", "산업명", "연도"]).항목코드.apply(lambda s: " ".join(sorted(s))).reset_index().rename(columns={"항목코드": "masked_items"})
# 각 산업 특성: 10차 2015–17 평균 사업체수·출하(비마스킹 기준), 결측 연도
k10 = ko[(ko.기준 == "10차") & (ko.마스킹 == 0)]
est = k10[k10.항목코드 == "T01"].groupby("세세분류코드").값.median(); shp = k10[k10.항목코드 == "T04"].groupby("세세분류코드").값.median()
mk["median_establishments_10th"] = mk.세세분류코드.map(est); mk["median_shipment_10th"] = mk.세세분류코드.map(shp)
# 각 산업×연도 존재 여부(행 자체 누락)
allyrs = {b: res["years_by_base"][b] for b in res["years_by_base"]}
miss = []
for b, g in ko.groupby("기준"):
    for (code, item), gg in g.groupby(["세세분류코드", "항목코드"]):
        ys = set(gg.연도)
        for y in allyrs[b]:
            if y not in ys: miss.append(dict(기준=b, 세세분류코드=code, 항목코드=item, 연도=y))
miss = pd.DataFrame(miss)
res["absent_rows_by_base_item"] = miss.groupby(["기준", "항목코드"]).size().to_dict() if len(miss) else {}
res["absent_rows_by_base_item"] = {f"{k[0]}|{k[1]}": int(v) for k, v in res["absent_rows_by_base_item"].items()}
if len(miss):
    am = miss.groupby(["기준", "세세분류코드", "연도"]).항목코드.apply(lambda s: " ".join(sorted(s))).reset_index().rename(columns={"항목코드": "absent_items"})
    mk = mk.merge(am, on=["기준", "세세분류코드", "연도"], how="outer")
mk = mk.sort_values(["기준", "세세분류코드", "연도"])
mk.to_csv(os.path.join(W, "kosis_mask.csv"), index=False, encoding="utf-8-sig")
res["masked_industry_years"] = int(mk.dropna(subset=["masked_items"]).shape[0]) if "masked_items" in mk else 0
res["masked_industries"] = sorted(mk.dropna(subset=["masked_items"]).세세분류코드.unique()) if "masked_items" in mk else []
# 표 5·사례 영향: m4가 쓰는 10차 2015–17 T04/T02, 08–09·16–17 성장
need = {"P2": [2015, 2016, 2017], "g_a": [2008, 2009], "g_b": [2016, 2017]}
aff = {}
for k, ys in need.items():
    for item in ["T04", "T02", "T01"]:
        g = ko[(ko.기준 == "10차") & (ko.항목코드 == item) & ko.연도.isin(ys)]
        bad = g[g.마스킹 != 0].groupby("세세분류코드").연도.apply(list).to_dict()
        missing = sorted(set(est.index) - set(g[g.마스킹 == 0].세세분류코드))
        aff[f"{k}_{item}"] = dict(masked=bad, all_years_missing=missing)
res["m4_split34_window_masking"] = aff
# ---------- SP-CASE 재계산 ----------
t5 = pd.read_csv(os.path.join(SPLIT34, "t5_cases.csv"), dtype={"ksic": str}, encoding="utf-8-sig").set_index("ksic")
cases = {}
for code in ["26211", "26212", "26111", "29280"]:
    row = {}
    for item, col in [("T04", "KOSIS출하_연성장%"), ("T02", "KOSIS종사자_연성장%(08-09→16-17)"), ("T01", "KOSIS사업체_연성장%")]:
        g = k10[(k10.항목코드 == item) & (k10.세세분류코드 == code)].set_index("연도").값
        a = g.reindex([2008, 2009]).mean(); b = g.reindex([2016, 2017]).mean()
        logg = np.log(b / a) / 8 * 100; cagr = ((b / a) ** (1 / 8) - 1) * 100
        row[item] = dict(log=round(float(logg), 2), compound=round(float(cagr), 2), t5=float(t5.at[code, col]),
                         diff_log_vs_t5=round(float(logg - t5.at[code, col]), 3), years_present=sorted(map(int, g.index)))
        # 11차 단절 확인: 같은 코드 10차 마지막 해 vs 11차 첫 해
        g11 = ko[(ko.기준 == "11차") & (ko.항목코드 == item) & (ko.세세분류코드 == code) & (ko.마스킹 == 0)].set_index("연도").값
        ov = sorted(set(g.index) & set(g11.index))
        row[item]["overlap_years_10_11"] = ov
        row[item]["ratio_11_over_10_overlap"] = [round(float(g11[y] / g[y]), 4) for y in ov]
    cases[code] = row
res["sp_cases"] = cases
# 10·11차 중복 연도 전 산업 비교(출하)
ov = []
for item in ["T01", "T02", "T04"]:
    a = ko[(ko.기준 == "10차") & (ko.항목코드 == item) & (ko.마스킹 == 0)].set_index(["세세분류코드", "연도"]).값
    b = ko[(ko.기준 == "11차") & (ko.항목코드 == item) & (ko.마스킹 == 0)].set_index(["세세분류코드", "연도"]).값
    j = pd.concat([a.rename("v10"), b.rename("v11")], axis=1).dropna()
    if len(j):
        r = (j.v11 / j.v10)
        ov.append(dict(item=item, n=int(len(j)), years=sorted(set(j.index.get_level_values(1))), median_ratio=round(float(r.median()), 4),
                       share_abs_gt5pct=round(float(((r - 1).abs() > 0.05).mean()), 3),
                       big=[(k, int(y), round(float(v), 3)) for (k, y), v in r[(r - 1).abs() > 0.2].items()][:15]))
res["overlap_10_11"] = ov
json.dump(res, open(os.path.join(CACHE, "s7_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:9000])
