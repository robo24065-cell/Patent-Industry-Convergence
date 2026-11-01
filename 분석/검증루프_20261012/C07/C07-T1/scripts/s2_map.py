# -*- coding: utf-8 -*-
r"""C07-T1 (c): 통계청 공식 KSIC 9↔10(및 10↔11) 연계표로 50개 코드 판정 + 9차 원표(DT_1F160622) 대 10차 소급표(DT_1FS1001) 값 비교.
출력: ksic9_10_map_check.csv, cache\s2_stats.json"""
import os, sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(B, r"분석\검증루프_20261012\C07\C07-T1")
sys.path.insert(0, os.path.join(B, r"분석\모델경쟁_20260927\_shared_report\label_validity"))
from lv_common import ksic_to_group
W = pd.read_csv(os.path.join(B, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
K50 = sorted(W.세세분류코드.unique()); NM = W.drop_duplicates("세세분류코드").set_index("세세분류코드").산업명
G = pd.Series({k: ksic_to_group(k) for k in K50}); gs = G.value_counts(); K34 = [k for k in K50 if gs[G[k]] > 1]
L = pd.read_csv(os.path.join(H, r"download\ksic_link\ksic_ksco_link_2025Q2.csv"), encoding="cp949", dtype=str)
L.columns = ["kind", "rev", "lvl", "new", "newnm", "old", "oldnm"]
def classify(rev):
    l = L[(L.kind == "KSIC") & (L.rev == rev) & (L.lvl == "5")]
    n_old_per_new = l.groupby("new").old.nunique(); n_new_per_old = l.groupby("old").new.nunique()
    return l, n_old_per_new, n_new_per_old
l10, a10, b10 = classify("10"); l11, a11, b11 = classify("11")
print("10차 5자리 신코드 수", l10.new.nunique(), "9차 구코드 수", l10.old.nunique(), "| 11차 신코드", l11.new.nunique())
def typ(code, l, a, b, direction="new"):
    olds = sorted(l[l.new == code].old.unique())
    if not olds:
        return "연계표에 없음", olds, []
    sibs = sorted(set(l[l.old.isin(olds)].new.unique()))
    split = any(b[o] > 1 for o in olds); merge = len(olds) > 1
    t = "1:1" if not split and not merge else ("분할(1:N)" if split and not merge else ("통합(N:1)" if merge and not split else "복합(N:M)"))
    if t == "1:1" and olds[0] != code:
        t = "1:1(코드 변경)"
    return t, olds, sibs
# 10→11 방향: 10차 코드가 '구코드'
def typ11(code):
    news = sorted(l11[l11.old == code].new.unique())
    if not news:
        return "연계표에 없음", news
    olds = sorted(set(l11[l11.new.isin(news)].old.unique()))
    split = len(news) > 1; merge = len(olds) > 1
    t = "1:1" if not split and not merge else ("분할(1:N)" if split and not merge else ("통합(N:1)" if merge and not split else "복합(N:M)"))
    if t == "1:1" and news[0] != code:
        t = "1:1(코드 변경)"
    return t, news
# 9차 원표 값
def lj(p):
    d = pd.DataFrame(json.load(open(p, encoding="utf-8")))
    return pd.DataFrame({"c": d.C2.str.lstrip("C"), "nm": d.C2_NM, "item": d.ITM_ID, "year": d.PRD_DE.astype(int), "v": pd.to_numeric(d.DT, errors="coerce")})
O9 = lj(os.path.join(H, r"download\kosis9\DT_1F160622_2005_2014.json"))
O10 = pd.concat([lj(os.path.join(H, r"download\kosis10\DT_1FS1001_2007_2020.json")), lj(os.path.join(H, r"download\kosis10\DT_1FS1001_siblings_2007_2020.json"))])
nm9 = O9.drop_duplicates("c").set_index("c").nm
p9 = O9.pivot_table(index=["c", "item"], columns="year", values="v"); p10 = O10.pivot_table(index=["c", "item"], columns="year", values="v")
gr = lambda P, c, it, a, b: float((np.log(P.loc[(c, it), b].mean()) - np.log(P.loc[(c, it), a].mean())) / (np.mean(b) - np.mean(a))) if (c, it) in P.index else np.nan
rows = []
for k in K50:
    t, olds, sibs = typ(k, l10, a10, b10)
    t11, news11 = typ11(k)
    r = {"ksic10": k, "name10": NM[k], "group": G[k], "in_K34(3.1.3)": k in K34, "type_9to10": t, "ksic9_codes": "|".join(olds),
         "ksic9_names": "|".join(l10[l10.old.isin(olds)].drop_duplicates("old").oldnm), "ksic10_siblings_of_9code": "|".join(sibs),
         "same_code_in_9th_table": k in nm9.index, "same_code_9th_name": nm9.get(k, ""),
         "type_10to11": t11, "ksic11_codes": "|".join(news11)}
    # 9차 원표 대 10차 소급: 2008·2009·2011–2014 같은 해 비율(9차 구코드 합 / 10차 코드), 1:1이면 동일 산업
    for it, lab in (("T01", "est"), ("T02", "emp"), ("T04", "ship")):
        if (k, it) in p10.index and all((o, it) in p9.index for o in olds) and olds:
            v9 = sum(p9.loc[(o, it)] for o in olds); miss = [s for s in sibs if (s, it) not in p10.index]
            v10s = sum(p10.loc[(s, it)] for s in sibs if (s, it) in p10.index)
            r["sibs_missing"] = "|".join(miss)
            yrs = [y for y in (2008, 2009, 2011, 2012, 2013, 2014) if y in v9.index and y in v10s.index]
            ratio = (v10s[yrs] / v9[yrs])
            r[f"{lab}_ratio10sib_over9_2008_14_min"] = round(float(ratio.min()), 4); r[f"{lab}_ratio10sib_over9_max"] = round(float(ratio.max()), 4)
            if t == "1:1":
                r[f"{lab}_g9_0809_1314"] = round(100 * gr(p9, olds[0], it, [2008, 2009], [2013, 2014]), 2)
                r[f"{lab}_g10_0809_1314"] = round(100 * gr(p10, k, it, [2008, 2009], [2013, 2014]), 2)
    rows.append(r)
M = pd.DataFrame(rows)
M.to_csv(os.path.join(H, "ksic9_10_map_check.csv"), index=False, encoding="utf-8-sig")
st = {"type_counts_50": M.type_9to10.value_counts().to_dict(), "type_counts_K34": M[M["in_K34(3.1.3)"]].type_9to10.value_counts().to_dict(),
      "non11_K34": M[M["in_K34(3.1.3)"] & (M.type_9to10 != "1:1")][["ksic10", "name10", "group", "type_9to10", "ksic9_codes"]].values.tolist(),
      "non11_50": M[M.type_9to10 != "1:1"][["ksic10", "type_9to10", "ksic9_codes", "ksic10_siblings_of_9code"]].values.tolist(),
      "type10to11_counts": M.type_10to11.value_counts().to_dict(),
      "non11_10to11": M[M.type_10to11 != "1:1"][["ksic10", "type_10to11", "ksic11_codes"]].values.tolist(),
      "code_collision(same code, different 9th name)": M[M.same_code_in_9th_table & (M.type_9to10 != "1:1")][["ksic10", "name10", "same_code_9th_name"]].values.tolist(),
      "K34_obs_nonone2one": int(2 * (M["in_K34(3.1.3)"] & (M.type_9to10 != "1:1")).sum())}
print(json.dumps(st, ensure_ascii=False, indent=1))
json.dump(st, open(os.path.join(H, r"cache\s2_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
c = [x for x in M.columns if x.startswith(("est_g", "emp_g", "ship_g"))]
print(M[M.type_9to10 == "1:1"][["ksic10"] + c].to_string())
print(M[[x for x in M.columns if "ratio" in x] + ["ksic10", "type_9to10"]].to_string())
