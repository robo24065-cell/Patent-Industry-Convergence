# -*- coding: utf-8 -*-
r"""C07-T1 (0)(a)(d): 작업 파일 ↔ 8월 원천(외부데이터\산업분류별) ↔ 현재 KOSIS 공식값(2026-10-13 OpenAPI) 칸 대조.
출력: official_compare.csv, cache\s1_stats.json. 쓰기는 C07-T1 안에서만."""
import os, sys, glob, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(B, r"분석\검증루프_20261012\C07\C07-T1")
W = pd.read_csv(os.path.join(B, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
W.columns = ["basis", "code", "name", "item", "itemnm", "year", "fv", "mask"]
K50 = sorted(W.code.unique())
# 8월 원천
fr = []
for p in sorted(glob.glob(os.path.join(B, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    d = pd.read_csv(p, encoding="cp949", dtype=str)
    d = d[d["[A]시도별"].str.strip() == "00"]
    yc = [c for c in d.columns if "년" in c]
    m = d.melt(id_vars=["[B]산업별", "[Item]항목"], value_vars=yc, var_name="yc", value_name="av")
    m["year"] = m.yc.str.extract(r"(\d{4})").astype(int)
    m["item"] = m["[Item]항목"].str[:3]
    m["c"] = m["[B]산업별"].str.strip()
    m["basis"] = "11차" if "1FS1101" in p else "10차"
    m["file"] = os.path.basename(p)
    fr.append(m[["basis", "c", "item", "year", "av", "file"]])
A = pd.concat(fr)
A = A.drop_duplicates(["basis", "c", "item", "year", "av"])
dupA = A.duplicated(["basis", "c", "item", "year"], keep=False)
print("8월 원천 행", len(A), "서로 다른 값 중복칸", int(dupA.sum()))
# 현재 공식
def lj(p, basis):
    j = json.load(open(p, encoding="utf-8"))
    d = pd.DataFrame(j)
    return pd.DataFrame({"basis": basis, "c": d.C2, "item": d.ITM_ID, "year": d.PRD_DE.astype(int), "ov": d.DT, "lst": d.LST_CHN_DE, "onm": d.C2_NM})
O = pd.concat([lj(os.path.join(H, r"download\kosis10\DT_1FS1001_2007_2020.json"), "10차"),
               lj(os.path.join(H, r"download\kosis11\DT_1FS1101_2020_2024.json"), "11차")])
O9 = lj(os.path.join(H, r"download\kosis9\DT_1F160622_2005_2014.json"), "9차")
num = lambda s: pd.to_numeric(s, errors="coerce")
# (0) 작업 파일 재생성 가능성: W ↔ A
Wc = W.assign(c="C" + W.code)
m0 = Wc.merge(A, on=["basis", "c", "item", "year"], how="left")
m0["avn"] = num(m0.av)
rep = {"W_rows": len(W), "W_in_aug": int(m0.av.notna().sum()),
       "W_eq_aug_exact": int((np.isclose(m0.fv, m0.avn, rtol=0, atol=1e-9) | (m0.fv.isna() & m0.avn.isna())).sum()),
       "W_mask1": int((W["mask"] == 1).sum()),
       "W_mask1_aug_X": int(((m0["mask"] == 1) & m0.av.isin(["X"])).sum()),
       "W_mask1_aug_blank": int(((m0["mask"] == 1) & (m0.av.isna() | (m0.av.str.strip() == ""))).sum()),
       "W_mask1_fv_values": sorted(m0.loc[m0["mask"] == 1, "fv"].astype(str).unique().tolist())[:10]}
# 원천에 있는데 작업 파일에 없는 50개 코드 칸
a50 = A[A.c.isin(["C" + k for k in K50]) & A["item"].isin(["T01", "T02", "T03", "T04", "T05", "T06", "T07"])]
mm = a50.merge(Wc[["basis", "c", "item", "year"]], on=["basis", "c", "item", "year"], how="left", indicator=True)
rep["aug50_cells"] = len(a50); rep["aug50_missing_in_W"] = mm[mm._merge == "left_only"][["basis", "c", "item", "year", "av"]].astype(str).values.tolist()[:40]
rep["aug50_missing_in_W_n"] = int((mm._merge == "left_only").sum())
print(json.dumps(rep, ensure_ascii=False))
# (a) 작업 파일 ↔ 현재 공식(50개 코드)
m1 = Wc.merge(O, on=["basis", "c", "item", "year"], how="left")
m1["ovn"] = num(m1.ov)
m1["comparable"] = m1.fv.notna() & m1.ovn.notna() & (m1["mask"] == 0)
m1["rel"] = np.where(m1.comparable, np.abs(m1.fv - m1.ovn) / np.where(m1.ovn != 0, np.abs(m1.ovn), np.nan), np.nan)
m1.loc[m1.comparable & (m1.ovn == 0), "rel"] = np.where(m1.loc[m1.comparable & (m1.ovn == 0), "fv"] == 0, 0.0, np.inf)
m1["status"] = np.select([m1.ov.isna(), ~m1.comparable, m1.rel <= 0.005], ["공식칸없음", "비교불가(X·결측·마스킹)", "일치(≤0.5%)"], "불일치")
st = {}
comp = m1[m1.comparable]
for key, g in [("전체", comp)] + [(f"item_{i}", g) for i, g in comp.groupby("item")] + [(f"basis_{b}", g) for b, g in comp.groupby("basis")]:
    st[key] = {"n": int(len(g)), "le0.5%": int((g.rel <= 0.005).sum()), "share": round(float((g.rel <= 0.005).mean()), 6),
               "exact": int((g.rel == 0).sum()), "max_rel": float(g.rel.max())}
st["T01T02T04"] = {"n": int(comp.item.isin(["T01", "T02", "T04"]).sum()),
                   "share": round(float((comp[comp.item.isin(["T01", "T02", "T04"])].rel <= 0.005).mean()), 6)}
st["noncomparable"] = m1[~m1.comparable].groupby(["status"]).size().to_dict()
st["mask_vs_officialX"] = {"mask1": int((m1["mask"] == 1).sum()), "mask1_officialX": int(((m1["mask"] == 1) & (m1.ov == "X")).sum()),
                           "officialX_mask0": int(((m1["mask"] == 0) & (m1.ov == "X")).sum()),
                           "mask1_rows": m1.loc[m1["mask"] == 1, ["basis", "code", "item", "year", "fv", "ov"]].astype(str).values.tolist()}
st["mismatch_rows"] = m1[m1.status == "불일치"][["basis", "code", "item", "year", "fv", "ov", "rel"]].astype(str).values.tolist()[:50]
st["LST_CHN_DE"] = O[O.c.isin(["C" + k for k in K50])].groupby("basis").lst.agg(lambda s: sorted(s.unique().tolist())).to_dict()
print(json.dumps(st, ensure_ascii=False, default=str)[:4000])
# 합계(광업·제조업 0, 제조업 C): 8월 원천 ↔ 현재 공식
t = A[A.c.isin(["0", "C"])].merge(O[O.c.isin(["0", "C"])], on=["basis", "c", "item", "year"], how="outer")
t["avn"], t["ovn"] = num(t.av), num(t.ov)
t["rel"] = np.abs(t.avn - t.ovn) / np.abs(t.ovn)
st["totals_aug_vs_official"] = {"n_both": int((t.avn.notna() & t.ovn.notna()).sum()), "le0.5%": int((t.rel <= 0.005).sum()),
                                 "exact": int((t.rel == 0).sum()), "max_rel": float(t.rel.max()),
                                 "official_only_years": sorted(t[t.avn.isna()].year.unique().tolist())}
# 50개 합 / 제조업 C 비중(참고)
s50 = O[O.c.isin(["C" + k for k in K50])].assign(v=lambda d: num(d.ov)).groupby(["basis", "item", "year"]).v.sum()
cC = O[O.c == "C"].assign(v=lambda d: num(d.ov)).set_index(["basis", "item", "year"]).v
sh = (s50 / cC).dropna().unstack("year").round(4)
sh2 = sh.loc[sh.index.get_level_values("item").isin(["T01", "T02", "T04"])]
st["share50_of_C"] = {f"{a}|{b}": {str(y): float(v) for y, v in r.items() if pd.notna(v)} for (a, b), r in sh2.iterrows()}
out = m1[["basis", "code", "name", "item", "year", "fv", "mask", "ov", "lst", "rel", "status"]].rename(
    columns={"fv": "file_value", "ov": "kosis_value_20261013", "lst": "kosis_LST_CHN_DE", "rel": "rel_diff"})
out = out.merge(m0[["basis", "code", "item", "year", "av"]].rename(columns={"av": "aug_raw_value_20260811"}), on=["basis", "code", "item", "year"], how="left")
out["table"] = np.where(out.basis == "10차", "DT_1FS1001", "DT_1FS1101")
tt = t.rename(columns={"c": "code", "av": "aug_raw_value_20260811", "ov": "kosis_value_20261013", "lst": "kosis_LST_CHN_DE", "rel": "rel_diff"})
tt["name"] = tt.onm; tt["status"] = np.where(tt.rel_diff <= 0.005, "합계 일치(≤0.5%)", np.where(tt.avn.isna() | tt.ovn.isna(), "한쪽만", "합계 불일치"))
tt["table"] = np.where(tt.basis == "10차", "DT_1FS1001", "DT_1FS1101")
out = pd.concat([out, tt[["basis", "code", "name", "item", "year", "aug_raw_value_20260811", "kosis_value_20261013", "kosis_LST_CHN_DE", "rel_diff", "status", "table"]]])
out.to_csv(os.path.join(H, "official_compare.csv"), index=False, encoding="utf-8-sig")
json.dump({"reproduce": rep, "official": st}, open(os.path.join(H, r"cache\s1_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(st["totals_aug_vs_official"], ensure_ascii=False)); print(json.dumps(st["share50_of_C"], ensure_ascii=False)[:1500])
