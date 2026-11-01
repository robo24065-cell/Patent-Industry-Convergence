# -*- coding: utf-8 -*-
r"""C10-T3 verify_0 v2: TXT 해외 배정의 독립 재계산(pandas). 규칙: 연계표 코드 없음 → 미배정, 코드 안 KSIC 1개 → 그 KSIC,
복수 → (T3 풀 안 문서만) 연결된 풀 안 같은 코드 KR 문서 ksic_txt 최빈 → (코드,sg)→(코드,mg)→(코드,sc)→코드 최빈표(풀 KR 'text' 문서, 키당 ≥3).
연결 = L_REG(fam_reg ∪ R8 CN) 또는 R8(결함, 재현용). 재현율(발명 2015–19)과 ρ(해외 RTA 2017–19 ↔ RCA)를 낸다. 쓰기: verify_0/v2_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
from scipy.stats import spearmanr
V = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(V, "..", "..", "..", "..", "모델경쟁_20260927"))
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
meta = ds.dataset(os.path.join(MC, "_shared", "meta"), format="parquet")
F = meta.to_table(columns=["cc", "appl_no", "appl_year", "ksic", "prio_cc"], filter=(ds.field("cc") != "KR") & (ds.field("appl_year") >= 2015) & (ds.field("appl_year") <= 2019)).to_pandas()
F = F.astype({c: object for c in F.columns if str(F[c].dtype) == "category"})
L_ = F.appl_no.str.len(); s5 = F.appl_no.str[4:5]
F = F[np.where(F.cc == "CN", L_.isin([12, 14]) & s5.isin(["1", "8"]), np.where(F.cc == "JP", L_.isin([8, 17]), True))].copy()
F["kr_origin"] = F.prio_cc.fillna("").str.split("|").str[0] == "KR"
rc = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), columns=["cc", "appl_no", "sg", "code"]).to_pandas()
km = pd.read_csv(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)
kc = km.groupby("code").ksic.agg(list); single = {c: v[0] for c, v in kc.items() if len(v) == 1}; multi = {c for c, v in kc.items() if len(v) > 1}
F = F.merge(rc, on=["cc", "appl_no"], how="inner"); F.loc[~F.code.isin(kc.index), "code"] = None
F["mg"] = F.sg.str.split("/").str[0]; F["sc"] = F.sg.str[:4]
ta = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet"), columns=["cc", "appl_no", "code", "ksic_txt", "method"]).to_pandas()
krt = ta[(ta.cc == "KR") & (ta.method == "text")].merge(rc[rc.cc == "KR"][["appl_no", "sg"]], on="appl_no")
krt["mg"] = krt.sg.str.split("/").str[0]; krt["sc"] = krt.sg.str[:4]


def tab(keys, min_n=3):
    g = krt.dropna(subset=keys)
    for k in keys: g = g[g[k] != ""]
    g = g.groupby(keys + ["ksic_txt"]).size().rename("n").reset_index()
    if min_n: g = g[g.groupby(keys).n.transform("sum") >= min_n]
    g = g.assign(kk=g.ksic_txt.astype(int)).sort_values(keys + ["n", "kk"], ascending=[True] * len(keys) + [False, True])
    return g.drop_duplicates(keys).set_index(keys).ksic_txt


lab = pd.Series(index=F.index, dtype=object)
for lvl in ("sg", "mg", "sc"):
    tb = tab(["code", lvl]); idx = pd.MultiIndex.from_frame(F[["code", lvl]])
    lab = lab.fillna(pd.Series(idx.map(tb), index=F.index))
lab = lab.fillna(F.code.map(tab(["code"], min_n=0)))
lab[F.code.isna()] = None
sg_ = F.code.map(single); lab[sg_.notna()] = sg_[sg_.notna()]
pool = pd.MultiIndex.from_frame(F[["cc", "appl_no"]]).isin(pd.MultiIndex.from_frame(ta[ta.cc != "KR"][["cc", "appl_no"]]))
reg = pq.read_table(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"]).to_pandas()
r8 = pq.read_table(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet")).to_pandas()
links = {"reg": pd.concat([reg, r8[r8.cc == "CN"]]).drop_duplicates(), "r8": r8}
krk = ta[ta.cc == "KR"][["appl_no", "code", "ksic_txt"]].rename(columns={"appl_no": "kr_appl_no", "code": "kcode"})
fo = ta[(ta.cc != "KR") & ta.code.isin(multi)][["cc", "appl_no", "code"]]
OUT = {}
ek = pd.read_csv(os.path.join(MC, "_shared_report", "fusion_advantage", "map5", "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); s_ = v / v.sum(); K50 = sorted(km.ksic); RCA = s_.div(s_.mean(1), axis=0).KR.reindex(K50)
P3 = F.appl_year.between(2017, 2019)
def abroad(l):
    d = pd.DataFrame({"k": l, "o": F.kr_origin})[P3 & l.notna()]
    k = d[d.o].groupby("k").size().reindex(K50).fillna(0); a = d.groupby("k").size().reindex(K50); return (k / a) / (k.sum() / a.sum())
def rho(x):
    d = pd.concat([x, RCA], axis=1).dropna(); return round(float(spearmanr(d.iloc[:, 0], d.iloc[:, 1]).statistic), 4)
for nm, L in links.items():
    x = L.merge(fo, on=["cc", "appl_no"]).merge(krk, on="kr_appl_no"); x = x[x.kcode == x.code]
    g = x.groupby(["cc", "appl_no", "ksic_txt"]).size().rename("n").reset_index().assign(kk=lambda d: d.ksic_txt.astype(int))
    g = g.sort_values(["cc", "appl_no", "n", "kk"], ascending=[True, True, False, True]).drop_duplicates(["cc", "appl_no"]).set_index(["cc", "appl_no"]).ksic_txt
    fam = pd.Series(pd.MultiIndex.from_frame(F[["cc", "appl_no"]]).map(g), index=F.index)
    fam[~pool | ~F.code.isin(multi)] = None
    l2 = fam.fillna(lab); l2[F.code.isna()] = None
    OUT[f"txt_recall_{nm}"] = {cc: round(float((l2[F.cc == cc] == F.ksic[F.cc == cc]).mean()) * 100, 3) for cc in ("US", "JP", "CN", "EP")}
    OUT[f"txt_rho_{nm}"] = rho(abroad(l2))
OUT["txt_recall_pure"] = {cc: round(float((lab[F.cc == cc] == F.ksic[F.cc == cc]).mean()) * 100, 3) for cc in ("US", "JP", "CN", "EP")}
OUT["txt_rho_pure"] = rho(abroad(lab))
json.dump(OUT, open(os.path.join(V, "v2_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(OUT)
