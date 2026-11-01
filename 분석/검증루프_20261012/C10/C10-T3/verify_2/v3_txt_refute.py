# -*- coding: utf-8 -*-
r"""C10-T3 verify_2: 'TXT 해외 재현율 29.6–31.1%는 결함 상속값, 올바른 연결로 29.4–30.7%' 반박 시도.
작업자 s2·verify_0 v2와 다른 경로: 저장 alt_assign(c5_R10)의 k_txt를 출발점으로, m_txt='family'(결함 상속) 행만 표 값으로 되돌리고
올바른 연결로 다시 상속. 연결은 (a) L_REG(fam_reg ∪ R8 CN) (b) 역방향 연결 L_REV(KR 문서 meta.family의 US/JP/EP 등록번호 토큰 → 해외 rgst_no, 독립 구축) ∪ R8 CN.
읽기만. 쓰기: verify_2/v3_out.json"""
import os, sys, json, time, re
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
V = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(V, "..", "..", "..", "..", "모델경쟁_20260927"))
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
OUT = {}
cols = ["cc", "appl_no", "appl_year", "is_inv", "ksic_prov", "code", "k_in_code", "k_txt", "m_txt"]
A = pd.concat([pq.read_table(os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10", "alt_assign", f"alt_{c}.parquet"), columns=cols).to_pandas() for c in ("US", "JP", "CN", "EP")])
A = A[A.is_inv & A.appl_year.between(2015, 2019)].reset_index(drop=True)
for c in A.columns:
    if str(A[c].dtype) == "category": A[c] = A[c].astype(object)
log("창 문서", A.groupby("cc").size().to_dict())
rec = lambda s: {cc: round(float((s[A.cc == cc] == A.ksic_prov[A.cc == cc]).mean()) * 100, 3) for cc in ("US", "JP", "CN", "EP")}
OUT["stored"] = rec(A.k_txt); log("저장", OUT["stored"])
OUT["m_txt_counts"] = A.groupby(["cc", "m_txt"]).size().unstack(fill_value=0).to_dict("index")
# ---- 표(fallback) 재구축: txt_assign KR 'text' × rec_conc sg ----
rc = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), columns=["cc", "appl_no", "sg"]).to_pandas()
ta = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet"), columns=["cc", "appl_no", "code", "ksic_txt", "method"]).to_pandas()
for d in (rc, ta):
    for c in d.columns:
        if str(d[c].dtype) == "category": d[c] = d[c].astype(object)
krt = ta[(ta.cc == "KR") & (ta.method == "text")].merge(rc[rc.cc == "KR"][["appl_no", "sg"]], on="appl_no")
krt["mg"] = krt.sg.str.split("/").str[0]; krt["sc"] = krt.sg.str[:4]
A = A.merge(rc, on=["cc", "appl_no"], how="left"); A["mg"] = A.sg.str.split("/").str[0]; A["sc"] = A.sg.str[:4]


def mode_tab(df, keys, min_n):
    g = df.dropna(subset=keys)
    for k in keys: g = g[g[k] != ""]
    g = g.groupby(keys + ["ksic_txt"]).size().rename("n").reset_index()
    if min_n: g = g[g.groupby(keys).n.transform("sum") >= min_n]
    g["kk"] = g.ksic_txt.astype(int)
    return g.sort_values(keys + ["n", "kk"], ascending=[True] * len(keys) + [False, True]).drop_duplicates(keys).set_index(keys).ksic_txt


fb = pd.Series([None] * len(A), index=A.index, dtype=object)
for lvl in ("sg", "mg", "sc"):
    m = pd.MultiIndex.from_frame(A[["code", lvl]].fillna("")).map(mode_tab(krt, ["code", lvl], 3))
    fb = fb.fillna(pd.Series(m, index=A.index))
fb = fb.fillna(A.code.map(mode_tab(krt, ["code"], 0)))
ipc = A.m_txt.isin(["ipc_sg", "ipc_mg", "ipc_sc", "code_mode"])
OUT["fallback_check_ipc_rows"] = {"n": int(ipc.sum()), "same": round(float((fb[ipc] == A.k_txt[ipc]).mean()), 5)}
log("표 재현(비상속 풀 행)", OUT["fallback_check_ipc_rows"])
fam_rows = A.m_txt == "family"
base = A.k_txt.copy(); base[fam_rows] = fb[fam_rows]
OUT["pure"] = rec(base); log("pure", OUT["pure"])
# ---- 연결 ----
r8 = pd.read_parquet(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet"))
reg = pd.read_parquet(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"])
# 역방향: KR meta.family 의 US/JP/EP 등록번호 토큰(숫자 ≤8자리, 8자리로 0 채움) → 해외 meta.rgst_no
meta = ds.dataset(os.path.join(MC, "_shared", "meta"), format="parquet")
fr = meta.to_table(columns=["cc", "appl_no", "rgst_no"], filter=ds.field("cc").isin(["US", "JP", "EP"])).to_pandas()
fr["cc"] = fr.cc.astype(object); fr = fr.dropna(subset=["rgst_no"]); fr["rg"] = fr.rgst_no.str.replace(r"\D", "", regex=True).str[-8:].str.zfill(8)
fr = fr[fr.rg.str.len() == 8]
dup = fr.duplicated(["cc", "rg"], keep=False); OUT["rev_rgst_dup_share"] = round(float(dup.mean()), 5); fr = fr[~dup]
pat = re.compile(r"\b(US|JP|EP)0*(\d{5,8})\b")
parts = []
for b in meta.to_batches(columns=["appl_no", "family"], filter=(ds.field("cc") == "KR"), batch_size=500000):
    k = b.to_pandas().dropna(subset=["family"])
    s = k.set_index("appl_no").family.str.split("|").explode()
    s = s[s.str.match(r"^(US|JP|EP)0*\d{5,8}$")]
    parts.append(pd.DataFrame({"kr_appl_no": s.index, "cc": s.str[:2].values, "rg": s.str[2:].str.lstrip("0").str.zfill(8).values}).drop_duplicates())
rv = pd.concat(parts).drop_duplicates().merge(fr[["cc", "rg", "appl_no"]], on=["cc", "rg"])[["cc", "appl_no", "kr_appl_no"]].drop_duplicates()
log("역방향 연결", rv.groupby("cc").size().to_dict())
krprov = meta.to_table(columns=["appl_no", "ksic"], filter=ds.field("cc") == "KR").to_pandas().rename(columns={"appl_no": "kr_appl_no", "ksic": "kk"})
krprov["kk"] = krprov.kk.astype(object)
fprov = A[["cc", "appl_no", "ksic_prov"]]
links = {"r8": r8, "reg": pd.concat([reg, r8[r8.cc == "CN"]]).drop_duplicates(), "rev": pd.concat([rv, r8[r8.cc == "CN"]]).drop_duplicates()}
links["reg_rev"] = pd.concat([links["reg"], links["rev"]]).drop_duplicates()
OUT["link_quality_window"] = {}
for nm, L in links.items():
    q = L.merge(fprov, on=["cc", "appl_no"]).merge(krprov, on="kr_appl_no")
    OUT["link_quality_window"][nm] = {cc: [int((q.cc == cc).sum()), round(float((q.ksic_prov == q.kk)[q.cc == cc].mean()), 4) if (q.cc == cc).any() else None] for cc in ("US", "JP", "EP", "CN")}
log("연결 품질(창 안 쌍 수, 제공 KSIC 일치)", OUT["link_quality_window"])
ov = links["reg"].merge(rv, on=["cc", "appl_no", "kr_appl_no"]); OUT["reg_rev_pair_overlap"] = ov.groupby("cc").size().to_dict()
# ---- 상속: 풀 안 해외 복수코드 문서 ← 풀 안 같은 코드 KR 문서 ksic_txt 최빈 ----
kc = pd.read_csv(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str).groupby("code").ksic.size()
multi = set(kc[kc > 1].index)
krk = ta[ta.cc == "KR"][["appl_no", "code", "ksic_txt"]].rename(columns={"appl_no": "kr_appl_no", "code": "kcode"})
poolkey = pd.MultiIndex.from_frame(ta[ta.cc != "KR"][["cc", "appl_no"]])
inpool = pd.MultiIndex.from_frame(A[["cc", "appl_no"]]).isin(poolkey) & A.code.isin(multi)
OUT["pool_multi_rows"] = int(inpool.sum()); OUT["stored_family_rows_outside_pool"] = int((fam_rows & ~inpool).sum())
for nm, L in links.items():
    x = L.merge(A.loc[inpool, ["cc", "appl_no", "code"]], on=["cc", "appl_no"]).merge(krk, on="kr_appl_no")
    x = x[x.kcode == x.code]
    g = x.groupby(["cc", "appl_no", "ksic_txt"]).size().rename("n").reset_index(); g["kk"] = g.ksic_txt.astype(int)
    g = g.sort_values(["cc", "appl_no", "n", "kk"], ascending=[True, True, False, True]).drop_duplicates(["cc", "appl_no"]).set_index(["cc", "appl_no"]).ksic_txt
    f = pd.Series(pd.MultiIndex.from_frame(A[["cc", "appl_no"]]).map(g), index=A.index); f[~inpool] = None
    lab = f.fillna(base)
    OUT[f"txt_{nm}"] = rec(lab); OUT[f"inherit_rows_{nm}"] = A.cc[f.notna()].value_counts().to_dict()
    OUT[f"inherit_acc_{nm}"] = {cc: round(float((f[f.notna() & (A.cc == cc)] == A.ksic_prov[f.notna() & (A.cc == cc)]).mean()), 4) for cc in ("US", "JP", "EP") if (f.notna() & (A.cc == cc)).any()}
    if nm == "r8": OUT["r8_rebuild_equals_stored_rows"] = round(float((lab == A.k_txt).mean()), 6)
    log(nm, OUT[f"txt_{nm}"], OUT[f"inherit_rows_{nm}"], OUT[f"inherit_acc_{nm}"])
rng = lambda d: f"{min(d.values()):.1f}–{max(d.values()):.1f}"
OUT["ranges"] = {k: rng(OUT[k]) for k in ("stored", "pure", "txt_r8", "txt_reg", "txt_rev", "txt_reg_rev")}
json.dump(OUT, open(os.path.join(V, "v3_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("done")
