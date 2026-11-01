# -*- coding: utf-8 -*-
r"""C10-T3 verify_4 v4c: 수정판 T3 TXT 캐시(t3_txt_reg)의 해외 라벨이 올바른 연결(fam_reg) TXT 상속 규칙과 맞는지 독립 대조.
상속 규칙(R9 s01 정의): 연계표 다중 코드 문서만, 같은 코드의 KR 문서 TXT 라벨 최빈(동률 KSIC 정수 작은 것). KR 원천 두 변형(모든 method / 'text'만).
읽기만: T3 TXT_R9 txt_assign, conc_ksic_map, fam_reg, R3 base_rows, c5 t3cache_hyb_txt, C10 t3_txt_reg. 쓰기: verify_4/v4c_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
SRC = os.path.join(MC, "T3_기술셀부상예측", "R3", "cache_pub", "base_rows.parquet")
STO = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10", "t3cache_hyb_txt", "base_rows.parquet")
FIX = os.path.join(W, "cache", "t3_txt_reg", "base_rows.parquet")
TA = ds.dataset(os.path.join(MC, "T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet"), format="parquet")
obj = lambda df: df.astype({c: object for c in df.columns if str(df[c].dtype) == "category"})
rd = lambda p, cc, cols: obj(ds.dataset(p, format="parquet").to_table(columns=cols, filter=ds.field("cc") == cc).to_pandas())
km = pd.read_csv(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)
multi = set(km.groupby("code").ksic.size().loc[lambda s: s > 1].index)
krt = rd(TA.files[0], "KR", ["appl_no", "code", "ksic_txt", "method"]).rename(columns={"appl_no": "kr_appl_no", "code": "kcode"})
reg = pq.read_table(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"]).to_pandas().drop_duplicates()
OUT = {}


def inh(fo, link, kr):
    x = link.merge(fo, on="appl_no").merge(kr, on="kr_appl_no"); x = x[x.kcode == x.code]
    g = x.groupby(["appl_no", "ksic_txt"]).size().rename("n").reset_index().assign(kk=lambda d: d.ksic_txt.astype(int))
    return g.sort_values(["appl_no", "n", "kk"], ascending=[True, False, True]).drop_duplicates("appl_no").set_index("appl_no").ksic_txt


for cc in ("US", "JP", "EP"):
    ta = rd(TA.files[0], cc, ["appl_no", "code", "ksic_txt", "method"])
    fo = ta[ta.code.isin(multi)][["appl_no", "code"]]
    link = reg[reg.cc == cc][["appl_no", "kr_appl_no"]]
    i_all = inh(fo, link, krt[["kr_appl_no", "kcode", "ksic_txt"]]); i_text = inh(fo, link, krt[krt.method == "text"][["kr_appl_no", "kcode", "ksic_txt"]])
    P = rd(SRC, cc, ["appl_no"]).drop_duplicates()
    P = P.merge(rd(STO, cc, ["appl_no", "ksic"]).rename(columns={"ksic": "k_sto"}), on="appl_no", how="left")
    P = P.merge(rd(FIX, cc, ["appl_no", "ksic"]).rename(columns={"ksic": "k_fix"}), on="appl_no", how="left")
    P = P.merge(ta[["appl_no", "method"]].drop_duplicates("appl_no"), on="appl_no", how="left")
    P["i_all"] = P.appl_no.map(i_all); P["i_text"] = P.appl_no.map(i_text)
    ha, ht = P.i_all.notna(), P.i_text.notna(); fam = P.method == "family"
    same = lambda a, b: (P[a] == P[b]) | (P[a].isna() & P[b].isna())
    r = dict(n_pool=int(len(P)), n_inh_all=int(ha.sum()), n_inh_text=int(ht.sum()), n_r8_family=int(fam.sum()),
             fix_eq_inh_all=round(float((P.k_fix[ha] == P.i_all[ha]).mean()), 5), fix_eq_inh_text=round(float((P.k_fix[ht] == P.i_text[ht]).mean()), 5),
             unchanged_elsewhere=round(float(same("k_fix", "k_sto")[~ha & ~fam].mean()), 5),
             r8fam_not_inh_changed=round(float((~same("k_fix", "k_sto"))[fam & ~ha].mean()), 4),
             changed_total=round(float((~same("k_fix", "k_sto")).mean()), 4))
    OUT[cc] = r; log(cc, r)
json.dump(OUT, open(os.path.join(V, "v4c_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done")
