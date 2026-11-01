# -*- coding: utf-8 -*-
# verify_5 (반박 검증) 2: T3 풀 해외 레코드 라벨 변화(대안 A·B)를 pyarrow+pandas로 독립 재계산, k_pure가 '상속 없음'인지 일관성 점검,
# 패널 셀 수와 R8에만 있는 셀의 크기 비교.
import os, sys, json, gc
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; W = BASE + "/분석/검증루프_20261012/C15/C15-T3"
SRC = MC + "/T3_기술셀부상예측/R3/cache_pub/base_rows.parquet"
out = {"pool": {}, "pure_consistency": {}}
tot = dict(n=0, ch_reg=0, ch_pure=0, fam_reg=0)
for cc in ("US", "EP", "JP", "CN"):
    b = pq.read_table(SRC, columns=["cc", "appl_no"], filters=[("cc", "=", cc)]).to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get)
    a = pq.read_table(W + f"/cache/pkr_alt_{cc}.parquet", columns=["appl_no", "k_stored", "k_r8", "k_reg", "k_pure", "fam_r8", "fam_reg"]).to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get)
    assert not a.appl_no.duplicated().any()
    # 상속 없음 일관성: fam_r8 거짓인 문서는 k_r8 == k_pure 이어야 하고, fam_reg 거짓이면 k_reg == k_pure
    nf = ~a.fam_r8; nr = ~a.fam_reg
    eq = lambda x, y: ((x == y).fillna(False).astype(bool) | (x.isna() & y.isna()).astype(bool))
    out["pure_consistency"][cc] = dict(r8_eq_pure_when_nofam=float(eq(a.k_r8[nf], a.k_pure[nf]).mean()),
                                       reg_eq_pure_when_nofam=float(eq(a.k_reg[nr], a.k_pure[nr]).mean()),
                                       stored_eq_r8=float(eq(a.k_stored, a.k_r8).mean()))
    m = b.merge(a, on="appl_no", how="left", validate="many_to_one")
    n = len(m); chr_ = int((~eq(m.k_reg, m.k_stored)).sum()); chp = int((~eq(m.k_pure, m.k_stored)).sum())
    out["pool"][cc] = dict(n_rec=n, missing_doc=int(m.k_stored.isna().sum() - 0), changed_reg=chr_ / n, changed_pure=chp / n, fam_reg=float(m.fam_reg.mean()))
    tot["n"] += n; tot["ch_reg"] += chr_; tot["ch_pure"] += chp; tot["fam_reg"] += int(m.fam_reg.sum())
    print(cc, out["pool"][cc], out["pure_consistency"][cc], flush=True)
    del a, b, m; gc.collect()
out["pool"]["ALL"] = dict(n_rec=tot["n"], changed_reg=tot["ch_reg"] / tot["n"], changed_pure=tot["ch_pure"] / tot["n"], fam_reg=tot["fam_reg"] / tot["n"])
print(out["pool"]["ALL"])
P = {a: pd.read_parquet(W + f"/cache/t3_pkr_{a}/panel_ccy.parquet", columns=["ksic", "mg", "cc", "year", "n"]) for a in ("r8", "reg", "pure")}
out["panel_rows"] = {a: len(d) for a, d in P.items()}
k = ["ksic", "mg", "cc", "year"]
s8 = P["r8"].set_index(k).n; sr = P["reg"].set_index(k).n; sp_ = P["pure"].set_index(k).n
only8 = s8[~s8.index.isin(sr.index)]; both = s8[s8.index.isin(sr.index)]; onlyr = sr[~sr.index.isin(s8.index)]
out["r8_vs_reg_cells"] = dict(only_r8=len(only8), only_reg=len(onlyr), common=len(both), only_r8_median_n=float(only8.median()), common_median_n=float(both.median()),
                              only_reg_median_n=float(onlyr.median()), only_r8_share_n1=float((only8 <= 1).mean()), common_share_n1=float((both <= 1).mean()),
                              only_r8_KR=int((only8.index.get_level_values("cc") == "KR").sum()))
print(out["panel_rows"], out["r8_vs_reg_cells"])
json.dump(out, open(W + "/verify_5/v5_labels.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
