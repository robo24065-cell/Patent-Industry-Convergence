# -*- coding: utf-8 -*-
"""아이디어 N1(특허 KSIC 브리지 산업 나우캐스트) 준비:
1) 기업별 전체 특허 보유 규모(제공범위 밖 포함) — business_rights.right_list(ro), 특허(10)만, 등록연도별 누적.
   ⚠ right_list 는 2026 스냅샷(현 권리자 기준)이라 가중치 분모에만 쓴다(결과변수와 무관한 규모 추정).
2) 기업×연도 KSIC 배분가중 w_ik(t) = (제공범위 생존특허 중 KSIC k 건수) / max(전체 등록특허 누적, 제공범위 생존 합).
출력: cache/alloc_w.parquet (corp_code, t0, ksic, w_share, w_full)"""
import os, sqlite3
import numpy as np, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
C = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
con = sqlite3.connect(f"file:{os.path.join(BASE, r'수집파이프라인/business_rights.sqlite')}?mode=ro", uri=True)
r = pd.read_sql("select replace(corp_rgstno,'-','') jurir, substr(rgst_date,1,4) y, count(distinct rgst_no) n from right_list "
                "where substr(rgst_no,1,2)='10' and corp_rgstno is not null group by 1,2", con)
con.close()
dart = pd.read_parquet(os.path.join(C, "dart_firms.parquet"))
r = r.merge(dart[["jurir_no", "corp_code"]], left_on="jurir", right_on="jurir_no")
r["y"] = pd.to_numeric(r.y, errors="coerce")
rows = []
for t0 in range(2010, 2024):
    s = r[r.y <= t0].groupby("corp_code").n.sum().rename("N_full"); s = s.reset_index(); s["t0"] = t0; rows.append(s)
N = pd.concat(rows)
S = pd.read_parquet(os.path.join(C, "ksic_share.parquet"))
P = pd.read_parquet(os.path.join(C, "patfeat.parquet"))[["corp_code", "t0", "n_alive"]]
W = S.merge(P, on=["corp_code", "t0"]).merge(N, on=["corp_code", "t0"], how="left")
W["n_k"] = W.share * W.n_alive
W["w_full"] = W.n_k / np.maximum(W.N_full.fillna(0), W.n_alive)   # 제공범위 밖 특허 비중만큼 축소
W = W.rename(columns={"share": "w_share"})
W[["corp_code", "t0", "ksic", "w_share", "w_full", "n_k", "N_full", "n_alive"]].to_parquet(os.path.join(C, "alloc_w.parquet"), index=False)
g = W.groupby(["corp_code", "t0"]).w_full.sum()
print("firms with N_full", W.N_full.notna().groupby(W.t0).mean().round(3).to_dict())
print("sum w_full per firm (quantiles)", g.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
