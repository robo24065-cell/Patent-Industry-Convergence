# -*- coding: utf-8 -*-
"""C07-T3 s02a: future()와 같은 방식으로 reg_lt20250101(사건 < 2025-01-01)에서 학습 원점 t0 2021·2022(레이블 ≤ 2024-12-31)와
채점 원점 t0=2024(no_label) 행을 만든다. 메모리 절약을 위해 필요한 열만 저장. 2025년 이후 사건은 읽지 않는다.
→ ../cache/s02a_tr2122.parquet, ../cache/s02a_te2024.parquet (내부용, 반출 금지)"""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c07lib import *  # noqa
t = time.time()
COLS = list(dict.fromkeys(["rgst_no", "t0", "y", "ksic", "owner_v2", "own_disp", "own_trial"] + NEED))
R = L.extract("20250101"); X = L.prep(R); del R
print("X", rss(), round(time.time() - t), flush=True)
assert X["rt"].d.max() < 20250101 and X["disp"].d.max() < 20250101
FT = L.family_tokens(X["pt"]); EXTU = L.external_unit(); KI = L.ksic_industry()
TRs = []
for t0 in (2021, 2022):
    B = L.to32(L.build_t0(t0, X, FT, EXTU, KI, label_disp=X["disp"], label_end=(t0 + 2) * 10000 + 1231))
    TRs.append(B[COLS].copy()); del B
    print("tr", t0, len(TRs[-1]), int(TRs[-1].y.sum()), rss(), round(time.time() - t), flush=True)
pd.concat(TRs, ignore_index=True).to_parquet(os.path.join(W, "cache", "s02a_tr2122.parquet"), index=False)
del TRs
TE = L.to32(L.build_t0(2024, X, FT, EXTU, KI, no_label=True)).reset_index(drop=True)
TE["y"] = 0
TE = TE[COLS + [c for c in ("mg",) if c in TE.columns]].copy()
# 분쟁 이력 하위집단(C01 규칙을 t0=2024로 이동): 무효·권리범위 예고등록 날짜 기준
D = X["disp"]
pre = set(D[D.d < 20230101].rgst_no); mid = set(D[(D.d >= 20230101) & (D.d <= 20241231)].rgst_no)
TE["H_pre"] = TE.rgst_no.isin(pre).astype(int)
TE["H_2324only"] = (TE.rgst_no.isin(mid) & ~TE.rgst_no.isin(pre)).astype(int)
TE["H_any"] = (TE.H_pre + TE.H_2324only).clip(upper=1)
TE.to_parquet(os.path.join(W, "cache", "s02a_te2024.parquet"), index=False)
out = dict(n_te=len(TE), H_pre=int(TE.H_pre.sum()), H_2324only=int(TE.H_2324only.sum()), own_disp_pos=int((TE.own_disp > 0).sum()),
           peak_rss=rss(), sec=round(time.time() - t))
json.dump(out, open(os.path.join(W, "cache", "s02a_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(out)
