# -*- coding: utf-8 -*-
r"""final 모드 패널(학습 t0 2015–2020 + 이미 채점된 시험 t0=2022)을 다시 만든다.
--which orig : 동결 코드 그대로(NPS 집계·시도는 원자료에서 새로 생성) → 동결 cache/final_panel.parquet와 비교
--which alias: NPS 별칭 보정판 → cache/final_panel_alias.parquet
🔒 새 결과 구간은 계산하지 않는다(t0=2021 제외는 동결 label_allowed 그대로)."""
import argparse
from w_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--which", required=True, choices=["orig", "alias"]); a = ap.parse_args()
D = build_final_panel(alias=(a.which == "alias"))
assert sorted(D.t0.unique().tolist()) == [2015, 2016, 2017, 2018, 2019, 2020, 2022]
fn = os.path.join(VC, "final_panel_orig.parquet" if a.which == "orig" else "final_panel_alias.parquet")
D.to_parquet(fn, index=False)
log("saved", fn, D.shape, {int(t): round(float(g.nps_present.mean()), 4) for t, g in D.groupby("t0")})
if a.which == "orig":
    F = pd.read_parquet(os.path.join(FROZEN_CACHE, "final_panel.parquet"))
    cols = ["y", "y1"] + FE
    res = {"rows_frozen": len(F), "rows_now": len(D)}
    m = F[["applicantCd", "t0"] + cols].merge(D[["applicantCd", "t0"] + cols], on=["applicantCd", "t0"], suffixes=("_f", "_n"))
    res["rows_matched"] = len(m)
    diff = {}
    for c in cols:
        x, z = m[c + "_f"].astype(float), m[c + "_n"].astype(float)
        eq = ((x - z).abs() < 1e-9) | (x.isna() & z.isna())
        diff[c] = float(eq.mean())
    res["col_match_min"] = min(diff.values()); res["not_full"] = {k: v for k, v in diff.items() if v < 1}
    json.dump(res, open(os.path.join(HERE, "w1_repro_panel.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(res)
