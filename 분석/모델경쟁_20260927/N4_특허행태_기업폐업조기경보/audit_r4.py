# -*- coding: utf-8 -*-
r"""N4 최종시험 전 감사(🔒 최종시험 레이블은 만들지 않는다 — dev 원점만 사용).
(b) 피처 시점 '시간여행' 검사: 원자료를 원점 t0 말에서 잘라(미래 사건·미래 폐업일·미래 스냅샷 제거) 피처를 다시 만들고,
    동결 dev 패널(cache/dev_panel.parquet)의 같은 원점 피처와 값이 같은지 비교한다. 다르면 그 피처가 t0 이후 정보를 쓴 것.
원자료 기간 점검(NPS 스냅샷 ym, DART avail_date, NTIS 연도)도 함께 기록 → audit_r4.json"""
import os, json
import numpy as np, pandas as pd
from n4lib import *
from n4feat import load_ctx, build_all

SPEC = json.load(open(os.path.join(OUT, "final_spec_r4.json"), encoding="utf-8"))
FE = SPEC["features"]
res = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "checks": {}}

# ---- 원자료 기간 점검 ----
ym = {}
for s in snaps_for("final"):
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s.replace('-', '')}.parquet"), columns=["ym"])
    ym[s] = sorted(map(str, d.ym.dropna().unique()))[:3]
res["checks"]["nps_snapshot_ym"] = ym
dd = pd.read_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"))
lag = (dd.avail_date - pd.to_datetime((dd.fy).astype(str) + "-12-31")).dt.days
res["checks"]["dart_avail_minus_fyend_days"] = dict(min=int(lag.min()), max=int(lag.max()), fy_max=int(dd.fy.max()))
pj = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["year"])
res["checks"]["ntis_project_year_range"] = [int(pj.year.min()), int(pj.year.max())]
log("raw checks", res["checks"])

# ---- 시간여행 검사 ----
P = pd.read_parquet(os.path.join(CACHE, "dev_panel.parquet"))
AF0 = load_app_firm(); AF0 = AF0[(AF0.applicantCd.astype(str).str[0] == "1") & AF0.known_date.notna()]
R0 = load_registry(); ns0 = load_nts(); A2K = a2key()
ctx0 = load_ctx("dev", set(P.applicantCd))
tt = {}
for t0 in [2017, 2021]:
    end = t0 * 10000 + 1231
    R = {k: v.copy() for k, v in R0.items()}
    for k in ["lapnp", "trf", "py", "rec"]:
        R[k] = R[k][R[k].d <= end]
    R["pt"] = R["pt"][R["pt"].rgstDate <= end].copy()
    R["pt"].loc[R["pt"].ext_d > end, "ext_d"] = np.nan
    ns = ns0.copy(); ns.loc[ns.end > end, "end"] = np.nan          # 미래 폐업일 숨김
    AF = AF0[AF0.known_date <= end]
    NPS = {s: v for s, v in load_nps("dev").items() if s <= f"{t0}-12"}
    ctx = dict(ctx0); ctx["R"] = R; ctx["NPS"] = NPS
    ctx["pj"] = ctx0["pj"][ctx0["pj"].year <= t0]
    ctx["dart"] = ctx0["dart"][ctx0["dart"].avail <= end]
    ctx["own"] = ctx0["own"][ctx0["own"].d <= end]; ctx["sur"] = ctx0["sur"][ctx0["sur"].d <= end]
    ctx["nl"] = ctx0["nl"][ctx0["nl"] <= t0]
    B = build_base(t0, "dev", AF, R, ns, NPS, A2K)
    X = build_all(B, t0, ctx, AF, ns)
    T = pd.concat([B, X], axis=1).set_index("applicantCd")
    Q = P[P.t0 == t0].set_index("applicantCd")
    same_pop = set(T.index) == set(Q.index)
    T = T.reindex(Q.index)
    diff = {}
    for c in FE:
        a, b = T[c].astype(float).values, Q[c].astype(float).values
        bad = ~((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-9, atol=1e-12))
        diff[c] = int(bad.sum())
    tt[str(t0)] = dict(n_panel=int(len(Q)), n_trunc=int(len(B)), same_population=bool(same_pop),
                       n_features_with_diff=int(sum(v > 0 for v in diff.values())), diffs={k: v for k, v in diff.items() if v > 0})
    log("time-travel", t0, tt[str(t0)])
res["checks"]["time_travel"] = tt

# ---- (d) 군집 단위 ----
res["checks"]["cluster"] = {str(t): dict(n=int((P.t0 == t).sum()), n_bizno=int(P.loc[P.t0 == t, "bizno"].nunique())) for t in sorted(P.t0.unique())}
# ---- (c) 기준선 정의 ----
res["checks"]["M0_matches_n4lib"] = SPEC["baseline_M0"] == M0
json.dump(res, open(os.path.join(OUT, "audit_r4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("saved audit_r4.json")
