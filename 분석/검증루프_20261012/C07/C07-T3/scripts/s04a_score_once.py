# -*- coding: utf-8 -*-
"""C07-T3 s04a: 🔒 1회 전향 채점(PREREG.txt sha256 505f53e6…). prospective_result.json이 있으면 거부.
- 2025-01-01 이후 무효·권리범위 예고등록을 sqlite(mode=ro)에서 처음 읽는다(이 스크립트가 유일한 결과 조회).
- 주창(~2026-06-30)·보조 A(~2026-08-15)·보조 B(~2026-08-10) 레이블
- 공식 evaluate 정의의 지표: AUC, skill vs M0/SB, 상위 1/5/10%, 상위 100, PR-AUC, 군집 부트스트랩 1,000회
→ prospective_result.json, ../cache/s04a_labels.parquet(내부), ../cache/s04a_events.parquet(내부), ../cache/s04a_rightpos.parquet(내부)"""
import os, sys, json, time, sqlite3, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c07lib import *  # noqa
from sklearn.metrics import average_precision_score
t = time.time()
OUTF = os.path.join(W, "prospective_result.json")
if os.path.exists(OUTF):
    sys.exit("prospective_result.json이 이미 있다 — 전향 채점은 1회만.")
PREREG_SHA = "505f53e6db1380981f0479ec15ec019f4fa288db98c01b8fc9ba9d16d717653f"
assert hashlib.sha256(open(os.path.join(W, "PREREG.txt"), "rb").read()).hexdigest() == PREREG_SHA, "PREREG가 바뀌었다"
s01 = json.load(open(os.path.join(W, "cache", "s01_out.json"), encoding="utf-8"))
s02 = json.load(open(os.path.join(W, "cache", "s02b_out.json"), encoding="utf-8"))
assert s01["R1_pass"] and s02["R2_pass"]
S = pd.read_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"))
FS = pd.read_parquet(os.path.join(L.HERE, "future_scores_r6.parquet"))
assert (S.rgst_no.values == FS.rgst_no.values).all() and np.array_equal(S.p_champ.values, FS.p.values)
# ---------------- 결과 조회(유일)
c = sqlite3.connect(f"file:{L.REG}?mode=ro", uri=True)
EV = pd.read_sql(f"select rgst_no, seq, rgstCsDate, rgstCsName from right where rgstCsDate >= '20250101' and rgstCsName in ('{L.DISP[0]}','{L.DISP[1]}')", c)
EV["d"] = pd.to_numeric(EV.rgstCsDate, errors="coerce")
EV.to_parquet(os.path.join(W, "cache", "s04a_events.parquet"), index=False)
pos_all = sorted(set(EV.rgst_no))
q = "select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason from right where rgst_no in (%s)" % ",".join("'%s'" % r for r in pos_all)
RP = pd.read_sql(q, c) if pos_all else pd.DataFrame(columns=["rgst_no", "seq", "rgstCsDate", "rgstCsName", "rgstCsReason"])
RP.to_parquet(os.path.join(W, "cache", "s04a_rightpos.parquet"), index=False)
WIN = {"main": 20260630, "auxA": 20260815, "auxB": 20260810}
DAYS = {"main": 546, "auxA": 592, "auxB": 588}
Lb = S[["rgst_no", "ksic"]].copy()
for k, hi in WIN.items():
    e = EV[(EV.d >= 20250101) & (EV.d <= hi)]
    Lb["y_" + k] = Lb.rgst_no.isin(set(e.rgst_no)).astype(int)
Lb.to_parquet(os.path.join(W, "cache", "s04a_labels.parquet"), index=False)
print("labels", {k: int(Lb["y_" + k].sum()) for k in WIN}, round(time.time() - t), flush=True)
# ---------------- 지표
g = L.cluster_key(S.assign(owner_v2=S.owner_v2.fillna("")))
P = {"champ": S.p_champ.values, "M0": S.p_M0.values, "SB": S.p_SB.values}
def metrics(y, p, pre):
    r = {f"{pre}auc": float(L.auc(y, p)), f"{pre}prauc": float(average_precision_score(y, p))}
    for nm, fr in (("1", 0.01), ("5", 0.05), ("10", 0.10)):
        prec, hit, k = L.topk(y, p, fr)
        r[f"{pre}top{nm}pct_prec"] = prec; r[f"{pre}top{nm}pct_hits"] = hit; r[f"{pre}top{nm}pct_capture"] = hit / max(int(y.sum()), 1); r[f"k_top{nm}pct"] = k
    r[f"{pre}hit100"] = L.topk(y, p, k=100)[1]
    return r
def cap_ci(y, p, groups, k, n=1000, seed=2026):
    rng = np.random.default_rng(seed); o = np.argsort(-p, kind="stable")[:k]; top = np.zeros(len(y), bool); top[o] = True
    codes, uniq = pd.factorize(np.asarray(groups)); G = len(uniq); v = []
    for _ in range(n):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[codes].astype(float)
        den = (w * y).sum()
        if den > 0:
            v.append((w * y * top).sum() / den)
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]
RES = dict(task="C07-T3", prereg_sha256=PREREG_SHA, t0=2024, n=int(len(S)), scored_at=time.strftime("%Y-%m-%d %H:%M:%S"),
           repro=dict(R1_t2022_maxdiff={k: s01[k]["max_abs_diff"] for k in MODELS}, R2_champ_vs_future=s02["champ_vs_future"]["max_abs_diff"]),
           n_owner_clusters=int(pd.Series(g).nunique()), windows={})
OFF = json.load(open(os.path.join(L.HERE, "final_result_r6.json"), encoding="utf-8"))
for k in WIN:
    y = Lb["y_" + k].values
    r = dict(window=["2025-01-01", str(WIN[k])], days=DAYS[k], pos=int(y.sum()), base_rate_permil=float(1000 * y.mean()),
             base_rate_permil_per730d=float(1000 * y.mean() * 730 / DAYS[k]))
    for m, p in P.items():
        r.update(metrics(y, p, m + "_"))
    for ref in ("M0", "SB"):
        r[f"skill_vs_{ref}"] = float(L.skill(r["champ_auc"], r[f"{ref}_auc"]))
    if k == "main":
        for ref, sd in (("M0", 2024), ("SB", 2025)):
            ci, cia, cib = L.boot_ci(y, P["champ"], P[ref], g, n=1000, seed=sd)
            r[f"skill_vs_{ref}_ci"] = ci; r[f"{ref}_auc_ci"] = cib
            if ref == "M0":
                r["champ_auc_ci"] = cia
            print("boot", ref, ci, round(time.time() - t), flush=True)
        r["champ_top1pct_capture_ci"] = cap_ci(y, P["champ"], g, r["k_top1pct"])
        r["SB_top1pct_capture_ci"] = cap_ci(y, P["SB"], g, r["k_top1pct"])
    RES["windows"][k] = r
# ---------------- 판정(주창)
m = RES["windows"]["main"]
def verdict(ref, off_lo):
    lo = m[f"skill_vs_{ref}_ci"][0]; alo = m["champ_auc_ci"][0]; pt = m[f"skill_vs_{ref}"]
    if lo > 0 and alo > 0.5:
        v = "전향 확인" + ("(약화)" if pt < off_lo else "")
    else:
        v = "전향 실패" + ("(약화)" if pt < off_lo else "")
    if m["pos"] < 30:
        v = "표본 부족 · " + v
    return dict(verdict=v, skill=pt, ci=m[f"skill_vs_{ref}_ci"], auc_ci=m["champ_auc_ci"], official_ci_lower=off_lo)
RES["verdict_primary_vs_M0"] = verdict("M0", OFF["skill_vs_M0_ci"][0])
RES["verdict_secondary_vs_SB"] = verdict("SB", OFF["skill_vs_SB_ci"][0])
RES["official_reference"] = dict(auc=OFF["champ_auc"], M0_auc=OFF["M0_auc"], SB_auc=OFF["SB_auc"], skill_vs_M0=OFF["skill_vs_M0"], skill_vs_M0_ci=OFF["skill_vs_M0_ci"],
                                 skill_vs_SB=OFF["skill_vs_SB"], skill_vs_SB_ci=OFF["skill_vs_SB_ci"], top1pct_capture=OFF["champ_top1pct_capture"],
                                 SB_top1pct_capture=OFF["SB_top1pct_capture"], pos=OFF["pos"], n=OFF["n"], base_rate_permil=OFF["base_rate_permil"])
RES["events_ge_2025"] = dict(rows=int(len(EV)), patents=int(EV.rgst_no.nunique()), max_date=int(EV.d.max()) if len(EV) else None,
                             in_population_main=int(m["pos"]))
RES["sec"] = round(time.time() - t); RES["rss"] = rss()
json.dump(RES, open(OUTF, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: RES[k] for k in ("verdict_primary_vs_M0", "verdict_secondary_vs_SB")}, ensure_ascii=False, indent=1))
print("main", {k: m[k] for k in ("pos", "champ_auc", "M0_auc", "SB_auc", "skill_vs_M0", "skill_vs_SB", "champ_top1pct_hits", "SB_top1pct_hits", "M0_top1pct_hits", "champ_hit100")})
