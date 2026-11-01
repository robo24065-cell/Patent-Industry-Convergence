# -*- coding: utf-8 -*-
"""C07-T3 s04b: 사전 고정 보조 분석 (i)–(iv). s04a가 저장한 레이블·사건 캐시만 읽는다(결과 재조회 없음).
→ subgroup_newdispute.csv, calibration.csv, ksic_rank_check.csv, ../cache/s04b_out.json (prospective_result.json에 'aux'로 병합)"""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c07lib import *  # noqa
from scipy.stats import spearmanr, chi2
t = time.time()
S = pd.read_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"))
Lb = pd.read_parquet(os.path.join(W, "cache", "s04a_labels.parquet"))
assert (S.rgst_no.values == Lb.rgst_no.values).all()
EV = pd.read_parquet(os.path.join(W, "cache", "s04a_events.parquet"))
RP = pd.read_parquet(os.path.join(W, "cache", "s04a_rightpos.parquet")); RP["d"] = pd.to_numeric(RP.rgstCsDate, errors="coerce")
g = L.cluster_key(S.assign(owner_v2=S.owner_v2.fillna("")))
y = Lb.y_main.values.astype(int); n = len(y); npos = int(y.sum())
P = {"champ": S.p_champ.values, "M0": S.p_M0.values, "SB": S.p_SB.values}
OUT = {}
# =============== (i) 분쟁 이력 하위집단
rows = []
k1 = int(round(0.01 * n)); top = {m: np.argsort(-p, kind="stable")[:k1] for m, p in P.items()}
Hpre, H23, Hany = S.H_pre.values.astype(bool), S.H_2324only.values.astype(bool), S.H_any.values.astype(bool)
Hnone = ~Hany
for m in P:
    o = top[m]
    rows.append(dict(section=f"원 상위1% 컷(k={k1}) 적중 구성", model=m, hits=int(y[o].sum()), hits_H_pre=int((y & Hpre)[o].sum()),
                     hits_H_2324only=int((y & H23)[o].sum()), hits_new_H_none=int((y & Hnone)[o].sum()),
                     pos_total=npos, pos_H_pre=int((y & Hpre).sum()), pos_H_2324only=int((y & H23).sum()), pos_new=int((y & Hnone).sum()),
                     new_capture_origcut=float((y & Hnone)[o].sum() / max((y & Hnone).sum(), 1))))
# 새 분쟁(C01 정의: 이력 양성만 빼고 이력 음성 유지)
mask = ~(y.astype(bool) & Hany); yn = (y.astype(bool) & Hnone)[mask].astype(int)
kn = int(round(0.01 * mask.sum()))
new = dict(section="새 분쟁(이력 양성 제외, 이력 음성 유지)", n=int(mask.sum()), pos=int(yn.sum()), k_recut=kn)
for m, p in P.items():
    new[f"{m}_auc"] = float(L.auc(yn, p[mask])); new[f"{m}_recut_top1pct_hits"] = int(L.topk(yn, p[mask], k=kn)[1])
    new[f"{m}_recut_top1pct_capture"] = new[f"{m}_recut_top1pct_hits"] / max(int(yn.sum()), 1)
new["skill_vs_M0"] = float(L.skill(new["champ_auc"], new["M0_auc"])); new["skill_vs_SB"] = float(L.skill(new["champ_auc"], new["SB_auc"]))
if yn.sum() >= 3:
    new["skill_vs_M0_ci"] = L.boot_ci(yn, P["champ"][mask], P["M0"][mask], g[mask], n=1000, seed=2029)[0]
    new["skill_vs_SB_ci"] = L.boot_ci(yn, P["champ"][mask], P["SB"][mask], g[mask], n=1000, seed=2030)[0]
rows.append(new)
# 단순 규칙: 이력 특허 전부
kr = int(Hany.sum())
rule = dict(section="단순 규칙: 분쟁 이력 특허 전부", n_flagged=kr, share_pop=kr / n, hits=int(y[Hany].sum()), capture=float(y[Hany].sum() / npos),
            precision=float(y[Hany].mean()))
for m, p in P.items():
    rule[f"{m}_hits_same_k"] = int(L.topk(y, p, k=kr)[1])
rows.append(rule)
# 공식 하위집단(fresh = own_disp=0 & own_trial=0)
fresh = (S.own_disp.values == 0) & (S.own_trial.values == 0)
for nm, s, sd in (("공식 하위집단: 신규(이력 없음)", fresh, 2027), ("공식 하위집단: 후속(이력 있음)", ~fresh, 2031)):
    ys = y[s]; d = dict(section=nm, n=int(s.sum()), pos=int(ys.sum()))
    if 3 <= ys.sum() < len(ys):
        for m, p in P.items():
            d[f"{m}_auc"] = float(L.auc(ys, p[s])); d[f"{m}_top1pct_capture"] = L.topk(ys, p[s], 0.01)[1] / ys.sum()
        d["skill_vs_M0"] = float(L.skill(d["champ_auc"], d["M0_auc"])); d["skill_vs_SB"] = float(L.skill(d["champ_auc"], d["SB_auc"]))
        d["skill_vs_M0_ci"] = L.boot_ci(ys, P["champ"][s], P["M0"][s], g[s], n=1000, seed=sd)[0]
        d["skill_vs_SB_ci"] = L.boot_ci(ys, P["champ"][s], P["SB"][s], g[s], n=1000, seed=sd + 1)[0]
    rows.append(d)
print("(i) done", round(time.time() - t), flush=True)
SG = pd.DataFrame(rows)
for c in SG.columns:
    SG[c] = SG[c].map(lambda v: json.dumps(v) if isinstance(v, list) else v)
SG.to_csv(os.path.join(W, "subgroup_newdispute.csv"), index=False, encoding="utf-8-sig")
OUT["subgroup"] = rows
# =============== (ii) 보정
def pci(nobs):
    lo = chi2.ppf(0.025, 2 * nobs) / 2 if nobs > 0 else 0.0; hi = chi2.ppf(0.975, 2 * nobs + 2) / 2
    return lo, hi
cal = []
for wk, days in (("main", 546), ("auxA", 592), ("auxB", 588)):
    yy = Lb["y_" + wk].values; f = days / 730
    lo, hi = pci(int(yy.sum()))
    for m, p in P.items():
        e = float(p.sum() * f)
        cal.append(dict(level="전체", window=wk, model=m, key="ALL", n=n, expected_adj=e, sum_p_24m=float(p.sum()), realized=int(yy.sum()),
                        ratio=e / max(yy.sum(), 1e-9), ratio_lo=e / hi, ratio_hi=(e / lo) if lo > 0 else np.inf))
f = 546 / 730
D = pd.DataFrame({"ksic": S.ksic.values, "y": y, "p": P["champ"]})
for k, gk in D.groupby("ksic"):
    e = float(gk.p.sum() * f); r = int(gk.y.sum()); lo, hi = pci(r)
    cal.append(dict(level="KSIC", window="main", model="champ", key=str(k), n=int(len(gk)), expected_adj=e, sum_p_24m=float(gk.p.sum()), realized=r,
                    ratio=(e / r) if r > 0 else np.inf, ratio_lo=e / hi, ratio_hi=(e / lo) if lo > 0 else np.inf))
D["dec"] = pd.qcut(pd.Series(P["champ"]).rank(method="first"), 10, labels=False)
for k, gk in D.groupby("dec"):
    e = float(gk.p.sum() * f); r = int(gk.y.sum()); lo, hi = pci(r)
    cal.append(dict(level="10분위(1=최저)", window="main", model="champ", key=str(int(k) + 1), n=int(len(gk)), expected_adj=e, sum_p_24m=float(gk.p.sum()),
                    realized=r, ratio=(e / r) if r > 0 else np.inf, ratio_lo=e / hi, ratio_hi=(e / lo) if lo > 0 else np.inf))
CAL = pd.DataFrame(cal); CAL.to_csv(os.path.join(W, "calibration.csv"), index=False, encoding="utf-8-sig")
OUT["calibration_overall"] = [r for r in cal if r["level"] == "전체"]
kc = CAL[CAL.level == "KSIC"]
OUT["calibration_ksic_summary"] = dict(n_ksic=int(len(kc)), n_ksic_realized0=int((kc.realized == 0).sum()), expected_in_realized0=float(kc[kc.realized == 0].expected_adj.sum()),
                                       median_ratio_nonzero=float(kc[kc.realized > 0].ratio.median()))
print("(ii) done", round(time.time() - t), flush=True)
# =============== (iii) 산업 지수 순위
K = pd.read_csv(os.path.join(L.HERE, "future_ksic_r6.csv"), encoding="utf-8-sig", dtype={"ksic": str})
FP = pd.read_parquet(os.path.join(L.HERE, "final_preds_r6.parquet"), columns=["ksic", "y"])
pers = FP.groupby("ksic").y.agg(["sum", "size"]); pers["rate"] = pers["sum"] / pers["size"]
ks = sorted(K.ksic); kidx = {k: i for i, k in enumerate(ks)}
codes_k = np.array([kidx[str(k)] for k in S.ksic.values])
real_n = np.bincount(codes_k, minlength=len(ks)).astype(float); real_y = np.bincount(codes_k, weights=y, minlength=len(ks))
real_rate = real_y / real_n
Ki = K.set_index("ksic").loc[ks]
mean_p = Ki.mean_p.values; exp_d = Ki.exp_disputes.values; pers_rate = pers.reindex(ks)["rate"].values
rho = lambda a, b: float(spearmanr(a, b).correlation)
res3 = dict(rho_meanp_vs_rate=rho(mean_p, real_rate), rho_exp_vs_count=rho(exp_d, real_y), rho_persist_vs_rate=rho(pers_rate, real_rate),
            rho_meanp_vs_persist=rho(mean_p, pers_rate), n_ksic=len(ks), n_ksic_with_pos=int((real_y > 0).sum()))
rng = np.random.default_rng(2027); codes_g, uq = pd.factorize(np.asarray(g)); G = len(uq)
B = {"meanp": [], "exp": [], "persist": [], "diff": []}
for _ in range(1000):
    w = rng.multinomial(G, np.full(G, 1.0 / G))[codes_g].astype(float)
    bn = np.bincount(codes_k, weights=w, minlength=len(ks)); by = np.bincount(codes_k, weights=w * y, minlength=len(ks))
    br = np.divide(by, bn, out=np.zeros_like(by), where=bn > 0)
    a1 = rho(mean_p, br); a3 = rho(pers_rate, br)
    B["meanp"].append(a1); B["exp"].append(rho(exp_d, by)); B["persist"].append(a3); B["diff"].append(a1 - a3)
for k2, v in B.items():
    v = np.array(v); v = v[~np.isnan(v)]
    res3[f"ci_{k2}"] = [float(x) for x in np.percentile(v, [2.5, 97.5])]
res3["verdict"] = "산업 지수 순위 전향 지지" if res3["ci_meanp"][0] > 0 else "산업 지수 순위 미지지"
KR = pd.DataFrame(dict(ksic=ks, n=real_n.astype(int), mean_p=mean_p, exp_disputes_24m=exp_d, n_top1pct=Ki.n_top1pct.values,
                       realized_main=real_y.astype(int), realized_rate_permil=1000 * real_rate, persist_rate_permil_2023_24=1000 * pers_rate))
KR["rank_index_meanp"] = KR.mean_p.rank(ascending=False, method="min").astype(int)
KR["rank_index_exp"] = KR.exp_disputes_24m.rank(ascending=False, method="min").astype(int)
KR["rank_realized_rate"] = KR.realized_rate_permil.rank(ascending=False, method="min").astype(int)
KR["rank_realized_count"] = KR.realized_main.rank(ascending=False, method="min").astype(int)
KR["rank_persist"] = KR.persist_rate_permil_2023_24.rank(ascending=False, method="min").astype(int)
KR = KR.sort_values("rank_index_exp"); KR.to_csv(os.path.join(W, "ksic_rank_check.csv"), index=False, encoding="utf-8-sig")
res3["top5_index_exp"] = KR.head(5)[["ksic", "exp_disputes_24m", "realized_main", "rank_realized_count", "rank_realized_rate"]].to_dict("records")
res3["top5_index_meanp"] = KR.sort_values("rank_index_meanp").head(5)[["ksic", "mean_p", "realized_rate_permil", "rank_realized_rate"]].to_dict("records")
res3["realized_top5_count"] = KR.sort_values("realized_main", ascending=False).head(5)[["ksic", "realized_main", "rank_index_exp"]].to_dict("records")
OUT["ksic_rank"] = res3
print("(iii) done", res3["rho_meanp_vs_rate"], res3["ci_meanp"], round(time.time() - t), flush=True)
# =============== (iv) 권리자 청구 혼입 (C01-T3 R1 규칙)
INV, SCO = L.DISP
posm = set(Lb.rgst_no[Lb.y_main == 1])
rows4 = []
for r in sorted(posm):
    gg = RP[RP.rgst_no == r].sort_values(["d", "seq"])
    w = gg[(gg.d >= 20250101) & (gg.d <= 20260630)]
    inv_d = sorted(w[w.rgstCsName == INV].d.astype(int)); sco_d = sorted(w[w.rgstCsName == SCO].d.astype(int))
    act = pas = 0
    if sco_d:
        dec = gg[gg.rgstCsName.str.contains("확정", na=False) & ~gg.rgstCsName.str.contains("기술평가|이의신청", na=False) & (gg.d >= sco_d[0])]
        rs = dec.rgstCsReason.fillna("").str.replace(" ", "")
        act = int(rs.str.contains("속함").any()); pas = int(rs.str.contains("무속").any())
    dec_any = gg[gg.rgstCsName.str.contains("확정", na=False) & (gg.d >= min(inv_d + sco_d))]
    rows4.append(dict(rgst_no=r, n_inv=len(inv_d), n_sco=len(sco_d), act=act, pas=pas, any_decision_after=int(len(dec_any) > 0)))
F4 = pd.DataFrame(rows4)
F4["dir"] = np.where(F4.n_sco == 0, "none", np.where(F4.act == 1, np.where(F4.pas == 1, "mixed", "active"), np.where(F4.pas == 1, "passive", "undetermined")))
only_sco = F4.n_inv == 0
res4 = dict(pos=int(len(F4)), inv_only=int(((F4.n_inv > 0) & (F4.n_sco == 0)).sum()), scope_only=int(only_sco.sum()), both=int(((F4.n_inv > 0) & (F4.n_sco > 0)).sum()),
            scope_dir=F4.dir.value_counts().to_dict(), any_decision_after_first_event=int(F4.any_decision_after.sum()),
            active_lower=int(F4.dir.isin(["active", "mixed"]).sum()), active_upper=int(F4.dir.isin(["active", "mixed", "undetermined"]).sum()),
            active_only_lower=int((only_sco & F4.dir.isin(["active", "mixed"])).sum()), active_only_upper=int((only_sco & F4.dir.isin(["active", "mixed", "undetermined"])).sum()))
res4["active_lower_share"] = res4["active_lower"] / max(res4["pos"], 1); res4["active_upper_share"] = res4["active_upper"] / max(res4["pos"], 1)
res4["active_only_upper_share"] = res4["active_only_upper"] / max(res4["pos"], 1)
# 무효 단독 레이블 재채점(참고, C01-T3과 같은 방식)
y_inv = Lb.rgst_no.isin(set(F4.rgst_no[F4.n_inv > 0])).values.astype(int)
res4["inv_only_label"] = dict(pos=int(y_inv.sum()), **{f"{m}_auc": float(L.auc(y_inv, p)) for m, p in P.items()},
                              **{f"{m}_top1pct_hits": int(L.topk(y_inv, p, 0.01)[1]) for m, p in P.items()})
OUT["owner_claim_mix"] = res4
print("(iv) done", round(time.time() - t), flush=True)
OUT["sec"] = round(time.time() - t); OUT["rss"] = rss()
json.dump(OUT, open(os.path.join(W, "cache", "s04b_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
R = json.load(open(os.path.join(W, "prospective_result.json"), encoding="utf-8"))
R["aux"] = OUT
json.dump(R, open(os.path.join(W, "prospective_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ("ksic_rank", "owner_claim_mix", "calibration_overall")}, ensure_ascii=False, indent=1, default=float)[:6000])
