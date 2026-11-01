# -*- coding: utf-8 -*-
"""사전 등록 점검(S2–S5, S7) — 채택 판단에 쓰지 않음. 사용: python r_report.py
XGB 챔피언(champ_xgb.json)과 로지스틱 챔피언(champ_logit.json)의 원점별 AUC·skill·권리자 군집 CI·상위 k·강한 기준선·하위집단."""
import sys, json
sys.dont_write_bytecode = True
from n8lib import *
from e_run import M0, M1, M2, load_panel, score, EXPS

HIST = ["own_disp", "own_trial", "owner_disp5"]
cx = json.load(open(f"{CACHE}/champ_xgb.json", encoding="utf-8"))
cl = json.load(open(f"{CACHE}/champ_logit.json", encoding="utf-8"))
need = list(dict.fromkeys(M0 + cx["feats"] + cl["feats"] + HIST))
D = load_panel(need)
t = time.time()
P = {}
P["M0"] = pd.read_parquet(f"{CACHE}/preds_xgb_E00.parquet")
P["CH"] = pd.read_parquet(f"{CACHE}/preds_xgb_{cx['exp']}.parquet")
P["LM0"] = pd.read_parquet(f"{CACHE}/preds_logit_E00.parquet")
P["LCH"] = pd.read_parquet(f"{CACHE}/preds_logit_{cl['exp']}.parquet")
sb_feats = M0 + [h for h in HIST]
psb = score(D, sb_feats, {}, "xgb")
log("SB fitted")
ps1_m0 = score(D, M0, {}, "xgb", seed=1); ps1_ch = score(D, cx["feats"], cx["cfg"], "xgb", seed=1)
log("seed1 fitted")
rows, sub_rows = [], []
for v in DEV:
    m = D.t0.values == v; te = D[m]; y = te.y.values
    g = cluster_key(te)
    get = lambda k: P[k][P[k].t0 == v].p.values
    p0, pc, pl0, plc = get("M0"), get("CH"), get("LM0"), get("LCH")
    assert (P["CH"][P["CH"].t0 == v].rgst_no.values == te.rgst_no.values).all()
    sb = psb[v]
    a0, ac, asb, al0, alc = auc(y, p0), auc(y, pc), auc(y, sb), auc(y, pl0), auc(y, plc)
    ci, cia, _ = boot_ci(y, pc, p0, g, n=200, seed=v)
    cisb, _, _ = boot_ci(y, pc, sb, g, n=200, seed=v + 1)
    cil, _, _ = boot_ci(y, plc, p0, g, n=200, seed=v + 2)
    npos = int(y.sum())
    r = dict(v=v, n=int(m.sum()), pos=npos, base_rate_permil=1000 * y.mean(),
             auc_M0=a0, auc_champ=ac, auc_champ_lo=cia[0], auc_champ_hi=cia[1], skill=skill(ac, a0), skill_lo=ci[0], skill_hi=ci[1],
             auc_SB=asb, skill_vs_SB=skill(ac, asb), skill_vs_SB_lo=cisb[0], skill_vs_SB_hi=cisb[1],
             auc_logitM0=al0, auc_logitCH=alc, skill_logitCH_vs_logitM0=skill(alc, al0), skill_logitCH_vs_xgbM0=skill(alc, a0), skill_logitCH_vs_xgbM0_lo=cil[0], skill_logitCH_vs_xgbM0_hi=cil[1],
             seed1_auc_M0=auc(y, ps1_m0[v]), seed1_auc_champ=auc(y, ps1_ch[v]), seed1_skill=skill(auc(y, ps1_ch[v]), auc(y, ps1_m0[v])))
    for tag, p in (("champ", pc), ("M0", p0), ("SB", sb), ("logitCH", plc)):
        p1, h1, k1 = topk(y, p, 0.01); p5, h5, k5 = topk(y, p, 0.05); _, h100, _ = topk(y, p, k=100); _, h10, _ = topk(y, p, 0.10)
        r.update({f"top1pct_prec_{tag}": p1, f"top1pct_hits_{tag}": h1, f"top1pct_capture_{tag}": h1 / max(npos, 1),
                  f"top5pct_prec_{tag}": p5, f"top5pct_capture_{tag}": h5 / max(npos, 1), f"hit100_{tag}": h100, f"top10pct_capture_{tag}": h10 / max(npos, 1)})
    r["k_top1pct"] = k1
    rows.append(r)
    # 하위집단: 신규 분쟁(과거 분쟁·심판 없음) / 후속 분쟁(이력 있음)
    fresh = (te.own_disp.values == 0) & (te.own_trial.values == 0)
    for nm, s in (("신규(이력 없음)", fresh), ("후속(이력 있음)", ~fresh)):
        ys = y[s]
        if ys.sum() < 3 or ys.sum() == len(ys):
            sub_rows.append(dict(v=v, group=nm, n=int(s.sum()), pos=int(ys.sum()))); continue
        b0, bc, bsb = auc(ys, p0[s]), auc(ys, pc[s]), auc(ys, sb[s])
        sub_rows.append(dict(v=v, group=nm, n=int(s.sum()), pos=int(ys.sum()), auc_M0=b0, auc_champ=bc, auc_SB=bsb, auc_logitCH=auc(ys, plc[s]),
                             skill_vs_M0=skill(bc, b0), skill_vs_SB=skill(bc, bsb),
                             top1pct_capture_champ=topk(ys, pc[s], 0.01)[1] / ys.sum(), top1pct_capture_M0=topk(ys, p0[s], 0.01)[1] / ys.sum()))
    log(v, round(ac, 4), round(skill(ac, a0), 4), ci)
O = pd.DataFrame(rows); S = pd.DataFrame(sub_rows)
O.to_csv(f"{HERE}/by_origin_champion.csv", index=False, encoding="utf-8-sig")
S.to_csv(f"{HERE}/subgroups_by_origin.csv", index=False, encoding="utf-8-sig")
summ = dict(
    xgb_champion=cx["exp"], logit_champion=cl["exp"], n_feats_xgb=len(cx["feats"]), n_feats_logit=len(cl["feats"]),
    mean_auc_M0=O.auc_M0.mean(), mean_auc_champ=O.auc_champ.mean(), mean_skill=O.skill.mean(), win=float((O.auc_champ > O.auc_M0).mean()),
    min_skill_ci_lo=O.skill_lo.min(), max_skill_ci_lo=O.skill_lo.max(),
    mean_auc_SB=O.auc_SB.mean(), mean_skill_vs_SB=O.skill_vs_SB.mean(), win_vs_SB=float((O.auc_champ > O.auc_SB).mean()), n_SB_ci_lo_gt0=int((O.skill_vs_SB_lo > 0).sum()),
    logit_mean_auc=O.auc_logitCH.mean(), logit_skill_vs_logitM0=O.skill_logitCH_vs_logitM0.mean(), logit_skill_vs_xgbM0=O.skill_logitCH_vs_xgbM0.mean(),
    logit_win_vs_xgb_champ=float((O.auc_logitCH > O.auc_champ).mean()),
    seed1_mean_skill=O.seed1_skill.mean(), seed1_win=float((O.seed1_auc_champ > O.seed1_auc_M0).mean()),
    top1pct_prec_champ=[O.top1pct_prec_champ.min(), O.top1pct_prec_champ.max()], top1pct_capture_champ=[O.top1pct_capture_champ.min(), O.top1pct_capture_champ.max()],
    top1pct_capture_M0=[O.top1pct_capture_M0.min(), O.top1pct_capture_M0.max()], top1pct_capture_SB=[O.top1pct_capture_SB.min(), O.top1pct_capture_SB.max()],
    hit100_champ=[int(O.hit100_champ.min()), int(O.hit100_champ.max())], hit100_M0=[int(O.hit100_M0.min()), int(O.hit100_M0.max())], hit100_SB=[int(O.hit100_SB.min()), int(O.hit100_SB.max())],
    top10pct_capture_champ_mean=O.top10pct_capture_champ.mean(), top10pct_capture_M0_mean=O.top10pct_capture_M0.mean(),
    subgroups={g: dict(pos_total=int(s.pos.sum()), mean_auc_M0=s.auc_M0.mean(), mean_auc_champ=s.auc_champ.mean(), mean_auc_SB=s.auc_SB.mean(),
                       mean_skill_vs_M0=s.skill_vs_M0.mean(), mean_skill_vs_SB=s.skill_vs_SB.mean(), win_vs_M0=float((s.auc_champ > s.auc_M0).mean()),
                       win_vs_SB=float((s.auc_champ > s.auc_SB).mean()), n_origins=int(s.auc_M0.notna().sum())) for g, s in S.groupby("group")},
)
json.dump(summ, open(f"{HERE}/report_dev_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(summ, ensure_ascii=False, indent=1, default=float)); log("done")
