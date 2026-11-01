# -*- coding: utf-8 -*-
r"""C11-T2 s8: 팔별 결과 모음 + 사전 판정(반올림 규칙) → res/c11t2_table.csv, res/c11t2_results.json"""
import os, sys, json
sys.dont_write_bytecode = True
import pandas as pd, numpy as np
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"
ARMS = ["stored", "r8", "reg", "pub", "pure"]
out = {}; rows = []
f4 = lambda x: f"{x:+.4f}"
for a in ARMS:
    j = json.load(open(f"{W}/res/s5_summary_mg__{a}.json", encoding="utf-8"))
    txt_auc = [m["auc"] for m in j["models"] if m["model"] == "txt"][0]
    P = j["pairs"]; o = {"txt_auc": txt_auc}
    for k in ("prov_vs_txt", "txt_vs_none", "txt_vs_conc", "prov_s3_vs_txt_s3"):
        if k in P: o[k] = dict(d=P[k]["dAUC"], lo=P[k]["ci"][0], hi=P[k]["ci"][1], wins=P[k]["wins"], verdict=P[k]["verdict"],
                              txt=f"{P[k]['dAUC']:+.4f} [{P[k]['ci'][0]:+.4f}, {P[k]['ci'][1]:+.4f}], {P[k]['wins']}/9 ({P[k]['verdict']})")
    s6 = f"{W}/res/s6/s6_external__{a}__p0.csv"
    if os.path.exists(s6):
        d = pd.read_csv(s6, encoding="utf-8-sig")
        for nm in ("사업체수", "종사자수", "출하액"):
            r = d[(d.외부지표 == nm) & (d.신호 == "cell_real")].iloc[0]
            o[f"ext_{nm}"] = dict(prov=r.A_rho, txt=r.B_rho, d=r.차이, lo=r.CI_lo, hi=r.CI_hi, wins=int(r.A우세_원점), q=r.q_BH, verdict=r.판정,
                                  text=f"ρ {r.A_rho:.3f} 대 {r.B_rho:.3f}, {r.차이:+.3f} [{r.CI_lo:+.3f}, {r.CI_hi:+.3f}], {int(r.A우세_원점)}/9, BH q {r.q_BH:.3f} ({r.판정})")
        o["ext_rows_CI_pos"] = int((d.CI_lo > 0).sum()); o["ext_rows_CI_neg"] = int((d.CI_hi < 0).sum())
    wf = f"{W}/res/s6/s6_within__{a}.csv"
    if os.path.exists(wf):
        w = pd.read_csv(wf, encoding="utf-8-sig"); r = w[(w.외부지표 == "사업체수") & (w.신호 == "cell_real")].iloc[0]
        o["within_사업체"] = dict(prov=r.제공_r, txt=r.TXT_r, d=r.차이, lo=r.CI_lo, hi=r.CI_hi, wins=int(r.제공우세_원점),
                                 text=f"제공 {r.제공_r:.3f} 대 TXT {r.TXT_r:.3f}, {r.차이:+.3f} [{r.CI_lo:+.4f}, {r.CI_hi:+.3f}], {int(r.제공우세_원점)}/9")
    out[a] = o
L = json.load(open(f"{W}/res/s7_landscape.json", encoding="utf-8")); out["landscape"] = {r["name"]: r for r in L["rows"]}; out["range_168"] = L["range_168"]
# 판정: 저장(원고) 대비 reg(A)·pure(B)
def judge_auc(stored, new):
    return "오류(숫자)" if abs(round(new, 4) - round(stored, 4)) >= 0.00005 or abs(new - stored) >= 0.00005 else "반올림 안"
S = out["stored"]
for a in ("reg", "pub", "pure"):
    N = out[a]; pv = N["prov_vs_txt"]; ps = S["prov_vs_txt"]
    num = any(abs(round(pv[k], 4) - round(ps[k], 4)) >= 0.00005 for k in ("d", "lo", "hi")) or pv["wins"] != ps["wins"]
    change = (pv["lo"] > 0 or pv["hi"] < 0) or (9 - pv["wins"]) >= 5
    rows.append(dict(arm=a, item="표10·T3NC-prov_vs_txt", stored=ps["txt"], new=pv["txt"], numeric_error=num, verdict_change=change))
    rows.append(dict(arm=a, item="T3NC-MG TXT AUC", stored=f"{S['txt_auc']:.4f}", new=f"{N['txt_auc']:.4f}", numeric_error=abs(N["txt_auc"] - S["txt_auc"]) >= 0.00005, verdict_change=False))
    tv = N["txt_vs_none"]; ts = S["txt_vs_none"]
    rows.append(dict(arm=a, item="T3NC-txt_vs_none", stored=ts["txt"], new=tv["txt"],
                     numeric_error=any(abs(round(tv[k], 4) - round(ts[k], 4)) >= 0.00005 for k in ("d", "lo", "hi")) or tv["wins"] != ts["wins"], verdict_change=tv["verdict"] != ts["verdict"]))
    for nm in ("사업체수", "종사자수"):
        e = N[f"ext_{nm}"]; es = S[f"ext_{nm}"]
        num = any(abs(round(e[k], 3) - round(es[k], 3)) >= 0.0005 for k in ("txt", "d", "lo", "hi")) or e["wins"] != es["wins"] or abs(e["q"] - es["q"]) >= 0.005
        rows.append(dict(arm=a, item=f"T3EXT-prov-txt-{nm}", stored=es["text"], new=e["text"], numeric_error=num, verdict_change=(e["verdict"] != es["verdict"]) or ((e["q"] < 0.05) != (es["q"] < 0.05))))
    e = N["within_사업체"]; es = S["within_사업체"]
    rows.append(dict(arm=a, item="T3EXT-34", stored=es["text"], new=e["text"],
                     numeric_error=any(abs(round(e[k], 3) - round(es[k], 3)) >= 0.0005 for k in ("txt", "d", "hi")) or abs(round(e["lo"], 4) - round(es["lo"], 4)) >= 0.00005 or e["wins"] != es["wins"],
                     verdict_change=(e["lo"] > 0) != (es["lo"] > 0)))
for a in ("reg", "pub", "pure"):
    for it in ("free_F", "txt_F", "free_H", "txt_H"):
        s = out["landscape"][f"{it}_r8"]; n = out["landscape"][f"{it}_{a}"]
        rows.append(dict(arm=a, item=f"C-LAND-{it}", stored=f"{s['kr_rta_gt1']}/{s['kr_leader']}/{s['kept']}/{s['rho_KR']:.3f}", new=f"{n['kr_rta_gt1']}/{n['kr_leader']}/{n['kept']}/{n['rho_KR']:.3f}",
                         numeric_error=(s['kr_rta_gt1'], s['kr_leader'], s['kept'], round(s['rho_KR'], 3)) != (n['kr_rta_gt1'], n['kr_leader'], n['kept'], round(n['rho_KR'], 3)), verdict_change=False))
    rows.append(dict(arm=a, item="168행 한국 RTA ρ 범위", stored=out["range_168"]["r8"]["text"], new=out["range_168"][a]["text"],
                     numeric_error=out["range_168"]["r8"]["text"] != out["range_168"][a]["text"], verdict_change=False))
T = pd.DataFrame(rows); T.to_csv(f"{W}/res/c11t2_table.csv", index=False, encoding="utf-8-sig")
json.dump(out, open(f"{W}/res/c11t2_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 120); print(T.to_string())
