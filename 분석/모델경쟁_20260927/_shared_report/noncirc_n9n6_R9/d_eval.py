# -*- coding: utf-8 -*-
"""평가: 팔별 원점 AUC, 사전 등록 비교(PROV-CW, PROV-TXT, PROV-NONE 등)의 군집 부트스트랩 CI와 판정.
산출: results.json, results_table.csv"""
import sys, os, glob
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r9_common import *

PAIRS = [("PROV", "TXT"), ("PROV", "CW"), ("PROV", "NONE"), ("TXT", "CW"), ("TXT", "NONE"), ("CW", "NONE"), ("PROV", "NONE_ALL"), ("NONE", "NONE_ALL")]


def topk(y, p, frac=0.01):
    y = np.asarray(y); k = max(1, int(round(frac * len(y)))); return float(y[np.argsort(-np.asarray(p), kind="stable")[:k]].mean())


def load_tabs(prefix, origins, arms, cl):
    tabs = {}
    for v in origins:
        mp = os.path.join(PRED, f"{prefix}_meta_{v}.parquet")
        if not os.path.exists(mp):
            return None
        t = pd.read_parquet(mp).rename(columns={cl: "cl"})
        for a in arms:
            fp = os.path.join(PRED, f"{prefix}_{a}_{v}.parquet")
            if os.path.exists(fp):
                t[f"p_{a}"] = pd.read_parquet(fp).p.values
        tabs[v] = t
    return tabs


def evaluate(prefix, origins, cl, n=200):
    arms = ["PROV", "CW", "TXT", "NONE", "NONE_ALL", "M0"]
    tabs = load_tabs(prefix, origins, arms, cl)
    if tabs is None:
        return None
    have = [a for a in arms if all(f"p_{a}" in t for t in tabs.values())]
    pairs = [pr for pr in PAIRS if pr[0] in have and pr[1] in have]
    res, am, ab = cluster_boot(tabs, pairs, "cl", n=n, seed=0)
    out = dict(prefix=prefix, n_origins=len(origins), arms=have, auc_mean=am, auc_by_origin=ab, comparisons=res,
               n_rows=int(sum(len(t) for t in tabs.values())), base_rate=float(np.mean([t.y.mean() for t in tabs.values()])))
    out["top1"] = {a: float(np.mean([topk(t.y, t[f"p_{a}"]) for t in tabs.values()])) for a in have}
    if "M0" in have:
        out["skill_vs_M0"] = {a: float(np.mean([(x - m) / (1 - m) for x, m in zip(ab[a], ab["M0"])])) for a in have if a != "M0"}
    return out


def evaluate_macro(prefixes, origins, cl, n=200):
    """섹션 평균: 키 (섹션, 원점). 부트스트랩 군집 가중치는 모든 키에 같게."""
    arms = ["PROV", "CW", "TXT", "NONE", "NONE_ALL"]
    tabs = {}
    for pf in prefixes:
        t = load_tabs(pf, origins, arms, cl)
        if t is None:
            return None
        for v, tt in t.items():
            tabs[(pf, v)] = tt
    have = [a for a in arms if all(f"p_{a}" in t for t in tabs.values())]
    pairs = [pr for pr in PAIRS if pr[0] in have and pr[1] in have]
    res, am, ab = cluster_boot(tabs, pairs, "cl", n=n, seed=0)
    keys = sorted(tabs)
    for k, r in res.items():   # 원점별(섹션 평균) 부호로 다시 셈
        per = {}
        for (pf, v), x in zip(keys, r["per_origin"]):
            per.setdefault(v, []).append(x)
        po = [float(np.mean(per[v])) for v in sorted(per)]
        r["per_origin_secmean"] = po; r["npos"] = int(sum(x > 0 for x in po)); r["nneg"] = int(sum(x < 0 for x in po)); r["n_origins"] = len(po)
        r["verdict"] = verdict(r["mean"], r["ci"][0], r["ci"][1], r["npos"], r["nneg"], len(po))
        del r["per_origin"]
    return dict(prefix="macro:" + ",".join(prefixes), arms=have, auc_mean=am, comparisons=res)


N9O = list(range(2014, 2021)); N6O = list(range(2014, 2020))
jobs = [("n9_y_all", N9O, "org"), ("n9_y_frozen", N9O, "org")] + [(f"n9_y_{s}", N9O, "org") for s in SECTIONS] + \
       [("n6_ntis_y_ntis", N6O, "bz"), ("n6_prov_y_ntis", N6O, "bz"), ("n6_prov_y", N6O, "bz")]
R = {}
for pf, org, cl in jobs:
    r = evaluate(pf, org, cl)
    if r is not None:
        R[pf] = r; log(pf, {k: (round(v["mean"], 4), [round(x, 4) for x in v["ci"]], v["npos"], v["verdict"]) for k, v in r["comparisons"].items()})
mac = evaluate_macro([f"n9_y_{s}" for s in SECTIONS], N9O, "org")
if mac:
    R["n9_sections_macro"] = mac; log("macro", {k: (round(v["mean"], 4), [round(x, 4) for x in v["ci"]], v["npos"], v["verdict"]) for k, v in mac["comparisons"].items()})
json.dump(R, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
rows = []
for pf, r in R.items():
    for k, v in r["comparisons"].items():
        rows.append(dict(set=pf, comparison=k, mean_dAUC=v["mean"], ci_lo=v["ci"][0], ci_hi=v["ci"][1], npos=v["npos"], n_origins=v["n_origins"], verdict=v["verdict"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "results_table.csv"), index=False, encoding="utf-8-sig")
rows = []
for pf, r in R.items():
    for a, x in r["auc_mean"].items():
        rows.append(dict(set=pf, arm=a, auc_mean=x, top1=r.get("top1", {}).get(a), skill_vs_M0=r.get("skill_vs_M0", {}).get(a),
                         auc_by_origin=" ".join(f"{z:.4f}" for z in r.get("auc_by_origin", {}).get(a, []))))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "auc_table.csv"), index=False, encoding="utf-8-sig")
log("end")
