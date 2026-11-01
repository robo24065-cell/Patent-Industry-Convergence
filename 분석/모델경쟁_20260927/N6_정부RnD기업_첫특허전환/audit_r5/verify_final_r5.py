# -*- coding: utf-8 -*-
"""N6 R5 최종시험 독립 재계산 + 위약(층 내 레이블 섞기). 모형은 다시 학습하지 않는다(동결 예측 파일 사용).
1) 레이블·위험집합을 final_run/n6_features와 다른 경로(duckdb SQL + 별도 연결 코드)로 다시 만들어 final_preds_r5.parquet의 y와 대조
2) AUC(Mann-Whitney 순위식)·skill·상위 k 적중률을 sklearn 없이 다시 계산해 final_result_r5.json과 대조
3) 위약: 시험 레이블을 층(연결 여부 × M0 점수 10분위) 안에서 1,000회 섞어 챔피언 vs M0 skill 분포
쓰기: 이 폴더(audit_r5)만."""
import os, sys, json
import numpy as np, pandas as pd, duckdb
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE)
sys.path.insert(0, OUT)
from n6_lib import norm, EXTF, EXT, R3I   # 경로·상호 정규화 함수만 재사용

res = json.load(open(os.path.join(OUT, "final_result_r5.json"), encoding="utf-8"))
P = pd.read_parquet(os.path.join(OUT, "final_preds_r5.parquet"))

# ---------- 1) 독립 레이블 재구성
pop = duckdb.sql(f"""select distinct regexp_replace(corp_rgst_no,'[^0-9]','','g') bz from read_parquet('{EXTF}/ntis_project_raw.parquet')
  where year between 2019 and 2021 and perform_grp in ('중소기업','중견기업','대기업')""").df()
pop = pop[pop.bz.str.len() == 10]
nm = duckdb.sql(f"""select bz, nm from (select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm, count(*) c,
   row_number() over (partition by regexp_replace(corp_rgst_no,'[^0-9]','','g') order by count(*) desc) r
   from read_parquet('{EXTF}/ntis_project_raw.parquet') where year between 2010 and 2021 and perform_grp in ('중소기업','중견기업','대기업') group by 1,2) where r=1""").df()
ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
ns["nn"] = ns.aname.map(norm)
uniq = ns.groupby("nn").applicantCd.nunique(); cnt = ns.groupby("nn").size()
uq = ns[ns.nn.isin(cnt[cnt == 1].index)].set_index("nn").applicantCd
link_b = ns[["bizno", "applicantCd"]].rename(columns={"bizno": "bz"})
nm["applicantCd"] = nm.nm.map(norm).map(uq)
link_n = nm[~nm.bz.isin(set(link_b.bz))].dropna(subset=["applicantCd"])[["bz", "applicantCd"]]
link = pd.concat([link_b[link_b.bz.isin(set(pop.bz))], link_n[link_n.bz.isin(set(pop.bz))]]).drop_duplicates()
af = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "appl_date", "known_date"])
af = af[af.appl_date <= 20231231]
a = link.merge(af, on="applicantCd")
known = set(a.loc[a.known_date <= 20211231, "bz"])
risk = pop[~pop.bz.isin(known)].copy()
pos = set(a.loc[(a.appl_date >= 20220101) & (a.appl_date <= 20231231), "bz"])
risk["y_ind"] = risk.bz.isin(pos).astype(int)
risk["linked_ind"] = risk.bz.isin(set(link.bz)).astype(int)
m = P.merge(risk, on="bz", how="outer", indicator=True)
lab = dict(n_frozen=int(len(P)), n_indep=int(len(risk)), both=int((m._merge == "both").sum()),
           only_frozen=int((m._merge == "left_only").sum()), only_indep=int((m._merge == "right_only").sum()),
           pos_frozen=int(P.y.sum()), pos_indep=int(risk.y_ind.sum()),
           y_mismatch=int((m[m._merge == "both"].y != m[m._merge == "both"].y_ind).sum()),
           linked_indep=int(risk.linked_ind.sum()))

# ---------- 2) 지표 재계산 (sklearn 없이)
def auc_mw(y, s):
    y = np.asarray(y); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def topk(y, s, fr):
    k = max(1, int(round(fr * len(y)))); o = np.argsort(-np.asarray(s), kind="stable")[:k]
    return float(np.asarray(y)[o].mean()), k, int(np.asarray(y)[o].sum())

y = P.y.values; pc = P.p_champ.values; pb = P.p_m0.values
a1, a0 = auc_mw(y, pc), auc_mw(y, pb); sk = (a1 - a0) / (1 - a0)
met = dict(auc=a1, auc_M0=a0, skill=sk, base_rate=float(y.mean()))
for fr in (0.01, 0.05, 0.10):
    t, k, h = topk(y, pc, fr); t0, _, h0 = topk(y, pb, fr)
    met[f"top{int(fr*100)}"] = t; met[f"top{int(fr*100)}_M0"] = t0; met[f"top{int(fr*100)}_k"] = k; met[f"top{int(fr*100)}_hits"] = h; met[f"top{int(fr*100)}_hits_M0"] = h0
diff = {k: abs(met[k] - res[k]) for k in ["auc", "auc_M0", "skill", "top1", "top5", "top10", "top1_M0", "top5_M0", "top10_M0", "base_rate"]}
# 독립 레이블로 다시 채점(예측은 동결 파일, bz 기준 결합)
mm = m[m._merge == "both"]
met_indep_label = dict(auc=auc_mw(mm.y_ind, mm.p_champ), auc_M0=auc_mw(mm.y_ind, mm.p_m0))
met_indep_label["skill"] = (met_indep_label["auc"] - met_indep_label["auc_M0"]) / (1 - met_indep_label["auc_M0"])
# 독립 부트스트랩(행 = 사업자번호 1개, 다른 시드)
rng = np.random.default_rng(777); bs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y)); b1, b0 = auc_mw(y[i], pc[i]), auc_mw(y[i], pb[i]); bs.append((b1 - b0) / (1 - b0))
met["skill_ci95_indep_seed777"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
# 10분위 첫 특허율
dec = pd.qcut(rankdata(pc) / len(pc), 10, labels=False)
met["decile_rate"] = [float(y[dec == d].mean()) for d in range(10)]

# ---------- 3) 위약: 층(연결 여부 × M0 10분위) 안에서 레이블 섞기
lk = mm.set_index("bz").linked_ind.reindex(P.bz).fillna(0).values.astype(int)
strata = lk * 10 + pd.qcut(rankdata(pb) / len(pb), 10, labels=False)
rng = np.random.default_rng(2026); pl = []
for _ in range(1000):
    ys = y.copy()
    for s in np.unique(strata):
        ii = np.where(strata == s)[0]; ys[ii] = rng.permutation(ys[ii])
    b1, b0 = auc_mw(ys, pc), auc_mw(ys, pb); pl.append((b1 - b0) / (1 - b0))
pl = np.array(pl)
plac = dict(strata="linked x M0-decile (20)", n_perm=1000, mean=float(pl.mean()), sd=float(pl.std()), q025=float(np.percentile(pl, 2.5)),
            q975=float(np.percentile(pl, 97.5)), max=float(pl.max()), p_ge_observed=float((pl >= sk).mean()), observed=float(sk))

out = dict(labels=lab, metrics_recomputed=met, abs_diff_vs_final_json=diff, rescored_with_independent_labels=met_indep_label, placebo=plac)
json.dump(out, open(os.path.join(HERE, "verify_final_r5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
