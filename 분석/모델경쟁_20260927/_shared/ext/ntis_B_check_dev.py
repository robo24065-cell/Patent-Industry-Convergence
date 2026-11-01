# 개발 구간 한정 의미 점검: KSIC별 R&D 신호 증가(t-2->t) vs KR 특허 출원 증가(t->t+1..t+3), t<=2014 (결과 출원연도 <=2017, 최종시험 구간 미사용)
import sys, json, numpy as np, pandas as pd, duckdb, os
from scipy.stats import spearmanr
import ntis_B_common as c
sys.stdout.reconfigure(encoding="utf-8")
META = c.EXT.replace(r"\ext", r"\meta")
cache = c.WORK + r"\ntis_B_kr_appl_ksic_year_le2017.parquet"
if not os.path.exists(cache):
    con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2")
    kr = con.sql(f"""select ksic, appl_year, count(distinct appl_no) n from read_parquet('{META.replace(chr(92), '/')}/*.parquet')
        where cc='KR' and appl_year between 2000 and 2017 group by 1,2""").df()
    kr.to_parquet(cache, index=False)
kr = pd.read_parquet(cache); kr["ksic"] = kr.ksic.astype(str)
P = kr.pivot(index="ksic", columns="appl_year", values="n").fillna(0)
sig = pd.read_parquet(c.EXT + r"\ntis_project_ksic_year.parquet")
res = []
for m in sig.method.unique():
    s = sig[sig.method == m].pivot(index="ksic", columns="year", values="gov_share_of_year")
    for t in range(2006, 2015):
        if t not in s.columns or (t - 2) not in s.columns: continue
        x = np.log((s[t] + s[t - 1] + 1e-6) / (s[t - 2] + s[t - 3] + 1e-6)) if (t - 3) in s.columns else np.log((s[t] + 1e-6) / (s[t - 2] + 1e-6))
        y = np.log((P[[t + 1, t + 2, t + 3]].sum(1) + 1) / (P[[t - 2, t - 1, t]].sum(1) + 1))
        base = np.log((P[[t - 2, t - 1, t]].sum(1) + 1) / (P[[t - 5, t - 4, t - 3]].sum(1) + 1)) if (t - 5) in P.columns else None
        d = pd.concat([x.rename("x"), y.rename("y")] + ([base.rename("mom")] if base is not None else []), axis=1).dropna()
        d = d[np.isfinite(d.x)]
        if len(d) < 15 or d.x.nunique() < 5: continue
        r = spearmanr(d.x, d.y).statistic
        # 모멘텀 통제 부분순위상관
        if "mom" in d:
            rk = d.rank(); from numpy.linalg import lstsq
            ex = rk.x - np.polyval(np.polyfit(rk.mom, rk.x, 1), rk.mom); ey = rk.y - np.polyval(np.polyfit(rk.mom, rk.y, 1), rk.mom)
            pr = float(np.corrcoef(ex, ey)[0, 1])
        else:
            pr = np.nan
        res.append((m, t, len(d), round(float(r), 3), round(pr, 3)))
res = pd.DataFrame(res, columns=["method", "t0", "n_ksic", "spearman", "partial_mom"])
res.to_csv(c.WORK + r"\ntis_B_check_dev.csv", index=False, encoding="utf-8-sig")
summ = res.groupby("method").agg(n_origins=("t0", "size"), mean_rho=("spearman", "mean"), pos=("spearman", lambda s: int((s > 0).sum())),
                                 mean_partial=("partial_mom", "mean"), pos_partial=("partial_mom", lambda s: int((s > 0).sum()))).round(3)
print(res.to_string(index=False)); print(summ.to_string())
summ.to_csv(c.WORK + r"\ntis_B_check_dev_summary.csv", encoding="utf-8-sig")
