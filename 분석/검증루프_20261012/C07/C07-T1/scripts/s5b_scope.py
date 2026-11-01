# (b) 조사 범위(10인 이상) 편향: 2016→2020 성장(로그 4년)을 전 규모(전국사업체조사) 대 광업제조업조사(10인 이상)로 비교, 34개·그룹 안 상관.
import os, sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1"
cv = pd.read_csv(os.path.join(H, r"cache\s5_coverage.csv"), dtype={"code": str})
cv = cv[cv.code != "C"]
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\label_validity")
sys.dont_write_bytecode = True
from lv_common import ksic_to_group
out = {}
for a, b in ((2016, 2019), (2017, 2019)):
    x = cv[cv.year.isin([a, b])].pivot_table(index="code", columns="year", values=["est_all", "emp_all", "est_mms", "emp_mms", "est_ge10_census", "emp_ge10_census"])
    d = pd.DataFrame({f"g_{m}": np.log(x[(m, b)] / x[(m, a)]) for m in ["est_all", "emp_all", "est_mms", "emp_mms", "est_ge10_census", "emp_ge10_census"]}).replace([np.inf, -np.inf], np.nan)
    d["grp"] = d.index.map(ksic_to_group); gs = d.grp.value_counts(); d34 = d[d.grp.map(gs) > 1].dropna()
    dm = d34.drop(columns="grp") - d34.groupby("grp").transform("mean")
    r = {}
    for it in ("est", "emp"):
        r[it] = {"r_all50": round(float(d[[f"g_{it}_all", f"g_{it}_mms"]].dropna().corr().iloc[0, 1]), 3),
                 "r_within_group_34": round(float(dm[[f"g_{it}_all", f"g_{it}_mms"]].corr().iloc[0, 1]), 3),
                 "r_within_group_34_census_ge10_vs_all": round(float(dm[[f"g_{it}_all", f"g_{it}_ge10_census"]].corr().iloc[0, 1]), 3),
                 "sign_agree_within_34": round(float((np.sign(dm[f"g_{it}_all"]) == np.sign(dm[f"g_{it}_mms"])).mean()), 3), "n": int(len(dm))}
    out[f"{a}-{b}"] = r
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(H, r"cache\s5b_scope.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
