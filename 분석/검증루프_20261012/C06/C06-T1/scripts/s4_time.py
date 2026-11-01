# -*- coding: utf-8 -*-
r"""C06-T1 s4 (보조, 판정 외): (d) 시점 정합 대리 — DART 2026 코드 고정, 문서 출원연도 구간별 제공·BEST 일치.
업종이 시간에 따라 바뀌면 2026 스냅샷과의 일치는 최근 문서일수록 높아야 한다. 개발 분할(2010–14)은 표 ≤2009, 평가(2015–19)는 표 ≤2014 BEST 라벨(R12 그대로).
쓰기: out/s4_time.csv"""
import os, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(HERE, "out")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12"); LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
KS5 = np.array(KSIC50 + [""], dtype=object)
LB = pd.read_parquet(os.path.join(R12, "cache", "labels.parquet"), columns=["appl_no", "ay", "split", "PROV", "H-mg-3", "H-sc-1"]); LB["appl_no"] = LB.appl_no.astype(str)
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet")); reg = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet"))
rows = []
for crit, col, best in (("DART5", "dart", "H-mg-3"), ("NPS5", "nps", "H-sc-1")):
    R = reg[["applicantCd", col]].dropna().rename(columns={col: "r"}); R = R[R.r.isin(KSIC50)]
    Z = pf.merge(LB, on="appl_no").merge(R, on="applicantCd")
    Z["mP"] = (KS5[Z.PROV.values.astype(int)] == Z.r.values).astype(float); Z["mB"] = (KS5[Z[best].values.astype(int)] == Z.r.values).astype(float)
    evf = set(Z[Z.split == "eval"].applicantCd); devf = set(Z[Z.split == "dev"].applicantCd); both = evf & devf
    for per, m in (("2010-14(dev)", Z.split == "dev"), ("2015-19(eval)", Z.split == "eval")):
        for scope, fs in (("구간 전체 기업", None), ("두 구간 모두 문서 있는 기업", both)):
            ZZ = Z[m] if fs is None else Z[m & Z.applicantCd.isin(fs)]
            T = ZZ.groupby("applicantCd")[["mP", "mB"]].mean()
            rng = np.random.default_rng(20261012); idx = rng.integers(0, len(T), (1000, len(T))); d = T.mP.values[idx].mean(1) - T.mB.values[idx].mean(1)
            rows.append({"기준": crit, "문서구간": per, "범위": scope, "기업": len(T), "PROV": round(T.mP.mean(), 4), "BEST": round(T.mB.mean(), 4),
                         "PROV-BEST": round(T.mP.mean() - T.mB.mean(), 4), "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)})
    for y in range(2010, 2020):
        ZZ = Z[Z.ay == y]; T = ZZ.groupby("applicantCd")[["mP", "mB"]].mean()
        rows.append({"기준": crit, "문서구간": str(y), "범위": "연도별", "기업": len(T), "PROV": round(T.mP.mean(), 4), "BEST": round(T.mB.mean(), 4), "PROV-BEST": round(T.mP.mean() - T.mB.mean(), 4)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "s4_time.csv"), index=False, encoding="utf-8-sig"); print("done")
