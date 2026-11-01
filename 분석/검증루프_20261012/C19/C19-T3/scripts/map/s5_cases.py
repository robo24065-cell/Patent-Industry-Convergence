# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\split34\s5_cases.py — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)
# -*- coding: utf-8 -*-
r"""split34 5단계: 사례 3개 표 + 그룹별 예측 쌍순서(개발) + 참고 부호검정.
사례: C2610(메모리·비메모리·LED·기타 반도체), C2620(LCD·OLED·PCB·실장기판·축전기), C2920(농기계·공작기계·건설기계·로봇).
특허는 개발 기간(2005–07→2015–17), KOSIS는 같은 창(2008–09→2016–17). 사후(2018–22) 특허 성장은 별도 열로만.
입력은 이 폴더의 중간 산출물(읽기). 출력은 이 폴더에만."""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = _C19_OUT
MC = os.path.abspath(os.path.join(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\split34", r"..\..\.."))
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group
M = pd.read_csv(os.path.join(HERE, "ksic_metrics_dev.csv"), index_col=0); M.index = M.index.astype(str)
P = pd.read_parquet(os.path.join(HERE, "panel_ksic_cc_year.parquet"))
P = P[(P.year >= 2005) & (P.year <= 2022)]
kr = P[P.cc == "KR"].groupby(["ksic", "year"]).n.sum().unstack().fillna(0)
tot = P.groupby(["ksic", "year"]).n.sum().unstack().fillna(0)
lm = lambda A, ys: np.log(A[ys].mean(axis=1) + 1)
post_kr = (lm(kr, [2020, 2021, 2022]) - lm(kr, [2015, 2016, 2017])) / 5
post_5 = (lm(tot, [2020, 2021, 2022]) - lm(tot, [2015, 2016, 2017])) / 5
CASES = ["C2610", "C2620", "C2920"]
rows = []
for g in CASES:
    m = M[M.grp == g]
    for k, r in m.iterrows():
        rows.append({"사례그룹": g, "ksic": k, "산업명": r["name"], "5국출원_2015_17(연평균)": round(r.n5_2015_17, 0),
                     "KR출원_연성장%(05-07→15-17)": round(100 * r.g_KR, 1), "5국출원_연성장%": round(100 * r["g_5국"], 1),
                     "CN출원_연성장%": round(100 * r.g_CN, 1),
                     "한국기원점유%_2015_17": round(100 * r["한국기원점유_2015_17"], 1), "한국기원점유_변화pp": round(r["한국기원점유_변화pp"], 1),
                     "KR출원_연성장%(08-09→16-17)": round(100 * r.gw_KR_lag0, 1),
                     "KOSIS종사자_연성장%(08-09→16-17)": round(100 * r.x_g_종사자수, 1), "KOSIS출하_연성장%": round(100 * r.x_g_출하액, 1),
                     "KOSIS사업체_연성장%": round(100 * r.x_g_사업체수, 1),
                     "수출_연성장%(배타HS6)": None if pd.isna(r.x_g_수출) else round(100 * r.x_g_수출, 1),
                     "수입_연성장%(배타HS6)": None if pd.isna(r.x_g_수입) else round(100 * r.x_g_수입, 1),
                     "사후_KR출원_연성장%(15-17→20-22)": round(100 * post_kr[k], 1), "사후_5국_연성장%": round(100 * post_5[k], 1)})
    # 연계표 해상도 값(그룹 하나의 값)
C = pd.DataFrame(rows)
C.to_csv(os.path.join(HERE, "t5_cases.csv"), index=False, encoding="utf-8-sig")
print(C.drop(columns=["산업명"]).to_string())

# 그룹별 예측 쌍순서(개발, 모형 M vs G=0.5)
PR = pd.read_parquet(os.path.join(HERE, "t3a_forecast_preds.parquet"))
PR = PR[(PR.t0 + 3 <= 2017) & (PR.모형 == "M_제공5자리")]
PR["grp"] = PR.ksic.map(ksic_to_group)
out = []
for (tgt, L, g), d in PR.groupby(["목표", "L", "grp"]):
    h = n = 0
    for t0, q in d.groupby("t0"):
        q = q.reset_index(drop=True)
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                if q.y[i] != q.y[j]:
                    dp = q.pred[i] - q.pred[j]; h += 0.5 if dp == 0 else float(np.sign(dp) == np.sign(q.y[i] - q.y[j])); n += 1
    out.append({"목표": tgt, "L": L, "grp": g, "쌍순서_M(개발 원점 합산)": round(h / n, 3), "n쌍": n})
F = pd.DataFrame(out).pivot_table(index="grp", columns=["목표", "L"], values="쌍순서_M(개발 원점 합산)")
F.columns = [f"{a}_L{b}" for a, b in F.columns]
F.to_csv(os.path.join(HERE, "t5_forecast_pairacc_by_group_dev.csv"), encoding="utf-8-sig")
print(F.round(3).to_string())

# 참고: 쌍 순서 일치(쌍끼리 독립 아님 → 참고용 이항검정)
meta = json.load(open(os.path.join(HERE, "t4_meta.json"), encoding="utf-8"))
ref = {}
for key, s in meta.items():
    a, b = map(int, s.split("/"))
    ref[key] = {"일치": s, "이항p_단측(참고,쌍 비독립)": round(binomtest(a, b, 0.5, alternative="greater").pvalue, 4)}
print(ref)
json.dump(ref, open(os.path.join(HERE, "t5_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
