# -*- coding: utf-8 -*-
r"""V3(보조): 4개 사례 출하 실질화(ECOS 생산자물가, 2026-10-13 재수신본). 창 2008–09→2016–17(÷8)과 2015–16(÷7)."""
import os, sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(B, r"분석\검증루프_20261012\C07\C07-T1")
KO = pd.read_csv(os.path.join(B, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
k10 = KO[(KO.기준 == "10차") & (KO.마스킹 == 0) & (KO.항목코드 == "T04")].pivot_table(index="세세분류코드", columns="연도", values="값")
def ppi(tb, it):
    j = json.load(open(os.path.join(H, rf"download\ecos\ecos_{tb}_{it}_2007_2018.json"), encoding="utf-8"))
    return pd.Series({int(r["TIME"]): float(r["DATA_VALUE"]) for r in j["StatisticSearch"]["row"]})
D = {"26211": [("404Y014", "309211AA", "LCD평판디스플레이")], "26212": [("404Y014", "309212AA", "기타전자표시장치"), ("404Y014", "309211AA", "LCD평판디스플레이")],
     "26111": [("404Y014", "309112AA", "집적회로"), ("404Y016", "30911201AA", "DRAM")], "29280": [("404Y014", "311253AA", "산업용로봇")]}
rows = []
g = lambda s, a, b, sp: 100 * (np.log(s[b].mean()) - np.log(s[a].mean())) / sp
for k, lst in D.items():
    s = k10.loc[k]
    for (ew, sp) in (([2016, 2017], 8.0), ([2015, 2016], 7.0)):
        nom = g(s, [2008, 2009], ew, sp)
        for tb, it, nm in lst:
            p = ppi(tb, it)
            real = g(s / p.reindex(s.index) * 100, [2008, 2009], ew, sp)
            rows.append({"ksic": k, "end_window": f"{ew[0]}-{ew[1]}", "deflator": f"{tb}:{it} {nm}", "nominal_g%": round(nom, 2),
                         "ppi_g%": round(g(p, [2008, 2009], ew, sp), 2), "real_g%": round(real, 2)})
R = pd.DataFrame(rows); print(R.to_string())
R.to_csv(os.path.join(H, r"cache\s3b_real.csv"), index=False, encoding="utf-8-sig")
