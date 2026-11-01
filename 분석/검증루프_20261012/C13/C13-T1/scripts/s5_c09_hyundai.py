# -*- coding: utf-8 -*-
"""C13-T1 s5: 지식재산통계연보 '권리별 다출원인 순위'에서 현대자동차·기아 특허 출원 공식값(2014–2019 자료연도)을 뽑는다(C09 이월).
입력: download/moip_unzip/* 출원 엑셀(읽기만). 출력: cache/s5_hyundai.json"""
import sys, os, glob, json, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = os.path.join(W, "download", "moip_unzip")
PUB = {"f_205_1": 2013, "idx_2015": 2015, "f_250_1": 2016, "f_267_2": 2017, "f_283_2": 2018, "f_312_2": 2019, "f_328_2": 2020}
out = []
for tag, ydata in PUB.items():
    for f in glob.glob(os.path.join(U, tag, "*")):
        b = os.path.basename(f)
        if "출원" not in b: continue
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            if "다출원" not in sh or "국제" in sh or "PCT" in sh: continue
            df = xl.parse(sh, header=None)
            head = " ".join(str(x) for x in df.iloc[:3, 0].tolist())
            m = re.search(r"in (\d{4})", head)
            for i in range(len(df)):
                for j in range(df.shape[1] - 1):
                    v = str(df.iat[i, j])
                    if ("현대자동차" in v or v.strip() in ("기아", "기아자동차", "기아 주식회사") or "기아자동차" in v) and len(v) < 30:
                        # 같은 행 오른쪽 첫 숫자
                        num = next((df.iat[i, k] for k in range(j + 1, min(j + 3, df.shape[1])) if str(df.iat[i, k]).replace(",", "").replace(".0", "").strip().isdigit()), None)
                        out.append(dict(src=tag, data_year=int(m.group(1)) if m else ydata, sheet=sh.strip(), row=i, col=j, name=v.strip(), value=num,
                                        header=head[:80]))
D = pd.DataFrame(out)
print(D.to_string())
D.to_json(os.path.join(W, "cache", "s5_hyundai.json"), orient="records", force_ascii=False, indent=1)
