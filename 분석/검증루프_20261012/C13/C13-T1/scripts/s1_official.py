# -*- coding: utf-8 -*-
"""C13-T1 s1: 지식재산통계연보 '심판종류별 청구 및 처리' 표에서 특허·실용신안의 무효·권리범위확인·정정 청구 건수를 뽑는다.
입력: download/moip_unzip/*/ 의 심판 엑셀(읽기만). 출력: cache/official_trials_raw.csv"""
import sys, os, glob, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(W, "download", "moip_unzip")
# 출판 폴더 → 자료 연도(표 제목의 'in YYYY'가 없을 때만 사용)
PUBYEAR = {"f_267_2": 2017, "f_283_2": 2018, "f_312_2": 2019, "f_328_2": 2020, "f_360_1": 2021,
           "f_380_2": 2022, "f_400_2": 2023, "f_416_2": 2024, "f_437_1": 2025}
SEC = [("무효", "무효"), ("권리범위", "권리범위"), ("정정무효", "정정무효"), ("정정", "정정"), ("취소", "취소"),
       ("거절", "거절불복"), ("통상실시", "통상실시권"), ("존속기간", "존속기간"), ("보정각하", "보정각하"), ("재심", "재심")]
rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    tag = os.path.basename(d)
    for f in glob.glob(os.path.join(d, "*")):
        if "심판" not in os.path.basename(f):
            continue
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            df = xl.parse(sh, header=None)
            head = " ".join(str(x) for x in df.iloc[:4, 0].tolist())
            if not ("심판종류별 청구" in head):
                continue
            m = re.search(r"in (\d{4})", head)
            year = int(m.group(1)) if m else PUBYEAR.get(tag)
            sec = None; part = ""
            for i in range(len(df)):
                c0 = str(df.iat[i, 0]) if pd.notna(df.iat[i, 0]) else ""
                c1 = str(df.iat[i, 1]) if df.shape[1] > 1 and pd.notna(df.iat[i, 1]) else ""
                if "(1)" in c1 or "당사자계" in c1 and "(" in c1: part = "당사자계"
                if "(2)" in c1 or "결정계" in c1 and "(" in c1: part = "결정계"
                if c0.strip():
                    s0 = c0.replace("\n", "").replace(" ", "")
                    sec = next((lab for key, lab in SEC if s0.startswith(key)), "기타:" + s0[:12])
                right = "특허" if c1.replace(" ", "").startswith("특허") else ("실용" if c1.replace(" ", "").startswith("실용") else None)
                if sec and right:
                    v = df.iat[i, 2]
                    try: v = float(str(v).replace(",", ""))
                    except Exception: v = float("nan") if str(v).strip() not in ("-", "") else 0.0
                    rows.append(dict(year=year, src=tag, file=os.path.basename(f), sheet=sh.strip(), row=i, part=part,
                                     trial=sec, right=right, request=v, label=c0.replace("\n", " ")[:30]))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(W, "cache", "official_trials_raw.csv"), index=False, encoding="utf-8-sig")
pv = R[R.trial.isin(["무효", "권리범위", "정정"])].pivot_table(index="year", columns=["trial", "right"], values="request", aggfunc="sum")
pd.set_option("display.width", 250)
print(pv.to_string())
print(R.groupby("year").src.unique())
print(sorted(R.trial.unique()))
