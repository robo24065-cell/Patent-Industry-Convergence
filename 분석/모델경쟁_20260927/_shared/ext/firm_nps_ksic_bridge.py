# -*- coding: utf-8 -*-
"""NPS 업종코드 개정(2019년 말~2020년: 국세청 업종코드가 KSIC 10차 체계로 바뀜) 보정 → 50 KSIC 산업 패널을 2015–2026 일관 시계열로.
방법: 같은 사업장(biz6+정규화상호+적용일자)이 2018-12(구코드)와 2021-12(신코드) 스냅샷에 모두 있으면,
      구코드별로 신코드→50KSIC 분포(사업장 수 비중)를 경험적 대응표로 만든다. 2019-12 이전 스냅샷은
      (a) 업종명이 50 KSIC와 바로 일치하면 그대로, (b) 아니면 대응표 비중으로 분할 배정.
출력: firm_nps_ksic_crosswalk.csv(구코드→KSIC50 비중), firm_nps_ksic50_panel.parquet(덮어씀: method 컬럼 포함)"""
import os, re, json, glob
import pandas as pd, numpy as np

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
d50 = json.load(open(os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"), encoding="utf-8"))
nn = lambda s: re.sub(r"[\s·,및]", "", str(s))
kmap = {nn(x["codenmKr"]): x["code"] for x in d50}
C = ["name", "biz6", "stat", "form", "indcd", "indnm", "adpt_dt", "n_emp", "n_new", "n_loss", "snap"]


def load(s):
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s.replace('-', '')}.parquet"), columns=C)
    d["key"] = d.biz6 + "|" + d.name.str.replace(r"\s", "", regex=True) + "|" + d.adpt_dt.fillna("")
    d["k50"] = d.indnm.map(lambda x: kmap.get(nn(x)))
    return d


old, new = load("2018-12"), load("2021-12")
j = old[["key", "indcd"]].drop_duplicates("key").merge(new[["key", "k50"]].drop_duplicates("key"), on="key")
j["k50"] = j.k50.fillna("other")
cw = j.groupby(["indcd", "k50"]).size().rename("n").reset_index()
cw["w"] = cw.n / cw.groupby("indcd").n.transform("sum")
cw = cw[(cw.k50 != "other")]
tot = j.groupby("indcd").size().rename("n_matched_wkpl")
cw = cw.merge(tot, on="indcd")
cw = cw[cw.n >= 3]  # 3개 사업장 이상 옮겨간 대응만 인정(잡음 제거)
cw.to_csv(os.path.join(EXT, "firm_nps_ksic_crosswalk.csv"), index=False, encoding="utf-8-sig")
print("대응표 행", len(cw), "구코드", cw.indcd.nunique(), "KSIC", cw.k50.nunique())

rows = []
for fp in sorted(glob.glob(os.path.join(EXT, "firm_nps_wkpl_*.parquet"))):
    s = os.path.basename(fp)[14:20]; s = f"{s[:4]}-{s[4:]}"
    d = load(s)
    sy = int(s[:4])
    d["born"] = (pd.to_numeric(d.adpt_dt.str[:4], errors="coerce") == sy).astype(float)
    d["act"] = (d.stat == "1").astype(float)
    d["one"] = 1.0
    direct = d[d.k50.notna()].assign(w=1.0, ksic=lambda x: x.k50, method="direct")
    parts = [direct]
    if sy <= 2019:
        rest = d[d.k50.isna()].merge(cw[["indcd", "k50", "w"]].rename(columns={"k50": "ksic"}), on="indcd")
        rest["method"] = "crosswalk"
        parts.append(rest)
    x = pd.concat(parts, ignore_index=True)
    for c in ["one", "act", "n_emp", "n_new", "n_loss", "born"]:
        x[c] = x[c].fillna(0) * x.w
    g = x.groupby("ksic").agg(n_wkpl=("one", "sum"), n_active=("act", "sum"), n_emp=("n_emp", "sum"), n_new_hire=("n_new", "sum"),
                              n_loss=("n_loss", "sum"), n_born_in_year=("born", "sum"),
                              share_crosswalk=("method", lambda m: (m == "crosswalk").mean())).reset_index()
    g["snap"] = s
    rows.append(g)
    print(s, g.ksic.nunique(), round(g.n_emp.sum()), flush=True)
P = pd.concat(rows, ignore_index=True)
P.to_parquet(os.path.join(EXT, "firm_nps_ksic50_panel.parquet"), index=False)
