# -*- coding: utf-8 -*-
"""연계표 배정 vs 제공 KSIC 라벨 일치율 (출원연도 <= 2021만 — 봉인 구간 미접근).
(1) 제공 KR 출원(메인 IPC): 연계표 소분류 vs 제공 라벨 소분류 일치율(free/in), 범위 판정 재현율(free 배정이 25개 소분류 안인 비율)
(2) NTIS KR 특허 출원 성과(첫 IPC): 연계표로 '50개 산업 특허'를 판정했을 때 제공 데이터 연결(=제공 50산업 특허) 대비 정밀도·재현율
산출: agree_r7.json"""
import os, json, sys
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r7_concord import *

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
EXT = os.path.join(MC, r"_shared\ext")
res = {}

kr = duckdb.sql(f"""select appl_no, any_value(main_ipc) ipc, list(distinct ksic) ks, any_value(appl_year) ay
                 from read_parquet('{META}') where cc='KR' and appl_year <= 2021 group by appl_no""").df()
gf, lf = assign(kr.ipc.values, "free"); gi, li = assign(kr.ipc.values, "in")
k3 = kr.ks.map(lambda L: {str(x)[:3] for x in L})
kr["gf"], kr["gi"] = gf, gi
mf = np.array([g in s for g, s in zip(gf, k3)]); mi = np.array([g in s for g, s in zip(gi, k3)])
has = np.array([g != "" for g in gi])
res["provided_kr"] = dict(n=int(len(kr)), assigned_in=float(has.mean()), lookup_level_in=pd.Series(li).value_counts(normalize=True).round(4).to_dict(),
                          agree3_in=float(mi[has].mean()), agree3_free=float(mf[np.array([g != '' for g in gf])].mean()),
                          in_scope_recall_free=float(np.isin(gf, G50).mean()),
                          multi_label_share=float((kr.ks.map(len) > 1).mean()))
# 5자리: 연계표는 5자리를 낼 수 없다. 참고로 '그룹 안 5자리 코드 수' 분포와 무작위 선택 시 기대 일치율
nper = pd.Series([k[:3] for k in KSIC50]).value_counts()
ks5 = kr.ks.map(lambda L: str(L[0]))
exp5 = float(np.mean([(1.0 / nper[g]) if (g != "" and g == k[:3]) else 0.0 for g, k in zip(gi, ks5)]))
res["provided_kr"]["codes_per_group"] = nper.to_dict()
res["provided_kr"]["agree5_expected_if_uniform_within_group"] = exp5
by = kr.assign(m=mi, h=has).groupby("ay").apply(lambda d: d.m[d.h].mean())
res["provided_kr"]["agree3_in_by_year"] = {int(k): round(float(v), 4) for k, v in by.items() if k >= 2003}
print(json.dumps(res["provided_kr"], ensure_ascii=False)[:1500])

# (2) NTIS 성과
R = pd.read_parquet(f"{EXT}/ntis_rpatent_raw.parquet", columns=["ProjectID", "Year", "is_kr", "IprType", "RegistType", "RegistNumber", "num_digits", "ipc_codes"])
R["Year"] = pd.to_numeric(R.Year, errors="coerce")
R = R[R.is_kr.fillna(False).astype(bool) & (R.IprType == "특허") & (R.RegistType == "출원") & (R.Year <= 2021)].drop_duplicates("num_digits")
L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "appl_year"])
linked = set(L.appl_no)
R["prov"] = R.num_digits.isin(linked)
R["ipc1"] = R.ipc_codes.map(first_ipc)
g, l = assign(R.ipc1.values, "free"); R["gf"] = g
R["conc50"] = np.isin(g, G50)
ok = R.ipc1.notna()
t = R[ok & (R.Year >= 2007)]
tp = int((t.conc50 & t.prov).sum())
res["ntis_kr_app_outputs"] = dict(n=int(len(R)), ipc_present=float(ok.mean()), n_eval=int(len(t)), provided_share=float(t.prov.mean()), conc50_share=float(t.conc50.mean()),
                                  precision_conc50_vs_provided=float(tp / max(1, t.conc50.sum())), recall_conc50_vs_provided=float(tp / max(1, t.prov.sum())),
                                  jaccard=float(tp / max(1, (t.conc50 | t.prov).sum())))
print(json.dumps(res["ntis_kr_app_outputs"], ensure_ascii=False))
json.dump(res, open(os.path.join(HERE, "agree_r7.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
