# -*- coding: utf-8 -*-
"""C12-T1 s4c: s4_dist의 KR 집계 정정. s4는 DIPG 단위 국가를 'first'로 잡아 다국 DIPG의 KR 몫을 과소 집계했다(kr_family_share 4.9%, kr_dipg_by_refl_year).
정정: KR 구성원 여부는 famgen(kr=max), KR DIPG 반영연도는 link_records(KR 행만 묶은 s2 기록). 출력: cache/s4c_out.json"""
import os, sys, json, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fm = pd.read_parquet(os.path.join(W, "cache", "fam_company_gen.parquet"))
R = pd.read_parquet(os.path.join(W, "cache", "link_records.parquet"))
O = {}
fk = fm.groupby("DIPG_PATF_ID").kr.max()
O["kr_family_share_corrected"] = float(fk.mean())
d = R.groupby("DIPG_ID").refl.min()
O["kr_dipg_n"] = int(len(d))
O["kr_dipg_by_refl_year_corrected"] = (d // 10000).value_counts().sort_index().astype(int).to_dict()
y = d // 10000
O["kr_dipg_2011_share"] = float((y == 2011).mean())
dd = pd.to_datetime(d.astype(int).astype(str))
O["kr_dipg_2011_top_date"] = dd[y == 2011].dt.strftime("%Y-%m-%d").value_counts().head(3).to_dict()
O["kr_dipg_max_single_day_share_2014_2024"] = {int(yy): round(float(dd[y == yy].dt.strftime("%Y-%m-%d").value_counts().iloc[0] / (y == yy).sum()), 4) for yy in range(2014, 2025)}
O["dipg_multi_country_note"] = "DIPG 하나에 여러 국가 특허가 들어 있음(전체 DIPG 234,453 중 KR 행을 가진 DIPG 61,269)"
json.dump(O, open(os.path.join(W, "cache", "s4c_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(O, ensure_ascii=False, indent=1))
