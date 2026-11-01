# -*- coding: utf-8 -*-
"""누수 재탐색(N4): NPS 연결키 = (사업자번호 앞6, CORP_APPLICANT 2026-07 상호). 원점 이후 상호를 바꾼 존속기업은 t0 스냅샷에서 못 찾는다(사후 목록 편향?).
시험 원점 t0=2023에서 nps_present=0인 기업 중 t0 이후 스냅샷(2024-12·2025-12·2026-08)에서 같은 키로 잡히고 최초적용일 adpt_dt ≤ t0인 기업(= t0에 이미 가입 사업장이었음)을 센다.
또 (biz6, adpt_dt)로 t0 스냅샷에서 옛 상호 행을 찾을 수 있는지 본다. 🔒 레이블은 이미 채점된 시험 레이블만 사용."""
import sys, os, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
import n4lib
from n4lib import norm, EXT
import numpy as np, pandas as pd
D = pd.read_parquet(os.path.join(HERE, "cache", "final_panel.parquet"))
res = {}
for t0 in [2017, 2019, 2021, 2023]:
    te = D[D.t0 == t0].copy()
    post = [s for s in ["2015-12","2016-12","2017-12","2018-12","2019-12","2020-12","2021-12","2022-12","2023-12","2024-12","2025-12","2026-08"] if s > f"{t0}-12"]
    if t0 == 2023:
        post = ["2024-12", "2025-12", "2026-08"]
    C = ["name", "biz6", "stat", "adpt_dt"]
    later = pd.concat([pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s.replace('-', '')}.parquet"), columns=C).assign(snap=s) for s in post])
    later["key"] = later.biz6.fillna("") + "|" + later.name.map(norm)
    cur = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{t0}12.parquet"), columns=C)
    cur["key"] = cur.biz6.fillna("") + "|" + cur.name.map(norm)
    lk = later[later.adpt_dt.fillna("9999") <= f"{t0}-12-31"]
    a_key = lk.groupby("key").adpt_dt.min()
    m0 = te[(te.nps_present == 0) & te.key.notna()]
    hit = m0[m0.key.isin(a_key.index)].copy()
    hit["adpt"] = hit.key.map(a_key)
    # t0 스냅샷에서 (biz6, adpt_dt) 일치·상호 다른 행
    cur_ba = cur.assign(ba=cur.biz6.fillna("") + "|" + cur.adpt_dt.fillna("")).groupby("ba").key.agg(lambda x: set(x))
    hit["ba"] = hit.key.str.split("|").str[0] + "|" + hit.adpt.fillna("")
    hit["found_oldname_t0"] = hit.ba.map(lambda b: b in cur_ba.index)
    r = dict(n=len(te), pos=int(te.y.sum()) if te.y.notna().all() else None, nps0=int((te.nps_present == 0).sum()),
             nps0_rate=float(te.loc[te.nps_present == 0, "y"].mean()) if te.y.notna().all() else None,
             nps0_with_key=len(m0), hidden_member=len(hit), hidden_found_oldname_in_t0=int(hit.found_oldname_t0.sum()),
             hidden_share_of_pop=len(hit) / len(te))
    if te.y.notna().all():
        r["hidden_closure_rate"] = float(hit.y.mean()) if len(hit) else None
        r["other_nps0_closure_rate"] = float(te.loc[(te.nps_present == 0) & ~te.applicantCd.isin(hit.applicantCd), "y"].mean())
        r["hidden_pos"] = int(hit.y.sum())
    res[t0] = r
    hit[["applicantCd", "key", "adpt", "found_oldname_t0"]].to_parquet(os.path.join(HERE, "cache", f"nps_hidden_{t0}.parquet"), index=False)
    print(t0, r, flush=True)
json.dump(res, open(os.path.join(HERE, "v4_nps_rename_probe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
