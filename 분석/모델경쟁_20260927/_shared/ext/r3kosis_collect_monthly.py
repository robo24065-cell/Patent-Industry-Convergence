# -*- coding: utf-8 -*-
"""r3kosis 월별 수집: 품목별 물량(DT_1F02012), 생산능력·가동률(DT_1F32001), 제조업 국내공급지수(DT_1MS2001/2003).
원자료 저장만 한다. 키 값은 출력하지 않는다."""
import os, sys, json, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3kosis_lib as L

LOG = open(os.path.join(L.EXT, "r3kosis_collect.log"), "a", encoding="utf-8")
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

JOBS = [
    # tbl, itm, objs, start_year, end_year, years_per_call, out
    ("DT_1F02012", "T40+T41+T42+T43+T44", ["ALL"], 1995, 2026, 4, "r3kosis_item_volume_monthly"),
    ("DT_1F32001", "T10+T20", ["ALL"], 2000, 2026, 8, "r3kosis_capacity_util_monthly"),
    ("DT_1MS2001", "T1+T2+T3", ["ALL"], 2010, 2026, 17, "r3kosis_domsupply_ind_monthly"),
    ("DT_1MS2003", "T1+T2+T3", ["ALL", "ALL"], 2010, 2026, 3, "r3kosis_domsupply_indxgoods_monthly"),
]

only = sys.argv[1:] or None
for tbl, itm, objs, y0, y1, step, out in JOBS:
    if only and tbl not in only:
        continue
    log(f"[{tbl}] 수집 시작 {y0}~{y1}")
    parts = []
    y = y0
    while y <= y1:
        ye = min(y + step - 1, y1)
        try:
            j = L.data(tbl, itm, objs, "M", f"{y}01", f"{ye}12")
        except L.KosisLimit as e:
            log(f"  한도/차단 신호 → 중단: {e}"); break
        if isinstance(j, list):
            parts.append(pd.DataFrame(j)); log(f"  {y}~{ye}: {len(j):,}행")
        else:
            log(f"  {y}~{ye}: 응답 {str(j)[:160]}")
            if step > 1 and str(j.get('err')) in ('31',):  # 40,000셀 초과 → 쪼개기
                step = max(1, step // 2); continue
        y = ye + 1
    if not parts:
        log("  수집 없음"); continue
    D = pd.concat(parts, ignore_index=True)
    keep = [c for c in ("TBL_ID", "PRD_DE", "ITM_ID", "ITM_NM", "C1", "C1_NM", "C2", "C2_NM",
                        "DT", "UNIT_NM", "LST_CHN_DE") if c in D.columns]
    D = D[keep].drop_duplicates()
    D["DT_num"] = pd.to_numeric(D["DT"], errors="coerce")
    D.to_parquet(os.path.join(L.EXT, out + ".parquet"), index=False)
    log(f"  저장 {out}.parquet {len(D):,}행, 기간 {D.PRD_DE.min()}~{D.PRD_DE.max()}, "
        f"분류 {D.C1.nunique()}종, 결측/비수치 {int(D.DT_num.isna().sum())}")
log(f"호출 누계 {L.CALLS}")
