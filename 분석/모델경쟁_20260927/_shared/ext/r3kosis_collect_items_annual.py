# -*- coding: utf-8 -*-
"""r3kosis 연간 품목편(광업제조업조사/경제총조사) 품목별 사업체수·생산액·출하액·연말재고액 2003–2024 수집.
표 ID는 KOSIS 통합검색 API로 확인한 것. 키 값은 출력하지 않는다."""
import os, sys, json, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3kosis_lib as L

LOG = open(os.path.join(L.EXT, "r3kosis_collect.log"), "a", encoding="utf-8")
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

TABLES = [  # (자료연도 시작, 끝, tblId, 대상 규모)
    (2003, 2003, "DT_1F1530", "5명이상"), (2004, 2004, "DT_1F1532", "5명이상"),
    (2005, 2006, "DT_1F1532A", "5명이상"), (2007, 2007, "DT_1F1632A", "10명이상"),
    (2008, 2008, "DT_1F1632C", "10명이상"), (2009, 2009, "DT_1632D", "10명이상"),
    (2010, 2010, "DT_1KI3021", "10명이상(경제총조사)"), (2011, 2011, "DT_1632E", "10명이상"),
    (2012, 2012, "DT_1632F", "10명이상"), (2013, 2013, "DT_1632G", "10명이상"),
    (2014, 2014, "DT_1632H", "10명이상"), (2015, 2015, "DT_1KI1530", "10명이상(경제총조사)"),
    (2016, 2016, "DT_1FP1016A", "10명이상"), (2017, 2017, "DT_1FP1017A", "10명이상"),
    (2018, 2018, "DT_1FP1018A", "10명이상"), (2019, 2019, "DT_1FP1019A", "10명이상"),
    (2020, 2020, "DT_2KI2029_S", "10명이상(경제총조사연계)"), (2021, 2021, "DT_1FP1021A", "10명이상"),
    (2022, 2022, "DT_1FP1022A", "10명이상"), (2023, 2023, "DT_1FP1116A", "10명이상"),
    (2024, 2024, "DT_1FP1124A", "10명이상"),
]
OUTD = os.path.join(L.RAW, "items_annual"); os.makedirs(OUTD, exist_ok=True)

for y0, y1, tbl, scope in TABLES:
    fp = os.path.join(OUTD, f"{tbl}.parquet")
    if os.path.exists(fp):
        log(f"[{tbl}] 이미 있음 — 건너뜀"); continue
    try:
        m = L.meta(tbl, "ITM")
    except L.KosisLimit as e:
        log(f"한도 신호 → 중단: {e}"); break
    if not isinstance(m, list):
        log(f"[{tbl}] 메타 실패 {str(m)[:150]}"); continue
    json.dump(m, open(os.path.join(OUTD, f"meta_{tbl}_ITM.json"), "w", encoding="utf-8"), ensure_ascii=False)
    items = [x["ITM_ID"] for x in m if x["OBJ_ID"] == "ITEM"]
    order = []
    for x in m:
        if x["OBJ_ID"] != "ITEM" and x["OBJ_ID"] not in [o for o, _ in order]:
            order.append((x["OBJ_ID"], x.get("OBJ_NM", "")))
    objs = []
    for oid, onm in order:
        xs = [x for x in m if x["OBJ_ID"] == oid]
        if "품목" in onm:
            objs.append("ALL")
        else:  # 시도·행정구역·규모 등 → 첫 코드(전국/계)
            objs.append(xs[0]["ITM_ID"])
    log(f"[{tbl}] {y0}~{y1} {scope} 분류순서 {[(o, n) for o, n in order]} → objs {objs}, 항목 {items}")
    try:
        se = "F" if "경제총조사" in scope or tbl.startswith("DT_2KI") else "Y"  # 5년 주기표는 prdSe=F
        j = L.data(tbl, "+".join(items), objs, se, str(y0), str(y1))
    except L.KosisLimit as e:
        log(f"한도 신호 → 중단: {e}"); break
    if not isinstance(j, list):
        log(f"  응답 {str(j)[:200]}")
        # 셀 초과면 항목별로 나눠 받기
        parts = []
        for it in items:
            jj = L.data(tbl, it, objs, se, str(y0), str(y1))
            if isinstance(jj, list):
                parts += jj
            else:
                log(f"   항목 {it} 실패 {str(jj)[:150]}")
        j = parts
    if not j:
        continue
    D = pd.DataFrame(j)
    D["tbl"] = tbl; D["scope"] = scope
    D.to_parquet(fp, index=False)
    log(f"  저장 {len(D):,}행, 연도 {sorted(D.PRD_DE.unique())}")
log(f"호출 누계 {L.CALLS}")
