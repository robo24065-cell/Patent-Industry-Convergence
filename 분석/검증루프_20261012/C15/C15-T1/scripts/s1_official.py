# -*- coding: utf-8 -*-
"""C15-T1 s1: 연보 원표(C13-T1 검증본, 읽기만)에서 특허 존속(1.7)·등록(1.5) 연도 계열을 판별로 읽고 판 사이 수정 여부, 최신판 4.1.3/4.1.11 내·외국인 값.
사용: python -I s1_official.py <moip_unzip> <out_dir>"""
import sys, os, glob, json, re
import openpyxl, pandas as pd
root, out = sys.argv[1], sys.argv[2]
ED = {"f_312_2": 2019, "f_328_2": 2020, "f_360_1": 2021, "f_380_2": 2022, "f_400_2": 2023, "f_416_2": 2024, "f_437_1": 2025}
rows = []
def norm(s): return re.sub(r"\s", "", str(s))
for src, ed in ED.items():
    f = [x for x in glob.glob(os.path.join(root, src, "*.xlsx")) if "총괄" in os.path.basename(x)][0]
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    for sn in wb.sheetnames:
        k = norm(sn)
        tab = "inforce" if "존속권리" in k else ("reg" if (k.endswith("연도별등록") or k.endswith("년도별등록")) else None)
        if tab is None: continue
        for r in wb[sn].iter_rows(values_only=True):
            if r and isinstance(r[0], (int, float)) and 1940 < r[0] < 2030 and isinstance(r[1], (int, float)):
                rows.append(dict(edition=ed, src=src, table=tab, year=int(r[0]), patent=float(r[1]), um=float(r[2]) if isinstance(r[2], (int, float)) else None))
            elif r and isinstance(r[0], str) and re.fullmatch(r"\s*\d{4}\s*", r[0]) and isinstance(r[1], (int, float)):
                rows.append(dict(edition=ed, src=src, table=tab, year=int(r[0]), patent=float(r[1]), um=float(r[2]) if isinstance(r[2], (int, float)) else None))
D = pd.DataFrame(rows).drop_duplicates(["edition", "table", "year"])
D.to_csv(os.path.join(out, "official_series_by_edition.csv"), index=False, encoding="utf-8-sig")
rev = {}
for tab in ["inforce", "reg"]:
    P = D[D.table == tab].pivot(index="year", columns="edition", values="patent")
    P = P[P.index >= 2000]
    diff = P.sub(P[2025], axis=0)
    nz = [(int(y), int(e), float(diff.loc[y, e])) for y in diff.index for e in diff.columns if pd.notna(diff.loc[y, e]) and diff.loc[y, e] != 0]
    rev[tab] = dict(n_cells=int(P.notna().sum().sum()), n_revised=len(nz), revised=nz[:30])
# 최신판 4.1.3 / 4.1.11 (내·외국인 특허)
f = [x for x in glob.glob(os.path.join(root, "f_437_1", "*.xlsx")) if "4-1" in os.path.basename(x)][0]
wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
res = {}
for sn in wb.sheetnames:
    k = norm(sn)
    if k.startswith("4.1.3.") or k.startswith("4.1.11."):
        cur = None; yrs = None; out_t = {}
        for r in wb[sn].iter_rows(values_only=True):
            r = list(r)
            if any(isinstance(v, (int, float)) and 2015 < v < 2030 for v in r[1:4]) and yrs is None:
                yrs = [int(v) for v in r if isinstance(v, (int, float)) and 2015 < v < 2030]
            lab = " ".join(str(v) for v in r[:2] if v is not None)
            if "내국인" in lab: cur = "dom"
            if "외국인" in lab: cur = "for"
            if "특허" in lab and "Patents" in lab:
                vals = [v for v in r[2:] if isinstance(v, (int, float))][:len(yrs)]
                key = cur if k.startswith("4.1.3.") else ("dom" if "내국인" in lab else "for" if "외국인" in lab else cur)
                out_t[key] = dict(zip(yrs, vals))
            if k.startswith("4.1.11.") and cur is None:
                pass
            if k.startswith("4.1.11.") and r[1] is not None and ("Resident" in str(r[1]) or "Non-resident" in str(r[1])) and "특허" in str(r[0] or prev0):
                vals = [v for v in r[2:] if isinstance(v, (int, float))][:len(yrs)]
                out_t["dom" if "내국인" in str(r[1]) else "for"] = dict(zip(yrs, vals))
            prev0 = r[0] if r[0] is not None else (prev0 if 'prev0' in dir() else "")
        res[k] = out_t
L = D[(D.edition == 2025)].pivot(index="year", columns="table", values="patent").rename(columns={"inforce": "I", "reg": "R"})
L.to_csv(os.path.join(out, "official_latest.csv"), encoding="utf-8-sig")
json.dump(dict(revisions=rev, resident_tables=res, files_total=len(D)), open(os.path.join(out, "s1_official.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(dict(revisions={k: {kk: vv for kk, vv in v.items() if kk != "revised"} for k, v in rev.items()}, resident_tables=res), ensure_ascii=False, indent=1))
print(L[L.index >= 2000].to_string())
