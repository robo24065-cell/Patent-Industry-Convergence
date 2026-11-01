# -*- coding: utf-8 -*-
"""facts.csv·facts_supplement(_R12).csv 각 행을 manuscript_v3.md 출처표(id 행)와 대조. 출력 cache/facts_vs_ms.csv"""
import os, re, csv, json, hashlib
import pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
M = os.path.join(RP, "manuscript_v3.md")
lines = open(M, encoding="utf-8").read().split("\n")
tab = {}
for i, ln in enumerate(lines, 1):
    m = re.match(r"^\| ([A-Za-z0-9][A-Za-z0-9_\-]*)( \([^|]*\))? \| ([^|]*) \| (.*) \| ([^|]*) \|\s*$", ln)
    if m and m.group(1) != "id":
        tab[m.group(1)] = dict(line=i, sections=m.group(3).strip(), value=m.group(4).strip(), src=m.group(5).strip(), tag=(m.group(2) or "").strip())
rows = []
for fn in ["facts.csv", "facts_supplement.csv", "facts_supplement_R12.csv"]:
    p = os.path.join(RP, "facts", fn)
    with open(p, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f); hdr = next(rd)
        for k, r in enumerate(rd, start=2):          # 파일 행 번호(헤더=1)
            d = dict(zip(hdr, r))
            fid = d.get("id", "")
            t = tab.get(fid)
            rows.append(dict(file=fn, row=k, fact_id=fid, value=d.get("수치", ""), unit=d.get("단위", ""), fragment=d.get("정확한 문장 조각", ""), cond=d.get("조건·경고", ""),
                             ms_line=t["line"] if t else "", ms_value=t["value"] if t else "", ms_tag=t["tag"] if t else "",
                             same=(t is not None and t["value"].replace(" ", "") == d.get("정확한 문장 조각", "").replace(" ", ""))))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(W, "cache", "facts_vs_ms.csv"), index=False, encoding="utf-8-sig")
print("ms table ids", len(tab), "facts rows", len(df), "in table", (df.ms_line != "").sum(), "same", int(df.same.sum()))
print("ms ids not in any facts file:", sorted(set(tab) - set(df.fact_id))[:80])
print(json.dumps(dict(sha1=hashlib.sha1(open(M, "rb").read()).hexdigest(), n_lines=len(lines)), ensure_ascii=False))
d = df[(df.ms_line != "") & ~df.same]
for r in d.itertuples():
    print(f"{r.file}:{r.row} {r.fact_id} L{r.ms_line}\n  F: {r.fragment[:230]}\n  M: {r.ms_value[:230]}")
