# -*- coding: utf-8 -*-
"""v1b: app13 확장으로 새로 매칭된 26429 특허(원 규칙 미매칭) 전체의 선언 기업-출원인명 일치 확인. 출력 verify_0/v1b_out.json"""
import os, re, glob, json
import pandas as pd
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
kr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(W, "cache", "etsi", "kr_part_*.parquet")))], ignore_index=True)
def orig(app):
    a = re.sub(r"[^0-9A-Z]", "", str(app or "").upper()); a = a[2:] if a.startswith("KR") else a
    return "10" + a if re.fullmatch(r"\d{11}", a) else None
def app13(app):
    raw = str(app or "").upper()
    if re.search(r"[UY]\d?\s*$", raw): return None
    d = re.sub(r"\D", "", raw); return d if (len(d) == 13 and d.startswith("10")) else None
kr["o"] = kr.PATT_APPLICATION_NUMBER.map(orig); kr["x"] = kr.PATT_APPLICATION_NUMBER.map(app13)
ex = kr[kr.o.isna() & kr.x.notna()]
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원인명", "세세분류코드"])
k["a"] = k["출원번호"].astype(str).str.strip()
m = ex.merge(k, left_on="x", right_on="a")
okeys = set(kr.o.dropna())
m["already_orig_key_elsewhere"] = m.a.isin(okeys)
al = {"LG": "엘지", "Innovative Technology": "아이티엘", "ASUSTeK": "아서스", "Samsung": "삼성", "Electronics and Telecommunications": "한국전자통신", "KT ": "케이티", "Pantech": "팬택", "SK": "에스케이"}
def ok(c, n):
    for a, b in al.items():
        if a.lower() in str(c).lower(): return b in str(n)
    return None
m["name_ok"] = [ok(c, n) for c, n in zip(m.COMP_LEGAL_NAME, m["출원인명"])]
u26 = m[m["세세분류코드"].astype(str) == "26429"].drop_duplicates("a")
R = dict(etsi_rows_app13_only=int(len(ex)), etsi_app13_distinct=int(ex.x.nunique()), matched_all_ksic=int(m.a.nunique()),
         matched_26429_new_and_not_orig=int((~u26.already_orig_key_elsewhere).sum()), matched_26429=int(len(u26)),
         name_ok_26429=u26.name_ok.map(str).value_counts().to_dict(), companies_26429=u26.COMP_LEGAL_NAME.value_counts().to_dict(),
         name_ok_all=m.drop_duplicates("a").name_ok.map(str).value_counts().to_dict(),
         unknown_name_examples=[ascii(x) for x in m[m.name_ok.isna()].drop_duplicates("a")[["COMP_LEGAL_NAME", "출원인명"]].head(8).values.tolist()])
json.dump(R, open(os.path.join(V, "v1b_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=True, indent=1))
