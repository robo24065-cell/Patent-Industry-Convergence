# -*- coding: utf-8 -*-
"""C12-T1 verify_0: 독립 재계산(정규식·pandas 파서·s1/s2 캐시 미사용).
1) ETSI zip을 csv 모듈로 한 줄씩 읽어 KR 행만 추출(문자열 분해만 사용)
2) 엄격·완전 키를 str 메서드로 만들고 duckdb로 제공 KR 전체에 조인
3) 기록 연결률, 26429 매칭 출원 수, 출원별 최초 반영일(엄격·완전) 저장
출력: verify_0/v0_link.json, verify_0/v0_refl.parquet"""
import os, sys, io, csv, json, time, zipfile
import duckdb, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
V = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
Z = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\trade_etsi_ISLD-export.zip")
KR = os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet")
csv.field_size_limit(10 ** 8)
DIG = set("0123456789")


def digits(s):
    return "".join(c for c in s if c in DIG)


def isd(s):
    return len(s) > 0 and all(c in DIG for c in s)


def strip_kind(p):
    """끝의 ' B1', ' A', 'B1' 같은 종류 기호 제거."""
    p = p.strip()
    parts = p.split(" ")
    if len(parts) > 1 and len(parts[-1]) <= 2 and parts[-1][:1].isalpha():
        return " ".join(parts[:-1]).strip(), parts[-1]
    if len(p) > 2 and p[-1].isdigit() and p[-2].isalpha() and not p[-3].isalpha():
        return p[:-2], p[-2:]
    if len(p) > 1 and p[-1].isalpha() and p[-2].isdigit():
        return p[:-1], p[-1]
    return p, ""


def app_keys(a):
    """(strict, full) 출원 키 집합."""
    s, f = set(), set()
    a0 = a.strip().upper()
    if not a0 or a0 in ("-", "NONE"):
        return s, f
    clean = "".join(c for c in a0 if c.isalnum())
    b = clean[2:] if clean.startswith("KR") else clean
    if isd(b) and len(b) == 11:
        s.add("10" + b)
    r = a0[2:].strip() if a0.startswith("KR") else a0
    r, kind = strip_kind(r)
    if kind[:1] in ("U", "Y"):
        return s, s | f
    seps = r.replace("/", "-").replace(" ", "-").replace(".", "-").replace(",", "-")
    toks = [x for x in seps.split("-") if x]
    if len(toks) == 1 and isd(toks[0]) and len(toks[0]) == 13 and toks[0].startswith("10"):
        f.add(toks[0])
    elif len(toks) == 3 and all(isd(x) for x in toks) and toks[0] == "10" and len(toks[1]) == 4 and 4 <= len(toks[2]) <= 7:
        f.add("10" + toks[1] + toks[2].zfill(7))
    elif len(toks) == 2 and all(isd(x) for x in toks) and "/" not in r:
        if len(toks[0]) == 4 and toks[0][:2] in ("19", "20") and 4 <= len(toks[1]) <= 7:
            f.add("10" + toks[0] + toks[1].zfill(7))
        elif len(toks[0]) == 2 and 4 <= len(toks[1]) <= 7:
            y = int(toks[0]); y = 1900 + y if y >= 50 else 2000 + y
            f.add("10" + str(y) + toks[1].zfill(7))
    return s, s | f


def publ_keys(pl):
    """(strict, full) 공보 키: ('R', 등록13) / ('P', 공개13)."""
    s, f = set(), set()
    for p in pl.split("|"):
        p = p.strip().upper()
        if not p or p in ("-", "NONE"):
            continue
        core, kind = strip_kind(p)
        if core.startswith("KR") and isd(core[2:]) and len(core[2:]) == 9 and core[2:4] == "10":
            s.add(("R", core[2:] + "0000"))
        if not p.startswith("KR") or kind[:1] in ("U", "Y"):
            continue
        q = core[2:].strip()
        if not all(c in DIG or c in "-/ ," for c in q):
            continue
        d = digits(q)
        if len(d) == 9 and d.startswith("10"):
            f.add(("R", d + "0000"))
        elif len(d) == 7:
            f.add(("R", "10" + d + "0000"))
        elif len(d) == 6:
            f.add(("R", "100" + d + "0000"))
        elif len(d) == 11 and d[:2] in ("19", "20"):
            f.add(("P", "10" + d))
        elif len(d) == 13 and d.startswith("10"):
            f.add(("P", d))
    return s, s | f


z = zipfile.ZipFile(Z)
rd = csv.reader(io.TextIOWrapper(z.open("ISLD-export.csv"), encoding="utf-8", errors="replace", newline=""), delimiter=";")
hdr = next(rd); ix = {c: i for i, c in enumerate(hdr)}
need = ["DIPG_ID", "PATT_APPLICATION_NUMBER", "PUBL_NUMBER", "Original_Application_Number", "Original_Publication_Number", "Reflected_Date", "Country_Of_Registration", "IPRD_ID"]
rec = {}
n = nkr = 0
for row in rd:
    n += 1
    if len(row) < len(hdr):
        continue
    if not row[ix["Country_Of_Registration"]].startswith("KR"):
        continue
    nkr += 1
    key = tuple(row[ix[c]] for c in need[:5])
    refl = int(digits(row[ix["Reflected_Date"]][:10]) or 99999999)
    if key in rec:
        rec[key] = min(rec[key], refl)
    else:
        rec[key] = refl
print("rows", n, "kr rows", nkr, "records", len(rec), round(time.time() - t0), flush=True)
out = []
for rid, (key, refl) in enumerate(rec.items()):
    sa, fa = app_keys(key[1]); sp, fp = publ_keys(key[2])
    for a in fa:
        out.append((rid, "A", a, int(a in sa), refl))
    for kt, kk in fp:
        out.append((rid, kt, kk, int((kt, kk) in sp), refl))
K = pd.DataFrame(out, columns=["rid", "kt", "key", "strict", "refl"])
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
con.register("K", K)
con.sql(f"""create table kr as select cast("출원번호" as varchar) appl_no, regexp_replace(cast("등록번호" as varchar), '[^0-9]', '', 'g') rg,
            regexp_replace(cast("공개번호" as varchar), '[^0-9]', '', 'g') pb, cast("세세분류코드" as varchar) ksic from read_parquet('{KR.replace(chr(92), '/')}')""")
m = con.sql("""select K.rid, K.strict, K.refl, kr.appl_no, kr.ksic from K join kr on K.kt='A' and K.key=kr.appl_no
             union all select K.rid, K.strict, K.refl, kr.appl_no, kr.ksic from K join kr on K.kt='R' and K.key=kr.rg and length(kr.rg)>0
             union all select K.rid, K.strict, K.refl, kr.appl_no, kr.ksic from K join kr on K.kt='P' and K.key=kr.pb and length(kr.pb)>0""").df()
ms = m[m.strict == 1]
rs = ms.groupby("appl_no").refl.min(); rf = m.groupby("appl_no").refl.min()
k26s = set(ms.loc[ms.ksic == "26429", "appl_no"]); k26f = set(m.loc[m.ksic == "26429", "appl_no"])
R = dict(rows=n, kr_rows=nkr, records=len(rec), records_linked_strict=int(ms.rid.nunique()), records_linked_full=int(m.rid.nunique()),
         rate_strict=ms.rid.nunique() / len(rec), rate_full=m.rid.nunique() / len(rec),
         provided_appl_strict=int(len(rs)), provided_appl_full=int(len(rf)), kr26429_strict=len(k26s), kr26429_full=len(k26f),
         kr26429_earlier_full=int((rf.reindex(list(k26s)) < rs.reindex(list(k26s))).sum()),
         out26429_share_full=float((m.drop_duplicates("appl_no").ksic != "26429").mean()), secs=round(time.time() - t0, 1))
pd.DataFrame({"appl_no": rf.index, "refl_full": rf.values}).merge(pd.DataFrame({"appl_no": rs.index, "refl_strict": rs.values}), how="left").to_parquet(os.path.join(V, "v0_refl.parquet"), index=False)
json.dump(R, open(os.path.join(V, "v0_link.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
