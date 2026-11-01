# -*- coding: utf-8 -*-
"""C12-T1 s2: ETSI KR 행(s1 산출) → 모든 표기 독립 파싱 → 제공 KR 전체(50개 KSIC, 1,087,511건) 연결.
규칙 strict = trade_06_etsi_link.py와 같은 범위(출원번호 'KR'+11자리 또는 숫자 11자리 → '10'+11, 등록공보 KR10+7자리 → 등록번호).
규칙 full   = strict ∪ 확장(13자리 10…, 하이픈·슬래시·공백 표기, 2자리 연도·6자리 일련, Original_* 필드, 공개번호 10yyyy+7, 구 등록번호 KR0xxxxxx).
실용신안(출원 20…, U·Y 접미)은 제외. 출력: cache/link_records.parquet, cache/link_matches.parquet, cache/s2_out.json"""
import os, re, sys, glob, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
kr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(W, "cache", "etsi", "kr_*.parquet")))], ignore_index=True)
for g in ["2G", "3G", "4G", "5G"]:
    kr["g" + g[0]] = (kr[g] == "1").astype("int8")  # 정정: 값 0/1/빈칸, "1"만 해당 세대
kr["refl"] = pd.to_numeric(kr.Reflected_Date.str[:10].str.replace("-", ""), errors="coerce")
kr["sig"] = pd.to_numeric(kr.IPRD_SIGNATURE_DATE.str[:10].str.replace("-", ""), errors="coerce")
kr["appd"] = pd.to_numeric(kr.PBPA_APP_DATE.str[:10].str.replace("-", ""), errors="coerce")
grp = ["DIPG_ID", "PATT_APPLICATION_NUMBER", "PUBL_NUMBER", "Original_Application_Number", "Original_Publication_Number"]
R = kr.groupby(grp, dropna=False).agg(refl=("refl", "min"), sig=("sig", "min"), comp=("COMP_LEGAL_NAME", "first"), iprd=("IPRD_ID", "first"),
                                       n_decl=("IPRD_ID", "nunique"), appd=("appd", "min"), fam=("DIPG_PATF_ID", "first"),
                                       g5=("g5", "max"), g4=("g4", "max"), g3=("g3", "max"), g2=("g2", "max")).reset_index()
R["rid"] = np.arange(len(R))


def yy4(y):
    y = int(y)
    return 1900 + y if y >= 50 else 2000 + y


def strict_keys(app, publ):
    """trade_06 그대로."""
    ks = []
    a = re.sub(r"[^0-9A-Z]", "", app or "")
    b = a[2:] if a.startswith("KR") else a
    if re.fullmatch(r"\d{11}", b):
        ks.append(("A", "10" + b))
    for p in str(publ).split("|"):
        p = p.strip()
        if not p or p in ("-", "None"):
            continue
        p = re.sub(r"\s+[A-Z]\d?$", "", p)
        if p.startswith("KR") and re.fullmatch(r"10\d{7}", p[2:]):
            ks.append(("R", p[2:] + "0000"))
    return ks


def app_full(s):
    """출원번호 표기 → 13자리 특허 출원번호(없으면 None, 사유)."""
    raw = str(s or "").strip().upper().replace("\u00a0", " ")
    if raw in ("", "-", "NONE"):
        return None, "blank"
    if raw.startswith(("US", "CN", "WO", "EP", "JP", "PCT", "DE")):
        return None, "foreign_number"
    if re.search(r"[UY]\d?$", raw) or raw.startswith("KR2") and len(re.sub(r"\D", "", raw)) == 13 and re.sub(r"\D", "", raw).startswith("20"):
        return None, "utility"
    r = raw[2:].strip() if raw.startswith("KR") else raw
    r = re.sub(r"\s*[A-Z]\d?$", "", r)  # 끝 종류 문자(A 등)
    d = re.sub(r"\D", "", r)
    parts = [x for x in re.split(r"[-/ ,\.]+", r) if x]
    if re.fullmatch(r"\d+", r.replace(" ", "")):
        if len(d) == 13 and d.startswith("10"):
            return d, "13"
        if len(d) == 13 and d.startswith("20"):
            return None, "utility"
        if len(d) == 11 and re.fullmatch(r"(19|20)\d{9}", d):
            return "10" + d, "11"
        if len(d) == 10 and re.fullmatch(r"(19|20)\d{8}", d):
            return "10" + d[:4] + "0" + d[4:], "10"
        return None, f"digits{len(d)}"
    if all(re.fullmatch(r"\d+", x) for x in parts):
        if len(parts) == 3 and parts[0] in ("10",) and len(parts[1]) == 4 and 4 <= len(parts[2]) <= 7:
            return "10" + parts[1] + parts[2].zfill(7), "10-yyyy-n"
        if len(parts) == 3 and parts[0] == "20":
            return None, "utility"
        if len(parts) == 2 and len(parts[0]) == 4 and parts[0][:2] in ("19", "20") and 4 <= len(parts[1]) <= 7:
            return "10" + parts[0] + parts[1].zfill(7), "yyyy-n"
        if len(parts) == 2 and len(parts[0]) == 2 and 4 <= len(parts[1]) <= 7 and "/" not in r:
            return "10" + str(yy4(parts[0])) + parts[1].zfill(7), "yy-n"
        if len(parts) == 2 and len(parts[1]) == 2 and 4 <= len(parts[0]) <= 7 and "/" in r:
            return "10" + str(yy4(parts[1])) + parts[0].zfill(7), "n/yy"
        if len(parts) == 3 and len(parts[1]) == 4 and len(parts[0]) == 2:
            return "10" + parts[1] + parts[2].zfill(7), "xx/yyyy/n"
    return None, "other"


def publ_full(s):
    """공개·등록 표기 → [(R, 등록13자리) | (P, 공개13자리)]."""
    out = []
    for p in str(s or "").replace("\u00a0", " ").split("|"):
        p = p.strip().upper()
        if not p or p in ("-", "NONE"):
            continue
        kr_pref = p.startswith("KR")
        q = p[2:].strip() if kr_pref else p
        q = re.sub(r"\s*[A-Z]\d?$", "", q)
        d = re.sub(r"\D", "", q)
        sep = bool(re.search(r"[-/ ]", q))
        if not re.fullmatch(r"[\d\-/ ,]+", q):
            continue
        if len(d) == 9 and d.startswith("10"):
            out.append(("R", d + "0000"))
        elif len(d) == 7 and (kr_pref or not sep):
            out.append(("R", "10" + d + "0000"))
        elif len(d) == 6 and kr_pref:
            out.append(("R", "100" + d + "0000"))
        elif len(d) == 11 and re.fullmatch(r"(19|20)\d{9}", d):
            out.append(("P", "10" + d))
        elif len(d) == 13 and d.startswith("10"):
            out.append(("P", d))
        elif sep and re.fullmatch(r"10-\d{7}", q):
            out.append(("R", d + "0000"))
    return out


APP_OK = {"13", "10-yyyy-n", "yy-n", "yyyy-n"}   # 이름 일치 검증(x_fullonly.py) 통과 형식만 완전 키에 넣는다(변경 기록 1)


def publ_ok(s):
    """완전 키용 공보 번호: PUBL_NUMBER의 'KR' 접두 항목만, 실용신안(U·Y 종류) 제외."""
    out = []
    for p in str(s or "").replace(" ", " ").split("|"):
        p = p.strip().upper()
        if not p.startswith("KR") or re.search(r"\s[UY]\d?$", p):
            continue
        out += publ_full(p)
    return out


recs, reasons = [], []
for r in R.itertuples(index=False):
    sk = set(strict_keys(r.PATT_APPLICATION_NUMBER, r.PUBL_NUMBER))
    fk = set(sk)
    a, why = app_full(r.PATT_APPLICATION_NUMBER); reasons.append(why)
    if a and why in APP_OK:
        fk.add(("A", a))
    for x in publ_ok(r.PUBL_NUMBER):
        fk.add(x)
    bk = set(fk)                       # broad(1차 시도, 검증에서 기각된 형식 포함) — 기록용
    if a:
        bk.add(("A", a))
    a2, _ = app_full(r.Original_Application_Number)
    if a2:
        bk.add(("A", a2))
    for x in publ_full(r.PUBL_NUMBER) + publ_full(r.Original_Publication_Number):
        bk.add(x)
    for kt, key in bk:
        lvl = 1 if (kt, key) in sk else (2 if (kt, key) in fk else 3)
        recs.append((r.rid, kt, key, int(lvl == 1), lvl))
R["app_reason"] = reasons
K = pd.DataFrame(recs, columns=["rid", "kt", "key", "strict", "lvl"])
print("records", len(R), "keys", len(K), {f"{a}_{b}": int(v) for (a, b), v in K.groupby(["kt", "lvl"]).size().items()}, round(time.time() - t0), flush=True)
# ---- 제공 KR 전체
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"),
                    columns=["출원번호", "출원연도", "등록번호", "공개번호", "등록일자", "공개일자", "출원인명", "메인IPC", "세세분류코드"])
k.columns = ["appl_no", "ay", "rg", "pb", "rgd", "pubd", "names", "ipc", "ksic"]
k["appl_no"] = k.appl_no.astype(str).str.strip(); k["ksic"] = k.ksic.astype(str)
k["rgk"] = k.rg.astype(str).str.replace(r"\D", "", regex=True); k["pbk"] = k.pb.astype(str).str.replace(r"\D", "", regex=True)
mA = K[K.kt == "A"].merge(k[["appl_no"]], left_on="key", right_on="appl_no")
mR = K[K.kt == "R"].merge(k.loc[k.rgk.str.len() > 0, ["appl_no", "rgk"]], left_on="key", right_on="rgk")
mP = K[K.kt == "P"].merge(k.loc[k.pbk.str.len() > 0, ["appl_no", "pbk"]], left_on="key", right_on="pbk")
M = pd.concat([x[["rid", "kt", "key", "strict", "lvl", "appl_no"]] for x in (mA, mR, mP)], ignore_index=True)
M = M.merge(R[["rid", "DIPG_ID", "refl", "sig", "comp", "iprd", "appd", "g5", "g4", "g3", "g2", "fam"]], on="rid")
M = M.merge(k[["appl_no", "ay", "ksic", "ipc", "names", "rgd", "pubd"]], on="appl_no")
R.to_parquet(os.path.join(W, "cache", "link_records.parquet"), index=False)
M.to_parquet(os.path.join(W, "cache", "link_matches.parquet"), index=False)
K.to_parquet(os.path.join(W, "cache", "link_keys.parquet"), index=False)
# ---- 요약
MB = M; M = MB[MB.lvl <= 2]
rec_s = set(M.loc[M.strict == 1, "rid"]); rec_f = set(M.rid)
ap_s = M.loc[M.strict == 1].groupby("appl_no").refl.min(); ap_f = M.groupby("appl_no").refl.min()
k26 = set(k.loc[k.ksic == "26429", "appl_no"])
out = dict(etsi_kr_rows=int(len(kr)), records=int(len(R)), dipg=int(R.DIPG_ID.nunique()), decl=int(kr.IPRD_ID.nunique()),
           app_reason=R.app_reason.value_counts().to_dict(),
           records_matched_strict=len(rec_s), records_matched_full=len(rec_f),
           dipg_matched_strict=int(M.loc[M.strict == 1, "DIPG_ID"].nunique()), dipg_matched_full=int(M.DIPG_ID.nunique()),
           provided_appl_matched_strict=int(len(ap_s)), provided_appl_matched_full=int(len(ap_f)),
           kr26429_matched_strict=int(len(set(ap_s.index) & k26)), kr26429_matched_full=int(len(set(ap_f.index) & k26)),
           kr26429_refl_earlier_in_full=int(((ap_f.reindex(list(set(ap_s.index) & k26)) < ap_s.reindex(list(set(ap_s.index) & k26)))).sum()),
           match_kt_counts={f"{a}_{b}": int(v) for (a, b), v in MB.drop_duplicates(["rid", "kt", "lvl"]).groupby(["kt", "lvl"]).size().items()},
           multi_appl_per_record=int((M.groupby("rid").appl_no.nunique() > 1).sum()),
           broad_kr26429=int(len(set(MB.appl_no) & k26)), broad_provided_appl=int(MB.appl_no.nunique()),
           secs=round(time.time() - t0, 1))
json.dump(out, open(os.path.join(W, "cache", "s2_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
