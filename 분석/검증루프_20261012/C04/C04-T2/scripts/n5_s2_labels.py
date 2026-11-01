# -*- coding: utf-8 -*-
"""C04-T2 N5 레이블 독립 재구성: ETSI 원본(KR 행, n5_s1 산출) × 제공 kr_patents(KSIC 26429). n5lib·trade_etsi_sep_link 미사용.
키 규칙 두 가지: strict(원 연결 규칙과 같은 범위: 출원번호 11자리 'yyyy+7' 또는 KR 접두 11자리 + 등록공보 10+7자리)
                ext(확장: 13자리 '10…' 출원번호·접미 문자 허용 + 공개번호 10+yyyy+7 매칭 추가)
출력: cache/n5_units_indep.parquet, cache/n5_s2_out.json"""
import os, re, glob, json, time
import numpy as np, pandas as pd
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
N5 = os.path.join(BASE, r"분석\모델경쟁_20260927\N5_표준특허선언레이더")
kr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(W, "cache", "etsi", "kr_part_*.parquet")))], ignore_index=True)
kr["refl"] = pd.to_numeric(kr.Reflected_Date.str[:10].str.replace("-", ""), errors="coerce")
kr["sig"] = pd.to_numeric(kr.IPRD_SIGNATURE_DATE.str[:10].str.replace("-", ""), errors="coerce")


def app_keys(s):
    raw = str(s or "").upper().strip()
    a = re.sub(r"[^0-9A-Z]", "", raw)
    strict = []
    b = a[2:] if a.startswith("KR") else a
    if re.fullmatch(r"\d{11}", b):
        strict.append("10" + b)
    ext = list(strict)
    d = re.sub(r"\D", "", raw)
    um = bool(re.search(r"[UY]\d?$", a))          # 실용신안(U·Y 접미)은 특허 출원번호가 아님 → 확장 규칙에서 제외
    if len(d) == 13 and d.startswith("10"):
        ext.append(d)
    elif len(d) == 11 and re.fullmatch(r"(19|20)\d{9}", d) and not um:
        ext.append("10" + d)
    return strict, ext


def pub_keys(s):
    reg, pub = [], []
    for p in str(s or "").split("|"):
        p = p.strip().upper()
        m = re.match(r"^KR\s*(\d+)\s*([A-Z]\d?)?$", p)
        if not m:
            continue
        n = m.group(1)
        if re.fullmatch(r"10\d{7}", n):
            reg.append(n + "0000")
        elif re.fullmatch(r"(19|20)\d{9}", n):
            pub.append("10" + n)
    return reg, pub


K = kr.drop_duplicates(["DIPG_ID", "PATT_APPLICATION_NUMBER", "PUBL_NUMBER", "refl"])[["DIPG_ID", "IPRD_ID", "PATT_APPLICATION_NUMBER", "PUBL_NUMBER", "refl", "sig", "COMP_LEGAL_NAME"]].copy()
ak = K.PATT_APPLICATION_NUMBER.map(app_keys); pk = K.PUBL_NUMBER.map(pub_keys)
recs = []
for (s, e), (rg, pb), refl, sig, dip, comp in zip(ak, pk, K.refl, K.sig, K.DIPG_ID, K.COMP_LEGAL_NAME):
    for x in set(s): recs.append(("A", x, refl, sig, dip, comp, 1))
    for x in set(e) - set(s): recs.append(("A", x, refl, sig, dip, comp, 0))
    for x in set(rg): recs.append(("R", x, refl, sig, dip, comp, 1))
    for x in set(pb): recs.append(("P", x, refl, sig, dip, comp, 0))
R_ = pd.DataFrame(recs, columns=["kt", "key", "refl", "sig", "dipg", "comp", "strict"])
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "등록번호", "공개번호", "공개일자", "등록일자", "출원인명", "세세분류코드"])
k.columns = ["appl_no", "ay", "rg", "pb", "pub", "rgd", "names", "ksic"]
k = k[k.ksic.astype(str) == "26429"].copy()
k["appl_no"] = k.appl_no.astype(str).str.strip()
for c in ["pub", "rgd"]:
    k[c] = pd.to_numeric(k[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
k["rgk"] = k.rg.astype(str).str.replace(r"\D", "", regex=True)
k["pbk"] = k.pb.astype(str).str.replace(r"\D", "", regex=True)
mA = k[["appl_no"]].merge(R_[R_.kt == "A"], left_on="appl_no", right_on="key")
mR = k[["appl_no", "rgk"]].merge(R_[R_.kt == "R"], left_on="rgk", right_on="key")
mP = k[["appl_no", "pbk"]].merge(R_[R_.kt == "P"], left_on="pbk", right_on="key")
M = pd.concat([mA[["appl_no", "refl", "sig", "dipg", "comp", "strict", "kt"]], mR[["appl_no", "refl", "sig", "dipg", "comp", "strict", "kt"]], mP[["appl_no", "refl", "sig", "dipg", "comp", "strict", "kt"]]], ignore_index=True)
M.to_parquet(os.path.join(W, "cache", "n5_kr26429_etsi_matches.parquet"), index=False)
r_strict = M[M.strict == 1].groupby("appl_no").refl.min(); r_ext = M.groupby("appl_no").refl.min()
k["refl_s"] = k.appl_no.map(r_strict); k["refl_e"] = k.appl_no.map(r_ext)
k["known"] = k[["pub", "rgd"]].min(axis=1)
END, OUT_END = 20221231, 20241231


def units(rc):
    U = k[(k.known <= END) & k.ay.between(2017, 2022) & ~(k[rc] <= END)].copy()
    U["y"] = ((U[rc] > END) & (U[rc] <= OUT_END)).astype(int)
    return U


Us, Ue = units("refl_s"), units("refl_e")
Us[["appl_no", "ay", "y", "refl_s", "refl_e"]].to_parquet(os.path.join(W, "cache", "n5_units_indep.parquet"), index=False)
F = pd.read_parquet(os.path.join(N5, "final_preds_r4.parquet"))
F["appl_no"] = F.appl_no.astype(str)


def cmp(U, name):
    m = F[["appl_no", "y"]].merge(U[["appl_no", "y"]], on="appl_no", how="outer", suffixes=("_frz", "_ind"), indicator=True)
    b = m[m._merge == "both"]
    return dict(rule=name, n_indep=int(len(U)), pos_indep=int(U.y.sum()), n_frozen=int(len(F)), pos_frozen=int(F.y.sum()),
                only_frozen=int((m._merge == "left_only").sum()), only_indep=int((m._merge == "right_only").sum()),
                label_mismatch=int((b.y_frz != b.y_ind).sum()),
                mismatch_examples=b[b.y_frz != b.y_ind].appl_no.head(5).tolist(), only_indep_examples=m[m._merge == "right_only"].appl_no.head(5).tolist(),
                only_frozen_examples=m[m._merge == "left_only"].appl_no.head(5).tolist())


out = dict(etsi_kr_rows=int(len(kr)), etsi_kr_dipg=int(kr.DIPG_ID.nunique()), etsi_kr_decl=int(kr.IPRD_ID.nunique()),
           key_records=R_.kt.value_counts().to_dict(), kr26429_n=int(len(k)),
           kr26429_matched_strict=int(k.refl_s.notna().sum()), kr26429_matched_ext=int(k.refl_e.notna().sum()),
           kr26429_refl_differs=int(((k.refl_s != k.refl_e) & k.refl_s.notna()).sum()),
           strict=cmp(Us, "strict"), ext=cmp(Ue, "ext"), secs=round(time.time() - t0, 1))
json.dump(out, open(os.path.join(W, "cache", "n5_s2_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
