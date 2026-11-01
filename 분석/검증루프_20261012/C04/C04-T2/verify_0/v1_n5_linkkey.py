# -*- coding: utf-8 -*-
"""반박 검증 v1: N5 연결 키 형식 누락 주장. ETSI KR 원행(kr_part)과 제공 kr_patents(26429)를 독자 키 코드로 다시 결합.
작업자 matches parquet·units parquet 미사용. 출력: verify_0/v1_out.json"""
import os, re, glob, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
t0 = time.time()
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
N5 = os.path.join(BASE, r"분석\모델경쟁_20260927\N5_표준특허선언레이더")
kr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(W, "cache", "etsi", "kr_part_*.parquet")))], ignore_index=True)
kr["refl"] = pd.to_numeric(kr.Reflected_Date.str[:10].str.replace("-", ""), errors="coerce")
R = {"etsi_kr_rows": int(len(kr))}

# 원 규칙(trade_06 그대로 옮김): app = [^0-9A-Z] 제거, cc 접두 제거 후 11자리 → '10'+a ; 공보 KR 10+7자리 → 등록 n+'0000'
def orig_keys(app, pub):
    a = re.sub(r"[^0-9A-Z]", "", str(app or "").upper()); a = a[2:] if a.startswith("KR") else a
    ks = []
    if re.fullmatch(r"\d{11}", a): ks.append(("A", "10" + a))
    for p in str(pub or "").split("|"):
        p = re.sub(r"\s+[A-Z]\d?$", "", p.strip())
        if p.startswith("KR") and re.fullmatch(r"10\d{7}", p[2:]): ks.append(("R", p[2:] + "0000"))
    return ks

# 주장한 확장: 출원번호 13자리 '10…' (구분자 제거 후), KR10… 형식만 추가 (공개번호 키는 넣지 않음)
def app13(app):
    raw = str(app or "").upper()
    if re.search(r"[UY]\d?\s*$", raw): return None
    d = re.sub(r"\D", "", raw)
    return d if (len(d) == 13 and d.startswith("10")) else None

K = kr.drop_duplicates(["DIPG_ID", "PATT_APPLICATION_NUMBER", "PUBL_NUMBER", "refl"])
rows = []
for app, pub, refl, comp, dip, ad in zip(K.PATT_APPLICATION_NUMBER, K.PUBL_NUMBER, K.refl, K.COMP_LEGAL_NAME, K.DIPG_ID, K.PBPA_APP_DATE):
    o = orig_keys(app, pub)
    for t, k in o: rows.append((t, k, refl, comp, dip, ad, app, pub, "orig"))
    x = app13(app)
    if x and ("A", x) not in o: rows.append(("A", x, refl, comp, dip, ad, app, pub, "app13"))
    # 공개번호 키(작업자 ext 규칙에 있던 것) — 별도 태그
    for p in str(pub or "").split("|"):
        m = re.match(r"^KR\s*((?:19|20)\d{9})\s*[A-Z]?\d?$", p.strip().upper())
        if m: rows.append(("P", "10" + m.group(1), refl, comp, dip, ad, app, pub, "pubno"))
E = pd.DataFrame(rows, columns=["kt", "key", "refl", "comp", "dipg", "etsi_appdate", "raw_app", "raw_pub", "src"])
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"))
cols = k.columns.tolist()
k = k.rename(columns={cols[0]: "appl_no"})
name_col = "출원인명"; date_col = "출원일"
assert name_col in cols and "세세분류코드" in cols, cols
k = k[k["세세분류코드"].astype(str) == "26429"].copy()
k["appl_no"] = k.appl_no.astype(str).str.strip()
k["rgk"] = k["등록번호"].astype(str).str.replace(r"\D", "", regex=True)
k["pbk"] = k["공개번호"].astype(str).str.replace(r"\D", "", regex=True)
R["kr26429_n"] = int(len(k))
mA = k.merge(E[E.kt == "A"], left_on="appl_no", right_on="key")
mR = k.merge(E[E.kt == "R"], left_on="rgk", right_on="key")
mP = k.merge(E[E.kt == "P"], left_on="pbk", right_on="key")
M = pd.concat([mA, mR, mP], ignore_index=True)
orig = M[M.src == "orig"]; c13 = M[M.src.isin(["orig", "app13"])]; allm = M
R["matched_orig"] = int(orig.appl_no.nunique()); R["matched_orig_plus_app13"] = int(c13.appl_no.nunique()); R["matched_orig_app13_pub"] = int(allm.appl_no.nunique())
R["matched_orig_plus_pub_only"] = int(M[M.src.isin(["orig", "pubno"])].appl_no.nunique())
r_o = orig.groupby("appl_no").refl.min(); r_13 = c13.groupby("appl_no").refl.min(); r_all = allm.groupby("appl_no").refl.min()
dt = lambda s: pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
k["pub"] = dt(k["공개일자"]); k["rgd"] = dt(k["등록일자"]); k["known"] = k[["pub", "rgd"]].min(axis=1)
END, OUT = 20221231, 20241231
F = pd.read_parquet(os.path.join(N5, "final_preds_r4.parquet")); F["appl_no"] = F.appl_no.astype(str)
S = pd.read_parquet(os.path.join(N5, r"VERIFY_R5\cache\sens_preds.parquet")); S["appl_no"] = S.appl_no.astype(str)
assert (S.appl_no.values == F.appl_no.values).all()
sk = lambda a, b: (a - b) / (1 - b)
def units(r):
    rr = k.appl_no.map(r)
    U = k[(k.known <= END) & k["출원연도"].between(2017, 2022) & ~(rr <= END)].copy()
    U["y"] = ((rr[U.index] > END) & (rr[U.index] <= OUT)).astype(int)
    return U
for nm, r in [("orig", r_o), ("app13", r_13), ("app13_pub", r_all)]:
    U = units(r)
    ind = F.appl_no.isin(U.appl_no).values
    yy = U.set_index("appl_no").y.reindex(F.appl_no[ind]).values.astype(int)
    pc, p0, pS = F.p_champ.values[ind], F.p_M0.values[ind], S.p_M0plus.values[ind]
    R["units_" + nm] = dict(n=int(len(U)), pos=int(U.y.sum()), frozen_n=int(len(F)), dropped_from_frozen=int((~ind).sum()), dropped_pos=int(F.y.values[~ind].sum()),
                           not_in_frozen=int((~U.appl_no.isin(F.appl_no)).sum()), label_flips=int((yy != F.y.values[ind]).sum()),
                           auc_champ=float(roc_auc_score(yy, pc)), auc_M0=float(roc_auc_score(yy, p0)), auc_strong=float(roc_auc_score(yy, pS)),
                           skill=sk(roc_auc_score(yy, pc), roc_auc_score(yy, p0)), skill_vs_strong=sk(roc_auc_score(yy, pc), roc_auc_score(yy, pS)))
# 탈락 단위 상세
Uo = set(units(r_o).appl_no); Ua = set(units(r_all).appl_no); U13 = set(units(r_13).appl_no)
drop_all = sorted(Uo - Ua); drop13 = sorted(Uo - U13)
R["drop_by_app13_only"] = len(drop13); R["drop_by_app13_pub"] = len(drop_all)
D = M[M.appl_no.isin(drop_all) & (M.refl <= END)].copy()
# 단위별: 어떤 키가 선언 이전 매칭을 만들었는가
cause = D.groupby("appl_no").src.agg(lambda s: "+".join(sorted(set(s))))
R["drop_cause_counts"] = cause.value_counts().to_dict()
# 이름·출원일 일치
def norm(s): return re.sub(r"[^a-z가-힣]", "", str(s).lower())
alias = {"samsung": ["삼성", "samsung"], "lg": ["엘지", "lg"], "qualcomm": ["퀄컴", "qualcomm"], "oppo": ["오포", "oppo", "광동"], "huawei": ["화웨이", "huawei"],
         "nokia": ["노키아", "nokia"], "ericsson": ["에릭슨", "ericsson"], "interdigital": ["인터디지탈", "interdigital"], "vivo": ["비보", "vivo"], "zte": ["zte", "중흥"],
         "sharp": ["샤프", "sharp"], "ntt": ["엔티티", "ntt", "도꼬모", "도코모"], "apple": ["애플", "apple"], "sony": ["소니", "sony"], "kt": ["케이티", "kt"], "etri": ["한국전자통신", "etri"],
         "xiaomi": ["샤오미", "xiaomi", "베이징"], "panasonic": ["파나소닉", "panasonic"], "fujitsu": ["후지쯔", "후지쓰", "fujitsu"], "nec": ["닛본덴끼", "일본전기", "nec"],
         "motorola": ["모토로라", "motorola"], "blackberry": ["블랙베리", "blackberry", "리서치인모션"], "intel": ["인텔", "intel"], "mediatek": ["미디어텍", "mediatek"]}
def name_ok(comp, names):
    c = norm(comp); n = norm(names)
    for kk, al in alias.items():
        if kk in c or any(a in c for a in al if len(a) > 2):
            return any(a in n for a in al)
    return None
D["name_ok"] = [name_ok(a, b) for a, b in zip(D.comp, D[name_col])]
D["k_appdate"] = dt(D[date_col]); D["e_appdate"] = pd.to_numeric(D.etsi_appdate.astype(str).str[:10].str.replace("-", ""), errors="coerce")
D["date_ok"] = (D.k_appdate == D.e_appdate)
per = D.sort_values("refl").groupby("appl_no").agg(src=("src", lambda s: "+".join(sorted(set(s)))), refl=("refl", "min"), comp=("comp", "first"), names=(name_col, "first"),
                                                     raw_app=("raw_app", "first"), raw_pub=("raw_pub", "first"), name_ok=("name_ok", "max"), date_ok=("date_ok", "max"),
                                                     k_appdate=("k_appdate", "first"), e_appdate=("e_appdate", "first"))
per["y_frozen"] = per.index.map(F.set_index("appl_no").y)
R["drop_units_name_ok"] = per.name_ok.value_counts(dropna=False).astype(int).to_dict().__repr__()
R["drop_units_date_ok"] = int(per.date_ok.sum()); R["drop_units_n"] = int(len(per))
per.reset_index().to_csv(os.path.join(V, "v1_drop_units.csv"), index=False, encoding="utf-8-sig")
# 원 규칙이 놓친 ETSI 원 표기 형태 분포(드롭 단위 app13 행)
a13 = D[D.src == "app13"].raw_app.map(lambda s: re.sub(r"\d", "9", str(s)))
R["app13_raw_shapes"] = a13.value_counts().head(10).to_dict()
p_ = D[D.src == "pubno"].raw_pub.head(5).tolist(); R["pubno_examples"] = p_
R["secs"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(V, "v1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
