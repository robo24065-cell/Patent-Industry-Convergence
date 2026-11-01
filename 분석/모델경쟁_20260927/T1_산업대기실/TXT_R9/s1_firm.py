# -*- coding: utf-8 -*-
"""[1] 기업 등록업종 일치: 기업 특허 KSIC 구성(제공 P / 연계표 CW / 해설서 텍스트 TXT, 민감도 TXT-R8) ↔ NPS(원점, 별칭)·DART·KIND.
label_validity s2_firm_agree.py 확장(NPS 연결·업종명 사전 규칙 그대로 복사). 사용: python s1_firm.py 2019 | 2015
출력: res/s1_firm_o<orig>.csv, res/s1_meta_o<orig>.json"""
from r9_common import *
import glob

ORIG = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
SNAP = f"{ORIG}12"
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def nname(s):
    return re.sub(r"[\s·ㆍ,\.\(\)]", "", str(s or ""))


nm2code = {}
for f in sorted(glob.glob(os.path.join(BASE, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    d = pd.read_csv(f, encoding="cp949", dtype=str, usecols=[2, 3]); d.columns = ["code", "name"]
    d = d[d.code.str.match(r"^[A-Z]\d{5}$", na=False)]
    for c, n in zip(d.code, d.name):
        nm2code.setdefault(nname(n), c[1:])
for r in json.load(open(KSIC_DEF, encoding="utf-8")):
    nm2code.setdefault(nname(r["codenmKr"]), r["code"])

# ---- 특허(세 배정 공통 집합: 연계표 배정 있음)
A = pd.read_parquet(os.path.join(CACHE, "kr_txt.parquet")).merge(pd.read_parquet(os.path.join(CACHE, "kr_txtfree.parquet")), on="appl_no", how="left")
af = pd.read_parquet(os.path.join(MC, r"N4_특허행태_기업폐업조기경보\cache\app_firm.parquet"), columns=["applicantCd", "appl_no", "src", "ay"])
af = af[(af.src == "registry") & af.applicantCd.astype(str).str.startswith("1") & (af.ay <= ORIG)]
P = af[["applicantCd", "appl_no"]].drop_duplicates().merge(A, on="appl_no", how="inner")
P = P[P.cw.notna()].copy()
assert P.k_txt.notna().all()
log("법인 특허-출원인 쌍", len(P), "출원인", P.applicantCd.nunique())

# ---- DART
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5:
            rows.append((p[0].strip(), p[3].replace("-", "").strip()))
CA = pd.DataFrame(rows, columns=["applicantCd", "jurir"])
c = ro(os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite"))
Dd = pd.read_sql("select jurir_no, induty_code from company where induty_code<>''", c); c.close()
Dd["jurir"] = Dd.jurir_no.str.replace("-", "").str.strip()
Dd = Dd[Dd.jurir.str.len() == 13].drop_duplicates("jurir")
dart = CA[CA.jurir.str.len() == 13].merge(Dd[["jurir", "induty_code"]], on="jurir")[["applicantCd", "induty_code"]].drop_duplicates("applicantCd")
dart.columns = ["applicantCd", "reg_code"]

# ---- NPS 원점 스냅샷 + 별칭(PROTOCOL_R6 §2)
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname
A2K = dict(zip(lk.applicantCd, lk.key)); Kset = set(A2K.values())
ALL = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
ident = []
for s in ALL:
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "adpt_dt"])
    d["key"] = d.biz6.fillna("") + "|" + d.name.map(norm)
    d["ba"] = d.biz6.fillna("") + "|" + d.adpt_dt.fillna("").astype(str)
    ident.append(d.loc[d.key.isin(Kset) & d.adpt_dt.notna(), ["ba", "key"]].drop_duplicates())
ident = pd.concat(ident).drop_duplicates()
nK = ident.groupby("ba").key.nunique(); ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
W = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{SNAP}.parquet"), columns=["name", "biz6", "adpt_dt", "indcd", "indnm", "n_emp", "stat"])
W["key"] = W.biz6.fillna("") + "|" + W.name.map(norm)
W["ba"] = W.biz6.fillna("") + "|" + W.adpt_dt.fillna("").astype(str)
have = set(W.key)
al = W[W.ba.isin(ident.index)].copy(); al["K"] = al.ba.map(ident)
al = al[(al.key != al.K) & ~al.K.isin(have)]; al["key"] = al.K
W2 = pd.concat([W, al.drop(columns="K")], ignore_index=True)
log("NPS 별칭 추가 사업장", len(al))
W2 = W2.sort_values("n_emp", ascending=False).drop_duplicates("key")
W2["reg_code"] = W2.indnm.map(lambda s: nm2code.get(nname(s)))
nps = pd.DataFrame({"applicantCd": list(A2K.keys()), "key": list(A2K.values())}).merge(W2[["key", "reg_code"]], on="key")[["applicantCd", "reg_code"]]

# ---- KIND (상장일 ≤ 2023-12-31, 업종명→KSIC 10/11차 2–4자리; _ideas_R7 p5_validity 규칙 복사)
nn = lambda s: re.sub(r"[\s,·\.;]", "", str(s)).replace("및", "")
L = pd.read_csv(os.path.join(EXT, "r7_data_ksic_ksco_revision_link_raw.csv"), encoding="cp949", dtype=str)
L = L[L["분류종류"] == "KSIC"]
names = pd.concat([L[["신분류코드", "신분류항목명"]].set_axis(["code", "nm"], axis=1), L[["구분류코드", "구분류항목명"]].set_axis(["code", "nm"], axis=1)])
names = names[names.code.str.fullmatch(r"\d{2,4}", na=False)]; names["k"] = names.nm.map(nn)
nm2c = names.groupby("k").code.agg(lambda s: s.value_counts().index[0])
K = pd.read_parquet(os.path.join(MC, r"_ideas_R7\cache\kind_linked.parquet"))
K["ind_code"] = K["업종"].map(nn).map(nm2c)
E = pd.read_parquet(os.path.join(MC, r"_ideas_R7\cache\ipo_events.parquet")); E["applicantCd"] = E.applicantCd.astype(str)
E = E.merge(K[["종목코드", "listing_date", "ind_code"]].drop_duplicates(["종목코드", "listing_date"]), on=["종목코드", "listing_date"])
E = E[E.listing_date <= "2023-12-31"].dropna(subset=["ind_code"]).sort_values("listing_date")
kind = E.drop_duplicates("applicantCd")[["applicantCd", "ind_code"]].rename(columns={"ind_code": "reg_code"})

# ---- 배정별 열
P["t5_P"] = P.ksic; P["t5_CW"] = P.cw.map(UNIQ); P["t5_TXT"] = P.k_txt; P["t5_TXTR8"] = P.k_txt_r8; P["t5_FREE"] = P.k_free
P["tg_P"] = P.grp; P["tg_CW"] = P.cw; P["tg_TXT"] = P.k_txt.map(K2G); P["tg_TXTR8"] = P.k_txt_r8.map(K2G); P["tg_FREE"] = P.k_free.map(K2G)
P["t3_P"] = P.ksic.str[:3]; P["t3_CW"] = P.cw.map(ksic9_crosswalk_3digit); P["t3_TXT"] = P.k_txt.str[:3]; P["t3_TXTR8"] = P.k_txt_r8.str[:3]; P["t3_FREE"] = P.k_free.str[:3]
ARM = ["P", "CW", "TXT", "TXTR8", "FREE"]
PAIRS = [("P", "TXT"), ("P", "CW"), ("TXT", "CW"), ("P", "TXTR8"), ("P", "FREE"), ("FREE", "TXT")]
rng = np.random.default_rng(0)
out, meta = [], {"origin": ORIG, "nps_snapshot": SNAP}


def summarize(G, label, src, B=1000):
    """G: 기업×(arm 일치율) + n. 기업평균·특허가중 + 쌍별 차이 CI(같은 재추출 공유)."""
    G = G.dropna(subset=[f"m_{a}" for a in ARM])
    nf = len(G)
    if nf < 20:
        return
    M = np.column_stack([G[f"m_{a}"].values for a in ARM]); n = G["n"].values.astype(float)
    idx = rng.integers(0, nf, (B, nf))
    for wt in ("기업평균", "특허가중"):
        if wt == "기업평균":
            val = M.mean(0); bs = M[idx].mean(1)
        else:
            val = (M * n[:, None]).sum(0) / n.sum(); nb = n[idx]
            bs = (M[idx] * nb[:, :, None]).sum(1) / nb.sum(1)[:, None]
        row = {"기준": src, "조건": label, "가중": wt, "기업수": nf, "특허수": int(n.sum())}
        for i, a in enumerate(ARM):
            row[a] = round(float(val[i]), 4)
        for a, b in PAIRS:
            i, j = ARM.index(a), ARM.index(b); d = bs[:, i] - bs[:, j]
            row[f"{a}-{b}"] = round(float(val[i] - val[j]), 4)
            row[f"{a}-{b}_lo"] = round(float(np.percentile(d, 2.5)), 4); row[f"{a}-{b}_hi"] = round(float(np.percentile(d, 97.5)), 4)
        out.append(row)


def firm_tab(sub, pre):
    g = sub.groupby("applicantCd")
    T = pd.DataFrame({"n": g.size()})
    for a in ARM:
        T[f"m_{a}"] = g[f"{pre}_{a}"].mean()
    return T


for src, R in (("NPS", nps), ("DART", dart), ("KIND", kind)):
    R = R.dropna(subset=["reg_code"]).copy()
    R["reg_code"] = R.reg_code.astype(str).str.replace(r"\D", "", regex=True)
    R["rg"] = R.reg_code.map(lambda c: ksic_to_group(c, include_other=True))
    X = P[P.applicantCd.isin(set(R.applicantCd))]
    firms = R[R.applicantCd.isin(set(X.applicantCd))]
    meta[src] = {"firms_with_patents": int(len(firms)), "share_group_in_space": round(float(firms.rg.isin(SPACE).mean()), 4),
                 "share_5digit": round(float((firms.reg_code.str.len() == 5).mean()), 4),
                 "share_5digit_in50": round(float(firms.reg_code.isin(KSIC50).mean()), 4),
                 "share_5digit_in34": round(float(firms.reg_code.isin(MULTI_K).mean()), 4)}
    F = firms[firms.rg.isin(SPACE)][["applicantCd", "reg_code", "rg"]]
    Y = X.merge(F, on="applicantCd")
    Y["r3"] = Y.reg_code.str[:3].where(Y.reg_code.str.len() >= 3)
    Y["r5"] = Y.reg_code.where(Y.reg_code.str.len() == 5)
    for a in ARM:
        Y[f"g_{a}"] = (Y[f"tg_{a}"] == Y.rg).astype(float)
        Y[f"d3_{a}"] = (Y[f"t3_{a}"] == Y.r3).astype(float).where(Y.r3.notna())
        Y[f"d5_{a}"] = (Y[f"t5_{a}"] == Y.r5).astype(float).where(Y.r5.notna())
    summarize(firm_tab(Y, "g"), "(a) 27그룹 [TXT그룹=CW그룹, 정의상 동일]", src)
    # (f2) 사후: FREE용 그룹 내 선택력 — 제공그룹=FREE그룹=등록그룹 (아래 Y5 정의 뒤에서 계산)
    summarize(firm_tab(Y[Y.r3.notna()], "d3"), "(b) 3자리", src)
    if Y.r5.notna().any():
        Y5 = Y[Y.r5.notna()]
        summarize(firm_tab(Y5, "d5"), "(c) 5자리 전체(등록 5자리 기업)", src)
        summarize(firm_tab(Y5[Y5.r5.isin(KSIC50)], "d5"), "(d) 5자리, 등록업종 50개 안 [주]", src)
        summarize(firm_tab(Y5[Y5.r5.isin(MULTI_K)], "d5"), "(e) 5자리, 등록업종 34개(그룹 공유) 안", src)
        Yf = Y5[Y5.r5.isin(MULTI_K) & (Y5.grp == Y5.rg) & (Y5.cw == Y5.rg)]
        summarize(firm_tab(Yf, "d5"), "(f) 그룹 내 선택력(제공그룹=연계표그룹=등록그룹 특허)", src)
        Yf2 = Y5[Y5.r5.isin(MULTI_K) & (Y5.grp == Y5.rg) & (Y5.tg_FREE == Y5.rg) & (Y5.cw == Y5.rg)]
        summarize(firm_tab(Yf2, "d5"), "(f2) 그룹 내 선택력, 제공=연계표=FREE=등록 그룹 특허[사후]", src)
        Yg = Y5[Y5.r5.isin(KSIC50)]
        T = pd.DataFrame({"n": Yg.groupby("applicantCd").size()})
        reg = Yg.drop_duplicates("applicantCd").set_index("applicantCd").r5
        for a in ARM:
            z = Yg.assign(_k=Yg[f"t5_{a}"].fillna("NA")).groupby(["applicantCd", "_k"]).size().reset_index(name="c")
            z = z.sort_values(["c", "_k"], ascending=[False, True]).drop_duplicates("applicantCd").set_index("applicantCd")._k
            T[f"m_{a}"] = (z.reindex(T.index) == reg.reindex(T.index)).astype(float)
        summarize(T, "(g) 기업 최빈 5자리 = 등록업종(50개 안)", src)
        Td = firm_tab(Yg, "d5")
        summarize(Td[Td.n <= 20], "(d2) (d) 중 특허 1~20건 기업", src)
    log(src, meta[src])

O = pd.DataFrame(out)
O.to_csv(os.path.join(RES, f"s1_firm_o{ORIG}.csv"), index=False, encoding="utf-8-sig")
json.dump(meta, open(os.path.join(RES, f"s1_meta_o{ORIG}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(O[O["가중"] == "기업평균"][["기준", "조건", "기업수", "P", "CW", "TXT", "P-TXT", "P-TXT_lo", "P-TXT_hi", "P-CW", "P-CW_lo", "TXT-CW", "P-TXTR8", "FREE", "P-FREE", "P-FREE_lo", "P-FREE_hi"]].to_string())
log("done")
