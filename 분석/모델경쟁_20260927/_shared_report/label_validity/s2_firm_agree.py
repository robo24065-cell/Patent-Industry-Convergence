# -*- coding: utf-8 -*-
"""2) 기업 단위: 기업 특허의 KSIC 구성(제공 라벨 vs 연계표 배정)이 기업 등록업종(국민연금 사업장 업종, DART induty_code)과 얼마나 맞는가.
- 특허→출원인코드: 등록원부 정확 코드(N4 cache app_firm, src=registry, 읽기만). 법인(코드 1로 시작)만.
- 원점: 출원 ≤ 2019, NPS 2019-12 게시분(자료 2019-11). NPS 연결은 PROTOCOL_R6 §2: 사업장 정체성(사업자번호 앞6+최초적용일)으로 옛 상호 별칭.
- DART induty_code는 2026-08 수집 스냅샷(원점 시점 값이 아님 — 한계로 기록).
- NTIS: 수행기관 업종코드 필드가 없다(ntis_project_raw 컬럼 확인) → 기업 업종 기준으로 쓸 수 없음.
"""
import sys, os, glob, re
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lv_common import *
import json

ORIG = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
SNAP = f"{ORIG}12"
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):  # N4 n4lib.norm 그대로
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
def nname(s):
    return re.sub(r"[\s·ㆍ,\.\(\)]", "", str(s or ""))

# ---- KSIC 이름 사전(KOSIS 광업제조업조사 5자리 + 제공 50개 해설)
nm2code = {}
for f in sorted(glob.glob(os.path.join(BASE, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    d = pd.read_csv(f, encoding="cp949", dtype=str, usecols=[2, 3])
    d.columns = ["code", "name"]
    d = d[d.code.str.match(r"^[A-Z]\d{5}$", na=False)]
    for c, n in zip(d.code, d.name):
        nm2code.setdefault(nname(n), c[1:])
for r in json.load(open(os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"), encoding="utf-8")):
    nm2code.setdefault(nname(r["codenmKr"]), r["code"])
log("KSIC 이름 사전", len(nm2code))

# ---- 특허(제공 라벨 + 연계표 배정)
cw = pd.read_parquet(os.path.join(OUT, "cache_kr_cw.parquet"))
af = pd.read_parquet(os.path.join(MC, r"N4_특허행태_기업폐업조기경보\cache\app_firm.parquet"))
af = af[(af.src == "registry") & af.applicantCd.astype(str).str.startswith("1") & (af.ay <= ORIG)]
P = af[["applicantCd", "appl_no"]].drop_duplicates().merge(cw, on="appl_no", how="inner")
P = P[P.cw.notna()]
log("법인 특허-출원인 쌍", len(P), "특허", P.appl_no.nunique(), "출원인", P.applicantCd.nunique())

# ---- DART
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5:
            rows.append((p[0].strip(), p[3].replace("-", "").strip(), p[4].replace("-", "").strip()))
CA = pd.DataFrame(rows, columns=["applicantCd", "jurir", "bizno"])
c = ro(os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite"))
D = pd.read_sql("select jurir_no, induty_code from company where induty_code<>''", c); c.close()
D["jurir"] = D.jurir_no.str.replace("-", "").str.strip()
D = D[D.jurir.str.len() == 13].drop_duplicates("jurir")
dart = CA[CA.jurir.str.len() == 13].merge(D[["jurir", "induty_code"]], on="jurir")[["applicantCd", "induty_code"]].drop_duplicates("applicantCd")
dart.columns = ["applicantCd", "reg_code"]

# ---- NPS (원점 스냅샷, 별칭 연결)
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
lk = lk[lk.nps_hit]
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
nK = ident.groupby("ba").key.nunique()
ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
W = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{SNAP}.parquet"), columns=["name", "biz6", "adpt_dt", "indcd", "indnm", "n_emp", "stat"])
W["key"] = W.biz6.fillna("") + "|" + W.name.map(norm)
W["ba"] = W.biz6.fillna("") + "|" + W.adpt_dt.fillna("").astype(str)
have = set(W.key)
al = W[W.ba.isin(ident.index)].copy()
al["K"] = al.ba.map(ident)
al = al[(al.key != al.K) & ~al.K.isin(have)]
al["key"] = al.K
W2 = pd.concat([W, al.drop(columns="K")], ignore_index=True)
log("NPS 별칭 추가 사업장", len(al))
W2 = W2.sort_values("n_emp", ascending=False).drop_duplicates("key")  # 기업키별 최대 가입자 사업장 = 주업종
W2["reg_code"] = W2.indnm.map(lambda s: nm2code.get(nname(s)))
nps = pd.DataFrame({"applicantCd": list(A2K.keys()), "key": list(A2K.values())}).merge(W2[["key", "indnm", "reg_code"]], on="key")
log("NPS 연결 출원인", len(nps), "업종명→KSIC 코드 식별률", round(float(nps.reg_code.notna().mean()), 4))

res = {"origin": ORIG, "nps_snapshot": SNAP}
out_rows, firm_tabs = [], {}
rng = np.random.default_rng(0)
for src, R in (("NPS", nps[["applicantCd", "reg_code", "indnm"]]), ("DART", dart)):
    X = P[P.applicantCd.isin(set(R.applicantCd))]
    firms = R[R.applicantCd.isin(set(X.applicantCd))].copy()
    firms["rg"] = firms.reg_code.map(lambda c: ksic_to_group(c, include_other=True))
    nf = len(firms)
    st = {"firms_with_patents": nf,
          "share_code_unknown": round(float(firms.reg_code.isna().mean()), 4),
          "share_group_in_space": round(float(firms.rg.isin(SPACE).mean()), 4),
          "share_manuf_outside_space": round(float(firms.rg.notna().mean() - firms.rg.isin(SPACE).mean()), 4)}
    F = firms[firms.rg.isin(SPACE)][["applicantCd", "reg_code", "rg"]]
    Y = X.merge(F, on="applicantCd")
    Y["r3"] = Y.reg_code.str[:3]; Y["r5"] = Y.reg_code.where(Y.reg_code.str.len() == 5)
    Y["cw3"] = Y.cw.map(ksic9_crosswalk_3digit)
    Y["p_g"] = Y.grp == Y.rg; Y["c_g"] = Y.cw == Y.rg
    Y["p_3"] = Y.ksic.str[:3] == Y.r3; Y["c_3"] = Y.cw3 == Y.r3
    Y["p_5"] = (Y.ksic == Y.r5).where(Y.r5.notna())
    uniq = {g: v[0] for g, v in cw.groupby("grp").ksic.unique().items() if len(v) == 1}
    Y["c_5"] = (Y.cw.map(uniq) == Y.r5).where(Y.r5.notna())
    # 기업별
    G = Y.groupby("applicantCd").agg(n=("appl_no", "size"), p_g=("p_g", "mean"), c_g=("c_g", "mean"),
                                    p_3=("p_3", "mean"), c_3=("c_3", "mean"), p_5=("p_5", "mean"), c_5=("c_5", "mean"))
    mp = Y.groupby(["applicantCd", "grp"]).size().reset_index(name="k").sort_values(["k", "grp"], ascending=[False, True]).drop_duplicates("applicantCd").set_index("applicantCd").grp
    mc = Y.groupby(["applicantCd", "cw"]).size().reset_index(name="k").sort_values(["k", "cw"], ascending=[False, True]).drop_duplicates("applicantCd").set_index("applicantCd").cw
    rgm = F.set_index("applicantCd").rg
    G["p_top1"] = (mp.reindex(G.index) == rgm.reindex(G.index)).astype(float)
    G["c_top1"] = (mc.reindex(G.index) == rgm.reindex(G.index)).astype(float)
    firm_tabs[src] = G
    def boot(sub, fn, B=1000):
        ids = sub.index.values
        return np.percentile([fn(sub.loc[rng.choice(ids, len(ids))]) for _ in range(B)], [2.5, 97.5])
    for lab, pc, cc_, w in (("연계표그룹(27) 특허가중", "p_g", "c_g", True), ("3자리 특허가중", "p_3", "c_3", True),
                           ("5자리 특허가중(등록업종 5자리 기업)", "p_5", "c_5", True),
                           ("연계표그룹(27) 기업평균", "p_g", "c_g", False), ("3자리 기업평균", "p_3", "c_3", False),
                           ("기업 최빈그룹=등록업종(top-1)", "p_top1", "c_top1", False)):
        sub = G[G[pc].notna()]
        if w:
            f = lambda s, a=pc, b=cc_: float(((s[a] - s[b]) * s.n).sum() / s.n.sum())
            pv = float((sub[pc] * sub.n).sum() / sub.n.sum()); cv = float((sub[cc_] * sub.n).sum() / sub.n.sum())
        else:
            f = lambda s, a=pc, b=cc_: float((s[a] - s[b]).mean())
            pv = float(sub[pc].mean()); cv = float(sub[cc_].mean())
        lo, hi = boot(sub, f)
        out_rows.append({"기준": src, "지표": lab, "기업수": len(sub), "특허수": int(sub.n.sum()),
                         "제공라벨_일치율": round(pv, 4), "연계표_일치율": round(cv, 4), "차이(제공-연계표)": round(pv - cv, 4),
                         "차이_95CI_lo": round(float(lo), 4), "차이_95CI_hi": round(float(hi), 4)})
    # 중소 단일 업종 근사: 특허 1~20건
    sm = G[G.n <= 20]
    out_rows.append({"기준": src, "지표": "연계표그룹(27) 기업평균 — 특허 1~20건 기업", "기업수": len(sm), "특허수": int(sm.n.sum()),
                     "제공라벨_일치율": round(float(sm.p_g.mean()), 4), "연계표_일치율": round(float(sm.c_g.mean()), 4),
                     "차이(제공-연계표)": round(float(sm.p_g.mean() - sm.c_g.mean()), 4), "차이_95CI_lo": None, "차이_95CI_hi": None})
    res[src] = st
    log(src, st)

O = pd.DataFrame(out_rows)
O.to_csv(os.path.join(OUT, f"lv2_firm_agreement_o{ORIG}.csv"), index=False, encoding="utf-8-sig")
print(O.to_string())
# 등록업종 그룹별(NPS) 일치율
G = firm_tabs["NPS"]
json.dump(res, open(os.path.join(OUT, f"lv2_meta_o{ORIG}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done")
