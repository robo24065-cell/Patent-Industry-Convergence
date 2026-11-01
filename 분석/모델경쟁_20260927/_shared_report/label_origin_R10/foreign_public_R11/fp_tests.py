# -*- coding: utf-8 -*-
r"""R11 foreign_public — 해외 문서 공개 라벨 vs 제공 해외 라벨의 출원인 등록업종 일치(PREREG.txt).
읽기만: cache/fdoc.parquet(fp_prep.py), label_origin_R10/cache/{kr_lo,pat_firm,firm_reg}.parquet, T3 conc_ksic_map.csv,
        _ideas_R7/cache/kind_linked.parquet, _shared/ext/r7_data_ksic_ksco_revision_link_raw.csv, CORP_APPLICANT.txt
쓰기: res_main.csv, res_sub.csv, res_tests.json (개별 기업명 없음, 집계만)"""
import os, sys, re, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")
MC = os.path.abspath(os.path.join(HERE, r"..\..\..")); BASE = os.path.abspath(os.path.join(MC, r"..\.."))
LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
_T = time.time()
def log(*a): print(*a, f"[{round(time.time()-_T)}s]", flush=True)
SEED = 20261012; NB = 1000
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
S3 = {k[:3] for k in KSIC50}
J = {}

# ---- 해외 문서 × KR 문서
fd = pd.read_parquet(os.path.join(CACHE, "fdoc.parquet")); fd["kr_appl_no"] = fd.kr_appl_no.astype(str)
km = pd.read_csv(os.path.join(MC, r"T3_기술셀부상예측\ESSENTIAL_R7\conc_ksic_map.csv"), dtype=str)[["ksic", "code"]]
CODESET = km.groupby("code").ksic.apply(list).to_dict()
kr = pd.read_parquet(os.path.join(LO, "kr_lo.parquet"), columns=["appl_no", "ay"]); kr["appl_no"] = kr.appl_no.astype(str)
fd = fd.merge(kr.rename(columns={"appl_no": "kr_appl_no"}), on="kr_appl_no", how="left")
log("fdoc", len(fd), fd.cc.value_counts().to_dict())

# ---- 등록업종: NPS·DART(R10 firm_reg), KIND(3자리)
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet")); reg = pd.read_parquet(os.path.join(LO, "firm_reg.parquet"))
nn = lambda s: re.sub(r"[\s,·\.;]", "", str(s)).replace("및", "")
L = pd.read_csv(os.path.join(MC, r"_shared\ext\r7_data_ksic_ksco_revision_link_raw.csv"), encoding="cp949", dtype=str)
L = L[L["분류종류"] == "KSIC"]
names = pd.concat([L[["신분류코드", "신분류항목명"]].set_axis(["code", "nm"], axis=1), L[["구분류코드", "구분류항목명"]].set_axis(["code", "nm"], axis=1)])
names = names[names.code.str.fullmatch(r"\d{2,4}", na=False)]; names["k"] = names.nm.map(nn)
nm2code = names.groupby("k").code.agg(lambda s: s.value_counts().index[0])
K = pd.read_parquet(os.path.join(MC, r"_ideas_R7\cache\kind_linked.parquet"))
K["ind"] = K["업종"].map(nn).map(nm2code); K["jurir"] = K.jurir_no.astype(str).str.replace("-", "").str.strip()
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5: rows.append((p[0].strip(), p[3].replace("-", "").strip()))
CA = pd.DataFrame(rows, columns=["applicantCd", "jurir"]); CA = CA[CA.jurir.str.len() == 13]
KK = K[(K.jurir.str.len() == 13) & K.ind.notna()].merge(CA, on="jurir")
KK["pre"] = KK.listing_date <= "2019-12-31"
KK = pd.concat([KK[KK.pre].sort_values("listing_date", ascending=False), KK[~KK.pre].sort_values("listing_date")]).drop_duplicates("applicantCd")
J["KIND"] = {"kind_rows": int(len(K)), "ind_coded": round(float(K.ind.notna().mean()), 3), "applicants_linked": int(len(KK))}
kind = KK[(KK.ind.str.len() >= 3) & KK.ind.str[:3].isin(S3)][["applicantCd", "ind"]].rename(columns={"ind": "kind"})
kind["kind"] = kind.kind.str[:3]
J["KIND"]["applicants_3digit_in_S3"] = int(len(kind))
reg = reg.merge(kind, on="applicantCd", how="outer")
log("KIND", J["KIND"])

ARMS = ["PROV", "CWEQ", "TXT", "FREE", "FFAM", "PKR"]
PUBLIC = ["CWEQ", "TXT", "FREE", "FFAM"]
COL = {"PROV": "prov", "TXT": "k_txt", "FREE": "k_free", "FFAM": "k_ffam", "PKR": "k_pkr"}

def scores(Z, r, digits):
    out = {}
    for a, c in COL.items():
        out[a] = (Z[c].fillna("").str[:digits] == Z[r]).astype(float).values
    cw = np.zeros(len(Z))
    for code, ks in CODESET.items():
        m = (Z.code == code).values
        if not m.any(): continue
        sub = Z.loc[m, r].values
        cw[m] = np.array([sum(k[:digits] == x for k in ks) / len(ks) for x in sub])
    out["CWEQ"] = cw
    return pd.DataFrame(out, index=Z.index)

rng = np.random.default_rng(SEED)
def evaluate(Z, r, digits, tag, boot=True):
    """Z: 해외문서×KR문서×기업 행. KR 문서별 해외 문서 평균 → 기업 평균."""
    S = scores(Z, r, digits); S["applicantCd"] = Z.applicantCd.values; S["kr"] = Z.kr_appl_no.values
    P = S.groupby(["applicantCd", "kr"])[ARMS].mean()
    g = P.groupby(level=0); T = g.mean(); n = g.size().values.astype(float)
    M = T[ARMS].values; val = M.mean(0); vw = (M * n[:, None]).sum(0) / n.sum()
    res = {"표본": tag, "기준": r, "자리": digits, "기업": len(T), "특허": int(n.sum()), "해외문서쌍": len(Z)}
    for i, a in enumerate(ARMS): res[a] = round(float(val[i]), 4); res[a + "_w"] = round(float(vw[i]), 4)
    pub_idx = [ARMS.index(a) for a in PUBLIC]; best = PUBLIC[int(np.argmax(val[pub_idx]))]; res["BEST"] = best
    if boot:
        idx = rng.integers(0, len(T), (NB, len(T))); B = M[idx].mean(1)
        nb = n[idx]; BW = (M[idx] * nb[:, :, None]).sum(1) / nb.sum(1)[:, None]
        for a in PUBLIC + ["PKR"]:
            i, j = ARMS.index("PROV"), ARMS.index(a); d = B[:, i] - B[:, j]; dw = BW[:, i] - BW[:, j]
            res[f"PROV-{a}"] = round(float(val[i] - val[j]), 4)
            res[f"PROV-{a}_lo"] = round(float(np.percentile(d, 2.5)), 4); res[f"PROV-{a}_hi"] = round(float(np.percentile(d, 97.5)), 4)
            res[f"PROV-{a}_w"] = round(float(vw[i] - vw[j]), 4)
            res[f"PROV-{a}_w_lo"] = round(float(np.percentile(dw, 2.5)), 4); res[f"PROV-{a}_w_hi"] = round(float(np.percentile(dw, 97.5)), 4)
        for k in ("", "_lo", "_hi"): res["PROV-BEST" + k] = res[f"PROV-{best}" + k]
    return res

X = pf.merge(fd, left_on="appl_no", right_on="kr_appl_no", suffixes=("_kr", ""))
X = X.drop(columns=["appl_no_kr"]) if "appl_no_kr" in X.columns else X
log("법인×쌍", len(X))
main, sub = [], []
SRC = {"nps": 5, "dart": 5, "kind": 3}
for src, dg in SRC.items():
    R = reg[["applicantCd", src]].dropna()
    R = R[R[src].isin(KSIC50)] if dg == 5 else R
    Y = X.merge(R.rename(columns={src: "r"}), on="applicantCd")
    # P0 재현(다수결, R10 fam_major와 같은 규칙)
    if dg == 5:
        pri = {"US": 0, "EP": 1, "JP": 2, "CN": 3}
        fm = fd.assign(pr=fd.cc.map(pri)).groupby(["kr_appl_no", "prov"]).agg(c=("prov", "size"), p=("pr", "min")).reset_index()
        fm = fm.sort_values(["kr_appl_no", "c", "p"], ascending=[True, False, True]).drop_duplicates("kr_appl_no").set_index("kr_appl_no").prov
        Q = Y.drop_duplicates(["applicantCd", "kr_appl_no"]).copy(); Q["maj"] = Q.kr_appl_no.map(fm)
        J[f"P0_{src}"] = {"firms": int(Q.applicantCd.nunique()), "patents": int(len(Q)),
                          "prov_majority_firm_avg": round(float((Q.maj == Q.r).astype(float).groupby(Q.applicantCd).mean().mean()), 4)}
        log("P0", src, J[f"P0_{src}"])
    main.append(evaluate(Y, "r", dg, "주: 전체")); log("main", src)
    for cc in ("US", "JP", "EP"):
        sub.append(evaluate(Y[Y.cc == cc], "r", dg, f"국가 {cc}"))
    sub.append(evaluate(Y[Y.ay.between(2015, 2019)], "r", dg, "KR 출원 2015-19"))
    if dg == 5:
        Y3 = Y.assign(r=Y.r.str[:3]); sub.append(evaluate(Y3, "r", 3, "3자리 비교"))
        # 다수결판(CW-eq는 문서 평균 그대로, 나머지 팔은 KR 문서별 다수결)
        Yv = Y.copy(); pri = {"US": 0, "EP": 1, "JP": 2, "CN": 3}; Yv["pr"] = Yv.cc.map(pri)
        for a, c in COL.items():
            t = fd.assign(pr=fd.cc.map(pri), v=fd[c].fillna("NA")).groupby(["kr_appl_no", "v"]).agg(c=("v", "size"), p=("pr", "min")).reset_index()
            t = t.sort_values(["kr_appl_no", "c", "p"], ascending=[True, False, True]).drop_duplicates("kr_appl_no").set_index("kr_appl_no").v
            Yv[c] = Yv.kr_appl_no.map(t).replace("NA", np.nan)
        sub.append(evaluate(Yv, "r", dg, "다수결판"))
    log("sub", src)
M_ = pd.DataFrame(main); S_ = pd.DataFrame(sub)
M_.insert(1, "src", list(SRC));
M_.to_csv(os.path.join(HERE, "res_main.csv"), index=False, encoding="utf-8-sig"); S_.to_csv(os.path.join(HERE, "res_sub.csv"), index=False, encoding="utf-8-sig")

# ---- 판정(PREREG)
def cond_up(r): return r["PROV-BEST"] >= 0.03 and r["PROV-BEST_lo"] > 0
def cond_down(r): return -r["PROV-BEST_hi"] > 0  # BEST−PROV CI 하한 > 0
mm = {s: r for s, r in zip(SRC, main)}
up = [cond_up(mm["nps"]), cond_up(mm["dart"])]; down = cond_down(mm["nps"]) or cond_down(mm["dart"])
kind_ok = mm["kind"]["PROV-BEST"] > 0; kind_n = mm["kind"]["기업"]
if down: verdict = "대체 우위(내림)"
elif all(up) and (kind_ok or kind_n < 50): verdict = "외부 기준 확인(올림)"
elif any(up): verdict = "부분(조건부 유지)"
else: verdict = "같다(조건부 유지)"
J["verdict"] = {"NPS_up": up[0], "DART_up": up[1], "KIND_same_direction": bool(kind_ok), "KIND_firms": int(kind_n), "down": bool(down), "판정": verdict}
json.dump(J, open(os.path.join(HERE, "res_tests.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 80)
cols = ["표본", "기준", "자리", "기업", "특허"] + ARMS + ["BEST", "PROV-BEST", "PROV-BEST_lo", "PROV-BEST_hi"]
print(M_[cols].to_string()); print(S_[cols].to_string()); print(json.dumps(J, ensure_ascii=False, default=str))
log("done")
