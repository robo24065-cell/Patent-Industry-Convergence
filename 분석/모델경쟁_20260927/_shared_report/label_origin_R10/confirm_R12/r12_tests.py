# -*- coding: utf-8 -*-
r"""R12 confirm — 근거 A 확인 시험(PREREG.txt). 읽기만: cache/labels.parquet, cache/h3all.parquet, cache/reg12.parquet(이 폴더), label_origin_R10/cache/pat_firm.parquet,
T3 conc_ksic_map.csv, foreign_public_R11/res_posthoc2.csv(재현 점검). 쓰기: res_dev.csv, res_main.csv, res_sub.csv, res_within.csv, res_app.csv, res_tests.json (집계만)"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")
MC = os.path.abspath(os.path.join(HERE, r"..\..\..")); LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
_T = time.time()
def log(*a): print(*a, f"[{round(time.time()-_T)}s]", flush=True)
SEED = 20261012; NB = 1000; rng = np.random.default_rng(SEED)
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
S3 = {k[:3] for k in KSIC50}
KS5 = np.array(KSIC50 + [""], dtype=object); KS3 = np.array([k[:3] for k in KSIC50] + [""], dtype=object)  # 인덱스 -1 = 미배정
prep = json.load(open(os.path.join(HERE, "res_prep.json"), encoding="utf-8")); CANDS = prep["candidates"]
km = pd.read_csv(os.path.join(MC, r"T3_기술셀부상예측\ESSENTIAL_R7\conc_ksic_map.csv"), dtype=str)
CMAP = km.groupby("code").ksic.apply(list).to_dict()
LB = pd.read_parquet(os.path.join(CACHE, "labels.parquet")); LB["appl_no"] = LB.appl_no.astype(str)
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet")); reg = pd.read_parquet(os.path.join(CACHE, "reg12.parquet"))
X = pf.merge(LB, on="appl_no"); log("법인×문서", len(X), X.split.value_counts().to_dict())
CRIT = {"NPS5": ("nps", 5), "DART5": ("dart", 5), "NPS3": ("nps", 3), "DART3": ("dart", 3), "KIND3": ("kind3", 3)}

def sample(split, crit):
    col, dg = CRIT[crit]; Rg = reg[["applicantCd", col, "n_emp"]].dropna(subset=[col]).rename(columns={col: "r"})
    if dg == 5: Rg = Rg[Rg.r.isin(KSIC50)]
    else: Rg = Rg.assign(r=Rg.r.str[:3]); Rg = Rg[Rg.r.isin(S3)]
    return X[X.split == split].merge(Rg, on="applicantCd")

def match(Z, cols, dg):
    KS = KS5 if dg == 5 else KS3; r = Z.r.values; out = {}
    for c in cols: out[c] = (KS[Z[c].values.astype(int)] == r).astype(float)
    cw = np.zeros(len(Z)); codes = Z.code.values
    for code, ks in CMAP.items():
        m = codes == code
        if m.any(): cw[m] = np.array([sum(k[:dg] == x for k in ks) / len(ks) for x in r[m]])
    out["CWEQ"] = cw
    return pd.DataFrame(out)

def firm(Z, A):
    A = A.copy(); A["f"] = Z.applicantCd.values; g = A.groupby("f"); T = g.mean(); n = g.size().values.astype(float)
    return T, n

def boot(T, n, pairs, idx=None):
    M = T.values; cols = list(T.columns); v = M.mean(0); vw = (M * n[:, None]).sum(0) / n.sum()
    if idx is None: idx = rng.integers(0, len(T), (NB, len(T)))
    B = M[idx].mean(1); nb = n[idx]; BW = (M[idx] * nb[:, :, None]).sum(1) / nb.sum(1)[:, None]
    res = {}
    for a, b in pairs:
        i, j = cols.index(a), cols.index(b); dd = B[:, i] - B[:, j]; dw = BW[:, i] - BW[:, j]
        res[f"{a}-{b}"] = (round(float(v[i] - v[j]), 4), round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4))
        res[f"{a}-{b}_w"] = (round(float(vw[i] - vw[j]), 4), round(float(np.percentile(dw, 2.5)), 4), round(float(np.percentile(dw, 97.5)), 4))
    return {c: round(float(x), 4) for c, x in zip(cols, v)}, {c: round(float(x), 4) for c, x in zip(cols, vw)}, res

# ---- 1) 개발 표본에서 BEST 선택
dev_rows, BEST = [], {}
for crit, (col, dg) in CRIT.items():
    Z = sample("dev", crit); A = match(Z, CANDS + ["PROV"], dg); T, n = firm(Z, A); v = T.mean(0)
    sc = [round(float(v[c]), 4) for c in CANDS]; BEST[crit] = CANDS[int(np.argmax(sc))]
    dev_rows.append({"기준": crit, "기업": len(T), "특허": int(n.sum()), "PROV": round(float(v["PROV"]), 4), "BEST": BEST[crit], **dict(zip(CANDS, sc))})
    log("dev", crit, BEST[crit], max(sc), "PROV", round(float(v["PROV"]), 4))
pd.DataFrame(dev_rows).to_csv(os.path.join(HERE, "res_dev.csv"), index=False, encoding="utf-8-sig")

# ---- 2) 평가 표본
J = {"BEST": BEST}; main, sub, within = [], [], []
EV = {}
for crit, (col, dg) in CRIT.items():
    Z = sample("eval", crit).reset_index(drop=True); A = match(Z, CANDS + ["PROV", "PKR"], dg)
    A["BEST"] = A[BEST[crit]]; A["BASE"] = A["H-sg-3"]; T, n = firm(Z, A); EV[crit] = (Z, A)
    pairs = [("PROV", "BEST"), ("PROV", "BASE"), ("PROV", "T0"), ("PROV", "PKR"), ("PROV", "CWEQ")]
    v, vw, res = boot(T, n, pairs)
    posthoc = max(CANDS, key=lambda c: v[c])
    row = {"기준": crit, "기업": len(T), "특허": int(n.sum()), "BEST": BEST[crit], "PROV": v["PROV"], "BEST값": v["BEST"], "BASE(H-sg-3)": v["BASE"],
           "T0(TXT-FREE)": v["T0"], "CWEQ": v["CWEQ"], "PKR(진단)": v["PKR"], "PROV_w": vw["PROV"], "BEST_w": vw["BEST"],
           "사후최강(판정X)": posthoc, "사후최강값": v[posthoc]}
    for k, (e, lo, hi) in res.items(): row[k] = e; row[k + "_lo"] = lo; row[k + "_hi"] = hi
    main.append(row); log("eval", crit, row["PROV"], row["BEST값"], res["PROV-BEST"])
    if dg == 5:
        # (2) 규모·특허 수 구간
        ne = Z.groupby("applicantCd").n_emp.first()
        sb = pd.cut(ne, [-1, 9, 49, 299, 1e12], labels=["<10", "10-49", "50-299", ">=300"]).astype(object).fillna("미상")
        pc = pd.cut(n, [0, 1, 4, 19, 1e12], labels=["1", "2-4", "5-19", ">=20"]).astype(str); pc = pd.Series(pc, index=T.index)
        for kind, lab in (("규모(NPS 가입자)", sb.reindex(T.index)), ("특허 수(평가 표본)", pc)):
            for b in sorted(lab.unique()):
                m = (lab == b).values
                if m.sum() < 20: sub.append({"기준": crit, "구간종류": kind, "구간": b, "기업": int(m.sum()), "비고": "기업<20, 생략"}); continue
                vv, vww, rr = boot(T[m], n[m], [("PROV", "BEST")])
                e, lo, hi = rr["PROV-BEST"]; ew, lw, hw = rr["PROV-BEST_w"]
                sub.append({"기준": crit, "구간종류": kind, "구간": b, "기업": int(m.sum()), "특허": int(n[m].sum()), "PROV": vv["PROV"], "BEST": vv["BEST"],
                            "PROV-BEST": e, "lo": lo, "hi": hi, "PROV-BEST_w": ew, "w_lo": lw, "w_hi": hw})
        # (3) 그룹 안 선택력
        gk = Z.code.map(lambda c: len(CMAP.get(c, [])) if isinstance(c, str) else 0).values
        ing = np.array([isinstance(c, str) and (r in CMAP.get(c, [])) for c, r in zip(Z.code.values, Z.r.values)])
        m = (gk >= 2) & ing
        Zw = Z[m]; Aw = pd.DataFrame({"PROV": A.PROV.values[m], "BEST": A.BEST.values[m], "BASE": A.BASE.values[m], "T1": A.T1.values[m], "RAND": 1.0 / gk[m]})
        Tw, nw = firm(Zw, Aw); vv, vww, rr = boot(Tw, nw, [("PROV", "BEST"), ("PROV", "RAND"), ("BEST", "RAND"), ("PROV", "T1")])
        w = {"기준": crit, "문서": int(m.sum()), "기업": len(Tw), "평가표본 중 비율": round(float(m.mean()), 4), **{k: vv[k] for k in vv}}
        for k, (e, lo, hi) in rr.items():
            if not k.endswith("_w"): w[k] = e; w[k + "_lo"] = lo; w[k + "_hi"] = hi
        within.append(w)
M_ = pd.DataFrame(main); M_.to_csv(os.path.join(HERE, "res_main.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(sub).to_csv(os.path.join(HERE, "res_sub.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(within).to_csv(os.path.join(HERE, "res_within.csv"), index=False, encoding="utf-8-sig")

# ---- 재현 점검 (b)
ph = pd.read_csv(os.path.join(MC, r"_shared_report\label_origin_R10\foreign_public_R11\res_posthoc2.csv"), encoding="utf-8-sig")
ph = ph[ph["표본"].str.startswith("법인")]
rep = {}
for _, r in ph.iterrows():
    mm = M_[M_["기준"] == r["기준"] + "5"].iloc[0]
    rep[r["기준"]] = {"firms": [int(r["기업"]), int(mm["기업"])], "prov": [r["KR제공"], mm["PROV"]], "freeipc": [r["KR_FREE-IPC"], mm["BASE(H-sg-3)"]],
                    "ok": bool(int(r["기업"]) == int(mm["기업"]) and abs(r["KR제공"] - mm["PROV"]) <= 0.0005 and abs(r["KR_FREE-IPC"] - mm["BASE(H-sg-3)"]) <= 0.0005)}
J["repro_R11_posthoc2"] = rep; log("repro", rep)

# ---- (4) 참고: 출원인 사전 최빈
H = pd.read_parquet(os.path.join(CACHE, "h3all.parquet")); H["appl_no"] = H.appl_no.astype(str)
HP = pf.merge(H, on="appl_no")
def fmode(col):
    t = HP[HP[col] >= 0].groupby(["applicantCd", col]).size().rename("c").reset_index().sort_values(["applicantCd", "c", col], ascending=[True, False, True])
    return t.drop_duplicates("applicantCd").set_index("applicantCd")[col]
mP, mH = fmode("PROV"), fmode("H3"); app = []
for crit in ("NPS5", "DART5"):
    Z, A = EV[crit]; has = Z.applicantCd.isin(mP.index).values
    lp = Z.applicantCd.map(mP).fillna(-1).astype(int).values; lh = Z.applicantCd.map(mH).fillna(-1).astype(int).values
    B5 = A.copy(); B5["PROV_APP"] = (KS5[lp] == Z.r.values).astype(float); B5["PUB_APP"] = (KS5[lh] == Z.r.values).astype(float)
    T, n = firm(Z[has], B5[has][["PROV", "BEST", "PROV_APP", "PUB_APP"]].reset_index(drop=True))
    vv, vww, rr = boot(T, n, [("PROV_APP", "PUB_APP"), ("PUB_APP", "PROV"), ("PROV", "BEST")])
    a = {"기준": crit, "기업(사전 특허 있음)": len(T), **vv}
    for k, (e, lo, hi) in rr.items():
        if not k.endswith("_w"): a[k] = e; a[k + "_lo"] = lo; a[k + "_hi"] = hi
    app.append(a)
pd.DataFrame(app).to_csv(os.path.join(HERE, "res_app.csv"), index=False, encoding="utf-8-sig")

# ---- 판정
mm = {r["기준"]: r for r in main}
up = {c: mm[c]["PROV-BEST_lo"] > 0 for c in ("NPS5", "DART5")}; rev = {c: mm[c]["PROV-BEST_hi"] < 0 for c in ("NPS5", "DART5")}
if any(rev.values()): verdict = "불충분(역전)"
elif all(up.values()): verdict = "강함"
elif any(up.values()): verdict = "부분"
else: verdict = "불충분"
J["verdict"] = {"NPS5_CI하한>0": up["NPS5"], "DART5_CI하한>0": up["DART5"], "역전": rev, "KIND3_반대증거": bool(mm["KIND3"]["PROV-BEST_hi"] < 0), "판정": verdict}
json.dump(J, open(os.path.join(HERE, "res_tests.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60)
print(M_[["기준", "기업", "특허", "BEST", "PROV", "BEST값", "BASE(H-sg-3)", "T0(TXT-FREE)", "PKR(진단)", "PROV-BEST", "PROV-BEST_lo", "PROV-BEST_hi",
          "PROV-BEST_w", "PROV-BEST_w_lo", "PROV-BEST_w_hi", "사후최강(판정X)", "사후최강값"]].to_string())
print(pd.DataFrame(sub).to_string()); print(pd.DataFrame(within).to_string()); print(pd.DataFrame(app).to_string())
print(json.dumps(J, ensure_ascii=False, default=str)); log("done")
