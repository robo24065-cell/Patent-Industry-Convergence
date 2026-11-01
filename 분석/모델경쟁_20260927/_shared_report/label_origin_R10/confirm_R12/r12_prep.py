# -*- coding: utf-8 -*-
r"""R12 confirm — 준비(PREREG.txt). 읽기만: c5_R10 alt_KR, T3 rec_conc, _ideas_R8 kr(명칭), T1 TXT_R9 cache, label_origin_R10 cache.
쓰기: 이 폴더 cache/labels.parquet(평가·개발 문서 × 후보 33개 + 진단), cache/h3all.parquet, res_prep.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
MC = os.path.abspath(os.path.join(HERE, r"..\..\..")); BASE = os.path.abspath(os.path.join(MC, r"..\.."))
f = lambda *p: os.path.join(MC, *p).replace("\\", "/")
_T = time.time()
def log(*a): print(*a, f"[{round(time.time()-_T)}s]", flush=True)
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
assert KSIC50 == sorted(KSIC50)
KI = {k: i for i, k in enumerate(KSIC50)}
R = {}
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
d = con.sql(f"""select a.appl_no, a.appl_year ay, a.ksic_prov, a.code, a.k_free k_free_alt, r.sg
   from read_parquet('{f("_shared_report","fusion_advantage","c5_R10","alt_assign","alt_KR.parquet")}') a
   left join (select appl_no, sg from read_parquet('{f("T3_기술셀부상예측","ESSENTIAL_R7","rec_conc.parquet")}') where cc='KR') r using(appl_no)""").df()
T1C = os.path.join(MC, r"T1_산업대기실\TXT_R9\cache")
d = d.merge(pd.read_parquet(os.path.join(T1C, "kr_txtfree.parquet")), on="appl_no", how="left")
d = d.merge(pd.read_parquet(os.path.join(T1C, "kr_txt.parquet"), columns=["appl_no", "k_txt"]), on="appl_no", how="left")
R["k_free_alt_eq_t1"] = float((d.k_free_alt == d.k_free).mean()); d = d.drop(columns="k_free_alt")
d["sg"] = d.sg.where(d.sg.notna() & (d.sg != ""))
d["mg"] = d.sg.str.split("/").str[0]; d["sc"] = d.sg.str[:4]
log("KR", len(d), R)
kf = d.k_free.map(KI).fillna(-1).astype(int).values; kp = d.ksic_prov.map(KI).fillna(-1).astype(int).values
ay = d.ay.values
IDX = {}
for lvl in ("sg", "mg", "sc"):
    codes, uniq = pd.factorize(d[lvl]); IDX[lvl] = (codes, len(uniq))   # -1 = 결측

# ---- (a) TF-IDF 제목 점수 재계산(t0b_txtfree.py 그대로)
from sklearn.feature_extraction.text import TfidfVectorizer
defs = json.load(open(os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"), encoding="utf-8"))
dtext = {x["code"]: " ".join([x.get(c) or "" for c in ("codenmKr", "codeIndex", "codeDefine", "codeExample")]) for x in defs}
cw = pd.read_parquet(os.path.join(T1C, "kr_txt.parquet"), columns=["appl_no", "appl_year"])
tt = pd.read_parquet(os.path.join(MC, r"_ideas_R8\cache\kr.parquet"), columns=["appl_no", "title"])
kk = cw.merge(tt, on="appl_no", how="left"); kk["title"] = kk.title.fillna("")
early = kk[kk.appl_year <= 2007].title
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True, min_df=1)
vec.fit(list(dtext.values()) + early.sample(min(200000, len(early)), random_state=0).tolist())
Dm = vec.transform([dtext[k] for k in KSIC50])
title = d[["appl_no"]].merge(tt, on="appl_no", how="left").title.fillna("").values
del cw, kk, early
TG = np.where((ay >= 2010) & (ay <= 2019))[0]
S = np.zeros((len(TG), 50), dtype=np.float32)
for s in range(0, len(TG), 100000):
    S[s:s + 100000] = (vec.transform(title[TG[s:s + 100000]]) @ Dm.T).toarray()
R["tfidf_argmax_eq_stored"] = float((S.argmax(1) == kf[TG]).mean()); log("TF-IDF", R["tfidf_argmax_eq_stored"])
del title
USE_TEXT = R["tfidf_argmax_eq_stored"] >= 0.995

# ---- 연계표 코드 마스크
km = pd.read_csv(f("T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)
cmap = km.groupby("code").ksic.apply(list).to_dict()
cc, cu = pd.factorize(d.code)
CM = np.ones((len(cu) + 1, 50), dtype=bool)
for i, c in enumerate(cu):
    if c in cmap: CM[i] = False; CM[i, [KI[k] for k in cmap[c]]] = True
cc = np.where(cc < 0, len(cu), cc)

def counts(lab, cutoff):
    m0 = (ay <= cutoff) & (lab >= 0); out = {}
    for lvl in ("sg", "mg", "sc"):
        codes, n = IDX[lvl]; m = m0 & (codes >= 0)
        C = np.zeros((n + 1, 50), dtype=np.float64); np.add.at(C, (codes[m], lab[m]), 1)
        out[lvl] = C
    out["P0"] = np.bincount(lab[m0], minlength=50) / m0.sum()
    return out

def rows(lvl, ids):
    codes, n = IDX[lvl]; c = codes[ids]; return np.where(c < 0, n, c)

def hard(C, ids, start, nmin):
    lv = ["sg", "mg", "sc"][["sg", "mg", "sc"].index(start):]
    out = np.full(len(ids), -1, dtype=np.int8); todo = np.ones(len(ids), bool)
    for l in lv:
        M = C[l][rows(l, ids)]; n = M.sum(1); ok = todo & (n >= nmin)
        out[ok] = M[ok].argmax(1); todo &= ~ok
    return out

def dirichlet(C, ids, a):
    Psc = (C["sc"][rows("sc", ids)] + a * C["P0"]) / (C["sc"][rows("sc", ids)].sum(1, keepdims=True) + a)
    M = C["mg"][rows("mg", ids)]; Pmg = (M + a * Psc) / (M.sum(1, keepdims=True) + a)
    M = C["sg"][rows("sg", ids)]; return (M + a * Pmg) / (M.sum(1, keepdims=True) + a)

CANDS = ["T0", "T1"] + [f"H-{s}-{n}" for s in ("sg", "mg", "sc") for n in (1, 3, 10, 30)] + [f"D-{a}" for a in (1, 3, 10, 30, 100)]
if USE_TEXT:
    CANDS += [f"C-{a}-{t}" for a in (3, 30) for t in (2, 5, 10, 20, 50)] + [f"G-{a}-{t}" for a in (3, 30) for t in (5, 20)]
R["candidates"] = CANDS
kt = d.k_txt.map(KI).fillna(-1).astype(int).values
parts = []
for tag, cut, lo, hi in (("dev", 2009, 2010, 2014), ("eval", 2014, 2015, 2019)):
    C = counts(kf, cut); CP = counts(kp, cut)
    ids = np.where((ay >= lo) & (ay <= hi))[0]; srow = np.searchsorted(TG, ids)
    L = {"appl_no": d.appl_no.values[ids], "ay": ay[ids], "split": tag, "PROV": kp[ids].astype(np.int8), "code": d.code.values[ids],
         "T0": kf[ids].astype(np.int8), "T1": kt[ids].astype(np.int8)}
    for s in ("sg", "mg", "sc"):
        for n in (1, 3, 10, 30): L[f"H-{s}-{n}"] = hard(C, ids, s, n)
    L["PKR"] = hard(CP, ids, "sg", 3)
    for a in (1, 3, 10, 30, 100):
        out = np.empty(len(ids), np.int8); outC = {t: np.empty(len(ids), np.int8) for t in (2, 5, 10, 20, 50)}; outG = {t: np.empty(len(ids), np.int8) for t in (5, 20)}
        for s in range(0, len(ids), 50000):
            sl = slice(s, s + 50000); P = dirichlet(C, ids[sl], a); out[sl] = P.argmax(1)
            if USE_TEXT and a in (3, 30):
                lp = np.log(P + 1e-12); Sx = S[srow[sl]]; msk = CM[cc[ids[sl]]]
                for t in outC: outC[t][sl] = (lp + t * Sx).argmax(1)
                for t in outG: outG[t][sl] = np.where(msk, lp + t * Sx, -np.inf).argmax(1)
        L[f"D-{a}"] = out
        if USE_TEXT and a in (3, 30):
            for t in outC: L[f"C-{a}-{t}"] = outC[t]
            for t in outG: L[f"G-{a}-{t}"] = outG[t]
    parts.append(pd.DataFrame(L)); log(tag, len(ids))
    if tag == "eval":
        allid = np.where(ay <= 2014)[0]
        pd.DataFrame({"appl_no": d.appl_no.values[allid], "PROV": kp[allid].astype(np.int8), "H3": hard(C, allid, "sg", 3)}).to_parquet(os.path.join(CACHE, "h3all.parquet"), index=False)
LB = pd.concat(parts, ignore_index=True)
LB.to_parquet(os.path.join(CACHE, "labels.parquet"), index=False)
R["doc_agree_with_prov"] = {c: {sp: round(float((g[c] == g.PROV).mean()), 4) for sp, g in LB.groupby("split")} for c in CANDS + ["PKR"]}
json.dump(R, open(os.path.join(HERE, "res_prep.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", len(LB))
