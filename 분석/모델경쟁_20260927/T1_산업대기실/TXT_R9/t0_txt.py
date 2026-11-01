# -*- coding: utf-8 -*-
"""TXT 배정 누출 제거판(어휘·IDF = 해설서 50 + 출원 ≤2007 KR 명칭). 그룹 = label_validity cw.
출력 cache/kr_txt.parquet: appl_no, appl_year, ksic, grp, cw, k_txt(주), k_txt_r8(R8 원판)."""
from r9_common import *
from sklearn.feature_extraction.text import TfidfVectorizer

defs = json.load(open(KSIC_DEF, encoding="utf-8"))
dtext = {d["code"]: " ".join([d.get(x) or "" for x in ("codenmKr", "codeIndex", "codeDefine", "codeExample")]) for d in defs}
assert set(dtext) == set(KSIC50)
cw = pd.read_parquet(os.path.join(LV, "cache_kr_cw.parquet"), columns=["appl_no", "appl_year", "ksic", "grp", "cw"])
tt = pd.read_parquet(os.path.join(R8, "cache", "kr.parquet"), columns=["appl_no", "title"])
kr = cw.merge(tt, on="appl_no", how="left"); kr["title"] = kr.title.fillna("")
assert len(kr) == len(cw)
log("kr", len(kr), "cw assigned", round(float(kr.cw.notna().mean()), 4))
early = kr[kr.appl_year <= 2007].title
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True, min_df=1)
vec.fit(list(dtext.values()) + early.sample(min(200000, len(early)), random_state=0).tolist())
log("vocab fit on <=2007 titles", min(200000, len(early)), "vocab", len(vec.vocabulary_))
Dm = vec.transform([dtext[k] for k in KSIC50]); kidx = {k: i for i, k in enumerate(KSIC50)}
k_txt = np.array([None] * len(kr), dtype=object)
for g, ks in G2K.items():
    m = (kr.cw == g).values
    if not m.any():
        continue
    if len(ks) == 1:
        k_txt[m] = ks[0]; continue
    idx = np.where(m)[0]; cols = [kidx[k] for k in ks]
    for s in range(0, len(idx), 100000):
        ii = idx[s:s + 100000]
        S = (vec.transform(kr.title.values[ii]) @ Dm[cols].T).toarray()
        k_txt[ii] = np.array(ks)[S.argmax(1)]
kr["k_txt"] = k_txt
r8 = pd.read_parquet(os.path.join(R8, "cache", "kr_assign.parquet"), columns=["appl_no", "k_txt"]).rename(columns={"k_txt": "k_txt_r8"})
kr = kr.merge(r8, on="appl_no", how="left")
kr.drop(columns="title").to_parquet(os.path.join(CACHE, "kr_txt.parquet"), index=False)
ok = kr.cw.notna()
res = {"n_kr": int(len(kr)), "cw_assigned": float(ok.mean()),
       "group_agree": float((kr.cw[ok] == kr.grp[ok]).mean()),
       "txt5_agree_all(미배정=불일치)": float((kr.k_txt == kr.ksic).mean()),
       "txt5_agree_multi34": float((kr.k_txt == kr.ksic)[kr.ksic.isin(MULTI_K)].mean()),
       "txt5_agree_given_group_correct_multi34": float((kr.k_txt == kr.ksic)[ok & (kr.cw == kr.grp) & kr.ksic.isin(MULTI_K)].mean()),
       "random_within_group_expect": float(kr.ksic[ok & (kr.cw == kr.grp) & kr.ksic.isin(MULTI_K)].map(lambda k: 1 / len(G2K[K2G[k]])).mean()),
       "cw5_agree_all(16개만 지목)": float((kr.cw.map(UNIQ) == kr.ksic).mean()),
       "txt_r8_5_agree_all": float((kr.k_txt_r8 == kr.ksic).mean()),
       "txt_v2_vs_r8_same": float((kr.k_txt == kr.k_txt_r8)[kr.k_txt.notna()].mean())}
json.dump(res, open(os.path.join(RES, "t0_txt_agree.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)); log("end")
