# -*- coding: utf-8 -*-
"""[사후 탐색, PREREG 밖] TXT-FREE: 연계표 그룹 제약 없이 50개 해설서 전체와 명칭 TF-IDF argmax(어휘 ≤2007, t0와 같음).
'공개 텍스트만으로 그룹까지 정하면 연계표 그룹보다 나은가'를 본다. 출력 cache/kr_txtfree.parquet(appl_no, k_free)."""
from r9_common import *
from sklearn.feature_extraction.text import TfidfVectorizer

defs = json.load(open(KSIC_DEF, encoding="utf-8"))
dtext = {d["code"]: " ".join([d.get(x) or "" for x in ("codenmKr", "codeIndex", "codeDefine", "codeExample")]) for d in defs}
cw = pd.read_parquet(os.path.join(CACHE, "kr_txt.parquet"))
tt = pd.read_parquet(os.path.join(R8, "cache", "kr.parquet"), columns=["appl_no", "title"])
kr = cw.merge(tt, on="appl_no", how="left"); kr["title"] = kr.title.fillna("")
early = kr[kr.appl_year <= 2007].title
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True, min_df=1)
vec.fit(list(dtext.values()) + early.sample(min(200000, len(early)), random_state=0).tolist())
Dm = vec.transform([dtext[k] for k in KSIC50])
k_free = np.empty(len(kr), dtype=object)
for s in range(0, len(kr), 100000):
    S = (vec.transform(kr.title.values[s:s + 100000]) @ Dm.T).toarray()
    k_free[s:s + 100000] = np.array(KSIC50)[S.argmax(1)]
kr["k_free"] = k_free
kr[["appl_no", "k_free"]].to_parquet(os.path.join(CACHE, "kr_txtfree.parquet"), index=False)
res = {"free5_agree_all": float((kr.k_free == kr.ksic).mean()),
       "free_group_agree_all": float((kr.k_free.map(K2G) == kr.grp).mean()),
       "cw_group_agree_all(미배정=불일치)": float((kr.cw == kr.grp).mean()),
       "txt5_agree_all": float((kr.k_txt == kr.ksic).mean())}
json.dump(res, open(os.path.join(RES, "t0b_txtfree_agree.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res); log("end")
