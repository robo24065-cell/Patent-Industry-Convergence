# -*- coding: utf-8 -*-
"""대체 배정 S-TXT: '제공 라벨 없이 공개 자료만으로 만들 수 있는 최선의 5자리 배정'.
= 공식 IPC-KSIC 연계표 그룹(2013) + 통계청 KSIC 해설서(50개 세세분류 정의·색인어·예시)와 KR 발명 명칭의 TF-IDF 유사도.
연계표 그룹 안 후보가 1개면 그대로, 2~5개면 해설서 유사도 argmax. KR만 가능(해설서가 한국어).
출력: cache/kr_assign.parquet (appl_no, ksic(제공), cw(연계표 그룹), k_txt)."""
import sys
sys.dont_write_bytecode = True
from r8_common import *
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer

defs = json.load(open(KSIC_DEF, encoding="utf-8"))
dtext = {}
for d in defs:
    parts = [d.get(x) or "" for x in ("codenmKr", "codeIndex", "codeDefine", "codeExample")]
    dtext[d["code"]] = " ".join(parts)
assert set(dtext) == set(KSIC50)
con = duckdb.connect(); con.sql("SET threads=2")
kr = con.sql(f"""SELECT k.appl_no, k.ksic, coalesce(k.title,'') title, c.cw
  FROM '{p('cache','kr.parquet')}' k
  JOIN '{p('cache','rec.parquet')}' r ON r.cc='KR' AND r.appl_no=k.appl_no
  LEFT JOIN '{p('cache','ipc_cw.parquet')}' c ON c.sg=r.sg AND c.mg=r.mg AND c.sc=r.sc""").df()
log("kr rows", len(kr))
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True, min_df=1)
vec.fit(list(dtext.values()) + kr.title.sample(200000, random_state=0).tolist())
D = vec.transform([dtext[k] for k in KSIC50])
kidx = {k: i for i, k in enumerate(KSIC50)}
k_txt = np.array([None] * len(kr), dtype=object)
for g, ks in G2K.items():
    m = (kr.cw == g).values
    if not m.any():
        continue
    if len(ks) == 1:
        k_txt[m] = ks[0]
        continue
    idx = np.where(m)[0]
    cols = [kidx[k] for k in ks]
    for s in range(0, len(idx), 100000):
        ii = idx[s:s + 100000]
        X = vec.transform(kr.title.values[ii])
        S = (X @ D[cols].T).toarray()
        k_txt[ii] = np.array(ks)[S.argmax(1)]
kr["k_txt"] = k_txt
kr[["appl_no", "ksic", "cw", "k_txt"]].to_parquet(p("cache", "kr_assign.parquet"), index=False)
ok = kr.cw.notna()
res = {
    "n_kr": int(len(kr)), "cw_assigned": float(ok.mean()),
    "group_agree(cw==K2G[ksic])": float((kr.cw[ok] == kr.ksic[ok].map(K2G)).mean()),
    "txt_5digit_agree_all": float((kr.k_txt == kr.ksic).mean()),
    "txt_5digit_agree_multi_ksic(제공 라벨이 34개 중 하나)": float((kr.k_txt == kr.ksic)[kr.ksic.isin(MULTI_K)].mean()),
    "txt_5digit_agree_given_group_correct_multi": float((kr.k_txt == kr.ksic)[ok & (kr.cw == kr.ksic.map(K2G)) & kr.ksic.isin(MULTI_K)].mean()),
}
# 그룹이 맞은 경우 그룹 내 무작위 추측의 기대 일치율
gm = ok & (kr.cw == kr.ksic.map(K2G)) & kr.ksic.isin(MULTI_K)
res["random_within_group_expect"] = float(kr.ksic[gm].map(lambda k: 1 / len(G2K[K2G[k]])).mean())
json.dump(res, open(p("p1a_txt_agree.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
log("end")
