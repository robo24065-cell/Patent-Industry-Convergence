# -*- coding: utf-8 -*-
"""(e) 보조: N7 동결 예측 V1·V2 재채점(판정 외), N6·T1 노출 수. 출력 cache/s10_out.json, s09_sens.csv에 N7 행 덧붙임 안 함(별도 csv)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("s9", os.path.join(os.path.dirname(__file__), "s09_sens.py"))
# s09 전체 재실행을 피하려고 필요한 함수만 다시 정의
def skill(a, b): return (a - b) / (1 - b)
def wauc(y, s, w=None):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    brk = np.r_[True, s[1:] != s[:-1]]; gid = np.cumsum(brk) - 1
    wp = np.bincount(gid, weights=w * y); wn = np.bincount(gid, weights=w * (~y)); cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
def boot(y, pm, pb, cl, n=1000, seed=0):
    rng = np.random.default_rng(seed); u, inv = np.unique(pd.Series(cl).astype(str).values, return_inverse=True); K = len(u); sk = []
    for _ in range(n):
        ww = rng.multinomial(K, np.full(K, 1.0 / K))[inv].astype(float)
        if (ww * y).sum() == 0 or (ww * (1 - y)).sum() == 0: continue
        sk.append(skill(wauc(y, pm, ww), wauc(y, pb, ww)))
    return [float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))]
P6 = pd.read_parquet(os.path.join(CACHE, "s06b_pairs.parquet")).set_index("cd")
R = pd.read_csv(os.path.join(W, "mislink_sample.csv"), dtype=str)
ev = json.load(open(os.path.join(CACHE, "s06b_evidence.json"), encoding="utf-8"))
R["cd"] = R.idx.astype(int).map({i: e["cd"] for i, e in enumerate(ev)})
lowR = R[R.strata.str.contains(r"(?:^|\|)low(?:$|\|)")]
V1 = set(lowR.cd[lowR.judgment.isin(["U", "M"])]); V2 = set(P6.index[P6.s_indep < 0.5])
ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet")).set_index("cd")
O = {}; ROWS = []
P = pd.read_parquet(os.path.join(MC, r"N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet")); P["applicantCd"] = P.applicantCd.astype(str)
OFF = 0.2856458057686246
for nm, drop in [("S0 공식 재현", set()), ("V1", V1), ("V2", V2)]:
    d = P[~P.applicantCd.isin(drop)]; a1, a0 = wauc(d.y, d.p_champ), wauc(d.y, d.p_M0); sk = skill(a1, a0)
    ci = boot(d.y.values, d.p_champ.values, d.p_M0.values, d.bizno.values)
    ROWS.append(dict(track="N7(보조)", variant=nm, n=len(d), dropped=len(P) - len(d), dropped_pos=int(P.y.sum() - d.y.sum()), skill=round(sk, 5), ci=[round(ci[0], 5), round(ci[1], 5)], delta=round(sk - OFF, 5),
                     threshold=round(0.25 * (0.3640120804312031 - 0.21059617612378412), 5)))
# N6: 사업자번호(bz) 기준 노출 — V2 코드의 사업자번호
v2biz = set(ca.biz.reindex(list(V2)).dropna()) - {""}
N6 = pd.read_parquet(os.path.join(MC, r"N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"))
O["N6_n"] = len(N6); O["N6_bz_in_V2"] = int(N6.bz.astype(str).isin(v2biz).sum()); O["N6_pos_in_V2"] = int(N6.y[N6.bz.astype(str).isin(v2biz)].sum())
# T1: DART 연결(법인번호) 노출 — 등록원부 범위안 특허 ≥3건(원점 2019) 국내 법인 중 CA–DART 연결 의심(이름 유사도 <0.5·사업자번호 불일치·법인번호 체크섬 실패)
U = pd.read_parquet(os.path.join(CACHE, "U.parquet")).set_index("cd")
A = pd.read_parquet(os.path.join(CACHE, "reg_app.parquet"))
c19 = A[(A.ay <= 2019) & (A.cd.str[:1] == "1")].drop_duplicates(["cd", "rgst_no"]).cd.value_counts()
t1 = set(c19[c19 >= 3].index)
dc = ro(DART_DB); D = pd.read_sql("select corp_name, jurir_no, bizr_no from company", dc)
D["jur"] = D.jurir_no.fillna("").str.replace(r"\D", "", regex=True); D["biz"] = D.bizr_no.fillna("").str.replace(r"\D", "", regex=True)
cj = ca[ca.jur != ""].reset_index()[["cd", "jur", "biz", "name", "jur_ok"]].merge(D, on="jur", suffixes=("", "_d"))
cj["s2"] = [sim(a, b) for a, b in zip(cj.name, cj.corp_name)]
bad = cj[(cj.s2 < 0.5) | ((cj.biz != "") & (cj.biz_d.str.len() == 10) & (cj.biz != cj.biz_d)) | (cj.jur_ok == False)]
O["T1_approx_firms(범위안≥3, ≤2019)"] = len(t1); O["T1_dart_linked"] = int(cj.cd.isin(t1).sum()); O["T1_dart_link_suspect"] = int(bad.cd.isin(t1).sum())
O["T1_in_V2"] = len(t1 & V2)
# CA–DART 이름 유사도 <0.5 사례 성격(상위 15개, 이름만)
O["ca_dart_lowsim_examples"] = cj[cj.s2 < 0.5][["name", "corp_name"]].head(15).values.tolist()
O["ca_dart_lowsim_n"] = int((cj.s2 < 0.5).sum())
pd.DataFrame(ROWS).to_csv(guard_path(os.path.join(CACHE, "s10_n7.csv")), index=False, encoding="utf-8-sig")
json.dump(O, open(guard_path(os.path.join(CACHE, "s10_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(O, ensure_ascii=False, indent=0)); print(pd.DataFrame(ROWS).to_string())
