# -*- coding: utf-8 -*-
"""(e) 오연결 의심 제외 민감도: N4(동결 예측 재채점, C05-T2 lib5 방식) + 근거 A DART5(R12 표본, C06-T1 s3 방식). 공식 성적 불변.
출력 cache/s09_sens.csv, cache/s09_out.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np
# --- C05-T2 lib5 의 wauc·boot 복사(동일 구현) ---
def skill(a, b): return (a - b) / (1 - b)
def wauc(y, s, w=None):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    brk = np.r_[True, s[1:] != s[:-1]]; gid = np.cumsum(brk) - 1
    wp = np.bincount(gid, weights=w * y); wn = np.bincount(gid, weights=w * (~y))
    cn = np.cumsum(wn) - wn
    num = (wp * (cn + 0.5 * wn)).sum(); den = wp.sum() * wn.sum()
    return float(num / den) if den > 0 else np.nan
def boot(y, pm, pb, cl, n=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    u, inv = np.unique(pd.Series(cl).astype(str).values, return_inverse=True); K = len(u); sk = []
    for _ in range(n):
        c = rng.multinomial(K, np.full(K, 1.0 / K)); ww = c[inv].astype(float)
        if (ww * y).sum() == 0 or (ww * (1 - y)).sum() == 0: continue
        sk.append(skill(wauc(y, pm, ww), wauc(y, pb, ww)))
    return [float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))]
ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet")).set_index("cd")
P6 = pd.read_parquet(os.path.join(CACHE, "s06b_pairs.parquet")).set_index("cd")
R = pd.read_csv(os.path.join(W, "mislink_sample.csv"), dtype=str)
ev = json.load(open(os.path.join(CACHE, "s06b_evidence.json"), encoding="utf-8"))
full_cd = {i: e["cd"] for i, e in enumerate(ev)}
R["cd"] = R.idx.astype(int).map(full_cd)
lowR = R[R.strata.str.contains(r"(?:^|\|)low(?:$|\|)")]
V1 = set(lowR.cd[lowR.judgment.isin(["U", "M"])])
V2 = set(P6.index[P6.s_indep < 0.5])
S = pd.read_parquet(os.path.join(EXT, "r3nts_status.parquet")); S["biz"] = S.bizno.astype(str).str.replace(r"\D", "", regex=True)
unreg = set(S.biz[S.b_stt.isna() | (S.b_stt.astype(str) == "")])
qual = set(ca.index[(ca.biz_ok == False) | ca.biz.isin(unreg) | ca.conflict])
V3 = V2 | qual
mer = pd.read_parquet(os.path.join(CACHE, "s07_merger_pairs.parquet"))
V4 = set(mer.cd_prev)
O = {"V1": len(V1), "V2": len(V2), "V3_n4": len(V3), "V4": len(V4)}
ROWS = []
# ---------------- N4 ----------------
P = pd.read_parquet(os.path.join(MC, r"N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet"))
P["applicantCd"] = P.applicantCd.astype(str)
OFF_N4 = (0.1769773096556013, [0.14745614490884804, 0.20857437535030798])
wid = OFF_N4[1][1] - OFF_N4[1][0]; thr = 0.25 * wid
def n4(name, drop):
    m = ~P.applicantCd.isin(drop)
    d = P[m]; a1, a0 = wauc(d.y, d.p_champ), wauc(d.y, d.p_M0); sk = skill(a1, a0)
    ci = boot(d.y.values, d.p_champ.values, d.p_M0.values, d.bizno.values, 1000, 0)
    ROWS.append(dict(track="N4", variant=name, n=int(m.sum()), dropped=int((~m).sum()), dropped_pos=int(P.y[~m].sum()), auc=round(a1, 5), auc_base=round(a0, 5),
                     metric="skill", value=round(sk, 5), ci_lo=round(ci[0], 5), ci_hi=round(ci[1], 5), delta=round(sk - OFF_N4[0], 5), threshold=round(thr, 5),
                     verdict=("수정 필요" if abs(sk - OFF_N4[0]) >= thr else "영향 없음") if not name.startswith("V3") and not name.startswith("V4") else "보조(판정 외)"))
n4("S0 공식 재현", set())
n4("V1 하위 층 표본 U·M 제외", V1)
n4("V2 하위 층 전체 제외", V2)
n4("V3 보조: V2+번호 품질 플래그", V3)
n4("V4 보조: 합병 소멸 코드 제외", V4)
O["N4_drop_in_test"] = {k: int(P.applicantCd.isin(v).sum()) for k, v in [("V1", V1), ("V2", V2), ("V3", V3), ("V4", V4)]}
# ---------------- 근거 A DART5 (C06-T1 s3 의 build·sample_sets·firm_table·boot 복사) ----------------
C06 = os.path.join(AN, r"검증루프_20261012\C06\C06-T1\cache")
R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924', '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902', '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312', '33301', '33309']
S50 = set(KSIC50); KS5 = np.array(KSIC50 + [""], dtype=object)
X = pd.read_parquet(os.path.join(C06, "docs_U.parquet")); reg = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet"))
Rg = reg[["applicantCd", "dart"]].dropna().rename(columns={"dart": "r"})
Z = X.merge(Rg, on="applicantCd")
Z["lp"] = KS5[Z.PROV.values.astype(int)]; Z["lb"] = KS5[Z["H-mg-3"].values.astype(int)]
Z["rs"] = Z.r.map(lambda c: {c} if c in S50 else set()); Z = Z[Z.rs.map(lambda s: bool(s & S50))].copy()
def firm_table(Z):
    mp = np.array([lp in s for lp, s in zip(Z.lp.values, Z.rs.values)], dtype=float); mb = np.array([lb in s for lb, s in zip(Z.lb.values, Z.rs.values)], dtype=float)
    return pd.DataFrame({"f": Z.applicantCd.values, "P": mp, "B": mb}).groupby("f").agg(P=("P", "mean"), B=("B", "mean"), n=("P", "size"))
def bootA(T, seed=20261012, NB=1000):
    rng = np.random.default_rng(seed); M = T[["P", "B"]].values; nf = len(T)
    v = M.mean(0); idx = rng.integers(0, nf, (NB, nf)); B = M[idx].mean(1); d = B[:, 0] - B[:, 1]
    return nf, int(T.n.sum()), float(v[0] - v[1]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
OFF_A = (0.0731, [0.0505, 0.0965]); thrA = 0.25 * (OFF_A[1][1] - OFF_A[1][0])
dc = ro(DART_DB); D = pd.read_sql("select corp_name, jurir_no, bizr_no from company", dc)
D["jur"] = D.jurir_no.fillna("").str.replace(r"\D", "", regex=True); D["biz"] = D.bizr_no.fillna("").str.replace(r"\D", "", regex=True)
cj = ca[ca.jur != ""].reset_index()[["cd", "jur", "biz", "name", "jur_ok"]].merge(D, on="jur", how="inner", suffixes=("", "_d"))
cj["s2"] = [sim(a, b) for a, b in zip(cj.name, cj.corp_name)]
badA = set(cj.cd[(cj.s2 < 0.5) | ((cj.biz != "") & (cj.biz_d.str.len() == 10) & (cj.biz != cj.biz_d)) | (cj.jur_ok == False)])
V3A = V2 | qual | badA
O["A_badA"] = len(badA)
T0 = firm_table(Z)
def arow(name, drop):
    T = T0[~T0.index.isin(drop)]
    nf, nd, dv, lo, hi = bootA(T)
    ROWS.append(dict(track="근거 A DART5", variant=name, n=nf, dropped=int(T0.index.isin(drop).sum()), dropped_pos=None, auc=None, auc_base=None,
                     metric="제공−공개최강(H-mg-3) 일치율 차", value=round(dv, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5), delta=round(dv - OFF_A[0], 5), threshold=round(thrA, 5),
                     verdict=("수정 필요" if abs(dv - OFF_A[0]) >= thrA else "영향 없음") if name[:2] in ("S0", "V1", "V2") else "보조(판정 외)", docs=nd))
arow("S0 공식 재현", set())
arow("V1 하위 층 표본 U·M 제외", V1)
arow("V2 하위 층 전체 제외", V2)
arow("V3 보조: V2+번호 품질+DART 연결 의심", V3A)
O["A_drop_in_sample"] = {k: int(T0.index.isin(v).sum()) for k, v in [("V1", V1), ("V2", V2), ("V3", V3A)]}
pd.DataFrame(ROWS).to_csv(guard_path(os.path.join(CACHE, "s09_sens.csv")), index=False, encoding="utf-8-sig")
json.dump(O, open(guard_path(os.path.join(CACHE, "s09_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(O, ensure_ascii=False)); print(pd.DataFrame(ROWS).to_string())
