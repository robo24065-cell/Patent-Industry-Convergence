# -*- coding: utf-8 -*-
"""verify_0 / v1: C11-T1 new_result(사업자번호 결측 = 2013년 단절) 반박 시도. 원천 직접 재계산.
입력 읽기 전용. 출력 verify_0/v1_out.json 만."""
import os, json, sqlite3, collections, math
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
CA_TXT = os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
OUT = os.path.join(BASE, r"분석\검증루프_20261012\C11\C11-T1\verify_0\v1_out.json")
O = {}

# ---- A. 원천 CA 직접 파싱(바이트 → ¶ 분리) ----
raw = open(CA_TXT, "rb").read().decode("utf-8")
lines = raw.split("\n")
hdr = lines[0].strip("\r").split("\u00b6")
rows = [l.strip("\r").split("\u00b6") for l in lines[1:] if l.strip("\r")]
O["hdr"] = hdr; O["rows"] = len(rows)
ca = pd.DataFrame(rows, columns=hdr)
ca.columns = ["cd", "nm", "en", "jur", "biz"]
for c in ["cd", "jur", "biz"]:
    ca[c] = ca[c].str.strip()
ca["jur"] = ca.jur.str.replace(r"\D", "", regex=True)
ca["biz"] = ca.biz.str.replace(r"\D", "", regex=True)
ca["hb"] = ca.biz != ""
ca["hj"] = ca.jur != ""
ca["iy"] = ca.cd.str[1:5]
ca["p1"] = ca.cd.str[:1]
O["biz_blank_all"] = [int((~ca.hb).sum()), round(float((~ca.hb).mean()), 4)]
O["cd_len"] = ca.cd.str.len().value_counts().to_dict()
# 연도별 보유율: 전체 행 / 법인번호 보유 행 / 코드1
def yr(d):
    g = d.groupby("iy").hb.agg(["size", "mean"])
    return {k: [int(r["size"]), round(float(r["mean"]), 4)] for k, r in g.iterrows() if "2000" <= k <= "2026"}
O["by_iy_all"] = yr(ca)
O["by_iy_jur"] = yr(ca[ca.hj])
O["by_iy_p1"] = yr(ca[ca.p1 == "1"])
# 결측 중 ≤2012 발급 비중
miss = ca[~ca.hb]
O["missing_share_le2012"] = round(float((miss.iy <= "2012").mean()), 4)
O["missing_by_p1"] = miss.p1.value_counts().to_dict()
O["missing_ge2013_n"] = int((miss.iy >= "2013").sum())
# 2012·2013 발급 코드 안에서 일련번호(6–11자리) 순서별 보유율(단절 위치 확인)
for y in ["2011", "2012", "2013"]:
    d = ca[(ca.iy == y) & (ca.p1 == "1")].copy()
    d["ser"] = pd.to_numeric(d.cd.str[5:11], errors="coerce")
    d = d.sort_values("ser")
    q = pd.qcut(d.ser.rank(method="first"), 10, labels=False)
    O[f"serial_decile_{y}"] = [[int(d.ser[q == i].min()), int(d.ser[q == i].max()), round(float(d.hb[q == i].mean()), 3)] for i in range(10)]

# ---- B. 등록원부에서 국내 법인 출원인 직접 구성 ----
con = sqlite3.connect("file:" + REG.replace("\\", "/") + "?mode=ro", uri=True)
A = pd.read_sql("select a.rgst_no, a.applicantCd cd, a.applicantNatl natl, a.applicantAddr addr, p.ksic, substr(p.applDate,1,4) ay from applicant a join patent p on a.rgst_no=p.rgst_no", con)
A["cd"] = A.cd.fillna("").astype(str).str.strip(); A = A[A.cd != ""]
A["ay"] = pd.to_numeric(A.ay, errors="coerce")
nco = A.groupby("rgst_no").cd.nunique(); A["co"] = A.rgst_no.map(nco).ge(2).astype(int)
g = A.groupby("cd")
pan = pd.DataFrame({"first": g.ay.min(), "last": g.ay.max(), "npat": g.rgst_no.nunique(), "co": g.co.mean(),
                    "natl": g.natl.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)})
kk = A.groupby(["cd", "ksic"]).size().rename("n").reset_index().sort_values(["cd", "n"], ascending=[True, False]).drop_duplicates("cd").set_index("cd").ksic
pan["ksic"] = kk
U = pan[(pan.index.str[:1] == "1") & (pan.natl == "대한민국")].copy()
cab = ca.set_index("cd")
U["L"] = cab.hb.reindex(U.index).fillna(False).astype(bool)
U["iy"] = U.index.str[1:5]
O["U_n"] = len(U); O["U_L1"] = int(U.L.sum()); O["U_L0"] = int((~U.L).sum())
def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float((a.mean() - b.mean()) / math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
def tvd(a, b):
    pa = pd.Series(a).fillna("NA").value_counts(normalize=True); pb = pd.Series(b).fillna("NA").value_counts(normalize=True)
    ix = pa.index.union(pb.index); return float(0.5 * (pa.reindex(ix, fill_value=0) - pb.reindex(ix, fill_value=0)).abs().sum())
a, b = U[U.L], U[~U.L]
O["smd"] = dict(first=round(smd(a["first"], b["first"]), 3), last=round(smd(a["last"], b["last"]), 3), logpat=round(smd(np.log1p(a.npat), np.log1p(b.npat)), 3),
                co=round(smd(a.co, b.co), 3), ksic_tvd=round(tvd(a.ksic, b.ksic), 3))
# 기제 시험: ≤2012 발급 코드에서 마지막 출원연도별 보유율(‘이후 활동 = 갱신’이면 ≥2013 활동 코드가 ~99%여야)
le = U[U.iy <= "2012"]
O["U_le2012_n"] = len(le)
O["U_le2012_by_last"] = {str(k): [int(v.size), round(float(v.mean()), 4)] for k, v in le.groupby(pd.cut(le["last"], [1900, 2004, 2008, 2012, 2014, 2016, 2018, 2020, 2022, 2026])).L}
O["U_le2012_last_ge2013_rate"] = round(float(le[le["last"] >= 2013].L.mean()), 4)
O["U_le2012_last_le2012_rate"] = round(float(le[le["last"] <= 2012].L.mean()), 4)
O["U_ge2013_rate"] = round(float(U[U.iy >= "2013"].L.mean()), 4)
# 2013년 단절이 '발급연도'와 '활동연도' 중 무엇에 붙는가: 같은 마지막 출원연도(2017–2020)에서 발급 2010–2012 vs 2013–2015
for lo, hi in [(2017, 2020), (2021, 2026)]:
    s = U[(U["last"] >= lo) & (U["last"] <= hi)]
    O[f"same_last_{lo}_{hi}"] = {y: [int(v.size), round(float(v.mean()), 4)] for y, v in s.groupby("iy").L if "2008" <= y <= "2016"}
# 연결 편향을 ≥2013 발급 층 안에서만 보면?
s = U[U.iy >= "2013"]
O["smd_within_ge2013"] = dict(n1=int(s.L.sum()), n0=int((~s.L).sum()))
s = U[U.iy <= "2012"]; a2, b2 = s[s.L], s[~s.L]
O["smd_within_le2012"] = dict(n1=len(a2), n0=len(b2), first=round(smd(a2["first"], b2["first"]), 3), last=round(smd(a2["last"], b2["last"]), 3))

# ---- C. N4·N7 동결 예측 층별 재채점(독립 AUC: 순위합) ----
def auc(y, s, w=None):
    y = np.asarray(y, float); s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    df = pd.DataFrame({"s": s, "y": y, "w": w}).groupby("s").apply(lambda d: pd.Series({"p": (d.w * d.y).sum(), "n": (d.w * (1 - d.y)).sum()})).sort_index()
    cn = df.n.cumsum() - df.n
    return float((df.p * (cn + 0.5 * df.n)).sum() / (df.p.sum() * df.n.sum()))
def auc_fast(y, s, w):
    o = np.lexsort((y, s)); s, y, w = s[o], y[o], w[o]
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, w * y); wn = np.bincount(inv, w * (1 - y)); cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
for tr, f in [("N4", r"N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet"), ("N7", r"N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet")]:
    P = pd.read_parquet(os.path.join(MC, f)); P["cd"] = P.applicantCd.astype(str)
    P["iy"] = P.cd.str[1:5]
    O[f"{tr}_iy_min_max"] = [P.iy.min(), P.iy.max()]
    O[f"{tr}_bizno_blank"] = int((P.bizno.astype(str).str.strip() == "").sum())
    for nm, m in [("all", P.iy > ""), ("le2012", P.iy <= "2012"), ("ge2013", P.iy >= "2013")]:
        d = P[m]; y = d.y.values.astype(float)
        a1 = auc_fast(y, d.p_champ.values.astype(float), np.ones(len(d))); a0 = auc_fast(y, d.p_M0.values.astype(float), np.ones(len(d)))
        sk = (a1 - a0) / (1 - a0)
        rng = np.random.default_rng(7); cl, inv = np.unique(d.bizno.astype(str).values, return_inverse=True); K = len(cl); bs = []
        for _ in range(600):
            ww = np.bincount(rng.integers(0, K, K), minlength=K)[inv].astype(float)
            b1 = auc_fast(y, d.p_champ.values.astype(float), ww); b0 = auc_fast(y, d.p_M0.values.astype(float), ww)
            bs.append((b1 - b0) / (1 - b0))
        O[f"{tr}_{nm}"] = dict(n=len(d), pos=int(y.sum()), rate=round(float(y.mean()), 4), auc=round(a1, 4), auc0=round(a0, 4), skill=round(sk, 4),
                               ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
json.dump(O, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
