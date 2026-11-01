# -*- coding: utf-8 -*-
r"""C06-T1 verify_0 반박 재계산 r1: '스냅샷 시차가 DART 우위를 만들지 않았다' 주장 검증.
독립 경로: DART 코드를 dart_company.sqlite(ro) + Corporate_20260720\CORP_APPLICANT.txt 법인번호로 직접 재구성(reg12.dart와 대조),
R12 labels.parquet + pat_firm.parquet. 계산: (1) 구간별 제공-BEST·CI 재현 (2) 연도 추세 기울기(기업 부트스트랩) DART vs NPS
(3) 설립 2020+ 제외 (4) 연도별 CI. 쓰기: verify_0\r1_*.json|csv 만."""
import os, sys, json, re, sqlite3
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"; MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12"); LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
KS = np.array(KSIC50 + [""], dtype=object); S50 = set(KSIC50); OUT = {}
# --- 독립 DART 연결
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5: rows.append((p[0].strip(), p[3].replace("-", "").strip()))
CA = pd.DataFrame(rows, columns=["applicantCd", "jurir"]).drop_duplicates("applicantCd")
c = sqlite3.connect("file:" + os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite") + "?mode=ro", uri=True)
D = pd.read_sql("select jurir_no, induty_code, est_dt, corp_cls from company", c); c.close()
D["jurir"] = D.jurir_no.fillna("").str.replace("-", "").str.strip(); D["ind"] = D.induty_code.fillna("").str.strip()
D = D[(D.ind != "") & (D.jurir.str.len() == 13)]
OUT["dart_jurir_conflict_groups"] = int(D.groupby("jurir").ind.nunique().gt(1).sum())
D = D.drop_duplicates("jurir")
L = CA[CA.jurir.str.len() == 13].merge(D[["jurir", "ind", "est_dt", "corp_cls"]], on="jurir")
reg = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet"))
cmp_ = reg[["applicantCd", "dart"]].merge(L[["applicantCd", "ind"]], on="applicantCd", how="outer")
OUT["recon_vs_reg12_dart_agree"] = float(((cmp_.dart == cmp_.ind) | (cmp_.dart.isna() & cmp_.ind.isna())).mean())
LB = pd.read_parquet(os.path.join(R12, "cache", "labels.parquet"), columns=["appl_no", "ay", "split", "PROV", "H-mg-3", "H-sc-1"]); LB["appl_no"] = LB.appl_no.astype(str)
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet")); pf["appl_no"] = pf.appl_no.astype(str); pf["applicantCd"] = pf.applicantCd.astype(str)
def build(R, best):
    Z = pf.merge(LB, on="appl_no").merge(R, on="applicantCd")
    Z["mP"] = (KS[Z.PROV.values.astype(int)] == Z.r.values).astype(float); Z["mB"] = (KS[Z[best].values.astype(int)] == Z.r.values).astype(float)
    Z["d"] = Z.mP - Z.mB; return Z
Rd = L[L.ind.isin(S50)][["applicantCd", "ind", "est_dt", "corp_cls"]].rename(columns={"ind": "r"})
Rn = reg[["applicantCd", "nps"]].dropna().rename(columns={"nps": "r"}); Rn = Rn[Rn.r.isin(S50)]
Z = {"DART5": build(Rd, "H-mg-3"), "NPS5": build(Rn, "H-sc-1")}
def boot_mean(T, B=1000, seed=20261012):
    rng = np.random.default_rng(seed); v = T.values; idx = rng.integers(0, len(v), (B, len(v))); d = v[idx].mean(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
res = []
for k, z in Z.items():
    ev = set(z[z.split == "eval"].applicantCd); dv = set(z[z.split == "dev"].applicantCd); both = ev & dv
    for per in ("dev", "eval"):
        for sc, fs in (("all", None), ("both", both)):
            zz = z[z.split == per] if fs is None else z[(z.split == per) & z.applicantCd.isin(fs)]
            T = zz.groupby("applicantCd").d.mean()
            res.append({"기준": k, "구간": per, "범위": sc, "기업": len(T), "diff": round(float(T.mean()), 4), "ci": boot_mean(T)})
    # 연도별 + CI
    for y in range(2010, 2020):
        T = z[z.ay == y].groupby("applicantCd").d.mean()
        res.append({"기준": k, "구간": str(y), "범위": "year", "기업": len(T), "diff": round(float(T.mean()), 4), "ci": boot_mean(T),
                    "mP": round(float(z[z.ay == y].groupby("applicantCd").mP.mean().mean()), 4), "mB": round(float(z[z.ay == y].groupby("applicantCd").mB.mean().mean()), 4)})
OUT["rows"] = res
# --- 추세 기울기: 기업-연도 평균 d, 연도별 기업평균의 OLS 기울기(10개 점), 기업 클러스터 부트스트랩
def slope_stats(z, years, B=1000, seed=20261012):
    fy = z[z.ay.isin(years)].groupby(["applicantCd", "ay"]).d.mean().reset_index()
    firms = fy.applicantCd.unique(); fidx = {f: i for i, f in enumerate(firms)}
    fy["fi"] = fy.applicantCd.map(fidx); Y = np.array(years)
    # 행렬 형태: firm x year 합과 개수
    S = np.zeros((len(firms), len(Y))); N = np.zeros_like(S); yi = {y: j for j, y in enumerate(Y)}
    np.add.at(S, (fy.fi.values, fy.ay.map(yi).values), fy.d.values); np.add.at(N, (fy.fi.values, fy.ay.map(yi).values), 1)
    def sl(w):
        m = (w[:, None] * S).sum(0) / (w[:, None] * N).sum(0); x = Y - Y.mean(); return float((x * (m - m.mean())).sum() / (x ** 2).sum())
    est = sl(np.ones(len(firms))); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(firms), len(firms)), minlength=len(firms)).astype(float); bs.append(sl(w))
    return est, bs
SL = {}
for nm, yrs in (("2010-19", list(range(2010, 2020))), ("dev 2010-14", list(range(2010, 2015))), ("eval 2015-19", list(range(2015, 2020)))):
    eD, bD = slope_stats(Z["DART5"], yrs); eN, bN = slope_stats(Z["NPS5"], yrs)
    bD, bN = np.array(bD), np.array(bN)
    SL[nm] = {"DART_slope_pp_per_yr": round(eD * 100, 3), "DART_ci": [round(np.percentile(bD, 2.5) * 100, 3), round(np.percentile(bD, 97.5) * 100, 3)],
              "NPS_slope_pp_per_yr": round(eN * 100, 3), "NPS_ci": [round(np.percentile(bN, 2.5) * 100, 3), round(np.percentile(bN, 97.5) * 100, 3)],
              "DART-NPS(독립 부트 근사)": round((eD - eN) * 100, 3), "DD_ci": [round(np.percentile(bD - bN, 2.5) * 100, 3), round(np.percentile(bD - bN, 97.5) * 100, 3)]}
OUT["slopes"] = SL
# --- 설립 2020+ 제외 (eval)
z = Z["DART5"]; ze = z[z.split == "eval"]
est19 = set(Rd[Rd.est_dt.fillna("") > "20191231"].applicantCd) & set(ze.applicantCd)
T = ze[~ze.applicantCd.isin(est19)].groupby("applicantCd").d.mean()
OUT["est2020_excluded"] = {"n_excl": len(est19), "firms": len(T), "diff": round(float(T.mean()), 4), "ci": boot_mean(T)}
# 설립 2015+ 제외(2010-14 문서 기준: 문서 이후 설립 기업 제외)
zd = z[z.split == "dev"]; est14 = set(Rd[Rd.est_dt.fillna("") > "20141231"].applicantCd) & set(zd.applicantCd)
T = zd[~zd.applicantCd.isin(est14)].groupby("applicantCd").d.mean()
OUT["dev_est2015_excluded"] = {"n_excl": len(est14), "firms": len(T), "diff": round(float(T.mean()), 4), "ci": boot_mean(T)}
json.dump(OUT, open(os.path.join(HERE, "r1_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in OUT.items() if k != "rows"}, ensure_ascii=True, default=str))
for r in res: print(json.dumps(r, ensure_ascii=True))
