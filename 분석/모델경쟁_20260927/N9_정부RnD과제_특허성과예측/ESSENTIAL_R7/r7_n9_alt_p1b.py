# -*- coding: utf-8 -*-
"""P1 대체판 민감도(alt_b): NTIS KR 특허 성과의 출원 레코드 + 등록 레코드(등록 레코드는 출원일이 없으므로 성과 Year=등록연도를 시점으로 사용)
중 첫 IPC 연계표 배정이 25개 소분류 안인 것. 제공판은 등록 레코드의 출원연도를 제공 데이터에서 가져오므로, 연계표 쪽에도 등록 레코드를 허용해 공정성을 높인 판.
시점: Year <= v-2 (등록연도 기준이라 출원연도 기준보다 보수적). 산출: alt_p1b_dev.parquet"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); N9 = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, N9)
from r7_concord import assign, first_ipc, G50
from n9_lib import EXT, CACHE, norm_name, log

HIST_MAX = 2019
D = pd.read_parquet(f"{CACHE}/panel_dev.parquet", columns=["pjt_no", "t0", "org", "pre_pjt", "has_pre"])
RAW = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "year", "research_agency", "pre_pjt", "secret_flag"])
P = RAW[(RAW.year >= 2007) & (RAW.year <= 2020) & (RAW.secret_flag.astype(float).fillna(0) == 0)].copy()
P["year"] = P.year.astype(int); P["org"] = P.research_agency.map(norm_name)
P["pre1"] = P.pre_pjt.astype(str).str.split(",").str[0].where(P.pre_pjt.notna())
R = pd.read_parquet(f"{EXT}/ntis_rpatent_raw.parquet", columns=["ProjectID", "Year", "is_kr", "IprType", "RegistType", "RegistNumber", "ipc_codes"])
R["Year"] = pd.to_numeric(R.Year, errors="coerce")
R = R[R.is_kr.fillna(False).astype(bool) & (R.IprType == "특허") & (R.Year <= HIST_MAX)].drop_duplicates(["ProjectID", "RegistNumber"])
Rc0 = R[R.RegistType == "출원"]
g, _ = assign(R.ipc_codes.map(first_ipc).values, "free"); R["c50"] = np.isin(g, G50)
PP = P[["pjt_no", "org", "year"]]
Rc = Rc0.groupby(["ProjectID", "Year"]).size().rename("n").reset_index().rename(columns={"Year": "appl_year"}).merge(PP, left_on="ProjectID", right_on="pjt_no")
H = R[R.c50].groupby(["ProjectID", "Year"]).size().rename("n").reset_index().rename(columns={"Year": "appl_year"}).merge(PP, left_on="ProjectID", right_on="pjt_no")
pre_map = RAW[RAW.year <= 2020].assign(pre1=lambda d: d.pre_pjt.astype(str).str.split(",").str[0].where(d.pre_pjt.notna())).set_index("pjt_no").pre1
pre_map = pre_map[~pre_map.index.duplicated()]
anc = []; cur = P[["pjt_no", "year"]].copy(); cur["anc"] = P.pre1.values; depth = 0
while depth < 15:
    cur = cur[cur.anc.notna() & (cur.anc != cur.pjt_no)]
    if len(cur) == 0:
        break
    depth += 1; anc.append(cur.assign(d=depth)); cur = cur.assign(anc=cur.anc.map(pre_map))
A = pd.concat(anc, ignore_index=True)
out = []
for v in sorted(D.t0.unique()):
    X = D[D.t0 == v].copy(); amax = v - 2
    nproj = P[P.year.between(v - 3, v - 1)].groupby("org").size()
    ra = Rc[Rc.year.between(v - 3, v - 1) & (Rc.appl_year <= amax)].groupby("org").n.sum()
    W = H[H.year.between(v - 3, v - 1) & (H.appl_year <= amax)]
    la = W.groupby("org").n.sum(); lp = W.groupby("org").pjt_no.nunique()
    X["org_p50_L2__altb"] = np.log1p(X.org.map(la).fillna(0))
    X["org_p50_rate_L2__altb"] = X.org.map(lp).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
    X["org_p50_share_L2__altb"] = X.org.map(la).fillna(0) / (X.org.map(ra).fillna(0) + 1)
    lpre = H[H.appl_year <= amax].groupby("ProjectID").n.sum()
    X["pre_p50_L2__altb"] = np.log1p(X.pre_pjt.map(lpre).astype(float).fillna(0)) * X.has_pre
    ch = A[A.year == v].merge(H[H.appl_year <= amax][["ProjectID", "n"]], left_on="anc", right_on="ProjectID").groupby("pjt_no").n.sum()
    X["chain_p50_L2__altb"] = np.log1p(X.pjt_no.map(ch).astype(float).fillna(0))
    out.append(X)
X = pd.concat(out).loc[D.index]
X[["pjt_no", "t0"] + [c for c in X.columns if c.endswith("__altb")]].reset_index(drop=True).to_parquet(os.path.join(HERE, "alt_p1b_dev.parquet"), index=False)
log("saved", (X.pre_p50_L2__altb > 0).mean())
