# -*- coding: utf-8 -*-
"""의미 검정(개발 구간): 국민연금 사업장 고용 신호가 특허 포기(등록료불납 소멸)를 선행하는가?
원점 L ∈ {2017-01-01, 2018-01-01, 2019-01-01} (T4 개발 원점 규칙 L ≤ 2019-01-01, 결과 3년 ≤ 2021-12).
표본: 원점에 살아 있는 KR 등록특허(제공범위) 중 첫 출원인이 사업자번호를 가진 법인.
피처(원점 이전 스냅샷만): 기본 = 등록 후 경과연수, 청구항수, 출원인 포트폴리오 크기(원점 전 등록건)
       NPS = 직전 스냅샷 존재·활성 여부, log(가입자수), 1년 가입자 증감률, 직전 1년 탈퇴, 사업장 업력(최초적용일)
검정: L=2017 학습 → 2018·2019 시험, L=2018 학습 → 2019 시험. 로지스틱 AUC 비교.
출력: firm_nps_test_lapse.csv"""
import os, sqlite3
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
con = sqlite3.connect(f"file:{os.path.join(BASE, r'수집파이프라인/registry_full.sqlite')}?mode=ro", uri=True)
pat = pd.read_sql("select rgst_no, rgstDate, cndrtExptnDate, claimCount, ksic from patent", con)
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName in ('소멸등록','권리의 말소등록')", con)
app = pd.read_sql("select rgst_no, applicantCd from applicant where seq=0", con)
con.close()
lapse = rt[(rt.rgstCsName == "소멸등록") & (rt.rgstCsReason == "등록료불납")].groupby("rgst_no").rgstCsDate.min().rename("lapse_dt")
dead = rt.groupby("rgst_no").rgstCsDate.min().rename("dead_dt")
pat = pat.merge(lapse, on="rgst_no", how="left").merge(dead, on="rgst_no", how="left").merge(app, on="rgst_no", how="left")
link = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
pat = pat[pat.applicantCd.isin(set(link.applicantCd))]  # 사업자번호 확보 법인 출원인
fp = pd.read_parquet(os.path.join(EXT, "firm_nps_firm_panel.parquet"))
SN = {2017: ("2016-12", "2015-12"), 2018: ("2017-12", "2016-12"), 2019: ("2018-12", "2017-12")}


def build(L):
    Ls = f"{L}0101"
    d = pat[(pat.rgstDate < Ls) & (pat.dead_dt.isna() | (pat.dead_dt >= Ls)) & (pat.cndrtExptnDate > f"{L+3}1231")].copy()
    d["y"] = (d.lapse_dt.notna() & (d.lapse_dt <= f"{L+2}1231")).astype(int)
    d["age"] = L - d.rgstDate.str[:4].astype(int)
    d["claims"] = pd.to_numeric(d.claimCount, errors="coerce").fillna(1).clip(upper=50)
    port = pat[pat.rgstDate < Ls].groupby("applicantCd").size().rename("port")
    d = d.merge(port, on="applicantCd", how="left")
    d["lport"] = np.log1p(d.port)
    s1, s0 = SN[L]
    a = fp[fp.snap == s1].set_index("applicantCd")
    b = fp[fp.snap == s0].set_index("applicantCd")
    d["present"] = d.applicantCd.isin(a.index).astype(int)
    d["active"] = d.applicantCd.map(a.active).fillna(False).astype(int)
    e1 = d.applicantCd.map(a.n_emp).fillna(0); e0 = d.applicantCd.map(b.n_emp).fillna(0)
    d["lemp"] = np.log1p(e1)
    d["gemp"] = np.log1p(e1) - np.log1p(e0)
    d["exit1y"] = (d.applicantCd.isin(b.index) & (d.active == 0)).astype(int)
    fa = pd.to_numeric(d.applicantCd.map(a.first_adpt).str[:4], errors="coerce")
    d["firm_age"] = (L - fa).clip(lower=0).fillna(0)
    d["L"] = L
    return d


BASEF = ["age", "claims", "lport"]
NPSF = ["present", "active", "lemp", "gemp", "exit1y", "firm_age"]
D = {L: build(L) for L in SN}
for L, d in D.items():
    print(L, len(d), "lapse율", round(d.y.mean(), 4), "NPS 존재율", round(d.present.mean(), 3))
    q = pd.qcut(d.loc[d.present == 1, "gemp"].rank(method="first"), 5, labels=False)
    print("  고용증감 5분위별 포기율", d.loc[d.present == 1].groupby(q).y.mean().round(4).tolist(),
          "| 탈퇴1년", round(d.loc[d.exit1y == 1, "y"].mean(), 4), f"(n={int(d.exit1y.sum())})",
          "| NPS 미존재", round(d.loc[d.present == 0, "y"].mean(), 4))
rows = []
for tr, tes in ((2017, (2018, 2019)), (2018, (2019,))):
    for name, F in (("base", BASEF), ("base+nps", BASEF + NPSF), ("nps_only", NPSF)):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        mdl.fit(D[tr][F], D[tr].y)
        for te in tes:
            auc = roc_auc_score(D[te].y, mdl.predict_proba(D[te][F])[:, 1])
            rows.append((tr, te, name, round(auc, 4), len(D[te])))
            print(tr, "->", te, name, round(auc, 4))
R = pd.DataFrame(rows, columns=["train_L", "test_L", "features", "auc", "n_test"])
R.to_csv(os.path.join(EXT, "firm_nps_test_lapse.csv"), index=False, encoding="utf-8-sig")
