# -*- coding: utf-8 -*-
"""C15-T1 verify_0 / v1: 'T4 최종시험 기저율 하락(18.8% vs 개발 ~25%)은 원부 산물이 아니며 공식 통계에도 같은 하락' 반박 시도.
독립 재계산(작업 스크립트·캐시 미사용):
 O  공식 연보 1.7 존속·1.5 등록 -> X_y=I_{y-1}+R_y-I_y, 연 이탈률, 3년 누적(창 y..y+2)
 P1 원부 T4형 레이블(납부기록 DE=2025-08-22 기준 paid-thru T in (L,L+3y], T<만료) 기준일 L=YYYY-01-01 2010..2022, 전체 '10' 존속
 P2 연령(등록 후 경과연수) 표준화 레이블률
 P3 실효연도별 불납 소멸 사건 수(소멸등록 사유 불납, E=rd+마지막 납부 연차)
사용: python -I v1_baserate.py  (sqlite mode=ro)"""
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
t0 = time.time(); sys.stdout.reconfigure(encoding="utf-8")
V = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
U = os.path.join(BASE, r"분석\검증루프_20261012\C13\C13-T1\download\moip_unzip\f_437_1")
R = {}
# ---------- O 공식
f1 = [os.path.join(U, x) for x in os.listdir(U) if "총괄" in x][0]
xl = pd.read_excel(f1, sheet_name=None, header=None)
def series(key):
    sh = [s for s in xl if key in s.replace(" ", "")][0]
    d = xl[sh]; d = d[pd.to_numeric(d[0], errors="coerce").between(1940, 2030) & pd.to_numeric(d[1], errors="coerce").notna()]
    return sh, pd.Series(pd.to_numeric(d[1]).values, index=pd.to_numeric(d[0]).astype(int).values)
shI, I = series("연도별존속권리"); shR, Rg = series("1.5.연도별등록")
R["sheets"] = [shI, shR]
X = {y: I[y-1] + Rg[y] - I[y] for y in range(2008, 2026)}
orate = {y: X[y] / I[y-1] for y in X}
R["official_X"] = {y: int(X[y]) for y in X}; R["official_rate"] = {y: round(orate[y], 5) for y in orate}
R["official_3yr"] = {y: round(1 - np.prod([1 - orate[t] for t in (y, y+1, y+2)]), 5) for y in range(2008, 2024)}
# ---------- 원부
src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
pt = pd.read_sql("select rgst_no, substr(rgstDate,1,8) rd, substr(cndrtExptnDate,1,8) ex from patent where substr(rgst_no,1,2)='10' and length(rgstDate)>=8", src)
pay = pd.read_sql("select rgst_no, max(cast(lastAnnl as integer)) l from pay where substr(rgst_no,1,2)='10' and substr(payDate,1,8) <= '20250822' group by rgst_no", src)
pt = pt.merge(pay, on="rgst_no", how="left"); pt["l"] = pt.l.fillna(0).astype(int)
def addyears(ds, k):
    y = ds.str[:4].astype(int) + k; md = ds.str[4:8]
    leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    md = md.where(~((md == "0229") & ~leap), "0228")
    return pd.to_datetime(y.astype(str) + md, format="%Y%m%d")
pt["rdt"] = pd.to_datetime(pt.rd, format="%Y%m%d"); pt["ext"] = pd.to_datetime(pt.ex, format="%Y%m%d", errors="coerce")
pt["T"] = addyears(pt.rd, pt.l)
R["n10"] = len(pt); R["l0"] = int((pt.l == 0).sum())
rows = []; agerows = []
for Ly in range(2010, 2023):
    L = pd.Timestamp(f"{Ly}-01-01"); L3 = pd.Timestamp(f"{Ly+3}-01-01")
    pop = (pt.rdt < L) & (pt.ext > L) & (pt["T"] > L)
    lab = pop & (pt["T"] <= L3) & (pt["T"] < pt.ext)
    age = ((L - pt.rdt).dt.days // 365).clip(upper=19)
    rows.append(dict(L=Ly, n=int(pop.sum()), rate=float(lab.sum() / pop.sum())))
    g = pd.DataFrame({"a": age[pop], "y": lab[pop].astype(int)}).groupby("a").y.agg(["count", "mean"])
    for a, r in g.iterrows(): agerows.append(dict(L=Ly, a=int(a), n=int(r["count"]), rate=float(r["mean"])))
br = pd.DataFrame(rows); ag = pd.DataFrame(agerows)
# 표준 연령 분포: 개발 원점 L=2012,2014,2016,2018 합동
w = ag[ag.L.isin([2012, 2014, 2016, 2018])].groupby("a").n.sum(); w = w / w.sum()
std = ag.pivot(index="L", columns="a", values="rate").reindex(columns=w.index).fillna(0).mul(w, axis=1).sum(axis=1)
br["rate_agestd"] = br.L.map(std)
br["mean_age"] = br.L.map(ag.assign(an=ag.a * ag.n).groupby("L").an.sum() / ag.groupby("L").n.sum())
br["off3"] = br.L.map(R["official_3yr"])
br["ratio_prov_off"] = br.rate / br.off3
R["baserate_by_L"] = br.round(5).to_dict("records")
ag.to_csv(os.path.join(V, "v1_rate_by_age.csv"), index=False)
dev = br[br.L.isin([2012, 2014, 2016, 2018])]; fin = br[br.L == 2022].iloc[0]
R["summary"] = dict(
    prov_dev_mean=float(dev.rate.mean()), prov_final=float(fin.rate), prov_rel=float(fin.rate / dev.rate.mean() - 1),
    prov_agestd_dev_mean=float(dev.rate_agestd.mean()), prov_agestd_final=float(fin.rate_agestd), prov_agestd_rel=float(fin.rate_agestd / dev.rate_agestd.mean() - 1),
    off3_dev_mean=float(dev.off3.mean()), off3_final=float(fin.off3), off3_rel=float(fin.off3 / dev.off3.mean() - 1),
    corr_prov_off3_2010_2022=float(np.corrcoef(br.rate, br.off3)[0, 1]),
    corr_prov_off3_dev_only=float(np.corrcoef(br[br.L <= 2019].rate, br[br.L <= 2019].off3)[0, 1]))
# ---------- P3 실효연도별 불납 소멸 사건
lap = pd.read_sql("select rgst_no, substr(rgstCsDate,1,8) cd from right where substr(rgst_no,1,2)='10' and rgstCsName='소멸등록' and rgstCsReason in ('등록료불납','연차료불납') and length(rgstCsDate)>=8", src)
pa = pd.read_sql("select rgst_no, cast(lastAnnl as integer) l, substr(payDate,1,8) pd from pay where substr(rgst_no,1,2)='10'", src)
m = lap.merge(pa, on="rgst_no"); m = m[m.pd < m.cd]
lm = m.groupby(["rgst_no", "cd"]).l.max().reset_index().merge(pt[["rgst_no", "rd"]], on="rgst_no")
lm["E"] = addyears(lm.rd, lm.l)
R["np_events_by_Eyear"] = {int(k): int(v) for k, v in lm.E.dt.year.value_counts().sort_index().items() if 2010 <= k <= 2026}
# 연 단위: 실효연도 E 불납 사건 / L=y-01-01 존속(P1 pop)
popn = dict(zip(br.L, br.n))
R["np_rate_by_Eyear"] = {y: round(R["np_events_by_Eyear"].get(y, 0) / popn[y], 5) for y in range(2010, 2023)}
R["secs"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(V, "v1_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
